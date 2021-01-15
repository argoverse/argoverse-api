
import copy
import csv
import pdb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat, quat_argo2scipy
from recordclass import RecordClass
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


class Box3d(RecordClass):
	center: np.ndarray # x, y, z
	width: float
	length: float
	height: float
	orientation_q: Tuple[float,float,float,float] # qw, qx, qy, qz
	name: str
	track_id: str
	timestamp: int
	orientation_yaw: Optional[float] = None
	v0: Optional[Tuple[float,float,float]] = None # vertex 0
	v1: Optional[Tuple[float,float,float]] = None # vertex 1
	v2: Optional[Tuple[float,float,float]] = None # vertex 2
	v3: Optional[Tuple[float,float,float]] = None # vertex 3


def rotmat2quat(R: np.ndarray) -> np.ndarray:
	"""  """
	q_scipy =  Rotation.from_matrix(R).as_quat()
	return quat_scipy2argo(q_scipy)


def quat_scipy2argo(q_scipy: np.ndarray) -> np.ndarray:
	"""Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
	x, y, z, w = q_scipy
	q_argo = np.array([w, x, y, z])
	return q_argo


def quaternion_yaw_scipy(q: Tuple[float,float,float,float]) -> float:
    """ """
    q_argo = q
    q_scipy = quat_argo2scipy(q_argo)
    yaw, _, _ = Rotation.from_quat(q_scipy).as_euler('zyx')
    return yaw




def remove_duplicate_instances(instances: List[Box3d]) -> List[Box3d]:
	"""Remove any duplicate cuboids in ground truth.
	Any ground truth cuboid of the same object class that shares the same centroid
	with another is considered a duplicate instance.
	We first form an (N,N) affinity matrix with entries equal to negative distance.
	We then find rows in the affinity matrix with more than one zero, and
	then for each such row, we choose only the first column index with value zero.
	
	Args:
		instances: array of length (M,), each entry is a Box3d

	Returns:
		array of length (N,) where N <= M, each entry is a unique ObjectLabelRecord
	"""
	if len(instances) == 0:
		return instances
	assert isinstance(instances, np.ndarray)

	# create affinity matrix as inverse distance to other objects
	affinity_matrix = compute_affinity_matrix(copy.deepcopy(instances), copy.deepcopy(instances))

	row_idxs, col_idxs = np.where(affinity_matrix == 0)
	# find the indices where each row index appears for the first time
	unique_row_idxs, unique_element_idxs = np.unique(row_idxs, return_index=True)
	# choose the first instance in each column where repeat occurs
	first_col_idxs = col_idxs[unique_element_idxs]
	# eliminate redundant column indices
	unique_ids = np.unique(first_col_idxs)
	return instances[unique_ids]


def compute_affinity_matrix(dts1: List[Box3d], dts2: List[Box3d]) -> np.ndarray:
	"""Calculate the affinity matrix between detections and ground truth labels,
	using a specified affinity function type.
	
	Args:
		dts1: set of detections (N,).
		dts2: other set of detections (M,).
	
	Returns:
		sims: Affinity scores between detections and ground truth annotations (N, M).
	"""
	dt_centers = np.array([dt.translation for dt in dts])
	gt_centers = np.array([gt.translation for gt in gts])
	sims = -cdist(dt_centers, gt_centers)
	return sims


def construct_argoverse_boxes_lidarfr(
	timestamp: int,
	sweep_labels: List[Dict[str,Any]],
	city_SE3_egovehicle: SE3
) -> List[Box3d]:
	""" Move egovehicle frame boxes, to live in the LiDAR frame instead"""
	# Make list of Box objects including coord system transforms.
	box_list = []
	for label in sweep_labels:

		x = label['center']['x']
		y = label['center']['y']
		z = label['center']['z']

		qw = label['rotation']['w']
		qx = label['rotation']['x']
		qy = label['rotation']['y']
		qz = label['rotation']['z']

		obj_label_record = json_label_dict_to_obj_record(label)
		vertices_egofr = obj_label_record.as_2d_bbox()

		box = Box3d(
			center = [x, y, z], # Argoverse and nuScenes use scalar-first
			width = label['width'],
			length = label['length'],
			height = label['height'],
			orientation_q = [qw, qx, qy, qz],
			name = label['label_class'],
			track_id = label['track_label_uuid'],
			timestamp = timestamp
		)

		# transform box from the egovehicle frame into the city frame
		box.center = city_SE3_egovehicle.transform_point_cloud(np.array(box.center).reshape(1,3)).squeeze()
		box.center = list(box.center)
		box.orientation_q = list(rotmat2quat(city_SE3_egovehicle.rotation @ quat2rotmat(list(box.orientation_q))))
		vertices_cityfr = city_SE3_egovehicle.transform_point_cloud(vertices_egofr)
		box.v0 = list(vertices_cityfr[0])
		box.v1 = list(vertices_cityfr[1])
		box.v2 = list(vertices_cityfr[2])
		box.v3 = list(vertices_cityfr[3])
		box.orientation_yaw = quaternion_yaw_scipy(box.orientation_q)
		box_list.append(box)

	return box_list


def create_tracks_csv(root_path: str, split_subdir: str) -> None:
	""" """
	split_root_path = f'{root_path}/{split_subdir}'
	dl = SimpleArgoverseTrackingDataLoader(data_dir=split_root_path, labels_dir=split_root_path)
	valid_log_ids = dl.sdb.get_valid_logs()
	# loop through all of the logs
	for log_id in valid_log_ids:

		log_row_dictionaries = []
		# for each log, loop through all of the LiDAR sweeps
		log_ply_fpaths = dl.get_ordered_log_ply_fpaths(log_id)
		num_log_sweeps = len(log_ply_fpaths)
		for sample_idx, sample_ply_fpath in enumerate(log_ply_fpaths):
			if sample_idx % 100 == 0:
				print(f'\t{log_id}: On {sample_idx}/{num_log_sweeps}')

			sample_lidar_timestamp = int(Path(sample_ply_fpath).stem.split('_')[-1])
			city_SE3_egovehicle = dl.get_city_SE3_egovehicle(log_id, sample_lidar_timestamp)
			sweep_labels = dl.get_labels_at_lidar_timestamp(log_id, sample_lidar_timestamp)
			boxes = construct_argoverse_boxes_lidarfr(sample_lidar_timestamp, sweep_labels, city_SE3_egovehicle)

			log_row_dictionaries.extend([box.__dict__ for box in boxes])

		csv_fpath = f'{split_subdir}_{log_id}.csv'
		write_csv(csv_fpath, log_row_dictionaries)

def write_csv(csv_fpath: str, dict_list: List[Dict[str,Any]], delimiter='\t'):
	""" """
	with open(csv_fpath, 'w', newline='') as csvfile:

		fieldnames = dict_list[0].keys()
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
		writer.writeheader()
		for row_dict in dict_list:
			writer.writerow(row_dict)


if __name__ == '__main__':

	root_path = '/home/ubuntu/argoverse/argoverse-tracking/val-1.1/argoverse-tracking'
	split_subdir = 'val'
	create_tracks_csv(root_path, split_subdir)

