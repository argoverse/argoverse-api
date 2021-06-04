# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""make util to project RGB values onto the point cloud"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, overload

import imageio
import numpy as np
from typing_extensions import Literal

from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.utils.camera_stats import CAMERA_LIST, RECTIFIED_STEREO_CAMERA_LIST, get_image_dims_for_camera
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

logger = logging.getLogger(__name__)


class CameraConfig(NamedTuple):
    """Camera config for extrinsic matrix, intrinsic matrix, image width/height.
    Args:
        extrinsic: extrinsic matrix
        intrinsic: intrinsic matrix
        img_width: image width
        img_height: image height
    """

    extrinsic: np.ndarray
    intrinsic: np.ndarray
    img_width: int
    img_height: int
    distortion_coeffs: np.ndarray


class Calibration:
    """Calibration matrices and utils.

    3d XYZ are in 3D egovehicle coord.
    2d box xy are in image coord, normalized by width and height
    Point cloud are in egovehicle coord

    ::

       xy_image = K * [R|T] * xyz_ego

       xyz_image = [R|T] * xyz_ego

       image coord:
        ----> x-axis (u)
       |
       |
       v y-axis (v)

    egovehicle coord:
    front x, left y, up z
    """

    def __init__(self, camera_config: CameraConfig, calib: Dict[str, Any]) -> None:
        """Create a Calibration instance.

        Args:
            camera_config: A camera config
            calib: Calibration data
        """
        self.camera_config = camera_config

        self.calib_data = calib

        self.extrinsic = get_camera_extrinsic_matrix(calib["value"])
        self.R = self.extrinsic[0:3, 0:3]
        self.T = self.extrinsic[0:3, 3]

        self.K = get_camera_intrinsic_matrix(calib["value"])

        self.cu = self.calib_data["value"]["focal_center_x_px_"]
        self.cv = self.calib_data["value"]["focal_center_y_px_"]
        self.fu = self.calib_data["value"]["focal_length_x_px_"]
        self.fv = self.calib_data["value"]["focal_length_y_px_"]

        self.bx = self.K[0, 3] / (-self.fu)
        self.by = self.K[1, 3] / (-self.fv)

        self.d = camera_config.distortion_coeffs

        self.camera = calib["key"][10:]

    def cart2hom(self, pts_3d: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to Homogeneous.

        Args:
            pts_3d: nx3 points in Cartesian

        Returns:
            nx4 points in Homogeneous by appending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_ego_to_image(self, pts_3d_ego: np.ndarray) -> np.ndarray:
        """Project egovehicle coordinate to image.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord

        Returns:
            nx3 points in image coord + depth
        """

        uv_cam = self.project_ego_to_cam(pts_3d_ego)
        return self.project_cam_to_image(uv_cam)

    def project_ego_to_cam(self, pts_3d_ego: np.ndarray) -> np.ndarray:
        """Project egovehicle point onto camera frame.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord.

        Returns:
            nx3 points in camera coord.
        """

        uv_cam = self.extrinsic.dot(self.cart2hom(pts_3d_ego).transpose())

        return uv_cam.transpose()[:, 0:3]

    def project_cam_to_ego(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Project point in camera frame to egovehicle frame.

        Args:
            pts_3d_rect: nx3 points in cam coord.

        Returns:
            nx3 points in ego coord.
        """
        return np.linalg.inv((self.extrinsic)).dot(self.cart2hom(pts_3d_rect).transpose()).transpose()[:, 0:3]

    def project_image_to_ego(self, uv_depth: np.ndarray) -> np.ndarray:
        """Project 2D image with depth to egovehicle coordinate.

        Args:
            uv_depth: nx3 first two channels are uv, 3rd channel
               is depth in camera coord. So basically in image coordinate.

        Returns:
            nx3 points in ego coord.
        """
        uv_cam = self.project_image_to_cam(uv_depth)
        return self.project_cam_to_ego(uv_cam)

    def project_image_to_cam(self, uv_depth: np.ndarray) -> np.ndarray:
        """Project 2D image with depth to camera coordinate.

        Args:
            uv_depth: nx3 first two channels are uv, 3rd channel
               is depth in camera coord.

        Returns:
            nx3 points in camera coord.
        """

        n = uv_depth.shape[0]

        x = ((uv_depth[:, 0] - self.cu) * uv_depth[:, 2]) / self.fu + self.bx
        y = ((uv_depth[:, 1] - self.cv) * uv_depth[:, 2]) / self.fv + self.by

        pts_3d_cam = np.zeros((n, 3))
        pts_3d_cam[:, 0] = x
        pts_3d_cam[:, 1] = y
        pts_3d_cam[:, 2] = uv_depth[:, 2]
        return pts_3d_cam

    def project_cam_to_image(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Project camera coordinate to image.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord

        Returns:
            nx3 points in image coord + depth
        """
        uv_cam = self.cart2hom(pts_3d_rect).T
        uv = self.K.dot(uv_cam)
        uv[0:2, :] /= uv[2, :]
        return uv.transpose()


def load_image(img_filename: Union[str, Path]) -> np.ndarray:
    """Load image.

    Args:
        img_filename (str): Image file name

    Returns:
        Image data
    """
    return imageio.imread(img_filename)


def load_calib(calib_filepath: Union[str, Path]) -> Dict[Any, Calibration]:
    """Load Calibration object for all camera from calibration filepath

    Args:
        calib_filepath (str): path to the calibration file

    Returns:
        list of Calibration object for all cameras
    """
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    calib_list = {}
    for camera in CAMERA_LIST:
        cam_config = get_calibration_config(calib, camera)
        calib_cam = next(
            (c for c in calib["camera_data_"] if c["key"] == f"image_raw_{camera}"),
            None,
        )

        if calib_cam is None:
            continue

        calib_ = Calibration(cam_config, calib_cam)
        calib_list[camera] = calib_
    return calib_list


def load_stereo_calib(calib_filepath: Union[str, Path]) -> Dict[Any, Calibration]:
    """Load Calibration object for the rectified stereo cameras from the calibration filepath

    Args:
        calib_filepath (str): path to the stereo calibration file

    Returns:
        list of stereo Calibration object for the rectified stereo cameras
    """
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    calib_list = {}
    for camera in RECTIFIED_STEREO_CAMERA_LIST:
        cam_config = get_calibration_config(calib, camera)
        calib_cam = next(
            (c for c in calib["camera_data_"] if c["key"] == f"image_raw_{camera}"),
            None,
        )

        if calib_cam is None:
            continue

        calib_ = Calibration(cam_config, calib_cam)
        calib_list[camera] = calib_
    return calib_list


def get_camera_extrinsic_matrix(config: Dict[str, Any]) -> np.ndarray:
    """Load camera calibration rotation and translation.

    Note that the camera calibration file contains the SE3 for sensor frame to the vehicle frame, i.e.
        pt_egovehicle = egovehicle_SE3_sensor * pt_sensor

    Then build extrinsic matrix from rotation matrix and translation, a member
    of SE3. Then we return the inverse of the SE3 transformation.

    Args:
       config: Calibration config in json, or calibration file path.

    Returns:
       Camera rotation and translation matrix.
    """
    vehicle_SE3_sensor = config["vehicle_SE3_camera_"]
    egovehicle_t_camera = np.array(vehicle_SE3_sensor["translation"])
    egovehicle_q_camera = vehicle_SE3_sensor["rotation"]["coefficients"]
    egovehicle_R_camera = quat2rotmat(egovehicle_q_camera)
    egovehicle_T_camera = SE3(rotation=egovehicle_R_camera, translation=egovehicle_t_camera)

    return egovehicle_T_camera.inverse().transform_matrix


def get_camera_intrinsic_matrix(camera_config: Dict[str, Any]) -> np.ndarray:
    """Load camera calibration data and constructs intrinsic matrix.

    Args:
       camera_config: Calibration config in json

    Returns:
       Camera intrinsic matrix.
    """
    intrinsic_matrix = np.zeros((3, 4))
    intrinsic_matrix[0, 0] = camera_config["focal_length_x_px_"]
    intrinsic_matrix[0, 1] = camera_config["skew_"]
    intrinsic_matrix[0, 2] = camera_config["focal_center_x_px_"]
    intrinsic_matrix[1, 1] = camera_config["focal_length_y_px_"]
    intrinsic_matrix[1, 2] = camera_config["focal_center_y_px_"]
    intrinsic_matrix[2, 2] = 1.0
    return intrinsic_matrix


def get_calibration_config(calibration: Dict[str, Any], camera_name: str) -> CameraConfig:
    """
    Get calibration config dumped with log.

    Args:
        calibration
        camera_name: name of the camera.

    Returns:
       instance of CameraConfig class
    """
    all_camera_data = calibration["camera_data_"]
    for camera_data in all_camera_data:
        if camera_name in camera_data["key"]:
            camera_calibration = camera_data["value"]
            break
    else:
        raise ValueError(f"Unknown camera name: {camera_name}")

    camera_extrinsic_matrix = get_camera_extrinsic_matrix(camera_calibration)
    camera_intrinsic_matrix = get_camera_intrinsic_matrix(camera_calibration)

    img_width, img_height = get_image_dims_for_camera(camera_name)
    if img_width is None or img_height is None:
        raise ValueError(f"Specified camera has unknown dimensions: {camera_name}")

    return CameraConfig(
        camera_extrinsic_matrix,
        camera_intrinsic_matrix,
        img_width,
        img_height,
        camera_calibration["distortion_coefficients_"],
    )


def point_cloud_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points: Numpy array of shape (N,3)

    Returns:
        Numpy array of shape (N,4)
    """
    num_pts = points.shape[0]
    return np.hstack([points, np.ones((num_pts, 1))])


def remove_nan_values(uv: np.ndarray, uv_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accept corrupt array"""

    uv = uv.T
    uv_cam = uv_cam.T
    x_valid_bool = np.logical_not(np.isnan(uv[:, 0]))
    y_valid_bool = np.logical_not(np.isnan(uv[:, 1]))
    xy_valid_bool = np.logical_and(x_valid_bool, y_valid_bool)

    uv = uv[xy_valid_bool]
    uv_cam = uv_cam[xy_valid_bool]
    return uv.T, uv_cam.T


def determine_valid_cam_coords(uv: np.ndarray, uv_cam: np.ndarray, camera_config: CameraConfig) -> np.ndarray:
    """
    Given a set of coordinates in the image plane and corresponding points
    in the camera coordinate reference frame, determine those points
    that have a valid projection into the image. 3d points with valid
    projections have x coordinates in the range [0,img_width-1], y-coordinates
    in the range [0,img_height-1], and a positive z-coordinate (lying in
    front of the camera frustum).

    Args:
       uv: Numpy array of shape (N,2)
       uv_cam: Numpy array of shape (N,3)
       camera_config: A camera configuration

    Returns:
       Numpy array of shape (N,) with dtype bool
    """
    x_valid = np.logical_and(0 <= uv[:, 0], uv[:, 0] < camera_config.img_width)
    y_valid = np.logical_and(0 <= uv[:, 1], uv[:, 1] < camera_config.img_height)
    z_valid = uv_cam[2, :] > 0
    valid_pts_bool = np.logical_and(np.logical_and(x_valid, y_valid), z_valid)
    return valid_pts_bool


# Make use of typing.overload so that we can robustly type-check the return
# value of this function.
# See https://mypy.readthedocs.io/en/latest/literal_types.html

# These tuples are:
# uv: Numpy array of shape (N,2) with dtype np.float32
# uv_cam: Numpy array of shape (N,3) with dtype np.float32
# valid_pts_bool: Numpy array of shape (N,) with dtype bool
# camera configuration : (only if you asked for it).
_ReturnWithConfig = Tuple[np.ndarray, np.ndarray, np.ndarray, CameraConfig]
_ReturnWithoutConfig = Tuple[np.ndarray, np.ndarray, np.ndarray]


@overload
def project_lidar_to_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    return_camera_config: Literal[True],
    remove_nan: bool = False,
) -> _ReturnWithConfig:
    ...


@overload
def project_lidar_to_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    return_camera_config: Literal[False],
    remove_nan: bool = False,
) -> _ReturnWithoutConfig:
    ...


@overload
def project_lidar_to_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    return_camera_config: bool = False,
    remove_nan: bool = False,
) -> Union[_ReturnWithConfig, _ReturnWithoutConfig]:
    ...


def project_lidar_to_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    return_camera_config: bool = False,
    remove_nan: bool = False,
) -> Union[_ReturnWithConfig, _ReturnWithoutConfig]:
    """
    Args:
        lidar_points_h: Numpy array of shape (4,N)
        calib_data: calibration data
        camera_name: representing name of this camera sensor
        return_camera_config: adds camera config to the return tuple
        remove_nan: filter out nan values from uv and uv_cam

    Returns:
       uv: Numpy array of shape (N,2) with dtype np.float32
       uv_cam: Numpy array of shape (3,N) with dtype np.float32
       valid_pts_bool: Numpy array of shape (N,) with dtype bool
    """
    camera_config = get_calibration_config(calib_data, camera_name)
    uv_cam = camera_config.extrinsic.dot(lidar_points_h)
    uv = camera_config.intrinsic.dot(uv_cam)

    if remove_nan:
        uv, uv_cam = remove_nan_values(uv, uv_cam)

    uv[0:2, :] /= uv[2, :]
    uv = uv.T
    uv = uv[:, :2]
    valid_pts_bool = determine_valid_cam_coords(uv, uv_cam, camera_config)

    if return_camera_config is True:
        return uv, uv_cam, valid_pts_bool, camera_config

    return uv, uv_cam, valid_pts_bool


SMALL_VALUE_THRESHOLD = 1e-9


def proj_cam_to_uv(
    uv_cam: np.ndarray, camera_config: CameraConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CameraConfig]:
    num_points = uv_cam.shape[0]
    uvh = np.zeros((num_points, 3))
    # (x_transformed_m, y_transformed_m, z_transformed_m)

    for idx in range(num_points):
        x_transformed_m = uv_cam[idx, 0]
        y_transformed_m = uv_cam[idx, 1]
        z_transformed_m = uv_cam[idx, 2]

        z_transformed_fixed_m = z_transformed_m

        # If we're behind the camera, z value (homogeneous coord w in image plane)
        # will be negative. If we are on the camera, there would be division by zero
        # later. To prevent that, move an epsilon away from zero.

        Z_EPSILON = 1.0e-4
        if np.absolute(z_transformed_m) <= Z_EPSILON:
            z_transformed_fixed_m = np.sign(z_transformed_m) * Z_EPSILON

        pinhole_x = x_transformed_m / z_transformed_fixed_m
        pinhole_y = y_transformed_m / z_transformed_fixed_m

        K = camera_config.intrinsic
        u_px = K[0, 0] * pinhole_x + K[0, 1] * pinhole_y + K[0, 2]

        v_px = K[1, 1] * pinhole_y + K[1, 2]

        uvh[idx] = np.array([u_px, v_px, z_transformed_m])

    uv = uvh[:, :2]
    valid_pts_bool = determine_valid_cam_coords(uv, uv_cam.T, camera_config)
    return uv, uv_cam.T, valid_pts_bool, camera_config


def distort_single(radius_undist: float, distort_coeffs: List[float]) -> float:
    """
    Calculate distortion for a single undistorted radius.
    Note that we have 3 distortion parameters.

    Args:
        radius_undist: undistorted radius
        distort_coeffs: list of distortion coefficients

    Returns:
        distortion radius
    """
    radius_dist = radius_undist
    r_u_pow = radius_undist
    for distortion_coefficient in distort_coeffs:
        r_u_pow *= radius_undist ** 2
        radius_dist += r_u_pow * distortion_coefficient

    return radius_dist


def project_lidar_to_undistorted_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    remove_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CameraConfig]:
    camera_config = get_calibration_config(calib_data, camera_name)

    R = camera_config.extrinsic[:3, :3]
    t = camera_config.extrinsic[:3, 3]
    cam_SE3_egovehicle = SE3(rotation=R, translation=t)

    points_egovehicle = lidar_points_h.T[:, :3]
    uv_cam = cam_SE3_egovehicle.transform_point_cloud(points_egovehicle)

    return proj_cam_to_uv(uv_cam, camera_config)


# Make use of typing.overload so that we can robustly type-check the return
# value of this function.
# See https://mypy.readthedocs.io/en/latest/literal_types.html

# These tuples are:
# uv: Numpy array of shape (N,2) with dtype np.float32
# uv_cam: Numpy array of shape (N,3) with dtype np.float32
# valid_pts_bool: Numpy array of shape (N,) with dtype bool
# camera configuration : (only if you asked for it).
_ReturnWithOptConfig = Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[CameraConfig],
]
_ReturnWithoutOptConfig = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]


@overload
def project_lidar_to_img_motion_compensated(
    pts_h_lidar_time: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    cam_timestamp: int,
    lidar_timestamp: int,
    dataset_dir: str,
    log_id: str,
    return_K: Literal[True],
) -> _ReturnWithOptConfig:
    ...


@overload
def project_lidar_to_img_motion_compensated(
    pts_h_lidar_time: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    cam_timestamp: int,
    lidar_timestamp: int,
    dataset_dir: str,
    log_id: str,
    return_K: Literal[False],
) -> _ReturnWithoutOptConfig:
    ...


@overload
def project_lidar_to_img_motion_compensated(
    pts_h_lidar_time: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    cam_timestamp: int,
    lidar_timestamp: int,
    dataset_dir: str,
    log_id: str,
    return_K: bool = False,
) -> Union[_ReturnWithOptConfig, _ReturnWithoutOptConfig]:
    ...


def project_lidar_to_img_motion_compensated(
    pts_h_lidar_time: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    cam_timestamp: int,
    lidar_timestamp: int,
    dataset_dir: str,
    log_id: str,
    return_K: bool = False,
) -> Union[_ReturnWithOptConfig, _ReturnWithoutOptConfig]:
    """
    Because of the high frame rate, motion compensation's role between the
    sensors is not very significant, moving points only by millimeters
    to centimeters. If the vehicle is moving at 25 miles per hour, equivalent
    to 11 meters/sec, then in 17 milliseconds (the max time between a lidar sweep
    and camera image capture) we should be able to move up to 187 millimeters.

    This can be verified in practice as the mean_change:
        mean_change = np.amax(pts_h_cam_time.T[:,:3] - pts_h_lidar_time ,axis=0)

    Adjust LiDAR points for ego-vehicle motion. This function accepts the
    egovehicle's pose in the city map both at camera time and also at
    the LiDAR time.

    We perform the following transformation, where "ego" stands for
    egovehicle reference frame

        pt_ego_cam_t = ego_cam_t_SE3_map * map_SE3_ego_lidar_t * pt_ego_lidar_t

    Note that both "cam_time_pts_h" and "lidar_time_pts_h" are 3D points in the
    vehicle coordinate frame, but captured at different times. These LiDAR points
    always live in the vehicle frame, but just in different timestamps. If we take
    a lidar point in the egovehicle frame, captured at lidar time, and bring it into
    the map at this lidar timestamp, then we know the transformation from map to
    egovehicle reference frame at the time when the camera image was captured.

    Thus, we move from egovehicle @ lidar time, to the map (which is time agnostic),
    then we move from map to egovehicle @camera time. Now we suddenly have lidar points
    living in the egovehicle frame @ camera time.

    Args:
        pts_h_lidar_time: Numpy array of shape (4,N)
        calib_data: Python dictionary
        camera_name: string, representing name of camera
        cam_timestamp: integer, representing time in nanoseconds when
           camera image was recorded
        lidar_timestamp: integer, representing time in nanoseconds when
            LiDAR sweep was recorded
        dataset_dir: string, representing path to where dataset is stored
        log_id: string, representing unique ID of vehicle log
        return_K: return a copy of the

    Returns:
        uv: Numpy array of shape (N,2) with dtype np.float32
        uv_cam: Numpy array of shape (N,3) with dtype np.float32
        valid_pts_bool: Numpy array of shape (N,) with dtype bool
    """
    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when camera image was recorded.
    city_SE3_ego_cam_t = get_city_SE3_egovehicle_at_sensor_t(cam_timestamp, dataset_dir, log_id)

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when the LiDAR sweep was recorded.
    city_SE3_ego_lidar_t = get_city_SE3_egovehicle_at_sensor_t(lidar_timestamp, dataset_dir, log_id)

    if city_SE3_ego_cam_t is None or city_SE3_ego_lidar_t is None:
        if return_K:
            return None, None, None, None
        else:
            return None, None, None

    # convert back from homogeneous
    pts_h_lidar_time = pts_h_lidar_time.T[:, :3]
    ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().right_multiply_with_se3(city_SE3_ego_lidar_t)
    pts_h_cam_time = ego_cam_t_SE3_ego_lidar_t.transform_point_cloud(pts_h_lidar_time)
    pts_h_cam_time = point_cloud_to_homogeneous(pts_h_cam_time).T

    return project_lidar_to_img(pts_h_cam_time, calib_data, camera_name, return_K)
