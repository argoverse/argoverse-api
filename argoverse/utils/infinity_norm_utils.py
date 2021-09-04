"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

import logging
import numpy as np

from argoverse.utils.interpolate import interp_arc

WPT_INFTY_NORM_INTERP_NUM = 50


def has_pts_in_infty_norm_radius(
	pts: np.ndarray,
	window_center: np.ndarray,
	window_sz: float
) -> bool:
	"""Check if a map entity has points within a search radius from a single query point.
	
	Note: This does NOT measure distance by Manhattan distance -- this is the infinity norm.

	Args:
	    pts: Nx2 array, representing map entity
	    window_center: 1x2, or (2,) array, representing query point
	    window_sz: search radius
	"""
	assert pts.ndim == 2
	if pts.shape[1] == 3:
		# take only x,y dimensions
		pts = pts[:,:2]
	assert pts.size % 2 == 0
	assert pts.shape[1] == 2
	
	if window_center.ndim == 2:
		window_center = window_center.squeeze()
	assert window_center.ndim == 1
	if window_center.size == 3:
		window_center = window_center[:2]
	assert window_center.size == 2
	# reshape just in case was given column vector
	window_center = window_center.reshape(1,2)

	dists = np.linalg.norm(pts - window_center, ord=np.inf, axis=1)
	return dists.min() < window_sz


def test_has_pts_in_infty_norm_radius1():
	""" No points within radius"""
	pts = np.array(
		[
			[5.1,0],
			[0,-5.1],
			[5.1,5.1]
		])
	within = has_pts_in_infty_norm_radius(pts, window_center=np.zeros(2), window_sz=5)
	assert within == False


def test_has_pts_in_infty_norm_radius2():
	""" 1 point within radius"""
	pts = np.array(
		[
			[4.9,0],
			[0,-5.1],
			[5.1,5.1]
		])
	within = has_pts_in_infty_norm_radius(pts, window_center=np.zeros(2), window_sz=5)
	assert within == True


def test_has_pts_in_infty_norm_radius3():
	""" All pts within radius"""
	pts = np.array(
		[
			[4.9,0],
			[0,-4.9],
			[4.9,4.9]
		])
	within = has_pts_in_infty_norm_radius(pts, window_center=np.zeros(2), window_sz=5)
	assert within == True


def test_has_pts_in_infty_norm_radius4():
	""" All pts within radius"""
	pts = np.array(
		[
			[4.9,4.9]
		])
	within = has_pts_in_infty_norm_radius(pts, window_center=np.zeros(2), window_sz=5)
	assert within == True


if __name__ == '__main__':

	# test_has_pts_in_infty_norm_radius1()
	# test_has_pts_in_infty_norm_radius2()
	# test_has_pts_in_infty_norm_radius3()
	test_has_pts_in_infty_norm_radius_from_traj()