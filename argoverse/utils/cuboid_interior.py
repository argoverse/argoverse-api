# Copyright (c) 2018 Charles R. Qi from Stanford University and Wei Liu from Nuro Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
# <Modifications copyright (C) 2019, Argo AI, LLC>

import copy
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay


def filter_point_cloud_to_bbox(bbox: np.ndarray, velodyne_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Given 2 orthogonal directions "u", "v" defined by 3 bbox vertices, s.t.::

        u = P1 - P2
        v = P1 - P4

    a point "x" in R^3 lies within the bbox iff::

        <u,P1> >= <u,x> >= <u,P2>
        <v,P1> >= <v,x> >= <v,P4>

    Args:
       bbox: Numpy array of shape (4,3) representing 3D bbox
       velodyne_pts: NumPy array of shape (N,3) representing Velodyne point cloud

    Returns:
       interior_pts: Numpy array of shape (N,3) representing velodyne points
            that fall inside the cuboid
    """
    P3 = bbox[0, :2]  # xmax, ymax
    P4 = bbox[1, :2]  # xmax, ymin
    P2 = bbox[2, :2]  # xmin, ymax
    P1 = bbox[3, :2]  # xmin, ymin

    u = P1 - P2
    v = P1 - P4

    pt_indices_to_plot = []

    u_low_bnd = u.dot(P2)
    u_upp_bnd = u.dot(P1)

    v_low_bnd = v.dot(P4)
    v_upp_bnd = v.dot(P1)

    for pt_idx in range(velodyne_pts.shape[0]):
        u_dot_x = u.dot(velodyne_pts[pt_idx, :2])
        v_dot_x = v.dot(velodyne_pts[pt_idx, :2])

        inside_u = u_low_bnd <= u_dot_x <= u_upp_bnd
        inside_v = v_low_bnd <= v_dot_x <= v_upp_bnd
        if inside_u and inside_v:
            pt_indices_to_plot.append(pt_idx)

    interior_pt_indices = np.array(pt_indices_to_plot)
    if interior_pt_indices.shape[0] == 0:
        return None
    else:
        interior_pts = velodyne_pts[interior_pt_indices]
        return interior_pts


def filter_point_cloud_to_bbox_2D_vectorized(bbox: np.ndarray, pc_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
       bbox: NumPy array of shape (4,2) representing 2D bbox
       pc_raw: NumPy array of shape (N,3) representing Velodyne point cloud

    Returns:
       pc_seg: NumPy array of shape (N,3) representing velodyne points
            that fall inside the cuboid
    """
    pc_2d = copy.deepcopy(pc_raw[:, :2])
    P3 = bbox[0]  # xmax, ymax
    P4 = bbox[1]  # xmax, ymin
    P2 = bbox[2]  # xmin, ymax
    P1 = bbox[3]  # xmin, ymin

    u = P1 - P2
    v = P1 - P4

    U = np.array([u[0:2]])
    V = np.array([v[0:2]])
    P1 = np.array([bbox[0][0:2]])
    P2 = np.array([bbox[1][0:2]])
    P4 = np.array([bbox[2][0:2]])

    dot1 = np.matmul(U, pc_2d.transpose(1, 0))
    dot2 = np.matmul(V, pc_2d.transpose(1, 0))
    u_p1 = np.tile((U * P1).sum(axis=1), (len(pc_2d), 1)).transpose(1, 0)
    v_p1 = np.tile((V * P1).sum(axis=1), (len(pc_2d), 1)).transpose(1, 0)
    u_p2 = np.tile((U * P2).sum(axis=1), (len(pc_2d), 1)).transpose(1, 0)
    v_p4 = np.tile((V * P4).sum(axis=1), (len(pc_2d), 1)).transpose(1, 0)

    flag = np.logical_and(in_between_matrix(dot1, u_p1, u_p2), in_between_matrix(dot2, v_p1, v_p4))
    flag = flag.squeeze()
    pc_seg = pc_raw[flag]
    return pc_seg, flag


def filter_point_cloud_to_bbox_3D(bbox: np.ndarray, pc_raw: np.ndarray) -> np.ndarray:
    """
    Args:
       bbox has shape object array: [(3,), (3,), (3,), height]
       pc_raw
    """
    u = bbox[1] - bbox[0]
    v = bbox[2] - bbox[0]
    w = np.zeros((3, 1))
    w[2, 0] += bbox[3]

    p5 = w + bbox[0]

    U = np.array([u[0:3, 0]])
    V = np.array([v[0:3, 0]])
    W = np.array([w[0:3, 0]])
    P1 = np.array([bbox[0][0:3, 0]])
    P2 = np.array([bbox[1][0:3, 0]])
    P4 = np.array([bbox[2][0:3, 0]])
    P5 = np.array([p5[0:3, 0]])

    dot1 = np.matmul(U, pc_raw.transpose(1, 0))
    dot2 = np.matmul(V, pc_raw.transpose(1, 0))
    dot3 = np.matmul(W, pc_raw.transpose(1, 0))
    u_p1 = np.tile((U * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p1 = np.tile((V * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p1 = np.tile((W * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    u_p2 = np.tile((U * P2).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p4 = np.tile((V * P4).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p5 = np.tile((W * P5).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)

    flag = np.logical_and(
        np.logical_and(in_between_matrix(dot1, u_p1, u_p2), in_between_matrix(dot2, v_p1, v_p4)),
        in_between_matrix(dot3, w_p1, w_p5),
    )

    pc_seg = pc_raw[flag[0, :]]
    return pc_seg


def in_between_matrix(x: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.logical_or(np.logical_and(x <= v1, x >= v2), np.logical_and(x <= v2, x >= v1))


def filter_point_cloud_to_bbox_3D_single_pt(bbox: np.ndarray, x: np.ndarray) -> np.ndarray:  # pc_raw):
    r"""

    Args:
       bbox: Numpy array of shape (8,1)
       x: Numpy array of shape (3,1)

    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

    ::

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    """
    # get 3 principal directions (edges) of the cuboid
    u = bbox[2] - bbox[6]
    v = bbox[2] - bbox[3]
    w = bbox[2] - bbox[1]

    # point x lies within the box when the following
    # constraints are respected

    # IN BETWEEN

    # do i need to check the other direction as well?
    valid_u1 = np.logical_and(u.dot(bbox[2]) <= u.dot(x), u.dot(x) <= u.dot(bbox[6]))
    valid_v1 = np.logical_and(v.dot(bbox[2]) <= v.dot(x), v.dot(x) <= v.dot(bbox[3]))
    valid_w1 = np.logical_and(w.dot(bbox[2]) <= w.dot(x), w.dot(x) <= w.dot(bbox[1]))

    valid_u2 = np.logical_and(u.dot(bbox[2]) >= u.dot(x), u.dot(x) >= u.dot(bbox[6]))
    valid_v2 = np.logical_and(v.dot(bbox[2]) >= v.dot(x), v.dot(x) >= v.dot(bbox[3]))
    valid_w2 = np.logical_and(w.dot(bbox[2]) >= w.dot(x), w.dot(x) >= w.dot(bbox[1]))

    valid_u = np.logical_or(valid_u1, valid_u2)
    valid_v = np.logical_or(valid_v1, valid_v2)
    valid_w = np.logical_or(valid_w1, valid_w2)

    valid = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)

    return valid


def filter_point_cloud_to_bbox_3D_vectorized(bbox: np.ndarray, pc_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""

    Args:
       bbox: Numpy array pf shape (8,3) representing 3d cuboid vertices, ordered
                as shown below.
       pc_raw: Numpy array of shape (N,3), representing a point cloud

    Returns:
       segment: Numpy array of shape (K,3) representing 3d points that fell
                within 3d cuboid volume.
       is_valid: Numpy array of shape (N,) of type bool

    https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d

    ::

            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    """
    # get 3 principal directions (edges) of the cuboid
    u = bbox[2] - bbox[6]
    v = bbox[2] - bbox[3]
    w = bbox[2] - bbox[1]

    # point x lies within the box when the following
    # constraints are respected

    # IN BETWEEN

    # do i need to check the other direction as well?
    valid_u1 = np.logical_and(u.dot(bbox[2]) <= pc_raw.dot(u), pc_raw.dot(u) <= u.dot(bbox[6]))
    valid_v1 = np.logical_and(v.dot(bbox[2]) <= pc_raw.dot(v), pc_raw.dot(v) <= v.dot(bbox[3]))
    valid_w1 = np.logical_and(w.dot(bbox[2]) <= pc_raw.dot(w), pc_raw.dot(w) <= w.dot(bbox[1]))

    valid_u2 = np.logical_and(u.dot(bbox[2]) >= pc_raw.dot(u), pc_raw.dot(u) >= u.dot(bbox[6]))
    valid_v2 = np.logical_and(v.dot(bbox[2]) >= pc_raw.dot(v), pc_raw.dot(v) >= v.dot(bbox[3]))
    valid_w2 = np.logical_and(w.dot(bbox[2]) >= pc_raw.dot(w), pc_raw.dot(w) >= w.dot(bbox[1]))

    valid_u = np.logical_or(valid_u1, valid_u2)
    valid_v = np.logical_or(valid_v1, valid_v2)
    valid_w = np.logical_or(valid_w1, valid_w2)

    is_valid = np.logical_and(np.logical_and(valid_u, valid_v), valid_w)
    segment_pc = pc_raw[is_valid]
    return segment_pc, is_valid


def extract_pc_in_box3d_hull(pc: np.ndarray, bbox_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points that fall within a 3d cuboid, by treating the 3d cuboid as a hull.
    Scipy.spatial's Delaunay class performs tesselation in N dimensions. By finding
    the simplices containing the given points, we also can determine which points
    lie outside the triangulation. Such invalid points obtain the value "-1". We
    threshold these to find the points that fall within the cuboid/hull.

    Please see Apache 2.0 license below, which governs this specific function.

    Args:
       pc: Numpy array of shape (N,3) representing point cloud
       bbox_3d: Numpy array of shape (8,3) representing 3D cuboid vertices

    Returns:
       segment: Numpy array of shape (K,3) representing 3d points that fell
                within 3d cuboid volume.
       box3d_roi_inds: Numpy array of shape (N,) of type bool, representing
            point cloud indices corresponding to points that fall within the
            3D cuboid.
    """
    if not isinstance(bbox_3d, Delaunay):
        hull = Delaunay(bbox_3d)

    box3d_roi_inds = hull.find_simplex(pc) >= 0
    return pc[box3d_roi_inds, :], box3d_roi_inds


"""
https://github.com/charlesq34/frustum-pointnets/blob/master/LICENSE
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2018 Charles R. Qi from Stanford University and
    Wei Liu from Nuro Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
