#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import open3d as o3d
import argparse
from reconstruct.utils import color_table, set_view, get_configs, get_decoder
from reconstruct.loss_utils import get_time
from reconstruct.kitti_sequence import KITIISequence
from reconstruct.argoverse2_sequence import Argoverse2Sequence
from reconstruct.optimizer import Optimizer, MeshExtractor
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-d', '--sequence_dir', type=str, required=True, help='path to kitti sequence')
    # parser.add_argument('-i', '--frame_id', type=int, required=True, help='frame id')
    return parser


# 2D and 3D detection and data association
if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    configs = get_configs(args.config)
    decoder = get_decoder(configs)
    # kitti_seq = KITIISequence(args.sequence_dir, configs)
    kitti_seq = Argoverse2Sequence(args.sequence_dir, configs)
    optimizer = Optimizer(decoder, configs)

    detections = {}
    for frame_id in range(156):
        try:
            det = kitti_seq.get_frame_by_id(frame_id)
            detections[frame_id] = det
            # detections += [det]
        except:
            print("failed", frame_id)
            pass
    print("len(detections)", len(detections))

    # start reconstruction
    objects_recon = {}
    start = get_time()
    for frame_id, dets in detections.items():
        det = dets[0]
        # No observed rays, possibly not in fov
        # if det.rays is None:
        #     continue
        # print("%d depth samples on the car, %d rays in total" % (det.num_surface_points, det.rays.shape[0]))
        obj = optimizer.reconstruct_object(det.T_cam_obj, det.surface_points)
        # in case reconstruction fails
        if obj.code is None:
            continue
        objects_recon[frame_id] = obj
    end = get_time()
    print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_ctr = vis.get_view_control()

    # Add coordinate_frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    t_velo = np.array([[-0, -1, -0, 0],
                        [-0,  0, -1, -0],
                        [ 1, -0, -0, -0],
                        [ 0,  0,  0,  1]])

    # Add LiDAR point cloud
    # car_pcd = o3d.geometry.PointCloud()
    # car_pcd.points = o3d.utility.Vector3dVector(velo_pts)
    # car_pcd.colors = o3d.utility.Vector3dVector(color_table[0])
    # vis.add_geometry(car_pcd)

    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    for frame_id, obj in objects_recon.items():
        if frame_id == 0:
            car_pcd = o3d.geometry.PointCloud()
            car_pcd.points = o3d.utility.Vector3dVector(obj.pts_surface)
            # car_pcd.colors = o3d.utility.Vector3dVector(color_table[0])
            vis.add_geometry(car_pcd)
        else:
            car_pcd.points = o3d.utility.Vector3dVector(obj.pts_surface)
            vis.update_geometry(car_pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.25)

    # must be put after adding geometries
    # set_view(vis, dist=20, theta=0.)
    vis.run()
    vis.destroy_window()
    
