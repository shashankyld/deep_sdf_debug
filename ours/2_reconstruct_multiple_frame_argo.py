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

    mesh_extractor = MeshExtractor(decoder, voxels_dim=64)
    first_frame = -1
    for frame_id, obj in objects_recon.items():
        if first_frame == -1:
            first_frame = frame_id
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])
        # mesh_o3d.paint_uniform_color(color_table[i])
        # Transform mesh from object to world coordinate
        mesh_o3d.transform(obj.t_cam_obj)
        mesh_o3d.transform(np.linalg.inv(t_velo))

        # Save
        mtx = np.linalg.inv(t_velo) @ obj.t_cam_obj
        x = mtx[0, 3]
        y = mtx[1, 3]
        z = mtx[2, 3]
        
        mtx = mtx[:, [2, 1, 0, 3]]
        opt_r = R.from_matrix(obj.t_cam_obj[:3, :3])
        yaw = opt_r.as_euler('zxy', degrees=True)[2]
        if yaw > 0:
            yaw2 = (yaw - 180)
        if yaw < 0:
            yaw2 = (yaw + 180)
        
        mtx_opt = np.array([[x, y, z, yaw2]])
        # if  not yaw2 < -150 or yaw2 > 150:
        #     print("frame_id", frame_id)
        #     print("yaw", yaw)
        #     print("yaw2", yaw2)
        # print("frame_id", frame_id)
        vis.add_geometry(mesh_o3d)
        opt_np = np.array([[x, y, z, yaw2]])
        if frame_id == first_frame:
            opt_nps = opt_np
        else:
            opt_nps = np.vstack((opt_nps, opt_np))
        # np.savetxt("../../P04_Instance-Completion-and-Motion-Estimation-with-Deep-Shape-Priors-for-Autonomous-Driving/dev_code/opt_nps_vanilla_motion_prior.txt", opt_nps)

    # must be put after adding geometries
    set_view(vis, dist=0., theta=90.)
    vis.run()
    vis.destroy_window()
    
