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
from bbox.metrics import iou_3d
from functools import partial
import os
from pathlib import Path
import numpy as np
import time
from typing import Callable, List, Tuple

from reconstruct.argoverse2_sequence import Argoverse2Sequence
from reconstruct.loss_utils import get_time
from reconstruct.optimizer import Optimizer, MeshExtractor
from reconstruct.utils import color_table, set_view, get_configs, get_decoder, \
                                translate_boxes_to_open3d_instance, config_parser, BoundingBox3D, \
                                change_bbox, get_bbox, get_bbox_gt, convert_to_lidar_cs




# visualizer
block_vis = True
play_crun = True


def quit(vis):
    print("Destroying Visualizer")
    vis.destroy_window()
    os._exit(0)

def next_frame(vis):
    global block_vis
    block_vis = not block_vis

def start_stop(vis):
    global play_crun
    play_crun = not play_crun

def register_key_callback(keys: List, callback: Callable):
    for key in keys:
        vis.register_key_callback(ord(str(key)), partial(callback))

def register_key_callbacks():
    register_key_callback(["Ā", "Q", "\x1b"], quit)
    register_key_callback([" "], start_stop)
    register_key_callback(["N"], next_frame)
    # register_key_callback(["V"], toggle_view)
    # register_key_callback(["C"], center_viewpoint)
    # register_key_callback(["F"], toggle_source)
    # register_key_callback(["K"], toggle_keypoints)
    # register_key_callback(["M"], toggle_map)
    # register_key_callback(["T"], toggle_trajectory)
    # register_key_callback(["B"], set_black_background)
    # register_key_callback(["W"], set_white_background)


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
parser = config_parser()
args = parser.parse_args()
configs = get_configs(args.config)
decoder = get_decoder(configs)
kitti_seq = Argoverse2Sequence(args.sequence_dir, configs)
optimizer = Optimizer(decoder, configs)


if args.sequence_dir == "data/P04/cleaned_data/000/000009/pcd.npy":
    # A car is driving in the opposite direction and is turning.
    gt = np.load("data/P04/gt/000/000009.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002001/pcd.npy":
    # no describtion.
    gt = np.load("data/P04/gt/002/002001.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002002/pcd.npy":
    # ortho.
    gt = np.load("data/P04/gt/002/002002.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002007/pcd.npy":
    # ortho.
    gt = np.load("data/P04/gt/002/002007.npy",  allow_pickle=True).item()
    
elif args.sequence_dir == "data/P04/cleaned_data/002/002022/pcd.npy":
    # opposite direction.
    gt = np.load("data/P04/gt/002/002022.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002028/pcd.npy":
    # opposite direction 
    gt = np.load("data/P04/gt/002/002028.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002031/pcd.npy":
    # turning car.
    gt = np.load("data/P04/gt/002/002031.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002038/pcd.npy":
    # turning on opposite direction.
    gt = np.load("data/P04/gt/002/002038.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002046/pcd.npy":
    # turning.
    gt = np.load("data/P04/gt/002/002046.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002048/pcd.npy":
    # turning.
    gt = np.load("data/P04/gt/002/002048.npy",  allow_pickle=True).item()

else:
    print("Ground truth not found")

pcd_track_uuids = np.load(args.sequence_dir,  allow_pickle=True).item()


# Magic number
x_rad = np.deg2rad(-90)
rot_x = np.array([[1, 0, 0], 
                    [0, np.cos(x_rad), -np.sin(x_rad)], 
                    [0, np.sin(x_rad), np.cos(x_rad)]])

z_rad = np.deg2rad(90)
rot_z = np.array([  [np.cos(z_rad), -np.sin(z_rad), 0], 
                    [np.sin(z_rad),  np.cos(z_rad), 0], 
                    [0       ,         0, 1]])
rot_velo_obj = rot_x @ rot_z 


t_velo = np.array([[-0, -1, -0, 0],
                    [-0,  0, -1, -0],
                    [ 1, -0, -0, -0],
                    [ 0,  0,  0,  1]])

############# Find track_uuids #############
for k, _ in pcd_track_uuids.items():
    track_uuid = k
    break
instances = pcd_track_uuids[track_uuid]
############# Find track_uuids #############
############# Find first instance #############
instance_id = 0
maximum_frame_number = 0
for i, _ in instances.items():
    frame_number = len(instances[i])
    if frame_number > maximum_frame_number:
        maximum_frame_number = frame_number
        first_instance = i
instance = instances[first_instance]
############# Find first instance #############

############# Get detection #############
detections = {}
for frame_id in instances[first_instance]:
    det = kitti_seq.get_frame_by_id(frame_id)
    detections[frame_id] = det
print("len(detections)", len(detections))
############# Get detection #############

############# Start reconstruction #############
objects_recon = {}
start = get_time()
t_velo = np.array([[-0, -1, -0, 0],
                    [-0,  0, -1, -0],
                    [ 1, -0, -0, -0],
                    [ 0,  0,  0,  1]])

for frame_id, dets in detections.items():
    det = dets[0]

    # r = R.from_matrix(det.pv_rcnn_debug[:3, :3])
    # euler_pv_rcnn_debug = r.as_euler('zxy', degrees=True)
    # print("euler_pv_rcnn_debug\n", euler_pv_rcnn_debug)

    # T_velo_obj = convert_to_lidar_cs(det.T_cam_obj, det.l)
    # r_t = R.from_matrix(T_velo_obj[:3, :3])
    # euler_T_velo_obj = r_t.as_euler('zxy', degrees=True)
    # print("euler_T_velo_obj\n", euler_T_velo_obj)

    obj = optimizer.reconstruct_object(det.T_cam_obj, det.surface_points)
    # obj.size = det.size
    # in case reconstruction fails
    if obj.code is None:
        continue
    objects_recon[frame_id] = obj
end = get_time()
# print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(objects_recon), end - start))
############# Start reconstruction #############

############# Find first frames #############
# Find the first frame in reconstruction object
first_frame = 0
for k, _ in objects_recon.items():
    first_frame = k
    break
############# Find first frames #############


# Visualize results
vis = o3d.visualization.VisualizerWithKeyCallback()
register_key_callbacks()
vis.create_window()
# Coordinate frame
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)

pts_str = 'pts_cam' # dict_keys(['T_cam_obj', 'pts_cam', 'surface_points', 'bbox'])


############# Evaluation #############
iou_gt_det = []
iou_gt_opt = []
############# Evaluation #############
                    
mesh_extractor = MeshExtractor(decoder, voxels_dim=64)

for (frame_id, points_scan), (_, obj) in zip(instance.items(), objects_recon.items()):

    print("frame_id\n", frame_id)
    if frame_id == first_frame:

        ############# Bounding boxes #############
        # Check wrong yaw detection
        mtx = instance[first_frame]['T_cam_obj']
        # Extracted point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(instance[first_frame][pts_str])
        vis.add_geometry(pcd)
        # Orientation
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        coordinate_frame.transform(mtx)
        prev_mtx = mtx
        vis.add_geometry(coordinate_frame)
        # Detected bounding box 
        bbox = get_bbox(instance[first_frame])
        line_set, box3d = translate_boxes_to_open3d_instance(bbox)
        line_set.paint_uniform_color(color_table[1]) # GREEN
        vis.add_geometry(line_set)
        ############# Bounding boxes #############


        ############# Ground truth #############
        gt_bbox = get_bbox_gt(gt[first_frame][track_uuid])
        gt_line_set, gt_box3d = translate_boxes_to_open3d_instance(gt_bbox)
        gt_line_set.paint_uniform_color(color_table[0])  # red
        gt_mtx = np.hstack((gt[first_frame][track_uuid]['rot'], np.array([[gt[first_frame][track_uuid]['x']], 
                                                                        [gt[first_frame][track_uuid]['y']], 
                                                                        [gt[first_frame][track_uuid]['z']]])))
        gt_mtx = np.vstack((gt_mtx, np.array([0, 0, 0, 1])))
        prev_gt_mtx = gt_mtx
        vis.add_geometry(gt_line_set)
        ############# Ground truth #############



        ############# Mesh #############
        mesh = mesh_extractor.extract_mesh_from_code(objects_recon[first_frame].code)
        mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])
        # Transform mesh from object to world coordinate
        mtx_opt = np.linalg.inv(t_velo) @ objects_recon[first_frame].t_cam_obj

        ############# Bbox of Mesh #############
        oriented_bbox_opt = mesh_o3d.get_oriented_bounding_box()
        # print("oriented_bbox_opt.extent", oriented_bbox_opt.extent)
        # obb = o3d.geometry.OrientedBoundingBox(
        #         [0, 0, 0], 
        #         rot_velo_obj,
        #         oriented_bbox_opt.extent)
        # obb.rotate(mtx_opt[:3, :3])
        # obb.translate(mtx_opt[:3, 3])
        # obb.color = np.array(color_table[3])
        # vis.add_geometry(obb)


        t_velo_obj = convert_to_lidar_cs(objects_recon[first_frame].t_cam_obj, 1)
        scale = np.sqrt(t_velo_obj[0, 0]**2 + t_velo_obj[1, 0]**2 + t_velo_obj[2, 0]**2)
        print("scale", scale)
        opt_line_bbox = BoundingBox3D(t_velo_obj[:3, 3][0], 
                            t_velo_obj[:3, 3][1], t_velo_obj[:3, 3][2],
                            scale * oriented_bbox_opt.extent[0], scale * oriented_bbox_opt.extent[1], 
                            scale * oriented_bbox_opt.extent[2], t_velo_obj[:3, :3] / scale)

        opt_line_set, opt_box3d  = translate_boxes_to_open3d_instance(opt_line_bbox)
        opt_line_set.paint_uniform_color(color_table[2])  # blue
        vis.add_geometry(opt_line_set)


        ############# Bbox of Mesh #############

        mesh_o3d.transform(mtx_opt)
        prev_mtx_opt = mtx_opt
        vis.add_geometry(mesh_o3d)
        ############# Mesh #############




    else:
        ############# Bounding boxes #############
        # Point Cloud
        pcd.points = o3d.utility.Vector3dVector(points_scan[pts_str])
        vis.update_geometry(pcd)
        
        # Orientation
        coordinate_frame.transform(np.linalg.inv(prev_mtx)) # undo previous transformation
        mtx = points_scan['T_cam_obj']
        
        coordinate_frame.transform(mtx)
        vis.update_geometry(coordinate_frame)
        
        # Detected bounding box 
        line_set.transform(np.linalg.inv(prev_mtx)) # undo previous transformation
        bbox = get_bbox(points_scan)
        change_bbox(line_set, bbox)
        line_set.transform(mtx)
        line_set.paint_uniform_color(color_table[1]) # GREEN
        vis.update_geometry(line_set)
        prev_mtx = mtx
        ############# Bounding boxes #############
    
    
        ############# Ground truth #############
        gt_mtx = np.hstack((gt[frame_id][track_uuid]['rot'], np.array([[gt[frame_id][track_uuid]['x']], 
                                                                          [gt[frame_id][track_uuid]['y']], 
                                                                          [gt[frame_id][track_uuid]['z']]])))
        gt_mtx = np.vstack((gt_mtx, np.array([0, 0, 0, 1])))
    
        gt_line_set.transform(np.linalg.inv(prev_gt_mtx)) # undo previous transformation
        gt_bbox = get_bbox_gt(gt[frame_id][track_uuid])
        change_bbox(gt_line_set, gt_bbox)
        gt_line_set.transform(gt_mtx)
        gt_line_set.paint_uniform_color(color_table[0])  # red
    
        vis.update_geometry(gt_line_set)
    
        prev_gt_mtx = gt_mtx
        ############# Ground truth #############

        ############# Mesh #############
        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d.transform(np.linalg.inv(prev_mtx_opt)) # undo previous transformation
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])
        # Transform mesh from object to world coordinate
        mtx_opt = np.linalg.inv(t_velo) @ obj.t_cam_obj
        
        ############# Bbox of Mesh #############
        # obb.translate(np.linalg.inv(prev_mtx_opt)[:3, 3]) # undo previous transformation
        # obb.rotate(np.linalg.inv(prev_mtx_opt)[:3, :3])
        oriented_bbox_opt = mesh_o3d.get_oriented_bounding_box()
        # print("oriented_bbox_opt.extent", oriented_bbox_opt.extent)
        # obb.center = np.array([0, 0, 0])
        # obb.extent = oriented_bbox_opt.extent
        # obb.rotate(mtx_opt[:3, :3])
        # obb.translate(mtx_opt[:3, 3])
        # obb.color = np.array(color_table[3])
        # vis.update_geometry(obb)



        t_velo_obj = convert_to_lidar_cs(obj.t_cam_obj, 1)
        scale = np.sqrt(t_velo_obj[0, 0]**2 + t_velo_obj[1, 0]**2 + t_velo_obj[2, 0]**2)
        print("scale", scale)
        # print("obj.t_cam_obj.size", obj.t_cam_obj.size)
        opt_line_bbox = BoundingBox3D(t_velo_obj[:3, 3][0], 
                            t_velo_obj[:3, 3][1], t_velo_obj[:3, 3][2],
                            scale * oriented_bbox_opt.extent[0], scale * oriented_bbox_opt.extent[1], 
                            scale * oriented_bbox_opt.extent[2], t_velo_obj[:3, :3] / scale)
        change_bbox(opt_line_set, opt_line_bbox)
        opt_line_set.paint_uniform_color(color_table[2])  # blue

        opt_line_set.transform(np.linalg.inv(prev_mtx_opt))  # undo previous transformation
        opt_line_set.transform(mtx_opt)
        vis.update_geometry(opt_line_set)

        ############# Bbox of Mesh #############

        mesh_o3d.transform(mtx_opt)
        vis.update_geometry(mesh_o3d)
        prev_mtx_opt = mtx_opt
        ############# Mesh #############
    

    ############# Evaluation #############
    iou_bbox_det = iou_3d(gt_bbox.iou, bbox.iou)
    iou_bbox_opt = iou_3d(gt_bbox.iou, opt_line_bbox.iou)
    # print("bbox, opt_line_bbox(For evaluate Visualization, ignored)", iou_3d(bbox.iou, opt_line_bbox.iou))
    print("IOU detection, optimization(Should be better)", iou_bbox_det, iou_bbox_opt)
    iou_gt_det.append(iou_bbox_det)
    iou_gt_opt.append(iou_bbox_opt)
    ############# Evaluation #############

    while block_vis:
        vis.poll_events()
        vis.update_renderer()
        if play_crun:
            break
    block_vis = not block_vis

print("Mean iou, Ground Truth vs Detection", np.mean(iou_gt_det))
print("Mean iou, Ground Truth vs Optimization", np.mean(iou_gt_opt))
vis.destroy_window()