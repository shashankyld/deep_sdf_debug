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
                                change_bbox, get_bbox, get_bbox_gt, convert_to_lidar_cs, ForceKeyErrorDict




# visualizer
block_vis = True
play_crun = False


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


if args.sequence_dir == "data/P04/cleaned_data/000/000003/pcd.npy":
    # The detected car is being overtaken.
    gt = np.load("data/P04/gt/000/000003.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/000/000009/pcd.npy":
    # A car is driving in the opposite direction and is turning.
    gt = np.load("data/P04/gt/000/000009.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/000/000011/pcd.npy":
    # The car drives in a perpendicular direction.
    gt = np.load("data/P04/gt/000/000011.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/000/000021/pcd.npy":
    # A car is turning right.
    gt = np.load("data/P04/gt/000/000021.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/000/000030/pcd.npy":
    # The car drives in a perpendicular direction.
    gt = np.load("data/P04/gt/000/000030.npy",  allow_pickle=True).item()
    
elif args.sequence_dir == "data/P04/cleaned_data/000/000049/pcd.npy":
    # The ego-car is being overtaken -> change to this car goes straight.
    gt = np.load("data/P04/gt/000/000049.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001000/pcd.npy":
    # The ego-car is turning left 
    gt = np.load("data/P04/gt/001/001000.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001005/pcd.npy":
    # The detected car is turning right and then goes straight
    gt = np.load("data/P04/gt/001/001005.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001006/pcd.npy":
    # The detected car comes from an alley and merges into the main road.
    gt = np.load("data/P04/gt/001/001006.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001007/pcd.npy":
    # The detected car goes straight and turns right.
    gt = np.load("data/P04/gt/001/001007.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001027/pcd.npy":
    # The detected car comes from the opposite direction and turns left.
    gt = np.load("data/P04/gt/001/001027.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001035/pcd.npy":
    # The detected car is turning right
    gt = np.load("data/P04/gt/001/001035.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001038/pcd.npy":
    # The detected car is turning left
    gt = np.load("data/P04/gt/001/001038.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001039/pcd.npy":
    # The car drives in perpendicular direction
    gt = np.load("data/P04/gt/001/001039.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/002/002001/pcd.npy":
    # The car drives in perpendicular direction
    gt = np.load("data/P04/gt/002/002001.npy",  allow_pickle=True).item()

elif args.sequence_dir == "data/P04/cleaned_data/001/001046/pcd.npy":
    # the car u-turns
    gt = np.load("data/P04/gt/001/001046.npy",  allow_pickle=True).item()

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
# Print detection frame numbers
print("detection frame numbers", list(detections.keys()))
############# Get detection #############

############# Start reconstruction #############
objects_recon = {}
start = get_time()
t_velo = np.array([[-0, -1, -0, 0],
                    [-0,  0, -1, -0],
                    [ 1, -0, -0, -0],
                    [ 0,  0,  0,  1]])

list_T_cam_obj = []
list_surface_points = []
frame_numbers_list = []

# dict_T_cam_obj = {}
# dict_surface_points = {}


for frame_id, dets in detections.items():
    det = dets[0]
    frame_numbers_list.append(frame_id)
    # print("det", det)
    list_T_cam_obj.append(det.T_cam_obj)
    # dict_T_cam_obj[frame_id] = det.T_cam_obj
    # Print scale factor of the t_cam_obj before dataset creation
    print("scale factor before dataset creation", np.sqrt(det.T_cam_obj[0, 0]**2 + det.T_cam_obj[1, 0]**2 + det.T_cam_obj[2, 0]**2))
    list_surface_points.append(det.surface_points)
    # dict_surface_points[frame_id] = det.surface_points

print("frame numbers list: ", frame_numbers_list)
sliding_window_size = 40
beginning_frame_number  = 0
sliding_window_start_frame = frame_numbers_list[beginning_frame_number]

frames_to_optimize = [i for i in range(sliding_window_start_frame, sliding_window_start_frame + sliding_window_size) if i in frame_numbers_list]
print("frames_to_optimize", frames_to_optimize)
number_of_frames_to_optimize = len(frames_to_optimize)
print("number_of_frames_to_optimize", number_of_frames_to_optimize)
# frames to optimize in the list, starts with beginning_frame_number, length is sliding_window_size or less, miss  frames if (frame number in list + i) not in frame_numbers_list
frames_to_optimize_in_list = [beginning_frame_number + i for i in range(number_of_frames_to_optimize)]
print("frames_to_optimize_in_list", frames_to_optimize_in_list) 
obj = optimizer.sliding_window_reconstruct_object(
    [list_T_cam_obj[i] for i in frames_to_optimize_in_list],
    [list_surface_points[i] for i in frames_to_optimize_in_list],
    sliding_window_size=sliding_window_size
)
end = get_time()
print("Reconstructed %d objects in the scene, time elapsed: %f seconds" % (len(obj.list_t_cam_obj), end - start))


print("obj: ", obj.keys())
# Load objects_recon as a list of objects created by the sliding window reconstruction. 
# This will be usefull for visualization and evaluation.

total_frames = len(obj.list_t_cam_obj)
''' 
list_t_cam_obj=list_t_cam_obj,
                                 list_pts_surface=list_pts_surface,
                                 unique_code=latent_vector.cpu().numpy(),
                                 is_good=True, loss=loss)
'''

list_pts_surface = obj["list_pts_surface"]
list_t_cam_obj = obj["list_t_cam_obj"]
optimized_code = obj["unique_code"]
loss = obj["loss"]
print("list_pts_surface", len(list_pts_surface))
print("list_t_cam_obj", len(list_t_cam_obj))
print("optimized_code", optimized_code)
print("loss", loss)

# for i in range(total_frames):
#     objects_recon[i] = {}
#     objects_recon[i]["pts_surface"] = list_pts_surface[i]
#     objects_recon[i]["t_cam_obj"] = list_t_cam_obj[i]
#     objects_recon[i]['code'] = optimized_code
#     objects_recon[i]["loss"] = loss
#     objects_recon[i]["is_good"] = True
first_frame = frame_numbers_list[0]

count = 0
for i in frames_to_optimize:
    
    print("frame_number added to object_recon: ", i)
    # frame_number = frame_numbers_list[i]
    # Create a ForceKeyErrorDict instance for each frame
    obj = ForceKeyErrorDict(t_cam_obj=list_t_cam_obj[count],
                            pts_surface=list_pts_surface[count],
                            code=optimized_code,
                            is_good=True,
                            loss=loss)
    # print("unique_code", obj.code)
    # Print scale factor of the t_cam_obj
    print("scale factor", np.sqrt(obj.t_cam_obj[0, 0]**2 + obj.t_cam_obj[1, 0]**2 + obj.t_cam_obj[2, 0]**2))
    objects_recon[i] = obj
    count += 1

# print("objects_recon", objects_recon)
print("objects_recon: ", objects_recon.keys())

############# Find first frames #############
# Find the first frame in reconstruction object

for k, _ in objects_recon.items():
    first_frame = k
    break
print("first_frame", first_frame)
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

print("Instance Items", instance.keys())
# Creating a Instance dict with only subset of the keys
instance_sliding_window = {k: instance[k] for k in frames_to_optimize}
for (frame_id, points_scan), (_, obj) in zip(instance_sliding_window.items(), objects_recon.items()):
    
    print("frame_id\n", frame_id)
    if frame_id == first_frame:
        print("coming from first frame")
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
        print("objects_recon[first_frame].code", objects_recon[first_frame].keys())
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
        print("updating")
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

   
