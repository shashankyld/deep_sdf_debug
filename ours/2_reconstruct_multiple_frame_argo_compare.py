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

import argparse
import open3d as o3d
from bbox import  BBox3D
from bbox.metrics import iou_3d
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import numpy as np
import time
from typing import Callable, List, Tuple
from scipy.spatial.transform import Rotation as R

from reconstruct.utils import color_table, set_view, get_configs, get_decoder, translate_boxes_to_open3d_instance
from reconstruct.loss_utils import get_time
from reconstruct.kitti_sequence import KITIISequence
from reconstruct.argoverse2_sequence import Argoverse2Sequence
from reconstruct.optimizer import Optimizer, MeshExtractor


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-d', '--sequence_dir', type=str, required=True, help='path to kitti sequence')
    # parser.add_argument('-i', '--frame_id', type=int, required=True, help='frame id')
    return parser

@dataclass
class BoundingBox3D:
    '''
    pose in s-frame
    '''
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rot: float
    iou: BBox3D = field(init=False, repr=False)

    def __post_init__(self):
        r = R.from_matrix(self.rot)
        q8d_xyzw = r.as_quat()
        q8d = np.array([q8d_xyzw[3], q8d_xyzw[0], q8d_xyzw[1], q8d_xyzw[2]])

        self.iou: BBox3D = BBox3D(self.x, self.y, self.z, 
                                         self.length, self.width, 
                                         self.height, q=q8d)


def translate_boxes_to_open3d_instance(bbox, crop=False):
    """
          4 -------- 6
         /|         /|
        5 -------- 3 .
        | |        | |
        . 7 -------- 1
        |/         |/
        2 -------- 0
    https://github.com/open-mmlab/OpenPCDet/blob/master/tools/visual_utils/open3d_vis_utils.py
    """
    center = [bbox.x, bbox.y, bbox.z]
    lwh = [bbox.length, bbox.width, bbox.height]
    if not crop:
        box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)
    else:
        lwh = [bbox.length, bbox.width, bbox.height * 0.9]
        box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)
    

    return line_set, box3d

def change_bbox(line_set, bbox):
    center = [bbox.x, bbox.y, bbox.z]
    lwh = [bbox.length, bbox.width, bbox.height]
    box3d = o3d.geometry.OrientedBoundingBox(center, bbox.rot, lwh)
        
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

def get_bbox(pcd):
    x = pcd['bbox'][0]
    y = pcd['bbox'][1]
    z = pcd['bbox'][2]
    l = pcd['bbox'][4]
    w = pcd['bbox'][3]
    h = pcd['bbox'][5]
    rot = pcd['T_cam_obj'][:3, :3]
    bbox = BoundingBox3D(x,y,z,l,w,h,rot)
    return bbox


def get_bbox_gt(pcd):
    x = pcd['x']
    y = pcd['y']
    z = pcd['z']
    l = pcd['length']
    w = pcd['width']
    h = pcd['height']
    rot = pcd['rot'][:3, :3]
    bbox = BoundingBox3D(x,y,z,l,w,h,rot)
    return bbox

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
    # The ego-car is being overtaken
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
else:
    print("Ground truth not found")
    # # the car u-turns - hard to clean #shashank
    # yaw_filter=[[0, 180], [-0, -50]]
    # pcd_track_uuids = np.load("data/P04/raw_data/001/001046/pcd.npy",  allow_pickle=True).item()
    # instance = pcd_track_uuids['945feec5-d2f5-49da-ab6d-c71f9402a23f']

pcd_track_uuids = np.load(args.sequence_dir,  allow_pickle=True).item()


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
for frame_id, dets in detections.items():
    det = dets[0]
    # print("det.T_cam_obj", det.T_cam_obj)
    obj = optimizer.reconstruct_object(det.T_cam_obj, det.surface_points)
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
line_set.paint_uniform_color(np.array([60. / 255., 180. / 255., 75. / 255.])) # GREEN
vis.add_geometry(line_set)
############# Bounding boxes #############


############# Ground truth #############
gt_bbox = get_bbox_gt(gt[first_frame][track_uuid])
gt_line_set, gt_box3d = translate_boxes_to_open3d_instance(gt_bbox)
gt_line_set.paint_uniform_color(np.array([230. / 255., 0., 0.]))  # red
gt_mtx = np.hstack((gt[first_frame][track_uuid]['rot'], np.array([[gt[first_frame][track_uuid]['x']], 
                                                                  [gt[first_frame][track_uuid]['y']], 
                                                                  [gt[first_frame][track_uuid]['z']]])))
gt_mtx = np.vstack((gt_mtx, np.array([0, 0, 0, 1])))
prev_gt_mtx = gt_mtx
vis.add_geometry(gt_line_set)
############# Ground truth #############


t_velo = np.array([[-0, -1, -0, 0],
                    [-0,  0, -1, -0],
                    [ 1, -0, -0, -0],
                    [ 0,  0,  0,  1]])
# t_velo = np.eye(4)
mesh_extractor = MeshExtractor(decoder, voxels_dim=64)


############# Mesh #############
mesh = mesh_extractor.extract_mesh_from_code(objects_recon[first_frame].code)
mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color(color_table[0])
# Transform mesh from object to world coordinate
mesh_o3d.transform(objects_recon[first_frame].t_cam_obj)
mtx_opt = np.linalg.inv(t_velo) @ objects_recon[first_frame].t_cam_obj
prev_mtx_opt = mtx_opt
vis.add_geometry(mesh_o3d)
############# Mesh #############

############# Bbox of Mesh #############
oriented_bbox_opt = mesh_o3d.get_oriented_bounding_box()
oriented_bbox_opt_x =  oriented_bbox_opt.center[2]
oriented_bbox_opt_y = -oriented_bbox_opt.center[0]
oriented_bbox_opt_z = -oriented_bbox_opt.center[1]
oriented_bbox_opt_l = oriented_bbox_opt.extent[0]
oriented_bbox_opt_w = oriented_bbox_opt.extent[1]
oriented_bbox_opt_h = oriented_bbox_opt.extent[2]

opt_line_bbox = BoundingBox3D(oriented_bbox_opt_x, 
                    oriented_bbox_opt_y, oriented_bbox_opt_z,
                    oriented_bbox_opt_l, oriented_bbox_opt_w, 
                    oriented_bbox_opt_h, np.eye(3))

opt_line_set, opt_box3d  = translate_boxes_to_open3d_instance(opt_line_bbox)
opt_line_set.paint_uniform_color(np.array([0., 0., 255. / 255.]))  # blue
vis.add_geometry(opt_line_set)
############# Bbox of Mesh #############

############# Evaluation #############
iou_gt_det = []
iou_gt_opt = []
iou_bbox_det = iou_3d(gt_bbox.iou, bbox.iou)
iou_bbox_opt = iou_3d(gt_bbox.iou, opt_line_bbox.iou)
iou_gt_det.append(iou_bbox_det)
iou_gt_opt.append(iou_bbox_opt)
############# Evaluation #############

# for frame_id, points_scan in points.items():
for (frame_id,points_scan), (frame_id_recon,obj) in zip(instance.items(), objects_recon.items()):

    if frame_id != first_frame:
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
        line_set.paint_uniform_color(np.array([60. / 255., 180. / 255., 75. / 255.])) # GREEN
        vis.update_geometry(line_set)
        prev_mtx = mtx
        ############# Bounding boxes #############
    
    
        ############# Ground truth #############
        # print("gt[frame_id][track_uuid]['x']", gt[frame_id][track_uuid]['x'])
        # print("gt[frame_id][track_uuid]['y']", gt[frame_id][track_uuid]['y'])
        # print("gt[frame_id][track_uuid]['z']", gt[frame_id][track_uuid]['z'])
        gt_mtx = np.hstack((gt[frame_id][track_uuid]['rot'], np.array([[gt[frame_id][track_uuid]['x']], 
                                                                          [gt[frame_id][track_uuid]['y']], 
                                                                          [gt[frame_id][track_uuid]['z']]])))
        gt_mtx = np.vstack((gt_mtx, np.array([0, 0, 0, 1])))
    
        gt_line_set.transform(np.linalg.inv(prev_gt_mtx)) # undo previous transformation
        gt_bbox = get_bbox_gt(gt[frame_id][track_uuid])
        change_bbox(gt_line_set, gt_bbox)
        gt_line_set.transform(gt_mtx)
        gt_line_set.paint_uniform_color(np.array([230. / 255., 0., 0.]))  # red
    
        vis.update_geometry(gt_line_set)
    
        prev_gt_mtx = gt_mtx
        ############# Ground truth #############

        mesh = mesh_extractor.extract_mesh_from_code(obj.code)
        mesh_o3d.transform(np.linalg.inv(prev_mtx_opt)) # undo previous transformation
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(color_table[0])
        # Transform mesh from object to world coordinate
        mtx_opt = np.linalg.inv(t_velo) @ obj.t_cam_obj
        # print("mesh x", mtx_opt)
        mesh_o3d.transform(mtx_opt)
        vis.update_geometry(mesh_o3d)

        ############# Bbox of Mesh #############
        oriented_bbox_opt = mesh_o3d.get_oriented_bounding_box()
        oriented_bbox_opt_x = oriented_bbox_opt.center[0]
        oriented_bbox_opt_y = oriented_bbox_opt.center[1]
        oriented_bbox_opt_z = oriented_bbox_opt.center[2]
        oriented_bbox_opt_l = oriented_bbox_opt.extent[0]
        oriented_bbox_opt_w = oriented_bbox_opt.extent[1]
        oriented_bbox_opt_h = oriented_bbox_opt.extent[2]

        opt_line_bbox = BoundingBox3D(oriented_bbox_opt_x, 
                            oriented_bbox_opt_y, oriented_bbox_opt_z,
                            oriented_bbox_opt_l, oriented_bbox_opt_w, 
                            oriented_bbox_opt_h, np.eye(3))

        # opt_line_set.paint_uniform_color(np.array([0., 0., 255. / 255.]))  # blue
        change_bbox(opt_line_set, opt_line_bbox)

        opt_line_set.transform(np.linalg.inv(prev_mtx_opt))  # undo previous transformation
        opt_line_set.transform(mtx_opt)
        vis.update_geometry(opt_line_set)
        prev_mtx_opt = mtx_opt
        ############# Bbox of Mesh #############


        ############# Evaluation #############
        iou_bbox_det = iou_3d(gt_bbox.iou, bbox.iou)
        iou_bbox_opt = iou_3d(gt_bbox.iou, opt_line_bbox.iou)
        iou_gt_det.append(iou_bbox_det)
        iou_gt_opt.append(iou_bbox_opt)
        ############# Evaluation #############

        while block_vis:
            vis.poll_events()
            vis.update_renderer()
            if play_crun:
                break
        block_vis = not block_vis

        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(0.1)

print("Mean iou, Ground Truth vs Detection", np.mean(iou_gt_det))
print("Mean iou, Ground Truth vs Optimization", np.mean(iou_gt_opt))
# vis.run()
vis.destroy_window()

    
