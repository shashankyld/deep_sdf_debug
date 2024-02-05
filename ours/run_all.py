from reconstruct_multiple_frame_argo_compare import Reconstruct_Argoverse2 


config = "configs/config_kitti.json"
sequence_s = [
    "000003",
    # "000009",
    # "000011",
    # "000021",
    # "000030",
    # "000049",
    "001000",
    # "001005",
    # "001006",
    # "001007",
    # "001027",
    # "001035",
    # "001038",
    # "001039",
    # "001046",
]
iou_gt_det_list = []
iou_gt_opt_list = []
for seq in sequence_s:
    iou_gt_det, iou_gt_opt = Reconstruct_Argoverse2(config=config, sequence_dir=seq)
    iou_gt_det_list.append(iou_gt_det)
    iou_gt_opt_list.append(iou_gt_opt)

print("sequence number      iou_gt_det        iou_gt_opt")
for seq_i in range(len(sequence_s)):
    print("seq", sequence_s[seq_i], iou_gt_det_list[seq_i], iou_gt_opt_list[seq_i])