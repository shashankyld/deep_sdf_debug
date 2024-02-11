from reconstruct_argo import Reconstruct_Argoverse2 
from rich.console import Console
from rich.table import Table



def find_max(att_list):

    max_att = max(att_list)

    att_list_str = []
    for att in att_list:
        if att == max_att:
            att_str = "[red]" + str(round(att, 4)) + "[/red]"
        else:
            att_str = str(round(att, 4))
        att_list_str.append(att_str)
    return att_list_str



def find_min(att_list):

    min_att = min(att_list)

    att_list_str = []
    for att in att_list:
        if att == min_att:
            att_str = "[red]" + str(round(att, 4)) + "[/red]"
        else:
            att_str = str(round(att, 4))
        att_list_str.append(att_str)
    return att_list_str
    
config = "configs/config_kitti.json"
sequence_s = [
    "000003",
    "000009",
    "000011",
    "000021",
    "000030",
    "000049",
    "001000",
    "001005",
    "001006",
    "001007",
    "001027",
    "001035",
    "001038",
    "001039",
    # "001046", no longer use
]

console = Console()
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Sequence", style="dim", width=12)
table.add_column("Detection(IOU)", justify="right")
table.add_column("Optization(IOU)", justify="right")
table.add_column("Optization with Code(IOU)", justify="right")
table.add_column("Detection(Yaw)", justify="right")
table.add_column("Optization(Yaw)", justify="right")
table.add_column("Optization with Code(Yaw)", justify="right")

for seq in sequence_s:
    iou = []
    yaw = []

    iou_gt_det, iou_gt_opt, yaw_det, yaw_opt = Reconstruct_Argoverse2(config=config, sequence_dir=seq)
    _, iou_gt_opt_code, _, yaw_opt_code = Reconstruct_Argoverse2(config=config, sequence_dir=seq, mode = "code")

    iou.append(iou_gt_det)
    iou.append(iou_gt_opt)
    iou.append(iou_gt_opt_code)

    yaw.append(yaw_det)
    yaw.append(yaw_opt)
    yaw.append(yaw_opt_code)

    iou_str = find_max(iou)
    yaw_str = find_min(yaw)

    table.add_row(
        seq, iou_str[0], iou_str[1], iou_str[2], yaw_str[0], yaw_str[1], yaw_str[2]
    )

console.print(table)
