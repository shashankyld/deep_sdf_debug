# install environment

```sh
./install.sh
```

## Deep_SDF in DSP-SLAM default

```bash
python dsp_slam/reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 0
python dsp_slam/reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 50
python dsp_slam/reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 100
python dsp_slam/reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 500
python dsp_slam/reconstruct_frame.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 1000
```

## Deep_SDF in DSP-SLAM without render term

```bash
python dsp_slam/reconstruct_frame_no_render_term.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 0
python dsp_slam/reconstruct_frame_no_render_term.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 50
python dsp_slam/reconstruct_frame_no_render_term.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 100
python dsp_slam/reconstruct_frame_no_render_term.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 500
python dsp_slam/reconstruct_frame_no_render_term.py --config configs/config_kitti.json --sequence_dir data/dsp_slam/kitti/07 --frame_id 1000
```

## Jupyter notebook for cleaning data and extract to pcd extension

1. Cleaning data -> Rotate the wrong orientation dectection.
2. Extract data -> .pcd to P04_pcd folder
