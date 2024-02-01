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

1. Evaluate optimization

```bash
python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000003/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000009/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000011/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000021/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000030/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/000/000049/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001000/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001005/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001006/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001007/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001027/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001035/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001038/pcd.npy

python ours/2_reconstruct_multiple_frame_argo_compare.py --config configs/config_kitti.json --sequence_dir data/P04/cleaned_data/001/001039/pcd.npy

```