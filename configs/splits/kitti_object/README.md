# KITTI 3D Object splits

- `train.txt` (3,712 frames) and `val.txt` (3,769 frames): the standard split of the 7,481 KITTI 3D Object training
  frames from Chen et al., *3D Object Proposals for Accurate Object Class Detection* (NeurIPS 2015), as shipped with
  OpenPCDet (`data/kitti/ImageSets`, Apache-2.0). Pinned here so runs never depend on a download.
- `mtrans_labelled_500.txt` / `mtrans_labelled_125.txt` (once created by `scripts/select_labelled_frames.py`): the
  KITTI train frames whose 3D labels MTrans may use. They are chosen from zero-shot 2D detections, never from `label_2`.
  The 125 frames are the first 125 of the 500, so the S1 ablation is nested.
