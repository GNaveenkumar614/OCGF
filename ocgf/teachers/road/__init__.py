"""Road teacher: trajectory-based LiDAR-camera road autolabeling (Alamikkotervo et al., IEEE Access 2025), adapted
from github.com/eerik98/lidar-camera-road-autolabeling (MIT; LICENSE-lidar-camera-road-autolabeling in this folder).

Training frames only: it needs the car's future path. Its labels are pseudo-labels, audited against KITTI-360's 2D
semantics and never used for evaluation. Steps: trajectory.py (path in the scan), lidar_label.py, camera_label.py,
label.py (one frame, outputs, audit).
"""
