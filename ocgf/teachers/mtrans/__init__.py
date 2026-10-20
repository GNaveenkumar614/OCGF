"""MTrans object teacher (vendored, MPL-2.0; see README.md and LICENSE in this folder).

2D box + LiDAR frustum -> 3D box in the LiDAR frame + confidence regressed to its 3D IoU. Runs offline; the student
(ocgf.models / training / evaluation) never imports it and reads its output files instead.
"""
