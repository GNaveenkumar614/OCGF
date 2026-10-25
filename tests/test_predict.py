"""The demo script's helpers: free-road decision, BEV colours, cell ground points back in the original image."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import predict  # noqa: E402


def test_free_road_needs_road_and_neither_occupied_nor_unknown():
    road = np.array([[0.9, 0.9, 0.9, 0.2]])
    occ = np.array([[0.1, 0.8, 0.1, 0.1]])
    unk = np.array([[0.1, 0.1, 0.7, 0.1]])
    probs = {"road": road, "occ": occ, "unk": unk}
    assert predict.d_free(probs).tolist() == [[True, False, False, False]]
    colours = predict.bev_classes(probs)
    assert tuple(colours[0, 0]) == predict.FREE and tuple(colours[0, 1]) == predict.OCCUPIED
    assert tuple(colours[0, 2]) == predict.UNKNOWN and tuple(colours[0, 3]) == predict.OTHER


def test_cell_ground_points_go_back_to_the_original_image():
    from ocgf.datasets.student_sample import IMAGE_H, IMAGE_W
    u_px, v_px = 700.0, 300.0                                   # a pixel of the cropped image
    g = np.array([[[2 * (u_px + 0.5) / IMAGE_W - 1, 2 * (v_px + 0.5) / IMAGE_H - 1], [-2.0, -2.0]]])
    u, v = predict.image_points(g, crop_left=4)
    assert u[0, 0] == u_px + 4 and v[0, 0] == v_px and np.isnan(u[0, 1])
