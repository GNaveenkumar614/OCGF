"""KITTI-360 semantic label ids, as used by the accumulated point clouds (data_3d_semantics, field 'semantic').

From kitti360Scripts helpers/labels.py (github.com/autonomousvision/kitti360Scripts, MIT License). The point clouds use
the 'id' column; the 3D box XML uses 'kittiId' instead (see boxes3d.KITTI_ID_NAMES).
"""
LABEL_NAMES = {
    0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi", 4: "static", 5: "dynamic",
    6: "ground", 7: "road", 8: "sidewalk", 9: "parking", 10: "rail track", 11: "building", 12: "wall", 13: "fence",
    14: "guard rail", 15: "bridge", 16: "tunnel", 17: "pole", 18: "polegroup", 19: "traffic light", 20: "traffic sign",
    21: "vegetation", 22: "terrain", 23: "sky", 24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus",
    29: "caravan", 30: "trailer", 31: "train", 32: "motorcycle", 33: "bicycle", 34: "garage", 35: "gate", 36: "stop",
    37: "smallpole", 38: "lamp", 39: "trash bin", 40: "vending machine", 41: "box", 42: "unknown construction",
    43: "unknown vehicle", 44: "unknown object",
}
LABEL_IDS = {name: i for i, name in LABEL_NAMES.items()}

ROAD = 7
PARKING = 9
# surfaces at ground level: they give the ground height and the ground class of a cell
GROUND_SURFACES = (6, 7, 8, 9, 10, 22)          # ground, road, sidewalk, parking, rail track, terrain
