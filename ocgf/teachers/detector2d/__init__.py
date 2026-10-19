"""Zero-shot 2D detector teacher: torchvision Faster R-CNN v2 with COCO weights (BSD-3), never fine-tuned in the main
configuration. Finds objects for MTrans, which cannot find them itself; a missed detection is a potential false-free cell.
"""
