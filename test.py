import numpy as np
import pickle

from pathlib import Path
from functools import reduce
from typing import List
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
import yaml
# import open3d
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

try:
    from nuscenes import NuScenes,NuScenesExplorer
    from nuscenes.utils import splits
    from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
        get_labels_in_coloring, create_lidarseg_legend, paint_points_label
    from nuscenes.utils.geometry_utils import transform_matrix, BoxVisibility
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")


color_map:{ # RGB values for each class
  0: [0,0,0],
  1: [70,130,180],
  2: [0,0,230],
  3: [135,206,235],
  4: [100,149,237],
  5: [219,112,147],
  6: [0,0,128],
  7: [240,128,128],
  8: [138,43,226],
  9: [112,128,144],
  10: [210,105,30],
  11: [105,105,105],
  12: [47,79,79],
  13: [188,143,143],
  14: [220,20,60],
  15: [255,127,80],
  16: [255,69,0],
  17: [255,158,0],
  18: [233,150,70],
  19: [255,83,0],
  20: [255,215,0],
  21: [255,61,99],
  22: [255,140,0],
  23: [255,99,71],
  24: [0,207,191],
  25: [175,0,75],
  26: [75,0,75],
  27: [112,180,60],
  28: [222,184,135],
  29: [255,228,196],
  30: [0,175,0],
  31: [255,240,245],
}


root =  'E:/Datasets/NuScenes/new_test_run/v1.0-mini'
version =  'v1.0-mini'
nusc = NuScenes(version=version,dataroot=root,verbose=True)

curr_sample = nusc.sample[0]

