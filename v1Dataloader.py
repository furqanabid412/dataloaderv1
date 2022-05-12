import numpy as np

from det3d.datasets.dataset_factory import get_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.datasets import build_dataset

from torch.utils.data import DataLoader
from det3d.torchie.parallel import collate, collate_kitti

import warnings
warnings.filterwarnings("ignore")

from tools.visualization import show_pts_in_box
import open3d


import time

def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                "anno_box", "ind", "mask", 'cat']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
            "img",
            "voxels_uv",
            "voxel_valid",
            "voxels_imgfeat",
            "bev_sparse"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def visualize_pcloud(scan_points,scan_labels):
    import yaml
    label_colormap = yaml.safe_load(open('colormap.yaml', 'r'))
    # rendering the pcloud in open3d
    pcd = open3d.geometry.PointCloud()
    # scan_points = scan_points.numpy()
    scan_points = scan_points[:,:2]
    pcd.points = open3d.utility.Vector3dVector(scan_points)
    # scan_labels = scan_labels.numpy()
    scan_labels = scan_labels[scan_labels != -1]
    colors = np.array([label_colormap[x] for x in scan_labels])
    pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(width=width, height=height, left=100)
    # vis.add_geometry(pcd)
    vis = open3d.visualization.draw_geometries([pcd])
    open3d.visualization.ViewControl()

if __name__ == "__main__":
    config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
    cfg = Config.fromfile(config_file)

    dataset = build_dataset(cfg.data.train)
    # dataset = build_dataset(cfg.data.val)
    print(dataset.__len__())

    i=10
    points,labels,front_image = dataset.__getitem__(i)

    visualize_pcloud(points,labels)


    for i in range(100):
        data = dataset.__getitem__(i)
        break
    print(11)

    data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_kitti,
            pin_memory=False,
        )

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = model.cuda()
    # model.eval()

    for i, data_batch in enumerate(data_loader):
        example = example_to_device(
            data_batch, 'cuda', non_blocking=False
        )
        losses = model(example, return_loss=True)
        print('loss', losses)
        break


