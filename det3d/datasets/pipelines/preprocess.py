import numpy as np
import cv2

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

from det3d.datasets.utils.cross_modal_augmentation import *


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            self.remove_points_after_sample = cfg.get('remove_points_after_sample', False)
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)
        self.use_img = cfg.get("use_img", False)
        self.doLidarSegmentation=cfg.doLidarSegmentation

    def __call__(self, res, info):

        res["mode"] = self.mode

        # copy the pointcloud data N*8(Nuscenes) into points

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError


        #  For training save ground truth annotations into gt_dict
        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
                "gt_frustums": anno_dict["frustums"],
            }

            # if camera images are also used then save those information into gt_dict too
            if self.use_img:
                cam_anno_dict = res["camera"]["annotations"]
                gt_dict["bboxes"] = cam_anno_dict["boxes_2d"]
                gt_dict["avail_2d"] = cam_anno_dict["avail_2d"]
                gt_dict["depths"] = cam_anno_dict["depths"]


        # If training + augmentation mode is on
        # if there are objects with annotations like  ["DontCare", "ignore", "UNKNOWN"]
        # then drop them from the gt_dict
        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name( gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"])

            _dict_select(gt_dict, selected)


            # Check if each gt_annotation contains lidar points more than min_points_in_gt
            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)


            # create a mask whether the gt_classes contains the classes for sampling(augmentation)
            gt_boxes_mask = np.array([n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

            if self.db_sampler:
                # calibration parameters (dictionary) of original ground truth
                # ref_to_global = (4,4) . ref is lidar here, global might be car
                # cams_from_global = 6*(4,4). global (car) to cameras
                # cam_intrinsic = (3,3). 3d to 2d projection

                calib = res["calib"] if "calib" in res else None

                # Pcloud features [(x,y,z,reflect,time),(proj_u,proj_v,camera_id)]
                pt_features=5+3
                if self.doLidarSegmentation:
                    pt_features+=1
                selected_feature = np.ones([pt_features])  # xyzrt, u v cam_id
                selected_feature[5:5 + 3] = 1. if self.use_img else 0.


                sampled_dict = self.db_sampler.sample_all_v2(res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"], gt_dict["gt_names"],gt_dict["gt_frustums"],
                    selected_feature,random_crop=False,revise_calib=True,
                    gt_group_ids=None,calib=calib,cam_name=res['camera']['name'],
                    road_planes=None,  # res["lidar"]["ground_plane"]
                    doLidarSegmentation=self.doLidarSegmentation
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    sampled_frustums = sampled_dict["gt_frustums"]
                    sampled_num = sampled_gt_boxes.shape[0]
                    origin_num = gt_dict['gt_boxes'].shape[0]
                    gt_dict["pasted"] = np.concatenate([np.zeros([gt_dict["gt_boxes"].shape[0]]), np.ones(sampled_num)],
                                                       axis=0)
                    if self.use_img:
                        gt_dict["avail_2d"] = np.concatenate([gt_dict["avail_2d"], sampled_dict["avail_2d"]], axis=0)
                        gt_dict["bboxes"] = np.concatenate([gt_dict["bboxes"], sampled_dict["bboxes"]], axis=0)
                        gt_dict["depths"] = np.concatenate([gt_dict["depths"], sampled_dict["depths"]], axis=0)
                        gt_dict["patch_path"] = np.concatenate(
                            [[['']*6 for i in range(origin_num)], sampled_dict["patch_path"]], axis=0)

                    # gt_boxes放最后
                    gt_dict["gt_names"] = np.concatenate([gt_dict["gt_names"], sampled_gt_names], axis=0)
                    gt_dict["gt_boxes"] = np.concatenate([gt_dict["gt_boxes"], sampled_gt_boxes])
                    gt_dict["gt_frustums"] = np.concatenate([gt_dict["gt_frustums"], sampled_frustums])
                    gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                    if self.remove_points_after_sample:
                        masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    if self.use_img:  # paste imgs
                        res['img'] = procress_image(res['img'], gt_dict)

                    # from tools.visualization import show_pts_in_box
                    # show_pts_in_box(points, sampled_points)

                    points, gt_boxes_mask = procress_points(points, sampled_points, gt_boxes_mask, gt_dict)

            if self.use_img:
                gt_dict.pop('avail_2d')
                gt_dict.pop('bboxes')
                gt_dict.pop('depths')

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        if self.shuffle_points:
            np.random.shuffle(points)

        if self.use_img:
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).astype(np.float32)
        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num
        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        voxels_uv = np.zeros([max_voxels, self.max_points_in_voxel, 3]).astype(np.float32)
        voxels_uv[:num_voxels[0]] = voxels[:, :, -4:-1]
        voxels_uv = voxels_uv.reshape((max_voxels, self.max_points_in_voxel, 1, 3))
        voxel_valid = np.zeros([max_voxels]).astype(np.bool_)
        voxel_valid[:num_voxels[0]] = 1

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            voxels_uv=voxels_uv,
            voxel_valid=voxel_valid,
            shape=grid_size,
            range=pc_range,
            size=voxel_size,
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            flip_voxels_uv = np.zeros([max_voxels, self.max_points_in_voxel, 3]).astype(np.float32)
            flip_voxels_uv[:flip_num_voxels[0]] = flip_voxels[:, :, -4:-1]
            flip_voxels_uv = flip_voxels_uv.reshape((max_voxels, self.max_points_in_voxel, 1, 3))
            flip_voxel_valid = np.zeros([max_voxels]).astype(np.bool_)
            flip_voxel_valid[:flip_num_voxels[0]] = 1

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                voxels_uv=flip_voxels_uv,
                voxel_valid=flip_voxel_valid,
                shape=grid_size,
                range=pc_range,
                size=voxel_size,
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            flip_voxels_uv = np.zeros([max_voxels, self.max_points_in_voxel, 3]).astype(np.float32)
            flip_voxels_uv[:flip_num_voxels[0]] = flip_voxels[:, :, -4:-1]
            flip_voxels_uv = flip_voxels_uv.reshape((max_voxels, self.max_points_in_voxel, 1, 3))
            flip_voxel_valid = np.zeros([max_voxels]).astype(np.bool_)
            flip_voxel_valid[:flip_num_voxels[0]] = 1

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                voxels_uv=flip_voxels_uv,
                voxel_valid=flip_voxel_valid,
                shape=grid_size,
                range=pc_range,
                size=voxel_size,
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            flip_voxels_uv = np.zeros([max_voxels, self.max_points_in_voxel, 3]).astype(np.float32)
            flip_voxels_uv[:flip_num_voxels[0]] = flip_voxels[:, :, -4:-1]
            flip_voxels_uv = flip_voxels_uv.reshape((max_voxels, self.max_points_in_voxel, 1, 3))
            flip_voxel_valid = np.zeros([max_voxels]).astype(np.bool_)
            flip_voxel_valid[:flip_num_voxels[0]] = 1

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                voxels_uv=flip_voxels_uv,
                voxel_valid=flip_voxel_valid,
                shape=grid_size,
                range=pc_range,
                size=voxel_size,
            )

        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"] 
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info

