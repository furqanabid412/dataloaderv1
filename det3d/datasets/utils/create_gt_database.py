import pickle
from pathlib import Path
import os 
import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

# import logging
# from datetime import datetime
#
# # datetime object containing current date and time
# #now we will Create and configure logger
# logging.basicConfig(filename="logs/latest_log.log", format='%(asctime)s %(message)s', filemode='w')
# #Let us Create an object
# logger=logging.getLogger()
# #Now we are going to Set the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)


dataset_name_map = {
    "NUSC": "NuScenesDataset",
    "WAYMO": "WaymoDataset"
}

# lidarseg labels to class names

general_to_label ={0: 'ignore', 1: 'ignore', 2: 'pedestrian', 3: 'pedestrian',
                   4: 'pedestrian', 5: 'ignore', 6: 'pedestrian', 7: 'ignore',
                   8: 'ignore', 9: 'barrier', 10: 'ignore', 11: 'ignore',
                   12: 'traffic_cone', 13: 'ignore', 14: 'bicycle', 15: 'bus',
                   16: 'bus', 17: 'car', 18: 'construction_vehicle', 19: 'ignore',
                   20: 'ignore', 21: 'motorcycle', 22: 'trailer', 23: 'truck',
                   24: 'road', 25: 'ignore', 26: 'sidewalk', 27: 'terrain',
                   28: 'building', 29: 'ignore', 30: 'vegetation', 31: 'ignore'}



def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    rate=1.,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    includeSeg=False,
    **kwargs,
):
    if includeSeg:
        pipeline = [
            {"type": "LoadPointCloudFromFile", "dataset": dataset_name_map[dataset_class_name], "use_img": True},
            {"type": "LoadLidarSegLabels","dataset":'NUSCENES',"doAugmentation": True},
            {"type": "LoadPointCloudAnnotations", "with_bbox": True, "use_img": True},
        ]
    else:
        pipeline = [
            {    "type": "LoadPointCloudFromFile","dataset": dataset_name_map[dataset_class_name],"use_img": True},
            {"type": "LoadPointCloudAnnotations", "with_bbox": True, "use_img": True},
        ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            use_img=True,
            nsweeps=kwargs["nsweeps"],
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name in ["WAYMO", "NUSC"]: 
        if db_path is None:
            db_path = root_path / "gt_database_{:03d}rate_{:02d}sweeps_withvelo_crossmodal".format(int(rate*100), nsweeps)
        if dbinfo_path is None:
            dbinfo_path = root_path / "dbinfos_{:03d}rate_{:02d}sweeps_withvelo_crossmodal.pkl".format(int(rate*100), nsweeps)
    else:
        raise NotImplementedError()

    if dataset_class_name == "NUSC":
        point_features = 5 + 3
    elif dataset_class_name == "WAYMO":
        point_features = 5 if nsweeps == 1 else 6 
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    cam_name = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        # if nsweeps > 1:
        #     points = sensor_data["lidar"]["combined"]
        # else:
        #     points = sensor_data["lidar"]["points"]

        points = sensor_data["lidar"]["combined"]
        if nsweeps == 0 and includeSeg:
            point_features+=1
            points = np.concatenate([points, sensor_data["lidar"]["lidarseg"]], axis=1)

            
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        gt_frustums = annos["frustums"]

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE') 
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]


        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        annos_img = sensor_data["camera"]["annotations"]
        avail_2d = annos_img["avail_2d"]
        boxes_2d = annos_img["boxes_2d"]  # N * 6 * 4
        depths = annos_img["depths"]
        import cv2
        imgs = [cv2.imread(sensor_data["camera"]["cam_paths"][cam]) for cam in cam_name]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue 
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                # original order was image -> pts
                # but i modified it to pts ->image
                # because we need to check the pts condition first

                # save pts
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                lidar_filename = filename
                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]

                totalPts = len(gt_points[:, -1])
                matchedPts = 0
                for j in range(totalPts):
                    segClass = general_to_label[gt_points[j, -1]]
                    if segClass == names[i]:
                        matchedPts += 1

                if matchedPts >= 5 and (matchedPts / totalPts) >= 0.90:
                    with open(filepath, "w") as f:
                        try:
                            gt_points[:, :point_features].tofile(f)
                        except:
                            print("process {} files".format(index))
                            break
                elif matchedPts < 5:
                    # logger.info("index # {} object # {} has less than 5 points".format(index, i))
                    continue
                elif (matchedPts / totalPts) < 0.90:
                    # logger.info("index # {} object # {} has {} matchedPts out of {} total points".
                    #       format(index, i, totalPts - matchedPts, matchedPts))
                    continue


                # save image patches
                cam_paths = [''] * len(cam_name)
                for cam_id, flag in enumerate(avail_2d[i]):
                    if flag:
                        cur_box = boxes_2d[i, cam_id]
                        img_patch = imgs[cam_id][cur_box[1]:cur_box[3], cur_box[0]:cur_box[2], :]
                        filename = '{}_{}_{}_{}.jpg'.format(image_idx, names[i], i, cam_id)
                        filepath = os.path.join(str(db_path), names[i], filename)
                        cam_paths[cam_id] = filepath
                        cv2.imwrite(filepath, img_patch)




            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], lidar_filename)
                else:
                    db_dump_path = str(lidar_filename)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    "frustum": gt_frustums[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }

                db_info.update({
                    "avail_2d": avail_2d[i],
                    "bbox": boxes_2d[i],
                    "depth": depths[i],
                    "cam_paths": np.array(cam_paths)
                })

                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)
