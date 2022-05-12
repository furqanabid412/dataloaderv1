import numpy as np
import cv2
from det3d.core.bbox import box_np_ops


def transform2Spherical(points):
    pts_r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
    pts_theta = np.arccos(points[:, 2] / pts_r)
    pts_phi = (np.arctan(points[:, 1] / points[:, 0]) + (points[:, 0] < 0) * np.pi + np.pi * 2) % (np.pi * 2)  # [0, 2*pi]
    # assert pts_phi.all() >= 0 and pts_phi.all() <= 2 * np.pi
    pts_rr = np.vstack([pts_r, pts_theta, pts_phi]).T
    return pts_rr


def procress_image(imgs, gt_dict):
    new_imgs = []
    for i in range(6):
        valid = gt_dict['avail_2d'][:, i]
        new_img = imgs[i].copy()
        cur_depths = gt_dict['depths'][:, i].copy()
        cur_depths[np.logical_not(valid)] = -1.
        idxs = np.argsort(cur_depths * -1)  # index of reversed sort

        for idx in idxs[:sum(valid > 0)]:
            bbox = gt_dict['bboxes'][idx, i]
            if gt_dict['pasted'][idx] > 0:
                patch = cv2.imread(gt_dict['patch_path'][idx, i])
                cur_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                assert cur_size[0] > 0 and cur_size[1] > 0
                patch = cv2.resize(patch, cur_size)
                new_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = patch

            else:
                new_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = imgs[i, bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        new_imgs.append(new_img)
        # cv2.imshow('t2', new_img)
        # cv2.waitKey(0)

    gt_dict.pop('patch_path')
    return np.array(new_imgs)


def procress_points(points, sampled_points, gt_boxes_mask, gt_dict):
    points = np.concatenate([sampled_points, points], axis=0)

    pts_rr = transform2Spherical(points)

    valid = np.zeros([points.shape[0]], dtype=np.bool_)
    valid_filter = np.zeros([points.shape[0]], dtype=np.bool_)

    # First "No. of sampled points" are valid samples for filtering
    valid_sample = np.zeros([points.shape[0]], dtype=np.bool_)
    valid_sample[:sampled_points.shape[0]] = 1

    # [No of points, No of gt_boxes]
    point_indices = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])

    depths = np.sqrt(np.square(gt_dict['gt_boxes'][:, 0]) + np.square(gt_dict['gt_boxes'][:, 1]) +
                     np.square(gt_dict['gt_boxes'][:, 2]))
    # sorting the depths in the ascending order (low to high)
    idxs = np.argsort(depths)

    for idx in idxs:
        cur_frus = gt_dict["gt_frustums"][idx]

        # STEP 1 : Find those points which are within the gt_box theta and phi

        # valid points in object frustum
        # points theta > gt_frustum min theta and points theta < gt_frustum max theta
        # TRUE if points are within the range of gt_frustum min and max
        val = (pts_rr[:, 1] > cur_frus[1, 0, 0]) & (pts_rr[:, 1] < cur_frus[1, 1, 0])

        # get phi values in sp_frus [min,max]
        sp_frus = [cur_frus[2, :, 0]] if cur_frus[2, 0, 1] < 0 else [cur_frus[2, :, 0], cur_frus[2, :, 1]]
        for frus in sp_frus:
            # if val and (points phi > gt_frustum min phi and points phi < gt_frustum max phi)
            val = val & (pts_rr[:, 2] > frus[0]) & (pts_rr[:, 2] < frus[1])

        # STEP 2 : Find out the points which belongs to current gt_box but not been filtered so far

        val1 = (point_indices[:, idx]) & (valid_filter < 1)  # points in 3D box and not filtered
        valid[val1] = 1  # remained points of current object - valid set to 1

        # STEP 3 : Take logical and of points in the current field of view (step1) and needs to be filtered (step2)

        val = val & (np.logical_not(valid))

        # STEP 4 : Since the gt_dict contains all the objects (original+sampled)
        # only filter those points which belongs to original foreground

        if not gt_dict["pasted"][idx]:  # sampled box -> filter bg and fg; original box -> only filter sampled fg
            val = val & valid_sample
        valid_filter[val] = 1

        # from tools.visual import show_pts_in_box
        # from tools.visualization import show_pts_in_box
        # show_pts_in_box(points, points[valid], points[val1], points[val])
        # red,light green,brown, purple
        # show_pts_in_box(None, points[val1], points[valid], points[val])

    # from tools.visual import show_pts_in_box
    # show_pts_in_box(points, points[valid], sampled_points, points[valid_filter])

    # remove those points which needs to be filtered
    points = points[valid_filter < 1]

    # After filtering if there is a gt_box (probably in original pcloud) remove it from list
    for i in range(gt_dict['gt_boxes'].shape[0]):
        val = (valid_filter < 1) & (point_indices[:, i])
        if not val.any():
            gt_boxes_mask[i] = False

    gt_dict.pop('pasted')
    gt_dict.pop('gt_frustums')
    return points, gt_boxes_mask




