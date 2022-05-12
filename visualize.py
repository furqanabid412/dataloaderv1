import numpy as np
import open3d
import cv2



def visualize_pcloud(scan_points,scan_labels):
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    import yaml
    label_colormap = yaml.safe_load(open('colormap.yaml', 'r'))
    label_colormap =label_colormap['color_map']
    # rendering the pcloud in open3d
    pcd = open3d.geometry.PointCloud()
    # scan_points = scan_points.numpy()
    scan_points = scan_points[:,:3]
    pcd.points = open3d.utility.Vector3dVector(scan_points)
    # scan_labels = scan_labels.numpy()
    # scan_labels = scan_labels[scan_labels != -1]
    scan_labels = scan_labels.astype(np.int)
    colors = np.array([label_colormap[x] for x in scan_labels])
    pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(width=width, height=height, left=100)
    # vis.add_geometry(pcd)
    vis = open3d.visualization.draw_geometries([pcd])
    open3d.visualization.ViewControl()


def visualize_camera(img):
    cv2.imwrite('data.jpg', img)
    cv2.imshow('img_show',img)
    cv2.waitKey(0)


# import cv2
# img = res["img"][cam_id]
# pts_uv_filtered = pts_uv[mask].astype(np.int)
# cur_rgb = img[pts_uv_filtered[:, 1], pts_uv_filtered[:, 0], :].astype(np.uint8)
#
# img_test = np.zeros_like(img, dtype=np.uint8)
# for i in range(pts_uv_filtered.shape[0]):
#     color = (int(cur_rgb[i, 0]), int(cur_rgb[i, 1]), int(cur_rgb[i, 2]))
#     p = (int(pts_uv_filtered[i, 0]), int(pts_uv_filtered[i, 1]))
#     cv2.circle(img_test, p, 1, color=color)
# img_show = np.concatenate([img, img_test], axis=0)
# img_show = cv2.resize(img_show, (800, 900))
# cv2.imshow('img_show', img_show)
# cv2.waitKey(0)
