import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

rate = 0.01
nsweeps = 10
data_root = "./data/nuscenes"
db_info_path = "/dbinfos_{:03d}rate_{:02d}sweeps_withvelo_crossmodal.pkl".format(int(rate*100), nsweeps)

train_anno = "/infos_train_{:02d}sweeps_withvelo_filter_True_{:03d}rate_crossmodal.pkl".format(nsweeps, int(rate*100))
val_anno = "/infos_val_10sweeps_withvelo_filter_True_crossmodal.pkl"
test_anno = ""
version = "v1.0-trainval"

use_aug = True
DOUBLE_FLIP = False

# voxel_size = [0.075, 0.075, 0.2]
# pc_range = [-54, -54, -5.0, 54, 54, 3.0]
voxel_size = [0.1, 0.1, 0.2]
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        share_conv_channel=64,
        dcn_head=False
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=pc_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_size[:2],
    double_flip=DOUBLE_FLIP
)


# dataset settings
dataset_type = "NuScenesDataset"

db_num = 2
db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path=data_root+db_info_path,
    sample_groups=[
        dict(car=2*db_num),
        dict(truck=3*db_num),
        dict(construction_vehicle=7*db_num),
        dict(bus=4*db_num),
        dict(trailer=6*db_num),
        dict(barrier=2*db_num),
        dict(motorcycle=6*db_num),
        dict(bicycle=6*db_num),
        dict(pedestrian=2*db_num),
        dict(traffic_cone=2*db_num),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) if use_aug else None


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
    remove_points_after_sample=True,  # False
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=pc_range,
    voxel_size=voxel_size,
    max_points_in_voxel=10,
    max_voxel_num=[60000, 120000],
    double_flip=DOUBLE_FLIP,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="DoubleFlip") if DOUBLE_FLIP else dict(type="Empty"),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat", double_flip=DOUBLE_FLIP),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+train_anno,
        ann_file=data_root+train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+val_anno,
        test_mode=True,
        ann_file=data_root+val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+test_anno,
        ann_file=data_root+test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
