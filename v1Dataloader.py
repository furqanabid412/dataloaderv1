import numpy as np

from det3d.datasets.dataset_factory import get_dataset
from det3d.torchie import Config
from det3d.datasets import build_dataset
from torch.utils.data import DataLoader
from det3d.torchie.parallel import collate, collate_kitti

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from visualize import visualize_pcloud,visualize_camera,plot_colormap

# def collate_nusc(batch_list, samples_per_gpu=1):
#     print("wait")


if __name__ == "__main__":

    config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
    cfg = Config.fromfile(config_file)
    dataset = build_dataset(cfg.data.train)
    # dataset = build_dataset(cfg.data.val)
    print("dataset length is : ",dataset.__len__())

    i=0
    data = dataset.__getitem__(i)
    #
    points, labels, front_image,calib = data["points"],data["labels"],data["front_image"],data["calib"]

    visualize_camera(front_image)
    plot_colormap()

    from rangeProjection import do_range_projection
    projected_labels = do_range_projection(points,labels)


    visualize_pcloud(points,labels)


    data_loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4,pin_memory=False,)

    for i, data_batch in tqdm(enumerate(data_loader)):
        print(i)

