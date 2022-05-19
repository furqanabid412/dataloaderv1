import torch
import numpy
from tqdm import tqdm
from models.pytorchLighteninig.plDataloader import NuscPLLoader



if __name__ == "__main__":
    config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
    dataset = NuscPLLoader(config_file,batch_size=2,num_workers=2)
    dataset.setup('fit')
    dataloader=dataset.train_dataloader()

    cnt = 0
    fst_moment = torch.empty(5)
    snd_moment = torch.empty(5)

    for data in tqdm(dataloader):
        images=data['projected_rangeimg']
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print("testing")
