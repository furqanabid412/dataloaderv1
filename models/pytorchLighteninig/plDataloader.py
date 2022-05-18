import pytorch_lightning as pl
from det3d.torchie import Config
from torch.utils.data import DataLoader
from det3d.datasets import build_dataset


class NuscPLLoader(pl.LightningDataModule):
    def __init__(self,config_file, batch_size=4,num_workers=2,shuffle=False):
        super(NuscPLLoader, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        # Loading the configuration files
        # config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
        self.cfg = Config.fromfile(config_file)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = build_dataset(self.cfg.data.train)
            self.val_dataset = build_dataset(self.cfg.data.val)
        if stage == 'test':
            self.test_dataset =  build_dataset(self.cfg.data.test)

    def train_dataloader(self):
        dataloader= DataLoader(self.train_dataset,batch_size=self.batch_size,
                                     shuffle=self.shuffle,num_workers=self.num_workers,
                                     pin_memory=False,drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                      pin_memory=False, drop_last=True)
        return dataloader