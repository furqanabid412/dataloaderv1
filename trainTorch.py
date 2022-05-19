import os
import torch
import numpy
import wandb
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.pytorchLighteninig.plDataloader import NuscPLLoader
from models.pytorchLighteninig.plTrainer import plModel


pl.seed_everything(hash("setting random seeds") % 2**32 - 1)


log_dirs = {'myPC':'E:/logs/dataloaderv2.0',} # add more for servers


def pl_train():

    # setting up the dataloader (PL)
    config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
    dataset = NuscPLLoader(config_file, batch_size=2, num_workers=4)
    dataset.setup('fit')

    # setting up the model (PL)
    model = plModel(n_channels=5, n_classes=17, ignore_class=0, arch='unet')

    # logger
    wandb = WandbLogger(project='NuscAug', entity='v1')
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(log_dirs['myPC'], 'checkpoints'), verbose=True, )
    bar = pl.callbacks.TQDMProgressBar()

    # trainer
    trainer = pl.Trainer(gpus=-1, auto_scale_batch_size=False,enable_checkpointing=True,
                         log_every_n_steps=1, logger=wandb, callbacks=[checkpoint, bar], max_epochs=10 )

    trainer.fit(model, dataset)



def testing(PATH):
    # setting up the dataloader for testing(PL)
    config_file = './configs/nusc/lidarseg/nusc_lidarseg.py'
    dataset = NuscPLLoader(config_file, batch_size=2, num_workers=2)
    dataset.setup('test')

    # load the model parameters from checkpoints (PL)
    model = plModel.load_from_checkpoint(PATH)
    model.eval()

    # logger
    wandb = WandbLogger(project='NuscAug', entity='v1')
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(log_dirs['myPC'], 'checkpoints'), verbose=True, )
    bar = pl.callbacks.TQDMProgressBar()

    # trainer
    trainer = pl.Trainer(gpus=-1, auto_scale_batch_size=False,
                         log_every_n_steps=1, logger=wandb, callbacks=[checkpoint, bar], max_epochs=20,
                         strategy="ddp_sharded", )

    trainer.test(model, dataset)


if __name__ == "__main__":
    # wandb.login()
    pl_train()


