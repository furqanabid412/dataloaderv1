import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import torch

from models.unet import UNet
from tools.iou import IouEval,AverageMeter


class plModel(pl.LightningModule):
    def __init__(self, n_channels,n_classes,ignore_class=0,arch='unet'):
        super(plModel, self).__init__()

        print("Configuring model")

        if arch=='unet':
            self.model = UNet(n_channels=n_channels,n_classes=n_classes,bilinear= True)
        else:
            raise TypeError("{} model is not valid".format(arch))

        # Loss function
        self.xentropy = nn.CrossEntropyLoss()

        # Metrics : iou + miou (manual)
        self.acc = AverageMeter()
        self.miou = AverageMeter()
        self.iou = AverageMeter()
        self.evaluator = IouEval(n_classes=n_classes, device="cuda", ignore=ignore_class)

        self.class_names = ['ignore','car','pedestrian','bicycle','motorcycle','bus','truck',
                            'construction_vehicle','trailer','barrier','traffic_cone','driveable_surface',
                            'other_flat','sidewalk','terrain','manmade','vegetation']

    def forward(self, x):
        out = self.model(x)  # x : [B,C,H,W]  --> out : [B,n,H,W]
        return out


    def loss(self, y_hat, y):
        loss = self.xentropy(y_hat, y)
        return loss


    def training_step(self, batch, batch_idx):

        x, y = batch["projected_rangeimg"], batch["projected_labels"]
        y = y.long()
        logits = self(x) # this calls self.forward
        loss = self.loss(logits,y)
        preds = torch.argmax(logits, 1)

        # getting metrics
        self.evaluator.reset()
        self.evaluator.addBatch(preds, y)
        train_miou, train_iou = self.evaluator.getIoU()
        self.acc.update(self.evaluator.getacc())
        self.miou.update(train_miou)
        self.iou.update(train_iou)

        # log step metric
        for i, iou in enumerate(self.iou.avg):
            self.log('{}-iou'.format(self.class_names[i]), iou, on_step=True, on_epoch=False)

        self.log('train_step_accuracy', self.acc.val, on_step=True, on_epoch=False)
        self.log('train_step_miou', self.miou.val, on_step=True, on_epoch=False)

        self.log('train_loss', loss)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # for i, iou in enumerate(self.iou.avg):
        #     self.log('iou_{}'.format(self.class_names[i - 1]), iou, on_step=False, on_epoch=True)

        self.log('train_epoch_avg_accuracy', self.acc.avg, on_step=False, on_epoch=True)
        self.log('train_epoch_avg_miou', self.miou.avg, on_step=False, on_epoch=True)

        self.acc.reset()
        self.miou.reset()
        self.iou.reset()

    def test_step(self, batch, batch_idx):

        # needs to be modified
        xs,y = batch["proj_scan_only"] , batch["proj_label_only"]
        logits = self(xs)

        # log step metric
        preds = torch.argmax(logits, 1)

        self.evaluator.reset()
        self.evaluator.addBatch(preds, y)
        self.acc.update(self.evaluator.getacc())
        train_miou, train_iou = self.evaluator.getIoU()
        self.miou.update(train_miou)
        self.iou.update(train_iou)

        self.log('test_step_accuracy', self.acc.val, on_step=True, on_epoch=False)
        self.log('test_step_miou', self.miou.val, on_step=True, on_epoch=False)


        for i, iou in enumerate(self.iou.avg):
            self.log('iou_{}'.format(self.class_names[i-1]), iou, on_step=True, on_epoch=False)

        return {'miou': self.miou.val}

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API

        self.log('test_epoch_avg_accuracy', self.acc.avg, on_step=False, on_epoch=True)
        self.log('test_epoch_avg_miou', self.miou.avg, on_step=False, on_epoch=True)

        self.acc.reset()
        self.miou.reset()
        self.iou.reset()

    def configure_optimizers(self):

        # optimizer = torch.optim.SGD(self.train_dicts,
        #                       lr=self.ARCH["train"]["lr"],
        #                       momentum=self.ARCH["train"]["momentum"],
        #                       weight_decay=self.ARCH["train"]["w_decay"])

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        return  [optimizer]

        #, "optimizer","lr_scheduler": scheduler, "monitor": monitor}
