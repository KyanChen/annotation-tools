import json
import os
import torch
import numpy as np
from torch import optim, nn, utils, Tensor
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
import pytorch_lightning as pl
from einops import repeat
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar


class SegModel(pl.LightningModule):
    def __init__(self, lr, n_class=2, in_channels=3):
        super().__init__()
        self.save_hyperparameters()
        # 256 16
        self.down_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=(2, 4), padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 64 8
        self.down_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 32 4
        self.down_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 16 2
        self.up_layer3 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 32 4
        self.up_layer2 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 64 8
        self.out_layer = nn.Conv2d(64+32, n_class, kernel_size=3, padding=1)

    def forward(self, x):
        # 256 16
        x1 = self.down_layer1(x)  # 64 8
        x2 = self.down_layer2(x1)  # 32 4
        x3 = self.down_layer3(x2)  # 16 2

        up_x2 = self.up_layer3(x3)  # 32 4
        up_x2 = torch.cat((up_x2, x2), dim=1)
        up_x1 = self.up_layer2(up_x2)  # 64 8
        up_x1 = torch.cat((up_x1, x1), dim=1)  # B 64+32 64 8
        up_x = F.interpolate(up_x1, scale_factor=(2, 4), mode='bilinear')
        x = self.out_layer(up_x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        acc = self._calculate_acc(batch, mode="val")
        return acc

    def test_step(self, batch, batch_idx):
        acc = self._calculate_acc(batch, mode="test")
        return acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        x, labels = batch['x'], batch['label']
        preds = self.forward(x)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        losses = {'loss': loss, 'acc': acc}
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return losses

    def _calculate_acc(self, batch, mode="test"):
        x, labels = batch['x'], batch['label']
        preds = self.forward(x)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log("%s_acc" % mode, acc)
        return acc


