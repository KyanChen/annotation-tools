import glob
import json
import os
import random

import PIL
from matplotlib import pyplot as plt
import cv2
import torch
import numpy as np
from skimage import io
from torch import optim, nn, utils, Tensor
from torchvision import transforms
import pytorch_lightning as pl
from plain_seg_model import SegModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar


def train_model(root_dir='EXP1', train_loader=None, val_loader=None, **model_kwargs):
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        default_root_dir=root_dir,
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir+'/checkpoints',
                filename='epoch_{epoch:02d}',
                auto_insert_metric_name=False,
                every_n_epochs=10,
                save_last=True
            ),
            LearningRateMonitor("epoch"),
            TQDMProgressBar()
        ],
    )
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir+'/checkpoints', "last.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = SegModel.load_from_checkpoint(pretrained_filename)
    else:
        # pl.seed_everything(42)  # To be reproducable
        model = SegModel(**model_kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SegModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(MyDataset, self).__init__()
        self.data = glob.glob(data_path+'/*.jpg')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data[idx]
        label_file = img_file.replace('.jpg', '.png')
        img = io.imread(img_file, as_gray=False)
        label = io.imread(label_file, as_gray=True).astype(np.uint8)

        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            area = cv2.contourArea(contours[i])
            if (w / h > 2 or w / h < 1 / 2) and area > 50:
                filtered_contours.append(contours[i])
        random_id = torch.randint(len(filtered_contours), size=[])
        contour = filtered_contours[random_id]
        x, y, w, h = cv2.boundingRect(contour)
        scale_w = random.randint(int(w * 0.05) + 1, int(w * 0.2) + 1)
        scale_h = random.randint(int(h * 0.1) + 1, int(h * 0.5) + 1)
        x = x - scale_w // 2
        y = y - scale_h // 2
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w += scale_w
        h += scale_h
        img = torch.from_numpy(img[y:y+h, x:x+w, ...]) / 255.
        label = torch.from_numpy(label[y:y+h, x:x+w, ...])
        # plt.imshow(img.numpy())
        # plt.show()
        img = torch.permute(img, (2, 0, 1))
        label = label.view(1, *label.shape)
        inputs = torch.cat((img, label), dim=0)
        # inputs = transforms.RandomResizedCrop(size=(16, 256), scale=(1, 1.0))(inputs)
        img = transforms.Resize(size=(16, 256))(inputs[:3])
        label = transforms.Resize(size=(16, 256), interpolation=PIL.Image.NEAREST)(inputs[3:])
        # plt.imshow(img.permute((1, 2, 0)).numpy())
        # plt.show()
        # plt.imshow(label.permute((1, 2, 0)).numpy())
        # plt.show()
        return {'x': img.float(), 'label': label.squeeze(0).long()}


if __name__ == '__main__':
    data_path = 'data'
    train_dataset = MyDataset(data_path=data_path)
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=2, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    model_kwargs = dict(lr=1e-4, n_class=2)
    model, result = train_model(
        root_dir='results/EXP1', train_loader=train_loader, **model_kwargs)
