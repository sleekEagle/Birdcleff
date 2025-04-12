import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD,Adam
from torch.optim.lr_scheduler import StepLR
import torchvision.models.video as t_models
from torchvision import models
import sys
current_module = sys.modules[__name__]

class BirdModel(pl.LightningModule):
    def __init__(self,cfg,fold,num_classes):
        super(BirdModel, self).__init__()
        self.cfg=cfg
        self.fold=fold
        self.model = models.efficientnet_b0(pretrained=True)
        self.lin = nn.Linear(1000 , num_classes)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss() 

    def forward(self, spec):
        spec = spec.repeat(1,3,1,1)
        x = self.model(spec)
        x = self.lin(x)
        x = self.sigmoid(x)
        return x


    def training_step(self, batch, batch_idx):
        spec = batch['spec']
        target = batch['label'].float()
        pred = self(spec)
        loss = self.loss_fn(pred, target)
        self.log(f'train_loss_fold{self.fold}',loss,
            on_step=False, on_epoch=True,
            prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        spec = batch['spec'].swapaxes(0,1)
        target = batch['label'].squeeze().float()
        pred = self(spec)
        mean_pred = torch.mean(pred,dim=0)
        loss = self.loss_fn(mean_pred, target)
        self.log(f'val_loss_fold{self.fold}',loss,
            on_step=False, on_epoch=True,
            prog_bar=False)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer 
