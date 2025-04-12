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
    def __init__(self,cfg,fold):
        super(BirdModel, self).__init__()
        self.cfg=cfg
        self.fold=fold
        self.model = models.efficientnet_b0(pretrained=True)
        self.lin = nn.Linear(1000 , 23)

    def forward(self, spec):
        spec = spec.repeat(1,3,1,1)
        emb = self.model(spec)
        output = self.lin(emb)
        return output


    def training_step(self, batch, batch_idx):

        
        return 1

    def validation_step(self, batch, batch_idx):
        spec = batch['spec'].swapaxes(0,1)
        pred = self(spec)
        return 1

    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer 
