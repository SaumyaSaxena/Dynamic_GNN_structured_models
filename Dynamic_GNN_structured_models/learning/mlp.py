
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

from .model_utils import make_mlp

import io
import PIL.Image

class MLP(LightningModule):
    def __init__(self, train_ds_gen=None, val_ds_gen=None, cfg_dict=None):
        super().__init__()
        self.save_hyperparameters(cfg_dict)

        self._train_ds_gen = train_ds_gen
        self._val_ds_gen = val_ds_gen
        
        self._N_O = cfg_dict["N_O"]
        self._N_R = cfg_dict["N_R"]
        self._D_S = cfg_dict["D_S"]
        self._D_R = cfg_dict["D_R"]
        self._D_G = cfg_dict["D_G"]
        self._D_S_d = cfg_dict["D_S_d"]

        self.input_dim = self._N_O * self._D_S + self._D_G + self._N_R*self._D_R
        self.output_dim = self._N_O * self._D_S_d + self._D_G

        self.hidden_layers = cfg_dict["hidden_layers"]

        self.lr = cfg_dict['lr']
        self.batch_size = cfg_dict['batch_size']

        self._make_model()

    @property
    def ds(self):
        return self._train_ds_gen()
    
    @property
    def val_ds(self):
        return self._val_ds_gen()

    def _make_model(self):
        self.fc = make_mlp(
                            self.input_dim, self.hidden_layers + [self.output_dim],
                            prefix='model', last_act='relu'
                        )

    def forward(self, obs):
        cond, action = obs

        feed = torch.cat((cond, action), dim = 1)
        if len(cond.shape) == 1:
            cond = cond.view(1, -1)
        batch_size = len(cond)
        dyn_mu = self.fc(feed)

        return {
            'mu': dyn_mu,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat['mu'], y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat['mu'], y)

        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)