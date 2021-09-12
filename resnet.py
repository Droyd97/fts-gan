"""
Module that implement ResNet Classifier
"""
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import os
import re
import math

from ftsgan.base_gan import BaseGAN
from ftsgan.multitcn import MultiTCN
from ftsgan.ganloss import GANLoss
from ftsgan.utils import next_path

from tqdm import tqdm
import time

try: 
    __IPYTHON__
    _in_ipython_session = True
    from IPython.display import clear_output
except NameError:
    _in_ipython_session = False


class ResNet(BaseGAN):
    def __init__(
        self,
        test_metrics=[],
        num_series=2,
        batch_size=512,
        num_epochs=2000,
        workers=4,
        ngpu=0,
        params=None,
        save_point=None,
        overwrite_save=True,
        save_dir=None,
        verbose=50,
        series_names=None
    ):
        """
        ResNet
        """
        super().__init__(
            "ResNet",
            num_series=num_series,
            test_metrics=test_metrics,
            batch_size=batch_size,
            num_epochs=num_epochs,
            workers=workers,
            ngpu=ngpu,
            params=params,
            verbose=verbose,
            save_point=save_point,
            save_dir=save_dir,
            overwrite_save=overwrite_save,
            series_names=series_names
        )

        # Setup parameters
        self.params = self.setup_params()

        # Set up network
        self.resnet = ResNetModel(
            self.params['layer_filters'],
            self.params['classes'],

        ).to(self.device)

        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.resnet.parameters(),
            self.params['lr_rate'],
            self.params['optim_betas'],
            self.params['optim_eps']
        )

        # Set tensors to store losses
        self.losses = self.setup_losses(['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'])

    def setup_params(self):
        setup_params = {
            'classes': 2,
            'layer_filters': [64, 128, 128],
            'lr_rate': 1e-4,
            'optim_betas': (0., 0.9),
            'optim_eps': 1e-08,
        }
        if self.params is not None:
            setup_params.update(self.params)
        return setup_params

    def train(self, dataset, indices):
        
        for epoch in tqdm(range(self.num_epochs)):
            self.epoch = epoch

            X_train, y_train, X_test, y_test = dataset.sample(self.batch_size)
            self.optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()

            result = self.resnet(X_train[:, indices])
            train_output = loss(result, y_train)

            train_output.backward()

            self.optimizer.step()
        
            self.losses['train_loss'][self.epoch] = train_output.item()
            self.losses['train_accuracy'][self.epoch] = (torch.argmax(result, dim=1) == y_train).sum().float() / y_train.shape[0]

            with torch.no_grad():
                test_result = self.resnet(X_test[:, indices])
                test_output = loss(test_result, y_test)
                self.losses['test_loss'][self.epoch] = test_output.item()
                self.losses['test_accuracy'][self.epoch] = (torch.argmax(test_result, dim=1) == y_test).sum().float() / y_test.shape[0]

            print("Train Accuracy: {}, Test Accuracy: {}, Train loss: {}, Test loss: {}".format(
                self.losses['train_accuracy'][self.epoch],
                self.losses['test_accuracy'][self.epoch],
                self.losses['train_loss'][self.epoch],
                self.losses['test_loss'][self.epoch]
                )
            )


class ResNetModel(nn.Module):
    def __init__(
        self,
        layer_filters,
        classes,
    ):
        super().__init__()
        layer_filters.insert(0, 1)
        layers = []
        for idx, i in enumerate(layer_filters[1:]):
            layers += [
                ResNetBlock(
                    layer_filters[idx],
                    layer_filters[idx + 1],
                    [8, 5, 3],
                    padding=(4, 3),
                )
            ]
        self.layers = nn.ModuleList(layers)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Squeeze(),
            nn.Linear(layer_filters[-1], classes),
        )

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return self.output(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernels,
        padding,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            spectral_norm(nn.Conv1d(in_channel, out_channel, kernels[0], padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv1d(out_channel, out_channel, kernels[1], padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv1d(out_channel, out_channel, kernels[2], padding=1)),
        )
        self.res_block = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channel, out_channel, 1))
        )

    def forward(self, x):
        res = x
        res = self.res_block(res)
        x = self.block(x)
        return x + res


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze()
