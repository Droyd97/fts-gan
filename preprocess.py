"""
Module to preprocess data.

Taken from:
M. Wiese, R. Knobloch, R.Korn, and P.Kretchmer, "Quant GANs: deep generation of 
financial time series," Quantitative Finance, Vol 20, no. 9, pp. 1419-1440, 2020,
arXiv: 1907.06673
"""

import torch

from ftsgan.gaussianize import Gaussianize as GZ

import numpy as np
import pandas as pd


class Transform():

    def fit(self, x):
        return x

    def transform(self, x):
        raise NotImplementedError('Transform needs to be a child class')

    def inverse_transform(self, x):
        raise NotImplementedError('Inverse transform needs to be a child class')


class RollingWindow():

    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def transform(self, x):
        x = x.transpose(1, 0)
        x = x.unfold(0, self.window_size, self.stride)
        return x

    def inverse_transform(self, x):
        return x.flatten()


class DiffTransform(Transform):

    def transform(self, x):
        return x[:, 1:] - x[:, :-1]

    def inverse_transform(self, x):
        return torch.cumsum(x, dim=2)


class LogTransform(Transform):
    def transform(self, x):
        return torch.log(x)

    def inverse_transform(self, x):
        return torch.exp(x)


class StandardScaler(Transform):
    
    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis, keepdim=True)
            self.std = torch.std(x, dim=self.axis, keepdim=True)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        std = self.std.repeat(x.shape[0], 1, x.shape[-1]).to(x.device)
        mean = self.mean.repeat(x.shape[0], 1, x.shape[-1]).to(x.device)
        return x * std + mean


class Pipeline():
    def __init__(self, steps):
        self.steps = steps

    def transform(self, x):
        for step in self.steps:
            x = step.transform(x)
        return x

    def inverse_transform(self, x, skip=()):
        for step in self.steps[::-1]:
            if isinstance(step, skip):
                continue
            x = step.inverse_transform(x)
        return x


class Gaussianize(Transform):
    def __init__(self):
        self.gaus = GZ(max_iter=1000)

    def transform(self, x):
        self.gaus.fit(x.cpu().numpy().T)
        return torch.transpose(torch.tensor(self.gaus.transform(x.cpu().numpy().T), device=x.device), 0, 1)

    def inverse_transform(self, x):
        output = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
        for i in range(x.shape[0]):
            output[i] = torch.transpose(torch.tensor(self.gaus.inverse_transform(x[i].cpu().numpy().T), device=x.device), 0, 1).unsqueeze(1)
        return output


class Condition():

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def fit(self, x):
        return x


class MaxDrawdown(Condition):

    def fit(self, x):
        temp_device = x.device
        x = x.cpu().numpy()
        max_draw = []
        for idx, sample in enumerate(x):
            max_draw.append(np.abs(np.min(drawdown(sample))))
        # print(max_draw)
        return torch.tensor(threshold(max_draw, self.thresholds), device=temp_device)


def threshold(x, condition):
    return pd.cut(x, bins=condition, labels=False)


def drawdown(returns):
    totalReturn = np.cumsum(returns)
    drawdown = totalReturn - np.maximum.accumulate(totalReturn)
    return drawdown
