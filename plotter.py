import torch 
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np


def plot_synthetic_paths(paths, dataset, figsize=(8, 8)):
    num_assets = len(paths)

    fig, ax = plt.subplots(num_assets, 1, figsize=(num_assets * 4, 8))
    for idx, (key, value) in enumerate(paths.items()):
        path = value[:, -1, :].squeeze().cpu().numpy().T
        path_len = path.shape[0]
        if num_assets > 1:
            ax[idx].plot(path, 'orange', alpha=0.5, )
            ax[idx].plot(dataset.series[key].raw_data[:, :path_len].T.cpu(), 'blue', alpha=0.8)
        else:
            ax.plot(path, 'orange', alpha=0.5, )
            ax.plot(dataset.series[key].raw_data[:, :path_len].T.cpu(), 'blue', alpha=0.8) 
    return ax


def xcorr(x, y):
    x = logrtn_torch(x)
    y = logrtn_torch(y)
    x = torch.mean(x, dim=0).squeeze()
    y = torch.mean(y, dim=0).squeeze()

    fig, ax = plt.subplots()
    ax.xcorr(x, y, normed=True)

    return ax


def logrtn_torch(x):
    x_log = torch.log(x)
    return x_log[..., 1:] - x_log[..., :-1]