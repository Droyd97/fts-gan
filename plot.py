"""
Module to generate plots that compare the statistical properties of real and fake data.

Taken from:
M. Wiese, R. Knobloch, R.Korn, and P.Kretchmer, "Quant GANs: deep generation of 
financial time series," Quantitative Finance, Vol 20, no. 9, pp. 1419-1440, 2020,
arXiv: 1907.06673
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluation import kurtosis_torch, skew_torch
from .evaluation import lev_eff_torch, cacf_torch


def to_numpy(x):
    return x.cpu().detach().numpy()


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def synthetic_paths(paths, title=None, figsize=(10, 4), start_from=1.):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    plt.suptitle(title)
    paths = paths * start_from
    for path in paths:
        ax.plot(path)
    ax.grid()
    set_style(ax)
    ax.spines['bottom'].set_visible(False)
    plt.xlim([0, paths.shape[1]])
    plt.ylabel('Spot price')
    plt.xlabel('days')


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=(0, 1)).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=(0, 1)).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[1:], label='Historical')
    ax.plot(acf_fake[1:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[1:, i].shape[0]),
                ub[1:, i], lb[1:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.grid(True)
    ax.legend()
    return ax


def compare_cacf(x_real, x_fake, assets, ax=None, legend=False, max_lag=128, figsize=(10, 8)):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)
    acf_fake_std = np.std(acf_fake_list, axis=0)

    n = x_real.shape[2]
    ind = torch.tril_indices(n, n).transpose(0, 1).cpu().numpy()
    for i, (j, k) in enumerate(ind):
        ax.plot(acf_real[:, i], label='Historical: ' + ' '.join([assets[j], assets[k]]))
        ax.plot(acf_fake[:, i], label='Generated: ' + ' '.join([assets[j], assets[k]]), alpha=0.8)
    # ax.set_ylim([-0.1, 0.35])
    """
    ub = acf_fake + acf_fake_std
    lb = acf_fake - acf_fake_std
    for i in range(acf_real.shape[-1]):
        ax.fill_between(
            range(acf_fake[1:, i].shape[0]),
            ub[1:, i], lb[1:, i],
            color='orange',
            alpha=.3
        )
    """
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.grid(True)
    if legend:
        ax.legend()
    return ax


def compare_lev_eff(x_real, x_fake, ax=None):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = lev_eff_torch(x_real, 32, dim=(1)).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)
    acf_fake_list = lev_eff_torch(x_fake, 32, dim=(1)).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)
    acf_fake_std = np.std(acf_fake_list, axis=0)
    ub = acf_fake + acf_fake_std
    lb = acf_fake - acf_fake_std

    ax.plot(acf_real[1:], label='Historical')
    ax.plot(acf_fake[1:], label='Generated', alpha=0.8)
    ax.fill_between(range(acf_fake[1:].shape[0]), ub[1:], lb[1:],
                    color='orange', alpha=.3)
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('Leverage Effect')
    ax.grid(True)
    ax.legend()
    return ax


def compare_hists(x_real, x_fake, ax=None, log=False, show_text=False):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.hist(
        [to_numpy(x_real).flatten(), to_numpy(x_fake).flatten()],
        bins=25, alpha=1, density=True, label=['Historical', 'Synthetic']
    )
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-PDF')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('PDF')
    ax.set_xlabel('Log-Return')

    def text_box(x, height, title):
        textstr = '\n'.join((
            r'%s' % (title,),
            # t'abs_metric=%.2f' % abs_metric
            r'$s=%.2f$' % (skew_torch(x).item(),),
            r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.05, height, textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=props
        )

    if show_text:
        text_box(x_real, 0.80, 'Historical')
        text_box(x_fake, 0.60, 'Generated')
    return ax


def compare_cross_corr(x_real, x_fake):
    """ Computes cross correlation matrices of x_real and x_fake and plots them. """
    cc_real = np.corrcoef(x_real.T)
    cc_fake = np.corrcoef(x_fake.T)
    plt.subplot(1, 2, 1)
    plt.matshow(cc_real, fignum=False)
    plt.subplot(1, 2, 2)
    plt.matshow(cc_fake, fignum=False)
    print(cc_real)
    print(cc_fake)
    print('cc_score: %s' % np.linalg.norm(cc_real - cc_fake, ord=2, axis=(0, 1)))
    plt.show()


def logrtn_torch(x):
    x_log = torch.log(x)
    return x_log[:, 1:] - x_log[:, :-1]


def comparison(x_fake_post, x_real_post, assets, rfs, figsize=(14, 8)):
    x_real = logrtn_torch(x_real_post)
    x_fake = logrtn_torch(x_fake_post)
    _, axes = plt.subplots(1, 5, figsize=figsize)
    compare_hists(x_real=x_real, x_fake=x_fake, ax=axes[0], show_text=True)
    compare_hists(x_real=x_real, x_fake=x_fake, ax=axes[1], log=True)
    compare_acf(x_real=x_real, x_fake=x_fake, ax=axes[2], max_lag=rfs)
    compare_acf(x_real=torch.abs(x_real), x_fake=torch.abs(x_fake), ax=axes[3], max_lag=rfs)
    compare_lev_eff(x_real=x_real, x_fake=x_fake, ax=axes[4])
    axes[0].title.set_text('Empirical PDF (linear-scale)')
    axes[1].title.set_text('Empirical PDF (log-scale)')
    axes[2].title.set_text('Serial ACF')
    axes[3].title.set_text('ACF of absolute log-returns')
    axes[4].title.set_text('Leverage effect')
