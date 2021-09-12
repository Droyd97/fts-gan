"""
Module with function to evalute the statistical properties of time series.

Taken from:
M. Wiese, R. Knobloch, R.Korn, and P.Kretchmer, "Quant GANs: deep generation of 
financial time series," Quantitative Finance, Vol 20, no. 9, pp. 1419-1440, 2020,
arXiv: 1907.06673
"""
from torch import nn


def acf_torch(x, max_lag, dim=(0, 1)):
    """

    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    # if dim == (0, 1):
    #    return torch.stack(cacf_list)
    # else:
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def lev_eff_torch(x, max_lag, dim=(0, 1)):
    lev_eff = list()
    x = x - x.mean((0, 1))

    mu = x.mean((0, 1))
    mu_pow = torch.abs(x).mean((0, 1))
    std = torch.std(x, unbiased=False, dim=(0, 1))
    std_pow = torch.std(torch.abs(x), unbiased=False, dim=(0, 1))

    for i in range(max_lag):
        if i > 0:
            y = (torch.abs(x[:, i:]) - mu_pow) * (x[:, :-i] - mu)
        else:
            y = (torch.abs(x) - mu_pow) * (x - mu)
        l_i = torch.mean(y, dim) / (std * std_pow)
        lev_eff.append(l_i)
    if dim == (0, 1):
        return torch.stack(lev_eff)
    else:
        return torch.cat(lev_eff, 1)


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


class Loss(nn.Module):
    def __init__(self, name, reg=1, transform=lambda x: x, threshold=10., norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class SpreadLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SpreadLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.spread_loss_real = spread_loss(x_real)

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        spread_loss_fake = spread_loss(x_fake)
        return self.norm_foo(spread_loss_fake - self.spread_loss_real.to(x_fake.device))

def spread_loss(x_real):
    x_min = torch.min(x_real, dim=2)[0]
    x_max = torch.max(x_real, dim=2)[0]
    return (x_max - x_min).mean()


class CorrelationLoss(Loss):
    def __init__(self, x_real, y_real, **kwargs):
        super(CorrelationLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.correlation_loss_real = correlation_torch(x_real, y_real)

    def forward(self, x_fake, y_fake):
        self.loss_componentwise = self.compute(x_fake, y_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake, y_fake):
        correlation_loss_fake = correlation_torch(x_fake, y_fake)
        return self.norm_foo(correlation_loss_fake - self.correlation_loss_real.to(x_fake.device))


def correlation_torch(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


acf_diff = lambda x: torch.sqrt(torch.pow(x, 2).sum(0))
cc_diff = lambda x: torch.abs(x).sum(0)


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.acf_real = acf_torch(self.transform(x_real), max_lag, dim=(0, 1))
        self.max_lag = max_lag

    def compute(self, x_fake):
        acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class LevEffLoss(Loss):
    def __init__(self, x_real, max_lag=16, **kwargs):
        super(LevEffLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.lef_eff_real = lev_eff_torch(x_real, max_lag)
        self.max_lag = max_lag

    def compute(self, x_fake):
        acf_fake = lev_eff_torch(self.transform(x_fake), self.max_lag)
        return self.norm_foo(acf_fake - self.lef_eff_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean_real = x_real.mean((0, 1))

    def compute(self, x_fake):
        return self.norm_foo(x_fake.mean((0, 1)))


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake):
        std_fake = x_fake.std((0, 1))
        return self.norm_foo(std_fake - self.std_real.to(x_fake.device))


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake)
        return self.norm_foo(skew_fake - self.skew_real.to(x_fake.device))


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real.to(x_fake.device))


import torch
from torch import nn


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].view(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise
