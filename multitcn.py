"""
Implements a Temporal Convolution Network
"""

import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv1d
from torch.nn.modules.upsampling import Upsample
from torch.nn.utils import spectral_norm

from copy import deepcopy


class PreTemporalBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            num_inner_blocks=2,
            hidden_skip_dim=0,
            batch_norm=False,
            series_cond=None,
            price_cond=False
    ):
        """
        Temporal Block
        """
        super().__init__()
        #  Set padding to ensure that the layers are the same dimensions
        self.padding = (kernel_size - 1) * dilation
        self.series_cond = series_cond
        self.price_cond = price_cond
        layers = []
        input_channel = in_channels
        for i in range(num_inner_blocks):
            layers.append(spectral_norm(nn.Conv1d(
                input_channel,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=self.padding,
                dilation=dilation)))
            layers.append(Chomp(self.padding))
            layers.append(nn.PReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            input_channel = out_channels
        self.temporal_block = nn.Sequential(*layers)
        if self.price_cond:
            self.conv1x1 = nn.Conv1d(in_channels - 1, 1, 1)
        self.eca = ECA(in_channels)

    def forward(self, inputs):
        """
        Parm
        """
        x, skip = inputs
        if self.price_cond:
            # temp = self.eca(torch.cumsum(x, dim=2))
            # x = x + temp
            temp = self.conv1x1(x)

            temp = torch.cumsum(temp, dim=-1)
            temp = (temp - temp.mean(dim=(2), keepdims=True))/ temp.std(dim=(2), keepdims=True)
            x = torch.cat((x, temp), dim=1)
        # if self.series_cond is not None:
        #     x = torch.cat(x, self.series_cond, dim=1)
        # print(x.shape)
        f_x = self.temporal_block(x)

        return x, f_x, skip


class PostTemporalBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_series,
            hidden_skip_dim=0,
            condition=False,
            # series_cond=False,
            batch_norm=False,
    ):
        """
        Temporal Block
        """
        super().__init__()
        #  Set padding to ensure that the layers are the same dimensions
        self.condition = condition
        # self.series_cond = series_cond
        self.batch_norm = batch_norm
        if hidden_skip_dim > 0:
            self.conv_1x1 = spectral_norm(nn.Conv1d(out_channels, hidden_skip_dim, kernel_size=1, stride=1))
        if self.condition:
            self.cond_1x1 = spectral_norm(nn.Conv1d(out_channels, hidden_skip_dim, kernel_size=1, stride=1))
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.condition_downsample = nn.Conv1d(out_channels, 1, 1)
        self.concat_downsample = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.eca1 = MultiECA(num_series, out_channels)
        self.eca2 = ECA(out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, inputs):
        """
        Parm
        """
        x, f_x, cond_global, skip, dim, labels = inputs
        if self.condition:
            # cond_global = torch.sum(cond_global, dim=2)
            f_x += self.eca1(cond_global)
            if labels is not None:
                # print(f_x.shape, labels.shape)
                f_x = torch.cat((labels, f_x), dim=2)
            # f_x = f_x + cond_global
            if self.batch_norm:
                f_x = self.norm(f_x)
            
            # cond = self.norm(self.condition_downsample(cond))
            # f_x = self.concat_downsample(torch.cat((f_x, cond), dim=1))
            f_x = self.act1(f_x)
        x = x if self.downsample is None else self.downsample(x)
        f_x += self.conv_1x1(x)

        skip[:, :, :] = skip[:, :, :] + f_x[..., -skip.shape[-1]:].clone()#+ self.conv_1x1(f_x[..., -skip.shape[-1]:].clone())

        f_x = f_x + x

        return self.act2(f_x), skip


class MultiTCN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            hidden_skip_dim,
            num_series=1,
            num_inner_blocks=2,
            dilation_factor=2,
            block_1x1=True,
            kernel_size=2,
            batch_norm=False,
            condition=None,
            num_labels=0,
            price_condition=None,
    ):
        """
        TCN
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_skip_dim = hidden_skip_dim
        hidden_channels = deepcopy(hidden_channels)

        pre_layers = []
        post_layers = []

        if block_1x1:
            hidden_channels.insert(0, hidden_channels[0])

        num_blocks = len(hidden_channels)
        self.num_blocks = num_blocks

        for i in range(num_blocks):
            if block_1x1:
                if i == 0:
                    dilation_size = 1
                else:
                    dilation_size = dilation_factor ** (i - 1)
            else:
                dilation_size = dilation_factor ** i
            ks = 1 if (block_1x1 and i == 0) else kernel_size

            # in_dim = in_dim if not cond else in_dim + 1
            # print(in_dim)
            out_dim = hidden_channels[i]

            pre_layers_series = []
            post_layers_series = []
            for series in range(num_series):
                in_dim = in_channels if i == 0 else hidden_channels[i - 1]
                if condition is not None:
                    cond = condition[series]
                    # cond = cond if i == 0 else False
                    cond = cond if (i != 0 and i != num_blocks - 1) else False
                else:
                    cond = False

                # if labels is not None:
                #     series_cond = labels[series]
                #     series_cond = series_cond if i == 0 else False
                #     # in_dim = in_dim + 1 if series_cond is True else in_dim
                # else:
                #     series_cond = False

                if price_condition is not None:
                    price_cond = price_condition[series]
                    price_cond = price_cond if i == 0 else False
                    in_dim = in_dim + 1 if price_cond is True else in_dim
                else:
                    price_cond = False

                pre_layers_series += [
                    PreTemporalBlock(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=ks,
                        num_inner_blocks=num_inner_blocks,
                        dilation=dilation_size,
                        hidden_skip_dim=hidden_skip_dim,
                        batch_norm=batch_norm,
                        # series_cond=series_cond,
                        price_cond=price_cond,
                    )
                ]
                post_layers_series += [
                    PostTemporalBlock(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        num_series=num_series,
                        hidden_skip_dim=hidden_skip_dim,
                        condition=cond,
                        # series_cond=series_cond,
                        batch_norm=batch_norm
                    )
                ]
            pre_layers += [nn.ModuleList(pre_layers_series)]
            post_layers += [nn.ModuleList(post_layers_series)]

        self.pre_tcn = nn.ModuleList(pre_layers)
        self.post_tcn = nn.ModuleList(post_layers)
        self.tcn_output = nn.Sequential(
            nn.PReLU(),
            spectral_norm(nn.Conv1d(hidden_skip_dim, hidden_skip_dim, kernel_size=1)),
            nn.PReLU(),
            spectral_norm(nn.Conv1d(hidden_skip_dim, out_channels, kernel_size=1)),
        )
        self.embed = ConditionalEncoder(hidden_skip_dim)

    def forward(self, x, labels=None):
        """
        params
        """
        b_size = x.shape[0]
        t_dim = x.shape[-1]
        n_series = x.shape[2]
        skip_list = {}
        f_x_list = {}
        x_list = {}
        # self.embed = self.embed
        # self.embed.weight = nn.Parameter(self.embed.weight.to(x.device))
        for series in range(n_series):
            skip_list[series] = torch.zeros(b_size, self.hidden_skip_dim, t_dim - self.receptive_field_size() + 1, device=x.device)
            x_list[series] = x[:, :, series, :]

        for module in range(len(self.pre_tcn)):
            cond = torch.zeros(b_size, self.hidden_skip_dim, n_series, t_dim, requires_grad=False, device=x.device)

            for series in range(n_series):
                x_series = x_list[series]
                if labels:
                    cond_label = labels[series] if series in labels else None
                else:
                    cond_label = None
                if cond_label is not None and module > 1:
                    cond_label = self.embed(cond_label.float(), x.shape)
                    # print(cond_label.shape, x_series.shape, module)
                    x_series = torch.cat((cond_label, x_series), dim=2)
                    x, f_x, skip_list[series] = self.pre_tcn[module][series]((x_series, skip_list[series]))
                    x = x[:, :, 1:]
                    f_x = f_x[:, :, 1:]
                else:
                    x, f_x, skip_list[series] = self.pre_tcn[module][series]((x_series, skip_list[series]))

                f_x_list[series] = f_x
                x_list[series] = x
                cond[:, :, series, :] = f_x.detach().clone()
            for series in range(n_series):
                series_mask = list(range(n_series))
                series_mask.remove(series)
                cond_temp = torch.index_select(cond, 2, torch.tensor(series_mask, device=cond.device))
                cond_temp.requires_grad = True
                cond_label = None
                # if module < 2:
                #     if labels:
                #         cond_label = labels[series] if series in labels else None
                #         if cond_label is not None:
                #             cond_label = self.embed(cond_label.float(), x.shape)
                            # print(cond_label)
                            # print(cond_label)
                x_list[series], skip_list[series] = self.post_tcn[module][series]((x_list[series], f_x_list[series], cond_temp, skip_list[series], series, cond_label))

        output_list = []
        for series in range(n_series):
            output_list.append(self.tcn_output(skip_list[series]).unsqueeze(2))
        return torch.cat(output_list, dim=2)

    def receptive_field_size(self):
        """ returns """
        drop = 1
        for block in self.pre_tcn:
            drop += block[0].padding * 2
        return drop


class ECA(nn.Module):
    """Constructs an ECA module.

    Implementation of ECA module from "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"

    Code can be found at:
    https://github.com/BangguWu/ECANet

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)#.unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MultiECA(nn.Module):
    def __init__(self, num_series, channel, k_size=3):
        super(MultiECA, self).__init__()
        self.num_series = num_series - 1
        eca_list = []
        for _ in range(num_series):
            eca_list += [ECA(channel, k_size)]
        self.eca_list = nn.ModuleList(eca_list)

    def forward(self, x):
        output = torch.zeros(x.shape[0], x.shape[1], x.shape[-1], device=x.device)
        for series in range(self.num_series):
            output += self.eca_list[series](x[:, :, series, :])
        return output


class Chomp(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        """
        Chomp
        """
        if self.padding != 0:
            return x[..., :-self.padding].contiguous()
        else:
            return x


class ConditionalEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Linear(1, channels)

    def forward(self, x, shape):
        x = x.unsqueeze(-1)
        x = self.upsample(x).unsqueeze(-1)
        return x
