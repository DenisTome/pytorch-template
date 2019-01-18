# -*- coding: utf-8 -*-
"""
Created on Sep 11 11:19 2018

@author: Denis Tome'

Set of custom layers that increase the convergence rate
when training a model.

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LinearWN',
    'ConvTranspose2dWNUB',
    'Conv2dWNUB',
    'glorot'
]


class LinearWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN, self).__init__(in_features, out_features, bias)
        self.g = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.linear(input, self.weight * self.g[:, None] / wnorm, self.bias)


class ConvTranspose2dWNUB(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, height, width, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvTranspose2dWNUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                  padding, dilation, groups, False)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x, **kwargs):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.conv_transpose2d(x, self.weight * self.g[None, :, None, None] / wnorm,
                                  bias=None, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups) + self.bias[None, ...]


class Conv2dWNUB(nn.Conv2d):
    def __init__(self, in_channels, out_channels, height, width, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2dWNUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, False)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.conv2d(x, self.weight * self.g[:, None, None, None] / wnorm,
                        bias=None, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups) + self.bias[None, ...]


def glorot(m, alpha):
    gain = math.sqrt(2. / (1. + alpha ** 2))

    if isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return

    # m.weight.data.normal_(0, std)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))
    m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, ConvTranspose2dWNUB) or isinstance(m, LinearWN):
        norm = torch.sqrt(torch.sum(m.weight.data[:] ** 2))
        m.g.data[:] = norm
