#*=============================================================================
#
# Author: wukun - 516420282@qq.com
#
# QQ : 516420282
#
# Last modified: 2020-06-09 07:41
#
# Filename: base_block.py
#
# Description: 
#
#*=============================================================================

import torch.nn as nn
import torch

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=(1,1)):
        super(CBL, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        cbl = self.relu(self.bn(self.conv(x)))
        return cbl

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=(1,1)):
        super(ResBlock, self).__init__()

        out_channels = in_channels // 2
        self.cbl1 = CBL(in_channels, out_channels, 1, 1, (0,0))
        self.cbl2 = CBL(out_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x):
        resblock = torch.add(x, self.cbl2(self.cbl1(x)))

        return resblock

class Block(nn.Module):
    def __init__(self, c, in_channels):
        super(Block, self).__init__()
        for i in range(c):
            self.add_module('res_'+str(i), ResBlock(in_channels))

    def forward(self, x):
        modules = self.modules()
        for module in modules:
            if type(module) == ResBlock:
                x = module(x)

        return x
