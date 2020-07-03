#*=============================================================================
#
# Author: wukun - 516420282@qq.com
#
# QQ : 516420282
#
# Last modified: 2020-06-09 07:42
#
# Filename: darknet53.py
#
# Description: 
#
#=============================================================================

import torch.nn as nn
import torch

from base_block import *

BLOCK=[1,2,8,8,4]

class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.conv1 = CBL(3, 32)
        self.down1 = CBL(32, 64, 3, 2)
        self.block1 = Block(BLOCK[0], 64)
        self.down2 = CBL(64, 128, 3, 2)
        self.block2 = Block(BLOCK[1], 128)
        self.down3 = CBL(128, 256, 3, 2)
        self.block3 = Block(BLOCK[2], 256)
        self.down4 = CBL(256, 512, 3, 2)
        self.block4 = Block(BLOCK[3], 512)
        self.down5 = CBL(512, 1024, 3, 2)
        self.block5 = Block(BLOCK[4], 1024)
        self.avg_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                      nn.LeakyReLU(inplace=True))
        self.linear1 = nn.Sequential(nn.Linear(1024, 256), 
                                     nn.BatchNorm1d(256),
                                     nn.LeakyReLU(inplace=True))
        self.dropout = nn.Dropout(0.7)
        self.linear2 = nn.Sequential(nn.Linear(256, 128), 
                                     nn.BatchNorm1d(128),
                                     nn.LeakyReLU(inplace=True))
        self.linear = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        x = self.down3(x)
        x = self.block3(x)
        x = self.down4(x)
        x = self.block4(x)
        x = self.down5(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear(x)
        
        return x
