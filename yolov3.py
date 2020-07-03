#*=============================================================================
#
# Author: wukun - 516420282@qq.com
#
# QQ : 516420282
#
# Last modified: 2020-06-08 07:44
#
# Filename: yolov3.py
#
# Description: 
#
#*=============================================================================

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from base_block import *

BLOCK=[1,2,8,8,4]
NECKS=[(1024, 512), (768, 256), (384, 256)]
ANCHORS = [(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]

class YoloBackbone(nn.Module):
    def __init__(self):
        super(YoloBackbone, self).__init__()

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        x = self.down3(x)
        x = self.block3(x)
        route_2 = x
        x = self.down4(x)
        x = self.block4(x)
        route_1 = x
        x = self.down5(x)
        x = self.block5(x)
        
        return x, route_1, route_2

class YoloNeck(nn.Module):
    def __init__(self, inout):
        super(YoloNeck, self).__init__()
        in_channels = inout[0]
        out_channels = inout[1]
        self.conv1 = CBL(in_channels, out_channels, 1, 1, (0,0))
        self.conv2 = CBL(out_channels, out_channels*2, 3)
        self.conv3 = CBL(out_channels*2, out_channels, 1, 1, (0,0))
        self.conv4 = CBL(out_channels, out_channels*2, 3)
        self.conv5 = CBL(out_channels*2, out_channels, 1, 1, (0,0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class YoloHead(nn.Module):
    def __init__(self, in_channels, out_channels, anchors=3):
        super(YoloHead, self).__init__()

        self.conv1 = CBL(in_channels, out_channels, 3)
        self.conv2 = CBL(out_channels, anchors*6, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()

        out_channels = in_channels // 2
        self.conv = CBL(in_channels, out_channels, 1)

    def forward(self, x, output_size):
        x = self.conv(x)
        x = F.interpolate(x, output_size)
        return x

class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()

        self.backbone = YoloBackbone()
        self.neck1 = YoloNeck(NECKS[0])
        self.head1 = YoloHead(512, 512)

        self.upsample1 = UpSample(512)
        self.neck2 = YoloNeck(NECKS[1])
        self.head2 = YoloHead(256, 256)

        self.upsample2 = UpSample(256)
        self.neck3 = YoloNeck(NECKS[2])
        self.head3 = YoloHead(256, 128)

    def forward(self, x):
        backbone, route_1, route_2 = self.backbone(x)
        neck1 = self.neck1(backbone)
        head1 = self.head1(neck1)
  
        shape = neck1.shape
        upsample1 = self.upsample1(neck1, (shape[2]*2, shape[3]*2))
        neck2 = self.neck2(torch.cat((route_1, upsample1), dim = 1))
        head2 = self.head2(neck2)
     
        shape = neck2.shape
        upsample2 = self.upsample2(neck2, (shape[2]*2, shape[3]*2))
        neck3 = self.neck3(torch.cat((route_2, upsample2), dim = 1))
        head3 = self.head3(neck3)

        return head1, head2, head3

def parse_detection(head, anchors, input_size=(416,416), num_class=1):

    shape = head.shape
    bbox_attr = 5 + num_class
    stride = [input_size[0] // shape[2], input_size[1] // shape[3]]
    anchors = torch.from_numpy(np.array([(anchor[0]/stride[0], anchor[1]/stride[1]) for anchor in anchors]))

    head = head.permute(0,2,3,1) 
    head = head.contiguous().view(-1, shape[2]*shape[3]*len(anchors), bbox_attr)

    bbox_center, bbox_size, confidence, classes = torch.split(head, [2,2,1,1], dim=-1)

    bbox_center = nn.Sigmoid()(bbox_center)
    x = torch.arange(shape[3])
    y = torch.arange(shape[2])
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x = grid_x.contiguous().view(-1,1)
    grid_y = grid_y.contiguous().view(-1,1)
    center = torch.cat((grid_x, grid_y), axis=-1)
    center = center.repeat(1,3).reshape(1,-1,2)
    bbox_center = bbox_center + center

    anchors = anchors.repeat(shape[2]*shape[3], 1)
    bbox_size = torch.exp(bbox_size) * anchors

    confidence = nn.Sigmoid()(confidence)
    classes = nn.Sigmoid()(classes)

if __name__ == '__main__':
    test = torch.ones(16, 3, 416, 416)

    yolov3 = YoloV3()

    head1, head2, head3 = yolov3(test)
    parse_detection(head1, ANCHORS[0:3])
    torch.save(yolov3.state_dict(), './wukun.pth')

