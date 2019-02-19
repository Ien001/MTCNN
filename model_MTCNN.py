#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, gradcheck


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        # 12 12 3 > 5 5 10
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 5 5 10 > 3 3 16
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # 3 3 16 > 1 1 32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        # face cls
        self.fcls_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1,bias=False)
        # bbox reg
        self.bbox_conv = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1,bias=False)

    def forward(self, x):
        x = self.conv1(x)
        #print('conv1 out:', x.shape)
        x = self.relu1(x)
        x = self.maxpool(x)
        #print('mp1 out:', x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        #print('conv2 out:', x.shape)
        x = self.conv3(x)
        #print('conv3 out:', x.shape)
        x = self.maxpool_1(x)
        #print('maxpool_1 out:', x.shape)

        cls_output = self.fcls_conv(x)
        

        bbox_output = self.bbox_conv(x)

        
        return cls_output, bbox_output


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        # 24 24 3 > 11 11 28
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 11 11 28 > 4 4 48
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # 4 4 48 > 3 3 64
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=1,bias=False)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        # 64 > 128
        self.fc = nn.Linear(64, 128)
        
        # face cls 
        self.fcls_fc = nn.Linear(128, 1)
        # bbox reg 
        self.bbox_fc = nn.Linear(128, 4)



    def forward(self, x):
        x = self.conv1(x)
        #print('conv1 out:', x.shape)
        x = self.relu1(x)
        x = self.maxpool(x)
        #print('mp1 out:', x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        #print('conv2 out:', x.shape)
        x = self.conv3(x)
        #print('conv3 out:', x.shape)
        x = self.maxpool_1(x)
        #print('maxpool_1 out:', x.shape)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #print('fc out:', x.shape)

        cls_output = self.fcls_fc(x)
        bbox_output = self.bbox_fc(x)
               

        return cls_output, bbox_output


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        # 48 48 3 > 23 23 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 23 23 32 > 10 10 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # 10 10 64 > 4 4 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        # 4 4 64 > 3 3 128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=6, stride=3, padding=1)
        # 128 > 256
        self.fc = nn.Linear(128, 256)
        
        # face cls 
        self.fcls_fc = nn.Linear(256, 1)
        # bbox reg 
        self.bbox_fc = nn.Linear(256, 4)
        # fll
        self.fll_fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print('conv1 out:', x.shape)
        x = self.relu1(x)
        x = self.maxpool(x)
        #print('mp1 out:', x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        #print('conv2 out:', x.shape)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool(x)
        #print('conv3 out:', x.shape)
        x = self.conv4(x)
        #print('conv4 out:', x.shape)
        x = self.maxpool_1(x)
        #print('maxpool_1 out:', x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #print('fc out:', x.shape)

        fcls_output = self.fcls_fc(x)
        bbox_output = self.bbox_fc(x)
        fll_output = self.fll_fc(x)                

        
        return fcls_output, bbox_output, fll_output

class HSMnet(nn.Module): 
    # for hard sample mining

    def __init__(self):
        super(HSMnet, self).__init__()
        # 12 12 3 > 5 5 10
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 5 5 10 > 3 3 16
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # 3 3 16 > 1 1 32
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        # face cls
        self.fcls_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1,bias=False)
        

    def forward(self, x):
        x = self.conv1(x)
        #print('conv1 out:', x.shape)
        x = self.relu1(x)
        x = self.maxpool(x)
        #print('mp1 out:', x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        #print('conv2 out:', x.shape)
        x = self.conv3(x)
        #print('conv3 out:', x.shape)
        x = self.maxpool_1(x)
        #print('maxpool_1 out:', x.shape)

        cls_output = self.fcls_conv(x)

        return cls_output


def main():
    print('*'*10, 'PNet', '*'*10)
    pnet = PNet()
    test_tensor = torch.ones(1,3,12,12)
    o1, o2 = pnet(test_tensor)
    print(o1.shape)
    print(o2.shape)
    
    print('*'*10, 'RNet', '*'*10)
    rnet = RNet()
    test_tensor = torch.ones(1,3,24,24)
    o1, o2 = rnet(test_tensor)
    print(o1.shape)
    print(o2.shape)

    print('*'*10, 'ONet', '*'*10)
    onet = ONet()
    test_tensor = torch.ones(1,3,48,48)
    o1, o2, o3 = onet(test_tensor)
    print(o1.shape)
    print(o2.shape)
    print(o3.shape)

    print('*'*10, 'HSMnet', '*'*10)
    pnet = HSMnet()
    test_tensor = torch.ones(1,3,12,12)
    o1 = pnet(test_tensor)
    print(o1.shape)

if __name__=="__main__":
    main()