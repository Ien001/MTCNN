#!/usr/bin/env python3
import time
import numpy as np
import torch

def HSM_sort(loss, thresh):
    loss = loss.cpu().numpy()
    #print(loss)
    index = loss.argsort()
    #print(index)
    index = index.tolist()[::-1]
    #print(index)
    num = len(index) * thresh
    hard_sample = index[0:int(num):1]

    return hard_sample
        

def main():
    # eq. pnet produce 10 bboxs, labels of bboxs range from 0 to 9 
    list_loss = torch.rand(10).cuda()
    th = 0.7

    hardsample = HSM_sort(list_loss, th)
    print('return hardsample id:\n',hardsample)


if __name__=="__main__":
    main()