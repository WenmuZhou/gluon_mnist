# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 18:24
# @Author  : zhoujun
from mxnet import gluon
import cv2
import numpy as np
from mxnet import nd

class custom_dataset(gluon.data.Dataset):
    def __init__(self, txt, data_shape, channel=3):
        fh = open(txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append((words[0], int(words[1])))
        self.data = data
        self.data_shape = data_shape
        self.channel = channel

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        if self.channel == 3:
            img = cv2.imread(img_path, 1)
        else:
            img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(img, (self.data_shape[0], self.data_shape[1], self.channel))

        return nd.array(img), label

    def __len__(self):
        return len(self.data)