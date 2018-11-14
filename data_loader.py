# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 18:24
# @Author  : zhoujun
from mxnet import gluon
import pathlib
import cv2
import numpy as np
from mxnet import nd

class custom_dataset(gluon.data.Dataset):
    def __init__(self, txt, data_shape, channel=3):
        self.data_list = []
        with open(txt, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                img_path = pathlib.Path(line[0])
                if img_path.exists() and img_path.stat().st_size > 0 and line[1] and str(img_path).endswith('.jpg'):
                    self.data_list.append((line[0], line[1]))
        self.data_shape = data_shape
        self.channel = channel

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        label = int(label)
        img = cv2.imread(img_path, 0 if self.channel == 1 else 3)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(img, (self.data_shape[0], self.data_shape[1], self.channel))
        return nd.array(img), label

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    from mxnet.gluon.data.vision.datasets import ImageFolderDataset