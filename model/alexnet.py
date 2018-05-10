# -*- coding: utf-8 -*-
# @Time    : 18-4-13 下午5:05
# @Author  : zhoujun
from mxnet.gluon import nn


class Lenet(nn.HybridBlock):
    def __init__(self, verbose=False, **kwargs):
        super(Lenet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
            net.add(nn.MaxPool2D(pool_size=3, strides=2))
            net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
            net.add(nn.MaxPool2D(pool_size=3, strides=2))
            net.add(nn.Flatten())
            net.add(nn.Dense(128, activation="relu"))
            net.add(nn.Dense(10))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out
