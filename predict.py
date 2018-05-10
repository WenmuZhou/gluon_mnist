# -*- coding: utf-8 -*-
# @Time    : 18-4-16 上午10:03
# @Author  : zhoujun
import time
import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
from mxnet import nd
import cv2
import os


def try_gpu(gpu):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


class Gluon_Model:
    def __init__(self, net, model_path, img_shape, img_channel=3, gpu_id=None, classes_txt=None):
        self.ctx = try_gpu(gpu_id)
        print(self.ctx)
        self.net = net
        self.net.load_params(model_path, ctx=self.ctx)
        self.net.hybridize()
        self.img_shape = img_shape
        self.img_channel = img_channel

        if classes_txt is not None:
            with open(classes_txt, 'r') as f:
                self.idx2label = dict(line.strip().split(' ') for line in f if line)
        else:
            self.idx2label = None

    def predict(self, img, is_numpy=False, topk=1):
        if len(self.img_shape) not in [2, 3] or self.img_channel not in [1, 3]:
            raise NotImplementedError

        if not is_numpy and self.img_channel in [1, 3]:  # read image
            if os.path.exists(img):
                img = cv2.imread(img, 0 if self.img_channel == 1 else 1)
            else:
                return 'file is not exists'

        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        if len(img.shape) == 2 and self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and self.img_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.reshape([self.img_shape[0], self.img_shape[1], self.img_channel])
        img = img.transpose([2, 0, 1])
        img = nd.array(img).expand_dims(axis=0)
        img = img.as_in_context(self.ctx)
        result = self.net(img)[0]
        result = result.asnumpy().argmax()
        return result


if __name__ == '__main__':
    img_path = '/data/datasets/mnist/train/0/0_1.png'
    model_path = 'models.AlexNet.params'

    model = Gluon_Model(models.AlexNet(classes=10), model_path, img_shape=[227, 227])
    start_cpu = time.time()
    epoch = 1000
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
        print('device: cpu, result:%s, time: %.4f' % (str(result), time.time() - start))
    end_cpu = time.time()

    model1 = Gluon_Model(models.AlexNet(classes=10), model_path, gpu_id=0, img_shape=[227, 227])
    start_gpu = time.time()
    for _ in range(epoch):
        start = time.time()
        result = model1.predict(img_path)
        print('device: gpu, result:%s, time: %.4f' % (str(result), time.time() - start))
    end_gpu = time.time()
    print('cpu avg time: %.4f' % ((end_cpu - start_cpu) / epoch))
    print('gpu avg time: %.4f' % ((end_gpu - start_gpu) / epoch))
