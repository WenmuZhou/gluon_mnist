# -*- coding: utf-8 -*-
# @Time    : 18-4-16 上午10:03
# @Author  : zhoujun
import time
import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
from mxnet import nd
import cv2
import os
from math import *
from mxnet.gluon.data.vision import transforms



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
        self.net.load_parameters(model_path, ctx=self.ctx)
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

        if len(img.shape) == 2 and self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and self.img_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        tensor = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        tensor = tensor.reshape([self.img_shape[0], self.img_shape[1], self.img_channel])

        tensor = transforms.ToTensor()(nd.array(tensor)).expand_dims(axis=0)
        tensor = tensor.as_in_context(self.ctx)
        result = self.net(tensor)[0]
        label = result.asnumpy().argmax()
        new_path = os.path.splitext(img_path)[0] + str(label) + "_rotate.png"
        tic = time.time()
        if label == 0:
            cv2.imwrite(new_path, img)
        elif label == 1:
            img = img_rotate(270, img)
            cv2.imwrite(new_path, img)
        elif label == 2:
            img = img_rotate(180, img)
            cv2.imwrite(new_path, img)
        elif label == 3:
            img = img_rotate(90, img)
            cv2.imwrite(new_path, img)
        print(new_path, time.time() - tic)
        return result

def img_rotate(degree, img):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation

if __name__ == '__main__':
    img_path = '/data/datasets/mnist/train/0/0_1.png'
    model_path = 'models/resnet50/2_1.0.params'

    model1 = Gluon_Model(models.resnet50_v1(classes=4), model_path, gpu_id=0, img_shape=[224, 224])

    for img in os.listdir('/data2/zj/pingan/t_xz/input1'):
        img_path = os.path.join('/data2/zj/pingan/t_xz/input1',img)
        start = time.time()
        result = model1.predict(img_path)