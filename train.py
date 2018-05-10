# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
from mxnet import gluon
import mxnet as mx
from mxnet import nd
from mxnet import autograd
import time
from data_loader.data_generator import custom_dataset
from mxnet import init
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
import time

def try_gpu(gpu):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


if __name__ == '__main__':
    train_data_path = '/data/datasets/mnist//train.txt'
    test_data_path = '/data/datasets/mnist/test.txt'

    # 初始化
    ctx = try_gpu(0)
    net = models.AlexNet(classes=10)
    net.hybridize()
    net.initialize()
    net.forward(nd.ones((1, 3, 227, 227)))

    sw = SummaryWriter('./log/%s'%(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
    sw.add_graph(net)
    # del net
    net = models.AlexNet(classes=10)
    net.hybridize()
    net.initialize(ctx=ctx, init=init.Xavier())

    print('initialize weights on', ctx)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
    # 获取数据
    batch_size = 64
    epochs = 3
    train_data = gluon.data.vision.ImageFolderDataset('/data/datasets/mnist/train', flag=1)
    # train_data = custom_dataset(txt='/data/datasets/mnist/train.txt', data_shape=(227, 227), channel=3)

    transforms_train = transforms.Compose([transforms.Resize(227),transforms.ToTensor()])
    train_data_loader = gluon.data.DataLoader(train_data.transform_first(transforms_train), batch_size=batch_size,
                                              shuffle=True, num_workers=3)
    # 训练
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    for epoch in range(epochs):
        # test_data.reset()
        start = time.time()
        train_loss = 0.0
        train_acc = 0.0
        n = train_data.__len__()
        for i, (data, label) in enumerate(train_data_loader):
            label = label.astype('float32').as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                outputs = net(data)
                loss = criterion(outputs, label)
            loss.backward()
            trainer.step(batch_size)

            cur_loss = loss.sum().asscalar()
            cur_acc = nd.sum(outputs.argmax(axis=1) == label).asscalar()
            train_acc += cur_acc
            train_loss += cur_loss
            # if i % 100 == 0:
            #     print('iter: %d, train_loss: %.4f, train_acc: %.4f' % (
            #         i, cur_loss / label.shape[0], cur_acc / label.shape[0]))
            cur_step = epoch * (n / batch_size) + i
            sw.add_scalar(tag='Train/loss', value=cur_loss/label.shape[0], global_step=cur_step)
            sw.add_scalar(tag='Train/acc', value=cur_acc/label.shape[0], global_step=cur_step)

        print('epoch: %d, train_loss: %.4f, train_acc: %.4f, time: %.4f' % (
            epoch + 1, train_loss / n, train_acc / n, time.time() - start))
    sw.close()
    # net.save_params('models.AlexNet.params')
    # net.load_params()
