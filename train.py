# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
from mxnet import gluon
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import init
from data_loader import custom_dataset
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


def evaluate_accuracy(test_data_loader, net, ctx):
    n = 0
    acc = 0
    for data, label in test_data_loader:
        label = label.astype('float32').as_in_context(ctx)
        data = data.as_in_context(ctx)
        acc += nd.sum(net(data).argmax(axis=1) == label).copyto(mx.cpu())
        n += label.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / n


def train():
    # 初始化
    ctx = try_gpu(2)
    net = models.resnet50_v1(classes=4)
    net.hybridize()
    net.initialize(ctx=ctx)
    # net.forward(nd.ones((1, 3, 227, 227)).as_in_context(ctx))

    sw = SummaryWriter('/data1/lsp/lsp/pytorch_mnist/log/rotate/mxnet_resnet18')
    # sw.add_graph(net)

    print('initialize weights on', ctx)

    # 获取数据
    batch_size = 64
    epochs = 10

    train_data = custom_dataset(txt='/data2/dataset/image/train.txt', data_shape=(224, 224), channel=3)
    test_data = custom_dataset(txt='/data2/dataset/image/val.txt', data_shape=(224, 224), channel=3)
    transforms_train = transforms.ToTensor()
    # transforms_train = transforms.Compose([transforms.Resize(227), transforms.ToTensor()])
    train_data_loader = gluon.data.DataLoader(train_data.transform_first(transforms_train), batch_size=batch_size,
                                              shuffle=True, num_workers=12)

    test_data_loader = gluon.data.DataLoader(test_data.transform_first(transforms_train), batch_size=batch_size,
                                             shuffle=True, num_workers=12)
    # 训练
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    steps = train_data.__len__() // batch_size

    schedule = mx.lr_scheduler.FactorScheduler(step=3 * steps, factor=0.1, stop_factor_lr=1e-6)
    sgd_optimizer = mx.optimizer.SGD(learning_rate=0.01, lr_scheduler=schedule)
    trainer = gluon.Trainer(net.collect_params(), optimizer=sgd_optimizer)

    for epoch in range(epochs):
        # test_data.reset()
        start = time.time()
        train_loss = 0.0
        train_acc = 0.0
        cur_step = 0
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
            if i % 100 == 0:
                batch_time = time.time() - start
                print('epoch [%d/%d], Iter: [%d/%d]. Loss: %.4f. Accuracy: %.4f, time:%0.4f, lr:%s' %
                      (epoch, epochs, i, steps, cur_loss, cur_acc / batch_size, batch_time, trainer.learning_rate))
                start = time.time()
            cur_step = epoch * steps + i
            sw.add_scalar(tag='Train/loss', value=cur_loss /
                                                  label.shape[0], global_step=cur_step)
            sw.add_scalar(tag='Train/acc', value=cur_acc /
                                                 label.shape[0], global_step=cur_step)
            sw.add_scalar(tag='Train/lr',
                          value=trainer.learning_rate, global_step=cur_step)

        val_acc = evaluate_accuracy(test_data_loader, net, ctx)
        sw.add_scalar(tag='Eval/acc', value=val_acc, global_step=cur_step)
        net.save_parameters("models/resnet501/{}_{}.params".format(epoch, val_acc))
        print('epoch: %d, train_loss: %.4f, train_acc: %.4f, val_acc: %.4f, time: %.4f, lr=%s' % (
            epoch, train_loss / n, train_acc / n, val_acc, time.time() - start, str(trainer.learning_rate)))
    sw.close()


if __name__ == '__main__':
    train()
