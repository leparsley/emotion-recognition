import os
import sys
import json
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from model import resnet34
from utils import train_and_val, plot_acc, plot_loss
import numpy as np



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 16

    """RandomResizedCrop:将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
       RandomHorizontalFlip()根据概率对图片进行水平（左右）翻转，每次根据概率来决定是否执行翻转；
       ToTensor()将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
       Normalize归一化
    """
    data_transform = {
        #串联多个图片变换
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("D:/emotion_resnet/training/training/", transform=data_transform["train"])  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4)  # 加载数据
    len_train = len(train_dataset)
    val_dataset = datasets.ImageFolder("D:/emotion_resnet/validation/validation/", transform=data_transform["val"])  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=4)  # 加载数据
    len_val = len(val_dataset)

    net = resnet34()
    loss_function = nn.CrossEntropyLoss()  #交叉熵损失函数
    """设置损失函数，nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合,可以直接使用它来替换网络中的这两个操作，这个函数可以用于多分类问题。"""
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 设置优化器和学习率
    """实现Adam算法:通过梯度的一阶矩和二阶矩自适应的控制每个参数的学习率的大小"""
    epoch = 70

    history = train_and_val(epoch, net, train_loader, len_train, val_loader, len_val, loss_function, optimizer, device)

    plot_loss(np.arange(0, epoch), history)
    plot_acc(np.arange(0, epoch), history)