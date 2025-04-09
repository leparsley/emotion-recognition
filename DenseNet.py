# -*- coding: utf-8 -*-

import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models
from PIL import Image
from torch.utils import data
from torchvision import transforms

# 使用DenseNet121进行分类，对情绪类图片进行分类
# 思路
# 卷积部分可认为是特征提取网络，一个图的特征提取出来后是Tensor
# 我们后面进行分类，是对图像特征进行分类

torchvision.models.densenet121()
imgs_path = glob.glob(r'D:\emotion_densenet\*\*.jpg')
# 获取前五图片
print(imgs_path[:5])

# 在路径当中提取出类别名称
img_p = imgs_path[100]  # 随便取，这里取第100张
# print(img_p)
a = img_p.split('\\')[2].split('.')[0]  # 对图片路径进行分割得到类名



# 列表推导式，提取出所有图片的类别
all_labels_name = [img_p.split('\\')[2].split('.')[0] for img_p in imgs_path]
print(all_labels_name)

unique_label = np.unique(all_labels_name)
# print(len(unique_label))

label_to_index = dict((v, k) for k, v in enumerate(unique_label))
index_to_label = dict((v, k) for k, v in label_to_index.items())
# print(label_to_index)
# print(index_to_label)

# 将图片类别完全转换成编码的形式
all_labels_name = [label_to_index.get(name) for name in all_labels_name]
print(all_labels_name)

print(len(all_labels_name))  # 显示所有图片总数量

#划分训练数据和测试数据
np.random.seed(2022)
random_index = np.random.permutation(len(imgs_path))
imgs_path = np.array(imgs_path)[random_index]                 #图片数据乱序
all_labels_name = np.array(all_labels_name)[random_index]               #标签乱序，保证两者乱序一致性

i = int(len(imgs_path)*0.8)
train_path = imgs_path[:i]
train_labels = all_labels_name[:i]
test_path = imgs_path[i:]
test_labels = all_labels_name[i:]

#创建输入
transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()
])

class EmotionDataset(data.Dataset):
    def __init__(self, imgs_path, labels):
        self.imgs = imgs_path
        self.labels = labels
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        #注意对黑白照片，要处理为channel为3的形式
        pil_img = Image.open(img)             #H, W,  ?     Channel = 3
        #对黑白图像进行处理
        np_img = np.asarray(pil_img, dtype=np.uint8)
        if len(np_img.shape)==2:
            img_data = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)
            pil_img = Image.fromarray(img_data)
        img_tensor = transform(pil_img)
        return img_tensor, label
    def __len__(self):
        return len(self.imgs)

train_ds = EmotionDataset(train_path, train_labels)
test_ds = EmotionDataset(test_path, test_labels)

BATCH_SIZE = 32
train_dl = data.DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
)
test_dl = data.DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
)

img_batch, label_batch = next(iter(train_dl))
print(img_batch.shape)

plt.figure(figsize=(12,8))
for i, (img, label) in enumerate(zip(img_batch[:6],label_batch[:6])):
    img =  img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
    plt.title(index_to_label.get(label.item()))
    plt.imshow(img)
#plt.show()


#调用DenseNet提取特征
my_densenet = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1).features

#GPU
if torch.cuda.is_available():
    my_densenet = my_densenet.cuda()

#设置不可训练
for p in my_densenet.parameters():
    p.requires_grad = False

train_features = []
train_feat_labels = []
for im, la in train_dl:
    out = my_densenet(im.cpu())          #一个批次一个批次的返回特征
    #将四维数据扁平成二维
    out = out.view(out.size(0), -1)
    train_features.extend(out.cpu().data)
    train_feat_labels.extend(la)

test_features = []
test_feat_labels = []
for im, la in test_dl:
    out = my_densenet(im.cpu())          #一个批次一个批次的返回特征
    #将四维数据扁平成二维
    out = out.view(out.size(0), -1)
    test_features.extend(out.cpu().data)
    test_feat_labels.extend(la)

print(len(train_features))         #9430

#创建特征Dataset和分类模型
class FeatureDataset(data.Dataset):
    def __init__(self, feat_list, label_list):
        self.feat_list = feat_list
        self.label_list = label_list
    def __getitem__(self, index):
        return self.feat_list[index], self.label_list[index]
    def __len__(self):
        return len(self.feat_list)



#使用提取之后的特征创建了两个DataSet
train_feat_ds = FeatureDataset(train_features, train_feat_labels)
test_feat_ds = FeatureDataset(test_features, test_feat_labels)

train_feat_dl = data.DataLoader(train_feat_ds,
                                batch_size=BATCH_SIZE,
                                shuffle=True
                                   )
test_feat_dl =  data.DataLoader(test_feat_ds,
                                batch_size=BATCH_SIZE
                                )

in_feat_size = train_features[0].shape[0]

class FCModel(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)      #可一个linear层，也可以多个
    def forward(self, input):
        return self.lin(input)



# 将FCModel实例化
net = FCModel(in_feat_size, 8)  # 要分类8个种类
print(net)
# 定义优化函数和损失函数
if torch.cuda.is_available():
    net.to('cuda')
net.to('cpu')
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.00001)


# 训练函数
def fit(epoch, model, trainloader, testloader):
    corret = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        """if torch.cuda.is_available():
            y = torch.tensor(y, dtype=torch.long)
            x, y = x.to('cuda'), y.to('cuda')"""
        y=torch.tensor(y,dtype=torch.long)
        x,y=x.to('cpu'),y.to('cpu')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            corret = corret + (y_pred == y).sum().item()
            total = total + y.size(0)
            running_loss = running_loss + loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)  # !!!!小心变量名错误
    epoch_acc = corret / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            y = torch.tensor(y, dtype=torch.long)
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct = test_correct + (y_pred == y).sum().item()
            test_total = test_total + y.size(0)
            test_running_loss = test_running_loss + loss.item()

        torch.save(model, "./weight/densenet_last.pth")

    epoch_test_loss = test_running_loss / len(testloader.dataset)  # !!!!小心变量名错误
    epoch_test_acc = test_correct / test_total

    print('epoch:', epoch,
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss', round(epoch_test_loss, 3),
          'test_accuracy', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = 50
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, net, train_feat_dl, test_feat_dl)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)


