import torch, glob, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']


def preict_one_img(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    img = cv2.resize(img, (224, 224))
    # 把图片由BGR变成RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 1.将numpy数据变成tensor
    tran = transforms.ToTensor()
    img = tran(img)
    img = img.to(device)
    # 2.将数据变成网络需要的shape
    img = img.view(1, 3, 224, 224)

    out1 = net(img)
    out1 = F.softmax(out1, dim=1)
    proba, class_ind = torch.max(out1, 1)

    proba = float(proba)
    class_ind = int(class_ind)
    img = img.cpu().numpy().squeeze(0)
    new_img = np.transpose(img, (1, 2, 0))
    plt.imshow(new_img)
    plt.title("预测的类别为： %s .  概率为： %3f" % (classes[class_ind], proba))
    plt.show()


if __name__ == '__main__':
    # 训练的时候类别是怎么放的，这里的classes也要对应写
    classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = "/UI/test/emoji/02.jpg"
    model_path = "./weight/best.pth"
    net = torch.load(model_path)
    preict_one_img(img_path)