from __future__ import print_function, division
from torch.utils.data import DataLoader
import torchvision

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import glob
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.ImageFolder(root='F:/1/emo/new_test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)  # num_workers：使用多进程加载的进程数，0代表不使用多进程

classes = test_dataset.classes
# print(classes)

model_path = "F:/1/emo/weight/best.pth"
model = torch.load(model_path)


def imshow(inp, i):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    # plt.xlabel('GroundTruth: {}'.format(ylabel))
    # plt.title('predicted: {}'.format(title))
    plt.imshow(inp)
    k = '%02d' % i
    plt.savefig("F:/1/emo/model_pic/res{}.jpg".format(k), bbox_inches='tight')


def show_pic_acc():
    correct = 0
    total = 0
    i = 1
    j = 0

    result = []
    true = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            out = torchvision.utils.make_grid(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(i, '.Predicted:', ''.join('%5s' % classes[predicted]), '  GroundTruth:',
                  ''.join('%5s' % classes[labels]))
            imshow(out, i)
            result.append(classes[predicted])
            true.append(classes[labels])
            i = i + 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc1 = 100 * correct / total
    # print(acc1)
    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    # print('预测:%s %%' % (result))
    #pre_list = [result[i:i + 4] for i in range(0, len(result), 4)]
    # print('真实:%s %%' % (true))
    #tru_list = [true[i:i + 4] for i in range(0, len(true), 4)]
    return acc1, result, true

# if __name__ == '__main__':
