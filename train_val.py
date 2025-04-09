import numpy as np
import os, sys
import os.path
from PIL import Image
url = 'D:/emotion_resnet/a/sadness/'#图片存储的文件夹名称
classname = 'sadness.'#图片的名字中字符部分
n=2988#图片的数量
array = np.arange(n)#产生长度为n的序列
np.random.shuffle(array)#将arrray序列随机排列
#把path文件夹下以及其子文件下的所有.jpg图片移动到new_path文件夹下
def moveImg(path,new_path):
    img=Image.open(path)
    if img.mode == "P":
        img = img.convert('RGB')
    img.save(os.path.join(new_path,os.path.basename(path)))
#30%的数据生成验证集
new_path='D:/emotion_resnet/validation/validation/sadness/'
i=0
while(i <= (int(n*0.3))):
    DatasetPath = url + classname + str(array[i]) + '.jpg'
    moveImg(DatasetPath,new_path)
    i=i+1
#70%的数据生成训练集
new_path='D:/emotion_resnet/training/training/sadness/'
while(i <= (n-1)):
    DatasetPath = url + classname + str(array[i]) + '.jpg'
    moveImg(DatasetPath,new_path)
    i=i+1