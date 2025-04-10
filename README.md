# emotion-recognition
### DenseNet vs machine learning based emotional image   
### Datasets:IAPSa   

### 运行环境：
    sklearn0.22.2、skimage0.16.2、numpy1.18.1、pandas1.0.3、scipy1.4.1、cv2、matplotlib、pytorch1.4.0、cv2、numpy1.18.1
### 实现：
    UI界面展示图像分类情况（基于训练图像数据集的标签：Amusement、Anger、Awe、Contentment、Disgust、Excitement、Fear、Sad）   
    
    后续可以更换其他模型替代，界面可以展示模型预测准确率，并将错误识别图像名字标红   

### 使用说明：
    运行 train.py 、DenseNet.py训练模型，训练出来的模型保存在weights文件夹中
    运行ui_2会出现UI界面，UI界面逻辑分离，根据不同的功能形成不同的py文件，例如predict.py、densenet_predict.py，分别对应了利用不同深度学习算法形成的模型文件对图像的预测效果
    通过从数据集中随机抽取80张图像，分别为amusement、anger、disgust、fear、sad、awe、excitement、contentment各10张，点击“多张图像预测”按钮，程序会自动调用模型对上述80张图像进行预测，为了验证实验的适用性，每次生成的80张图像是随机的，因此，识别准确率在一定范围内进行波动。（ps:超出数据集范围会存在一定误差，“单张图片预测”功能同样如此）
    
### 问题：
    训练的数据量较小，对于相似情感会存在一定的误判情况
    模型参数未调整

### 总结：
    缺少现实意义
