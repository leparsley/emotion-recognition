import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

def predict_densenet(img):

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load("densenet_model.pth", map_location='cpu')
    model.eval()
    model.to(DEVICE)

    img = transform_test(img)
    img = torch.unsqueeze(img, dim=0)
    img = Variable(img).to(DEVICE)



    classes = {'0': 'anger', '1': 'disgust', '2': 'fear', '3': 'joy', '4': 'sadness', '5': 'surprise'}

    with torch.no_grad():
        # 维度压缩
        output = torch.squeeze(model(img))
        # print(output)
        # Softmax函数是一个非线性转换函数，通常用在网络输出的最后一层，输出的是概率分布（比如在多分类问题中，Softmax输出的是每个类别对应的概率）
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    return classes[str(predict_cla)], predict[predict_cla].item()


if __name__ == '__main__':
    file_name="F:/1/pythonProject/test/3/232.jpg"
    img = Image.open(file_name)
    a, b = predict_densenet(img)
    print('Image Name:{},predict:{}'.format(a,b))




    # path = 'Dataset/Test/Bald/'
    # testList = os.listdir(path)
    # for file in testList:
    #     img = Image.open(path + file)
    #     img = transform_test(img)
    #     img.unsqueeze_(0)
    #     img = Variable(img).to(DEVICE)
    #     out = model(img)
    #     # Predict
    #     _, pred = torch.max(out.data, 1)
    #     print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))