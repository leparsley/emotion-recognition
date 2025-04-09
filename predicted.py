import torch
import torchvision.transforms as transforms


def predict_(img):

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model_path = "./weight/best.pth"
    model = torch.load(model_path)

    model.eval()
    classes = {'0': 'amusement', '1': 'anger', '2': 'awe', '3': 'contentment', '4': 'disgust', '5': 'excitement', '6': 'fear', '7': 'sad'}
    with torch.no_grad():
        # 维度压缩
        output = torch.squeeze(model(img))
        #print(output)
        # Softmax函数是一个非线性转换函数，通常用在网络输出的最后一层，输出的是概率分布（比如在多分类问题中，Softmax输出的是每个类别对应的概率）
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    return classes[str(predict_cla)], predict[predict_cla].item()


