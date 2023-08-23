import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #图片预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256), #将图片最小的一个边转化为256 另一个同比例缩小
         transforms.CenterCrop(224),#中心裁剪
         transforms.ToTensor(), #转为Tensor格式 此时可以进行训练了
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) #对像素值进行归一化处理

    # load image
    img_path = "./pea.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=292, has_logits=False).to(device)
    # load model weights
    #训练之后生成的权重
    model_weight_path = "./weights1/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    sortdit = {}
    for i in range(len(predict)):
         # print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
         #                                           predict[i].numpy()))
        sortdit[class_indict[str(i)]] = predict[i].numpy()
    sortdit = dict(sortdit)
    rank = sorted(sortdit.items(), key=lambda x: x[1], reverse=True)
    rank = dict(rank)
    i = 0
    restr = ''
    for (key,value) in (rank.items()):
        #print("class: {:10}   prob: {:.3}".format(key,value))
        restr += "class: {:10}   prob: {:.3}\n".format(key,value)
        i = i + 1
        if(i == 3):
            break
    #plt.show()
    print(restr)
  
if __name__ == '__main__':
    main()