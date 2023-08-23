from flask import Flask,jsonify,render_template,request,redirect,url_for,jsonify
import os
import json
import sys
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import collections
import numpy as np


#sys.path.append("..")
from vit_model import vit_base_patch16_224_in21k as create_model
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './pics/'
app.config['JSON_SORT_KEYS'] = False
app.config['JSON_AS_ASCII'] = False 
#@app.route('/')
#def Home():
#    return render_template("test.html")

@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],'pripic.jpg'))
        return redirect(url_for('success'))
        #return render_template('success.html')
    else:
        return render_template('home.html')

#@app.route('/uploaded',methods=['GET','POST'])

def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

@app.route("/success",methods=['GET','POST'])
def success():
    """加载模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #图片预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256), #将图片最小的一个边转化为256 另一个同比例缩小
         transforms.CenterCrop(224),#中心裁剪
         transforms.ToTensor(), #转为Tensor格式 此时可以进行训练了
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) #对像素值进行归一化处理

    # load image
    img_path = "/mnt/pics/pripic.jpg"
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
    model_weight_path = "./weights4/model-46.pth"
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
    #restr = ''
    top1 = list(rank.keys())[0]
    top2 = list(rank.keys())[1]
    top3 = list(rank.keys())[2]
    top01 = rank[top1]
    top02 = rank[top2]
    top03 = rank[top3]
    restr = {}
   
    for (key,value) in (rank.items()):
        #print("class: {:10}   prob: {:.3}".format(key,value))
        #restr += "class: {:10}   prob: {:.3}\n".format(key,value)
        restr[key] = format(value, '.3f')
        i = i + 1
        if(i == 3):
            break
    #plt.show()
    dict(sorted(restr.items(), key=lambda x: x[1], reverse=True))
    print(restr)
    print(top01)
    img_stream = return_img_stream(img_path)
    #res_json = json.dumps(restr)
    #return jsonify(restr)
    return render_template('success.html', prediction_text = restr,
    img_stream=img_stream,top1=top1,
    top2=top2,top3=top3,
    top01=np.round(top01,3),top02=np.round(top02,3),
    top03=np.round(top03,3))


if __name__=='__main__':

    app.run(debug = True)
