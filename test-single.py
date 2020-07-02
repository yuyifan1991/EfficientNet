from __future__ import print_function, division

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import time


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # find the mapping of folder-to-label
    data = datasets.ImageFolder('home_data/train')
    mapping = data.class_to_idx
    print(mapping)

    # start testing
    net_name = 'efficientnet-b0'
    img_dir = '028018.jpg'

    # load model
    save_dir = 'home_data/model'
    modelft_file = save_dir + "/" + net_name + '.pth'

    # load image
    time5 = time.time()
    img = Image.open(img_dir)
    inputs = data_transforms(img)
    inputs.unsqueeze_(0)
    time6 = time.time()

    # use GPU
    time7 = time.time()
    model = torch.load(modelft_file).cuda()
    time8 = time.time()
    #model = torch.load(modelft_file)
    model.eval()
    # use GPU
    inputs = Variable(inputs.cuda())
    #inputs = Variable(inputs)

    # forward
    time1 =time.time()
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    time2 = time.time()
    class_name = get_key(mapping, preds.item())
    # use the mapping

    print(img_dir)
    print('prediction_label:', class_name)
    print(time2-time1)
    print(time6-time5)
    print(time8-time7)
    print(30*'--')

