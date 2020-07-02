from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as FUN
import os
from scipy import io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

input_size = 224


# Load Test images
def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}

    # num_workers=0 if CPU else = 1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    ##################################
#    print('dataset_loaders is:',dataset_loaders)
    print('image_datasets is:',image_datasets)
    print('data_set_sizes is :',data_set_sizes)
    ##################################
    return dataset_loaders, data_set_sizes


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, labels = data
        ################
       # print(inputs)
       # print(labels)
        ################
        labels = torch.squeeze(labels.type(torch.LongTensor))
        # GPU
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # CPU
        # inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
#        print(outputs)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        cont += 1

    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))

    return FUN.softmax(Variable(outPre),dim=1).data.numpy(), outLabel.numpy()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Start Testing
    net_name = 'efficientnet-b0'
    data_dir = 'home_data'
    save_dir = 'home_data/model'
    modelft_file = save_dir + "/" + net_name + '.pth'
    batch_size = 16

    # GPU时
    model_ft = torch.load(modelft_file).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # CPU时
    # model_ft = torch.load(modelft_file, map_location='cpu')
    # criterion = nn.CrossEntropyLoss()

    outPre, outLabel = test_model(model_ft, criterion)

    # Save result
    np.save(save_dir + '/Pre', outPre)
    np.save(save_dir + '/Label', outLabel)

    # Change the result and scores to .mat
    mat = np.load(save_dir + '/Pre.npy')
    io.savemat(save_dir + '/Pre.mat', {'gene_features': mat})

    label = np.load(save_dir + '/Label.npy')
    io.savemat(save_dir + '/Label.mat', {'labels': label})
