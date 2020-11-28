# import torch
# import urllib
# from PIL import Image
# from torchvision import transforms
# import urllib
#
# model_dir = 'C:\\code\\Gardner2019\\Files\\densenet121.pth'
# if __name__ == '__main__':
#     # model = torch.load(model_dir)
#     model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
#     model.eval()
#     # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#     # try: urllib.URLopener().retrieve(url, filename)
#     # except: urllib.request.urlretrieve(url, filename)
#     input_img = Image.open('../Files/dog.jpg')
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = preprocess(input_img)
#     input_batch = input_tensor.unsqueeze(0)
#     input_batch = input_batch.to('cuda')
#     print(input_batch.shape)
#     model.to('cuda')
#
#     output = model(input_batch)
#     print(output[0].shape)
#     print(torch.argmax(output[0]))
#     dict = model.state_dict()
#     for k, v in dict.items():
#         print(k)
#
#
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import math
import time
from torch.utils.data import DataLoader, random_split
from Dataset.Dataset import *
from Networks.Network import *
from Networks.Losses import *


if __name__ == '__main__':
    dataset_path = 'C:\\datasets\\DeepMaterialsData\\Data_Deschaintre18\\trainBlended'
    device = 'cuda:0'
    print("creating model...")
    model = ParamLENet().to(device)
    print("loading data...")
    data = LEDataset(data_dir=cropped_imgs_dir, param_file=cropped_param_file)
    split_ratio = 0.2
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    training_data_loader = DataLoader(training_data, batch_size=48, pin_memory=True, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=48, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(training_data)/training_data_loader.batch_size))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lsc_loss_func = LSCLoss()

    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # ckp_path = 'modelsavings/checkpoint.pt'
    # print('loading model...')
    # model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)  # load_epoch: [1,100]
    # print("loaded model, epoch: ", load_epoch, ", training started...")

    load_epoch = 0

    model.train()
    epochs = 150
    for epoch in range(load_epoch, epochs):
        print('----------------------training, EPOCH {}---------------------'.format(epoch+1))
        epoch_loss = 0
        epoch_start_time = time.time()
        for batch_idx, sample in enumerate(training_data_loader):
            img_batch = sample["img"].to(device)
            target_svbrdf_batch = sample["svbrdf"].to(device)
            estimated_svbrdf_batch = model(img_batch)
            loss = loss_func(estimated_svbrdf_batch, target_svbrdf_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0:
                print("Epoch {:d}, Batch {:d}...".format(epoch+1, batch_idx+1))
            if batch_idx == batch_num-1:
                print("Epoch {:d}, Batch {:d}...".format(epoch+1, batch_idx+1))
        epoch_end_time = time.time()
        epoch_time = epoch_end_time-epoch_start_time
        print('----------------------EPOCH{}---------------------'.format(epoch+1))
        print("epoch mean loss: "+str(epoch_loss/batch_num))
        print("epoch time: "+str(epoch_time))
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        is_best = True if epoch == epochs-1 else False
        save_ckp(checkpoint, is_best, 'modelsavings', 'modelsavings')
        print('model saved, epoch: ', epoch+1)
        print('--------------------------------------------------');
