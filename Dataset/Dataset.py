import os
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
from DataPreprocess.CropPano import get_cropped_and_param
from DataPreprocess.ProcessEXR import read_crop_param
from DataPreprocess.ProcessEXR import text_param2list_param
from DataPreprocess.Consts import *

transformer = transforms.Compose([transforms.ToTensor()])
real_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])


class LEDataset(Dataset):
    def __init__(self, data_dir):
        self.filenames = os.listdir(data_dir)  # datadir: path to crop imgs
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        # self.param_file = param_file
        self.transformer = transformer  # resize & to tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        crop_img_name = self.filenames[idx]
        img = Image.open(crop_img_name)  # open jpg file
        img = self.transformer(img)  # [Nc, 3, 256, 256]

        crop_img_nojpg_name = crop_img_name.split("/")[-1].split("|")[0]
        ambient_string = crop_img_name.split("/")[-1].split("|")[-1].replace(".jpg", "")
        # theta_phi_string = crop_img_name.split("/")[-1].split("|")[-1].replace(".jpg", "")
        ambient = np.fromstring(ambient_string.split(']')[0].split('[')[1], sep=' ')
        # text_param = read_crop_param(
        #     crop_param_file=self.param_file,
        #     crop_img_name=crop_img_nojpg_name)
        # param = text_param2list_param(text_param)
        # for i in range(LIGHT_N - len(param)):
        #     param.append([np.array([0.0, 0.0, 0.0]), 0.0, np.array([0.0, 0.0, 0.0]), 0.0])
        # return {'img': img, 'param':param, 'theta_phi':theta_phi}
        return {'img': img, 'name':crop_img_nojpg_name, 'ambient':ambient}


if __name__ == '__main__':
    data = LEDataset(data_dir=cropped_imgs_dir)
    split_ratio = 0.2
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    training_data_loader = DataLoader(training_data, batch_size=48, pin_memory=True, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=48, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(training_data)/training_data_loader.batch_size))
    device = "cuda:0"

    for batch_idx, sample in enumerate(training_data_loader):
        img_batch = sample['img'].to(device)
        name_batch = sample['name']
        ambient_batch = sample['ambient'].to(device)
        print(name_batch)
        print(ambient_batch)
        # print(theta_phi_batch.shape)
        # print(len(param_batch))  # param_batch: 3 elem for 3 lights
        # print(len(param_batch[0]))  # param_batch[0]: 4 elem for l,s,c,d of first light
        # print(param_batch[0][0].shape)  # param_batch[0][0]: 48 l for first light in batch
        # for i in range(len(img_batch)):
        #     print(name_batch[i])
        #     print(ambient_batch[i])
            # print(theta_phi_batch[i])
            # #                             light  lscd  batch
            # print("l of light0: ", param_batch[0][0][i])
            # print("l of light1: ", param_batch[1][0][i])
            # print("l of light2: ", param_batch[2][0][i])
            # print("s of light0: ", param_batch[0][1][i])
            # print("s of light1: ", param_batch[1][1][i])
            # print("s of light2: ", param_batch[2][1][i])
            # print("c of light0: ", param_batch[0][2][i])
            # print("c of light1: ", param_batch[1][2][i])
            # print("c of light2: ", param_batch[2][2][i])
            # print("d of light0: ", param_batch[0][3][i])
            # print("d of light1: ", param_batch[1][3][i])
            # print("d of light2: ", param_batch[2][3][i])
            # print('---------------------------------------')
