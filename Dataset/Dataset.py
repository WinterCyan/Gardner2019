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
    def __init__(self, data_dir, param_file):
        self.filenames = os.listdir(data_dir)  # datadir: path to crop imgs
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        self.param_file = param_file
        self.transformer = transformer  # resize & to tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        crop_img_name = self.filenames[idx]
        img = Image.open(crop_img_name)  # open jpg file
        img = self.transformer(img)  # [Nc, 3, 256, 256]

        text_param = read_crop_param(
            crop_param_file=self.param_file,
            crop_img_name=crop_img_name.split("/")[-1].replace(".jpg", ""))
        param = text_param2list_param(text_param)
        for i in range(LIGHT_N - len(param)):
            param.append([np.array([0.0,1.0,0.0]), 0.0, np.array([0.0, 0.0, 0.0]), 0.0])
        return {'img': img, 'param': param}


if __name__ == '__main__':
    data = LEDataset(data_dir=cropped_imgs_dir, param_file=cropped_param_file)
    split_ratio = 0.2
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    training_data_loader = DataLoader(training_data, batch_size=48, pin_memory=True, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=48, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(training_data)/training_data_loader.batch_size))

    for batch_idx, sample in enumerate(training_data_loader):
        img_batch = sample['img']
        param_batch = sample['param']
        print(param_batch)
