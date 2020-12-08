import torch.nn as nn
import torch.nn.functional as F
from DataPreprocess.WarpUtils import warp_hdr
from DataPreprocess.SphericalGaussian import render_sg
from DataPreprocess.ProcessEXR import exr2array
from DataPreprocess.Consts import *


class LSCLoss(nn.Module):
    def __init__(self, weight_light=20.0, weight_ambient=1.0):
        super(LSCLoss, self).__init__()
        self.weight_light = weight_light
        self.weight_ambient = weight_ambient

    def forward(self, estimated_light_param, estimated_ambient, crop_img_nojpg_name, gt_ambient):  # get the hdr_data, theta, phi from name & file
        gt_light_env = exr2array(warped_exr_dir+crop_img_nojpg_name+".exr")
        sg_light_env = render_sg(estimated_light_param, None, None, False)
        light_loss = F.mse_loss(gt_light_env, sg_light_env)
        ambient_loss = F.mse_loss(gt_ambient, estimated_ambient)
        lsc_loss = self.weight_light*light_loss + self.weight_ambient*ambient_loss
        return lsc_loss
