import torch.nn as nn
import torch.nn.functional as F
from DataPreprocess.WarpUtils import warp_hdr
from DataPreprocess.SphericalGaussian import render_sg
from DataPreprocess.RetrieveLights import get_env_ambient


class LSCLoss(nn.Module):
    def __init__(self, weight_light=20, weight_ambient=1):
        super(LSCLoss, self).__init__()
        self.weight_light = weight_light
        self.weight_ambient = weight_ambient

    def forward(self, estimated_light_param, estimated_ambient, theta, phi, hdr_data):  # get the hdr_data, theta, phi from name & file
        # TODO: get input batch file names from dataset when training
        warp_hdr_data = warp_hdr(hdr_data, theta, phi)
        gt_light_env, gt_ambient = get_env_ambient(warp_hdr_data)
        sg_light_env = render_sg(estimated_light_param, None, None, False)
        light_loss = F.mse_loss(gt_light_env, sg_light_env)
        ambient_loss = F.mse_loss(gt_ambient, estimated_ambient)
        lsc_loss = self.weight_light*light_loss + self.weight_ambient*ambient_loss
        return lsc_loss

