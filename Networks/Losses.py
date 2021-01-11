import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from DataPreprocess.WarpUtils import warp_hdr
from DataPreprocess.SphericalGaussian import render_sg
from DataPreprocess.ProcessEXR import exr2array
from DataPreprocess.Consts import *
from DataPreprocess.ProcessEXR import read_crop_param, text_param2list_param
from DataPreprocess.SphericalGaussian import render_sg_tensor
from DataPreprocess.WarpUtils import delta_between_l


class LSCLoss(nn.Module):
    def __init__(self, weight_light=0.8, weight_ambient=0.2):
        super(LSCLoss, self).__init__()
        self.weight_light = weight_light
        self.weight_ambient = weight_ambient

    def forward(self,
                gt_light_env_batch,
                estimated_l_batch,
                estimated_s_batch,
                estimated_c_batch,
                gt_ambient_batch,
                estimated_a_batch):  # get the hdr_data, theta, phi from name & file
        light_loss = 0
        for batch_idx in range(gt_light_env_batch.shape[0]):
            # get single light_env, move to GPU
            # gt_light_env_name = gt_light_env_name_batch[batch_idx]
            gt_light_env = gt_light_env_batch[batch_idx]
            # t1 = time.time()
            # gt_light_env = exr2array(warped_exr_dir+gt_light_env_name+".exr")
            # # gt_light_env_tensor = torch.Tensor(gt_light_env).permute(2,0,1).to(device)  # [3,H,W]
            # gt_light_env_tensor = torch.Tensor(gt_light_env).permute(2,0,1).unsqueeze(0).to(device)
            # gt_light_env_tensor_resize = F.interpolate(gt_light_env_tensor, size=(RESIZE_H, RESIZE_W), mode='bilinear', align_corners=True)
            # gt_light_env_tensor_resize = gt_light_env_tensor_resize.squeeze(0)
            # t2 = time.time()
            # print("load light env time: ", t2-t1)

            # calculate single sg_env in GPU
            single_img_l = estimated_l_batch[batch_idx]  # 9
            single_img_s = estimated_s_batch[batch_idx]  # 3
            single_img_c = estimated_c_batch[batch_idx]  # 9
            single_img_l = single_img_l.reshape(3, 3)  # [3,3]
            # normalize l
            l_length = torch.sqrt((single_img_l**2.0).sum(1, keepdim=True))
            single_img_l = torch.div(single_img_l, l_length)
            single_img_c = single_img_c.reshape(3, 3)  # [3,3]
            sg_light_env_tensor = render_sg_tensor(single_img_l, single_img_s, single_img_c)

            # calculate loss
            sample_light_loss = F.mse_loss(gt_light_env, sg_light_env_tensor)
            light_loss += float(sample_light_loss)
            # light_loss += sample_light_loss
        # light_loss = F.mse_loss(gt_light_env_batch, sg_light_env_batch)
        ambient_loss = F.mse_loss(gt_ambient_batch, estimated_a_batch)
        lsc_loss = self.weight_light*light_loss + self.weight_ambient*ambient_loss
        return lsc_loss


class RefineLoss(nn.Module):
    def __init__(self):
        super(RefineLoss, self).__init__()

    def forward(self, estimated_light_param, estimated_ambient, crop_img_nojpg_name, gt_ambient):
        gt_light_param = text_param2list_param(read_crop_param(cropped_param_file, crop_img_nojpg_name))  # get gt param from file
        a_loss = F.mse_loss(gt_ambient, estimated_ambient)  # calculate ambient loss
        dsc_loss = 0
        gt_ls = [param[0] for param in gt_light_param]
        gt_ss = [param[1] for param in gt_light_param]
        gt_cs = [param[2] for param in gt_light_param]
        gt_ds = [param[3] for param in gt_light_param]
        for light_param in estimated_light_param:  # for N lights in estimated results calculate and accumulate loss
            pred_l, pred_s, pred_c, pred_d = light_param
            # calculate angular distance, get closest light, delta_sort[0] is the index of the min distance
            delta_sort = np.argsort([delta_between_l(pred_l, gt_l) for gt_l in gt_ls])
            closest_l = gt_ls[delta_sort[0]]
            if delta_between_l(pred_l, closest_l) > np.pi/4.0:
                continue
            closest_d = gt_ds[delta_sort[0]]
            closest_s = gt_ss[delta_sort[0]]
            closest_c = gt_cs[delta_sort[0]]
            d_loss = F.mse_loss(pred_d, closest_d)
            s_loss = F.mse_loss(pred_s, closest_s)
            c_loss = F.mse_loss(pred_c, closest_c)
            dsc_loss += (d_loss+s_loss+c_loss)
        refine_loss = a_loss + dsc_loss
        return refine_loss
