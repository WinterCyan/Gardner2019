import numpy as np

WIDTH = 7768
HEIGHT = 3884

SIZE_CROP = 1024
CROP_DISTRIB_MU = 0.5
CROP_DISTRIB_SIGMA = 0.08
LIGHT_N = 3

# WIDTH = 2048
# HEIGHT = 1024

GAUSSIAN_KERNEL_SIZE = 35


data_home_dir = "/media/winter/_hdd/LavalHDRDataset/"
hdr_dataset_dir = data_home_dir+"IndoorHDRDataset2018/"
hdr_dataset_hdrformat_dir = data_home_dir+"IndoorHDRDataset2018/HDRFormat/"
hdr_sample_dir = data_home_dir+"HDRSample10/"
light_param_file = data_home_dir+"HDR_results/light_param.txt"
light_sg_renderings_dir = data_home_dir+"HDR_results/light_sg_renderings/"
light_masks_dir = data_home_dir+"HDR_results/light_masks/"
hdr_jpgs_dir = data_home_dir+"HDR_results/hdr_jpgs/"
fusion_hdr_jpgs_dir = data_home_dir+"HDR_results/fusion_hdr_jpgs/"
single_file = hdr_dataset_dir+"AG8A8752-5cd7fafa87.exr"
depth_files_dir = data_home_dir+'HDR_DepthFile/'
cropped_imgs_dir =data_home_dir+'CroppedImgs/'
cropped_param_file = data_home_dir+"HDR_results/cropped_light_param.txt"
