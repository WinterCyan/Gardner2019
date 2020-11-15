import numpy as np
import Imath
from DataPreprocess.WarpUtils import *


WIDTH = 7768
HEIGHT = 3884

# WIDTH = 2048
# HEIGHT = 1024

GAUSSIAN_KERNEL_SIZE = 35

pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

hdr_dataset_dir = "/media/winter/_hdd/LavalHDRDataset/IndoorHDRDataset2018/"
param_file = "/media/winter/_hdd/LavalHDRDataset/IndoorHDRDataset2018/param_results.txt"
single_file = "/media/winter/_hdd/LavalHDRDataset/IndoorHDRDataset2018/AG8A8752-5cd7fafa87.exr"
depth_files_dir = '/home/winter/Documents/LavalIndoorHDR_DepthAnnotation/'
