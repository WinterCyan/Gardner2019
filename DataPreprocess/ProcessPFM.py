import numpy as np
import re
import sys
from PIL import Image
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from DataPreprocess.Consts import *


def load_pfm(file_name):
    file = open(depth_files_dir+file_name, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


if __name__ == '__main__':
    pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    for file in pfm_files:
        file_full_name = join(depth_files_dir, file)
        data, scale = load_pfm(file_full_name)
        max_depth = np.amax(data)
        min_depth = np.amin(data)
        waist_line = data[int(data.shape[0]/2), data.shape[1]-5:data.shape[1]+5]
        ave = np.average(data)
        ave_waist = np.average(waist_line)
        print(file+": ", max_depth, ave, ave_waist)
        # data_gray = ((data-min_depth)/(max_depth-min_depth))*255.0
        # plt.imsave(file_full_name.replace(".pfm", ".jpg"), data_gray)
