import os
import imageio
import cv2
import matplotlib.pyplot as plt

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import numpy as np
from Consts import *
from os.path import *
from os import listdir
from ProcessEXR import exr2array


def hdr2ldr(hdr_file_name):
    # hdr = cv2.imread(hdr_dataset_dir+hdr_file_name)
    hdr = exr2array(hdr_dataset_dir+hdr_file_name)
    print(type(hdr[0][0][0]))

    # clamp
    # ldr = np.clip(hdr,0.0,1.0)
    # ldr = ldr ** (1/2.2)
    # ldr = 255.0 * ldr

    # drago
    tonemap = cv2.createTonemapDrago(2.2)
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr*255,0,255).astype('uint8')

    # imageio
    # hdr = imageio.imread(hdr_dataset_dir+hdr_file_name)
    # ldr = np.clip(hdr, 0, 1)
    # ldr = ldr**(1/2.2)
    # ldr = 255.0 * ldr

    return ldr


if __name__ == '__main__':
    exr_files = [f for f in listdir(hdr_dataset_dir) if isfile(join(hdr_dataset_dir,f)) and f.endswith(".exr")]
    for f in exr_files[:100]:
        print(f)
        ldr = hdr2ldr(f)
        plt.imshow(ldr)
        plt.show()
        # cv2.imwrite(ldr_dir+f.replace(".exr", "_ldr.jpg"), ldr)
