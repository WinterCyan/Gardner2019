from shutil import copyfile
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from DataPreprocess.Consts import *

srcfiles = [f for f in listdir(cropped_imgs_dir)]
des_dir = '/media/winter/Winter SSD/LEDataset/CroppedImgs/'
f = open(join(des_dir, 'filenames.txt'), 'a')
for i in range(len(srcfiles)):
    img = plt.imread(join(cropped_imgs_dir, srcfiles[i]))
    f.write(srcfiles[i]+'\n')
    plt.imsave(join(des_dir, i.__str__()+'.jpg'), img)
    if i % 100 == 0:
        print(i)
    # plt.imshow(img)
    # plt.show()