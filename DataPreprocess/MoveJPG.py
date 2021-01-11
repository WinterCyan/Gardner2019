from shutil import copyfile
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from DataPreprocess.Consts import *

# srcfiles = [f for f in listdir(cropped_imgs_dir)]
# des_dir = '/media/winter/Winter SSD/LEDataset/CroppedImgs/'
# f = open(join(des_dir, 'filenames.txt'), 'a')
# for i in range(len(srcfiles)):
#     img = plt.imread(join(cropped_imgs_dir, srcfiles[i]))
#     f.write(srcfiles[i]+'\n')
#     plt.imsave(join(des_dir, i.__str__()+'.jpg'), img)
#     if i % 100 == 0:
#         print(i)
    # plt.imshow(img)
    # plt.show()

f = open(join(data_home_dir, "HDR_results/filenames.txt"))
names = f.readlines()
print(len(names))
for i in range(len(names)):
    if isfile(join(cropped_imgs_dir, i.__str__()+".jpg")):
        os.rename(join(cropped_imgs_dir, i.__str__()+".jpg"), join(cropped_imgs_dir, names[i].strip()))
