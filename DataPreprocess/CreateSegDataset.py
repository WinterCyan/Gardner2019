import matplotlib.pyplot as plt
import numpy as np

from WarpUtils import *
from ProcessEXR import *
from CropPano import *
from Consts import *

def createDataset():
    exr_files = [f for f in listdir(hdr_dataset_dir) if isfile(join(hdr_dataset_dir,f)) and f.endswith(".exr")]
    random.shuffle(exr_files)
    for f in exr_files[:4]:
        name = f[:-4]
        processHDR(name)


def processHDR(name):
    # load HDR & segmap
    count = 16
    hdr_data = exr2array(hdr_dataset_dir+name+".exr")
    # segmap = plt.imread(light_masks_dir+name+"_light_mask.jpg")
    gray_segmap = plt.imread(light_masks_dir+name+"_light_semantic_map.jpg")
    segmap = np.expand_dims(gray_segmap, axis=-1)
    print(np.unique(segmap))
    print(segmap.shape)

    # cropped_hdr = []
    converted_ldr = []
    cropped_segmap = []
    for i in range(count):
        c1 = 0.125*(i-1)  # (-0.25, 0, 0.25, 0.5, ..., 1.5), total: 8
        # c2 = 0.2
        c2 = np.clip(np.random.normal(loc=CROP_DISTRIB_MU, scale=CROP_DISTRIB_SIGMA), a_min=0.2, a_max=0.55)
        # c2 = np.random.uniform(low=0.2, high=0.5)
        # c2 = np.random.normal(loc=CROP_DISTRIB_MU, scale=CROP_DISTRIB_SIGMA)
        center_point = np.array([c1, c2])  # camera center point (valid range [0,2])
        center_row, center_col = crop_center2row_col(center_point)
        crop_theta, crop_phi = row_col2theta_phi(center_row, center_col, WIDTH, HEIGHT)
        partial_hdr = nfov.toNFOV(hdr_data, center_point, True).astype('float32')
        partial_segmap = nfov.toNFOV(segmap, center_point, False)
        # cropped_hdr.append(partial_hdr)
        cropped_segmap.append(partial_segmap)
        print(np.min(cropped_segmap, axis=(0,1)))
        # print(cropped_segmap[-1].shape)
        ldrDurand = tonemap_drago.process(partial_hdr)
        partial_ldr = np.clip(ldrDurand*255, 0, 255).astype('uint8')
        converted_ldr.append(partial_ldr)
    for i in range(count):
        ldr_img_name = ldr_imgs + name + "_partial_{}.jpg".format(i)
        seg_label_name = seg_labels + name + "_partial_{}_segmap.jpg".format(i)
        plt.imsave(ldr_img_name, converted_ldr[i])
        plt.imsave(seg_label_name, cropped_segmap[i])


if __name__ == '__main__':
    # createDataset()
    processHDR(name="9C4A1707-0f4b3a9a59")