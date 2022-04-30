import matplotlib.pyplot as plt
from WarpUtils import *
from ProcessEXR import *
from CropPano import *
from Consts import *

def createDataset():
    pass


if __name__ == '__main__':
    # load HDR & segmap
    name = "9C4A0316-01fdb8157d"
    count = 8
    hdr_data = exr2array(hdr_dataset_dir+name+".exr")
    segmap = plt.imread(light_masks_dir_old+name+"_light_mask.jpg")

    # cropped_hdr = []
    converted_ldr = []
    cropped_segmap = []
    for i in range(count):
        c1 = 0.25*(i-1)  # (-0.25, 0, 0.25, 0.5, ..., 1.5), total: 8
        c2 = min(0.8, np.random.normal(loc=CROP_DISTRIB_MU, scale=CROP_DISTRIB_SIGMA))
        center_point = np.array([c1, c2])  # camera center point (valid range [0,2])
        center_row, center_col = crop_center2row_col(center_point)
        crop_theta, crop_phi = row_col2theta_phi(center_row, center_col, WIDTH, HEIGHT)
        partial_hdr = nfov.toNFOV(hdr_data, center_point, True)
        partial_segmap = nfov.toNFOV(segmap, center_point, False)
        # cropped_hdr.append(partial_hdr)
        cropped_segmap.append(partial_segmap)
        ldrDurand = tonemap_drago.process(partial_hdr)
        partial_ldr = np.clip(ldrDurand*255, 0, 255).astype('uint8')
        converted_ldr.append(partial_ldr)
    for i in range(count):
        ldr_img_name = name + "_partial_{}.jpg".format(i)
        seg_label_name = name + "_partial_{}_segmap.jpg".format(i)
        plt.imsave(ldr_img_name, converted_ldr[i])
        plt.imsave(seg_label_name, cropped_segmap[i])

