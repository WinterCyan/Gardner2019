from DataPreprocess.ProcessEXR import *
from DataPreprocess.WarpUtils import *
import cv2
import imageio as im


tonemap_drago = cv2.createTonemapDrago(2.2, 0.5)
def new_exr2jpg(hdr_file_name, full_jpg_name):
    exr_data = exr2array(hdr_dataset_dir+hdr_file_name)
    im.imwrite(hdr_dataset_hdrformat_dir+hdr_file_name.replace(".exr", ".hdr"), exr_data, format='hdr')
    hdr_data = cv2.imread(hdr_dataset_hdrformat_dir+hdr_file_name.replace(".exr", ".hdr"), cv2.IMREAD_ANYDEPTH)
    ldrDurand = tonemap_drago.process(hdr_data)
    ldr_8bit = np.clip(ldrDurand*255, 0, 255).astype('uint8')
    cv2.imwrite(full_jpg_name, ldr_8bit)


if __name__ == '__main__':
    pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    for file in pfm_files:
        print(file)
        new_exr2jpg(file.replace("-depth.pfm", ".exr"), fusion_hdr_jpgs_dir+file.replace("-depth.pfm", ".jpg"))
