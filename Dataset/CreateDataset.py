from DataPreprocess.CropPano import *


if __name__ == '__main__':
    pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    # idx = random.randrange(0, len(pfm_files))
    for file in pfm_files:
        hdr_file_name = file.replace("-depth.pfm", ".exr")
        print(hdr_file_name)
        crop_results = get_cropped_and_param(hdr_file_name)
        imgs = crop_results["imgs"]
        params = crop_results["params"]
        for i in range(len(imgs)):
            plt.imsave(cropped_imgs_dir+hdr_file_name.replace(".exr", "_crop_"+i.__str__()+".jpg"), imgs[i])
            write_crop_param(hdr_file_name.replace(".exr", "_crop_"+i.__str__()), params[i])
