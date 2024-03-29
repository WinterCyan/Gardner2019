from DataPreprocess.CropPano import *
import torch
import torch.nn.functional as F
import cv2
import os


if __name__ == '__main__':
    warped_exrs = [f for f in listdir(warped_exr_dir) if isfile(join(warped_exr_dir, f)) and f.endswith(".exr")]
    # warped_exr_files = [f for f in listdir(warped_exr_dir) if isfile(join(warped_exr_dir,f)) and f.endswith(".exr")]
    for file in warped_exrs:
    #     # if not OpenEXR.isOpenExrFile(warped_exr_dir+file):
    #     #     print("not read")
        print(file)
        exr_data = exr2array(warped_exr_dir + file)
        exr_data_tensor = torch.Tensor(exr_data).permute(2,0,1).unsqueeze(0)
        exr_data_tensor_resize =  F.interpolate(exr_data_tensor, size=(RESIZE_H, RESIZE_W), mode='bilinear', align_corners=True)
        exr_data_tensor_resize = exr_data_tensor_resize.squeeze(0)  # [3, H, W]
        exr_data_tensor = exr_data_tensor_resize.permute(1,2,0)
        exr_data = exr_data_tensor.cpu().numpy()
        cv2.imwrite(resized_warped_exr_dir+file, exr_data.astype(np.float32))
    #
    #     im.imwrite("../Files/temp.hdr", exr_data, format='hdr')
    #     hdr_data = cv2.imread("../Files/temp.hdr", cv2.IMREAD_ANYDEPTH)
    #     ldrDurand = tonemap_drago.process(hdr_data)
    #     ldr_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
    #     cv2.imwrite(warped_exr_jpg_dir+file.replace(".exr",".jpg"), ldr_8bit)

    # TODO: resize warped pano
    # cropped_imgs = [f for f in listdir(cropped_imgs_dir) if isfile(join(cropped_imgs_dir,f)) and f.endswith(".jpg")]
    # threshed_hdr = [f for f in listdir(warped_exr_dir) if isfile(join(warped_exr_dir,f)) and f.endswith(".exr")]
    # for crop_img_name in cropped_imgs:
    #     if "]|[" in crop_img_name:
    #         print("skip ", crop_img_name)
    #         continue
    #     crop_img_nojpg_name = crop_img_name.split("/")[-1].split("|")[0]
    #     crop_img_prefix_name = crop_img_nojpg_name.split("_")[0]
    #     print(crop_img_nojpg_name)
    #     theta_phi_string = crop_img_name.split("/")[-1].split("|")[-1].replace(".jpg", "")
    #     theta_phi = np.fromstring(theta_phi_string.split(']')[0].split('[')[1], sep=',')
    #     hdr_data = exr2array(hdr_dataset_dir+crop_img_prefix_name+".exr")
    #     warp_hdr_data = warp_hdr(hdr_data, delta_theta=theta_phi[0], delta_phi=theta_phi[1])
    #     light_hdr_data, ambient = get_threshold_ambient(warp_hdr_data)
    #     ambient_string = ambient.__str__()
    #     cv2.imwrite(warped_exr_dir+crop_img_nojpg_name+".exr", light_hdr_data.astype(np.float32))
    #     os.rename(cropped_imgs_dir+crop_img_name, cropped_imgs_dir+crop_img_name.replace(".jpg", "|"+ambient_string+".jpg"))


    # pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    # hdr_data = exr2array(hdr_dataset_dir+"9C4A0358-ffcdfa74fe.exr")
    # hdr_file_name = "../Files/threshed_pano.hdr"
    #
    # crop_img_name = "AG8A4509-2d64992a75_crop_1|[1.543699685730206, 1.5703918993163193].jpg"
    # crop_img = im.imread(cropped_imgs_dir+crop_img_name)
    # crop_img_nojpg_name = crop_img_name.split("/")[-1].split("|")[0]
    # theta_phi_string = crop_img_name.split("/")[-1].split("|")[-1].replace(".jpg", "")
    # theta_phi = np.fromstring(theta_phi_string.split(']')[0].split('[')[1], sep=',')
    # hdr_data = exr2array(hdr_dataset_dir+"AG8A4509-2d64992a75.exr")
    # warp_hdr_data = warp_hdr(hdr_data, delta_theta=theta_phi[0], delta_phi=theta_phi[1])
    # hdr_data, _ = get_threshold_ambient(warp_hdr_data)
    #
    # im.imwrite(hdr_file_name, hdr_data.astype(np.float32), format='hdr')
    # threshed_hdr = cv2.imread(hdr_file_name, cv2.IMREAD_ANYDEPTH)
    # ldrDurand = tonemap_drago.process(threshed_hdr)
    # ldr_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
    # cv2.imwrite(hdr_file_name.replace(".hdr", ".jpg"), ldr_8bit)
    # hdr_jpg_name = "9C4A0034-a460e29cd9.jpg"
    # hdr_fusion_img = im.imread(fusion_hdr_jpgs_dir+hdr_jpg_name)
    # plt.imshow(hdr_fusion_img)
    # plt.show()
    # idx = random.randrange(0, len(pfm_files))
    # for file in pfm_files:
    #     hdr_file_name = file.replace("-depth.pfm", ".exr")
    #     print(hdr_file_name)
    #     crop_results = get_cropped_and_param(hdr_file_name)
    #     imgs = crop_results["imgs"]
    #     params = crop_results["params"]
    #     thetas = crop_results["thetas"]
    #     phis = crop_results["phis"]
    #     for i in range(len(imgs)):
    #         theta_phi_string = [thetas[i], phis[i]].__str__()
            # plt.imsave(cropped_imgs_dir+hdr_file_name.replace(".exr", "_crop_"+i.__str__()+"|"+theta_phi_string.__str__()+".jpg"), imgs[i])
            # write_crop_param(hdr_file_name.replace(".exr", "_crop_"+i.__str__()), params[i])
