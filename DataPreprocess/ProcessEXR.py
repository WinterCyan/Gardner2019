import OpenEXR
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from os import listdir
from os.path import isfile, join
from scipy.ndimage.filters import gaussian_filter
from DataPreprocess.RetrieveLights import *
from DataPreprocess.SphericalGaussian import *
from DataPreprocess.ProcessPFM import *
import Imath
import cv2
import imageio as im

pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
tonemap_drago = cv2.createTonemapDrago(2.2, 0.5)
# im.plugins.freeimage.download()


def exr2array(full_file_name):
    data = OpenEXR.InputFile(full_file_name)
    data_window = data.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    # (r_str, g_str, b_str) = data.channels('RGB', pixel_type)
    d_rgb = data.channels('RGB', pixel_type)
    # r_data = np.frombuffer(r_str, dtype=np.float32).reshape(size[1],size[0])
    # g_data = np.frombuffer(g_str, dtype=np.float32).reshape(size[1],size[0])
    # b_data = np.frombuffer(b_str, dtype=np.float32).reshape(size[1],size[0])
    # rgb_data = np.stack((r_data, g_data, b_data), axis=-1)  # shape: [1024,2048,3]

    rgb_data = np.stack((np.frombuffer(d, dtype=np.float32).reshape(size[1],size[0]) for d in d_rgb), axis=-1)
    return rgb_data


def rgb2gray(rgb_data, blur=True):
    r = rgb_data[:, :, 0]
    g = rgb_data[:, :, 1]
    b = rgb_data[:, :, 2]
    gray = r * 0.2989 + g * 0.587 + b * 0.114
    blurred = gaussian_filter(gray, sigma=GAUSSIAN_KERNEL_SIZE)  # shape: [1024,2048]
    return blurred if blur else gray


def tone_mapping(rgb_data):
    mapped = rgb_data / (rgb_data + 1.0)
    gamma_correction = pow(mapped, 1.0 / 2.4)
    return gamma_correction


def exr2jpg(hdr_file_name, full_jpg_name):
    exr_data = exr2array(hdr_dataset_dir+hdr_file_name)
    im.imwrite(hdr_dataset_hdrformat_dir+hdr_dataset_dir.replace(".exr", ".hdr"), exr_data, format='hdr')
    hdr_data = cv2.imread(hdr_dataset_hdrformat_dir+hdr_dataset_dir.replace(".exr", ".hdr"), cv2.IMREAD_ANYDEPTH)
    ldrDurand = tonemap_drago.process(hdr_data)
    ldr_8bit = np.clip(ldrDurand*255, 0, 255).astype('uint8')
    cv2.imwrite(full_jpg_name, ldr_8bit)

    # data = OpenEXR.InputFile(full_file_name)
    # data_window = data.header()['dataWindow']
    # size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    # rgbf = [np.frombuffer(data.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    # for i in range(3):
    #     rgbf[i] = np.where(rgbf[i] <= 0.0031308, (rgbf[i] * 12.92) * 255.0,
    #                        (1.055 * (rgbf[i] ** (1.0 / 2.2)) - 0.055) * 255.0)
    # rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgbf]
    # Image.merge("RGB", rgb8).save(full_jpg_name, "JPEG", quality=95)


def get_threshold_ambient(hdr_data, threshold=0.05, blur=False):
    gray_hdr_data = rgb2gray(hdr_data, blur=False)
    peak = np.amax(gray_hdr_data)
    light_data = np.copy(hdr_data)
    ambient_data = np.copy(hdr_data)
    ambient_area = gray_hdr_data<threshold*peak
    light_data[ambient_area] = 0
    light_area = gray_hdr_data>=threshold*peak
    ambient_data[light_area] = 0
    ambient = np.sum(ambient_data, axis=(0,1))/(np.count_nonzero(ambient_area)/3.0)
    if blur: light_data = gaussian_filter(light_data, sigma=THRESH_GAUSSIAN_KERNEL_SIZE)
    return light_data, ambient


def write_result(hdr_file_name, param_full_file_name):
    rgb_data = exr2array(hdr_dataset_dir+hdr_file_name)
    gray_data = rgb2gray(rgb_data)
    depth_data, _ = load_pfm(hdr_file_name.replace(".exr", "-depth.pfm"))

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, regions = retrieve_lights(gray_data, count=3, percentage=0.333)
    param = get_parametric_lights(rgb_data, depth_data, regions)
    plt.imsave(light_masks_dir+hdr_file_name.replace(".exr", "_light_mask.jpg"), state_map)
    f = open(param_full_file_name, "a")
    print(hdr_file_name)
    print(param)
    f.write("file_"+hdr_file_name+";")
    for light in param:
        f.write("light:")
        for p in light:
            f.write(p.__str__()+",")
        f.write(";")
    f.write('\n')
    f.close()
    print('------------------------------------')


def getLightSemanticMap(hdr_file_name:str, count:int, percentage:float):
    rgb_data = exr2array(hdr_dataset_dir+hdr_file_name)
    # blurred = gaussian_filter(rgb_data, sigma=7)
    gray_data = rgb2gray(rgb_data)
    # depth_data, _ = load_pfm(hdr_file_name.replace(".exr", "-depth.pfm"))

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, _ = retrieve_lights(gray_data, count, percentage)
    return state_map



def write_crop_param(crop_name, param):
    f = open(cropped_param_file, "a")
    print(crop_name)
    print(param)
    f.write("file_"+crop_name+";")
    for light in param:
        f.write("light:")
        for p in light:
            f.write(p.__str__()+",")
        f.write(";")
    f.write('\n')
    f.close()
    print('------------------------------------')


def read_result(param_file_full_name, hdr_file_name):
    param_file = open(param_file_full_name, "r")
    while True:
        line = param_file.readline()
        if line == '':
            print("result for {} not found.".format(hdr_file_name))
            return None
        if line.split(';')[0].split('_')[1] == hdr_file_name:
            num_light = len(line.split(';'))-2
            param = []
            for i in range(num_light):
                light_param_str = line.split(';')[i+1].split(':')[1]
                l = light_param_str.split(',')[0]
                s = light_param_str.split(',')[1]
                c = light_param_str.split(',')[2]
                d = light_param_str.split(',')[3]
                light_param = [l, s, c, d]
                param.append(light_param)
            return param
        else:
            continue


def read_crop_param(crop_param_file, crop_img_name):
    param_file = open(crop_param_file, "r")
    while True:
        line = param_file.readline()
        if line == '':
            print("result for {} not found.".format(crop_img_name))
            return None
        if line.split(';')[0].split("file_")[1] == crop_img_name:
            num_light = len(line.split(';'))-2
            param = []
            for i in range(num_light):
                light_param_str = line.split(';')[i+1].split(':')[1]
                l = light_param_str.split(',')[0]
                s = light_param_str.split(',')[1]
                c = light_param_str.split(',')[2]
                d = light_param_str.split(',')[3]
                light_param = [l, s, c, d]
                param.append(light_param)
            return param
        else:
            continue


def text_param2list_param(param):
    list_param = []
    for light_text_param in param:
        l_text = light_text_param[0]
        s_text = light_text_param[1]
        c_text = light_text_param[2]
        d_text = light_text_param[3]
        l = np.fromstring(l_text.split('[')[1].split(']')[0], dtype=float, sep=' ')
        s = float(s_text)
        c = np.fromstring(c_text.split('[')[1].split(']')[0], dtype=float, sep=' ')
        d = float(d_text)
        light_param = [l, s, c, d]
        list_param.append(light_param)
    return list_param



if __name__ == '__main__':
    exr_files = [f for f in listdir(hdr_dataset_dir) if isfile(join(hdr_dataset_dir,f)) and f.endswith(".exr")]
    for f in exr_files:
        print(f)
        semantic_map = getLightSemanticMap(f, 5, 0.4).astype('uint8')
        # binary_map = np.ones_like(semantic_map)
        # binary_map[np.argmin(semantic_map)] = 0
        # print(np.unique(binary_map))
        # binary_segmap = np.clip(semantic_map * 1000, 0, 255)
        binary_segmap = np.clip(semantic_map*260, 0, 255).astype('uint8')
        # print(np.unique(binary_segmap))
        # print(semantic_map.shape)
        # print(type(semantic_map))
        # print(type(semantic_map[0][0]))
        # plt.imshow(semantic_map)
        # plt.show()
        # plt.imsave(light_masks_dir+f.replace(".exr", "_light_semantic_map.jpg"), binary_segmap)
        Image.fromarray(binary_segmap).save(light_masks_dir+f.replace(".exr", "_light_semantic_map.png"))


    # exr_files = [f for f in listdir(warped_exr_dir) if isfile(join(warped_exr_dir, f)) and f.endswith(".exr")]
    # pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    # print(pfm_files)
    # for file in pfm_files:
    #     write_result(file.replace("-depth.pfm", ".exr"), light_param_file)

    # for file in exr_files:
    #     hdr_file_name = warped_exr_dir + file
    #     try:
    #         hdr_data = exr2array(hdr_file_name)
    #         im.imwrite("../Files/temp.hdr", hdr_data.astype(np.float32), format='hdr')
    #         hdr_data = cv2.imread("../Files/temp.hdr", cv2.IMREAD_ANYDEPTH)
    #         ldrDurand = tonemap_drago.process(hdr_data)
    #         ldr_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
    #         cv2.imwrite("../Files/threshed_pano.jpg", ldr_8bit)
    #         plt.imshow(plt.imread("../Files/threshed_pano.jpg"))
    #         plt.gcf().canvas.set_window_title(file)
    #         plt.show()
    #     except:
    #         print(file+" failed.")
    #     text_param = read_result(light_param_file, file)
    #     param = text_param2list_param(text_param)
    #     render_sg(param, file)
    #     print("rendered "+file)
    # pass
    # crop_img_name = "AG8A9956-30e880bb24_crop_2|[1.2476587714114975, 0.7849937359188709].jpg"
    # crop_img = im.imread(cropped_imgs_dir+crop_img_name)
    # crop_img_nojpg_name = crop_img_name.split("/")[-1].split("|")[0]
    # theta_phi_string = crop_img_name.split("/")[-1].split("|")[-1].replace(".jpg", "")
    # theta_phi = np.fromstring(theta_phi_string.split(']')[0].split('[')[1], sep=',')
    # hdr_data = exr2array(hdr_dataset_dir+"AG8A9956-30e880bb24.exr")
    # warp_hdr_data = warp_hdr(hdr_data, delta_theta=theta_phi[0], delta_phi=theta_phi[1])
    # hdr_file_name = warped_exr_dir+"9C4A8676-1428be837d_crop_3.exr"
    # hdr_data = exr2array(hdr_file_name)
    # im.imwrite(hdr_file_name, hdr_data.astype(np.float32), format='hdr')
    # hdr_data = cv2.imread(hdr_file_name, cv2.IMREAD_ANYDEPTH)
    # ldrDurand = tonemap_drago.process(hdr_data)
    # ldr_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
    # cv2.imwrite("../Files/threshed_pano.jpg", ldr_8bit)
    # pass
