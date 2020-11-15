import OpenEXR
import Imath
from os import listdir
from os.path import isfile, join
import array
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from DataPreprocess.RetrieveLights import *
from DataPreprocess.Consts import *


def exr2array(full_file_name):
    data = OpenEXR.InputFile(full_file_name)
    data_window = data.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    r_str = data.channel('R', pixel_type)
    r_data = np.frombuffer(r_str, dtype=np.float32)
    r_data.shape = (size[1], size[0])

    g_str = data.channel('G', pixel_type)
    g_data = np.frombuffer(g_str, dtype=np.float32)
    g_data.shape = (size[1], size[0])

    b_str = data.channel('B', pixel_type)
    b_data = np.frombuffer(b_str, dtype=np.float32)
    b_data.shape = (size[1], size[0])

    rgb_data = np.stack((r_data, g_data, b_data), axis=-1)  # shape: [1024,2048,3]
    return rgb_data


def rgb2gray(rgb_data):
    r = rgb_data[:, :, 0]
    g = rgb_data[:, :, 1]
    b = rgb_data[:, :, 2]
    gray = r * 0.2989 + g * 0.587 + b * 0.114
    blurred = gaussian_filter(gray, sigma=GAUSSIAN_KERNEL_SIZE)  # shape: [1024,2048]
    return blurred


def tone_mapping(rgb_data):
    mapped = rgb_data / (rgb_data + 1.0)
    gamma_correction = pow(mapped, 1.0 / 2.4)
    return gamma_correction


def exr2jpg(full_file_name, full_jpg_name):
    data = OpenEXR.InputFile(full_file_name)
    data_window = data.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    rgbf = [np.frombuffer(data.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgbf[i] = np.where(rgbf[i] <= 0.0031308, (rgbf[i] * 12.92) * 255.0,
                           (1.055 * (rgbf[i] ** (1.0 / 2.2)) - 0.055) * 255.0)
    rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgbf]
    Image.merge("RGB", rgb8).save(full_jpg_name, "JPEG", quality=95)


def write_result(hdr_file_name, param_full_file_name):
    rgb_data = exr2array(hdr_dataset_dir+hdr_file_name)
    gray_data = rgb2gray(rgb_data)
    # jpg_rgb_data = np.asarray(Image.open(hdr_file_name.replace(".exr", ".jpg")))

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, regions = retrieve_lights(gray_data, count=3, percentage=0.333)
    param = get_parametric_lights(rgb_data, regions)
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


def read_result(param_file_full_name, hdr_file_name):
    file = open(param_file_full_name, "r")
    while True:
        line = file.readline()
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
                light_param = [l, s, c]
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
        l = np.fromstring(l_text.split('[')[1].split(']')[0], dtype=float, sep=' ')
        s = float(s_text)
        c = np.fromstring(c_text.split('[')[1].split(']')[0], dtype=float, sep=' ')
        light_param = [l, s, c]
        list_param.append(light_param)
    return list_param



if __name__ == '__main__':
    exr_files = [f for f in listdir(hdr_sample_dir) if isfile(join(hdr_dataset_dir, f)) and f.endswith(".exr")]
    for file in exr_files:
        exr2jpg(hdr_dataset_dir+file, hdr_jpgs_dir+file.replace(".exr", ".jpg"))
        write_result(file, light_param_file)

    # param = read_result(light_param_file, file_name)
    # list_param = text_param2list_param(param)
    # print(list_param)
