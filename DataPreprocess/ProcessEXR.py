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

GAUSSIAN_KERNEL_SIZE = 35
pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)


def exr2array(file_dir):
    data = OpenEXR.InputFile(file_dir)
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


def exr2jpg(file_dir, jpg_dir):
    data = OpenEXR.InputFile(file_dir)
    data_window = data.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    rgbf = [np.frombuffer(data.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgbf[i] = np.where(rgbf[i] <= 0.0031308, (rgbf[i] * 12.92) * 255.0,
                           (1.055 * (rgbf[i] ** (1.0 / 2.2)) - 0.055) * 255.0)
    rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgbf]
    Image.merge("RGB", rgb8).save(jpg_dir, "JPEG", quality=95)


def write_result(hdr_file_name, param_file_name):
    rgb_data = exr2array(hdr_file_name)
    gray_data = rgb2gray(rgb_data)
    # jpg_rgb_data = np.asarray(Image.open(hdr_file_name.replace(".exr", ".jpg")))

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, regions = retrieve_lights(gray_data, count=3, percentage=0.333)
    param = get_parametric_lights(rgb_data, regions)
    plt.imsave(hdr_file_name.replace(".exr", "_state.jpg"), state_map)
    f = open(param_file_name, "a")
    hdr_file_justname = hdr_file_name.split("/")[-1]
    print(hdr_file_justname)
    print(param)
    f.write("file_"+hdr_file_justname+"\n")
    for item in param:
        f.write("light\n")
        for p in item:
            f.write(p.__str__()+"\n")
    f.close()


if __name__ == '__main__':
    dataset_dir = "/home/winter/Documents/HDR_Dataset/"
    # dataset_dir = "/home/winter/Downloads/IndoorHDRDatasetPreview/100samplesDataset/"
    param_file = "/home/winter/Documents/results/param_result.txt"
    # single_file = "../Files/9C4A9627-545d0bdbb0.exr"
    exr_files = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f)) and f.endswith(".exr")]
    for file in exr_files:
        exr2jpg(join(dataset_dir, file), join(dataset_dir, file.replace(".exr", ".jpg")))
        write_result(join(dataset_dir, file), param_file)
    # exr2jpg(single_file, single_file.replace(".exr", ".jpg"))
    # print_result(single_file)
