import OpenEXR
import Imath
from os import listdir
from os.path import isfile, join
import array
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from DataPreprocess.LightsToParam import *
from DataPreprocess.RetrieveLights import *

GAUSSIAN_KERNEL_SIZE = 15
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


def plot_result(file_name):
    rgb_data = exr2array(file_name)
    gray_data = rgb2gray(rgb_data)
    jpg_rgb_data = np.asarray(Image.open(file_name.replace(".exr", ".jpg")))

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, regions = retrieve_lights(gray_data, count=3, percentage=0.333)
    frame_value = 3
    for region in regions:
        sphere_corners = get_fitted_ellipse(region, gray_data.shape[1], gray_data.shape[0])["sphere"]
        center = [(sphere_corners[2] + sphere_corners[3]) / 2.0, (sphere_corners[0] + sphere_corners[1]) / 2.0]  # phi, theta
        print("center:", center)
        s = (abs(sphere_corners[1] - sphere_corners[0]) + abs(sphere_corners[2] - sphere_corners[3])) / 2.0
        print("s:", s)
        l = theta_phi2xyz(center[1], center[0])
        print("l:", l)
        rectangle_corners = get_fitted_ellipse(region, gray_data.shape[1], gray_data.shape[0])["rectangle"]
        draw_rectangle(state_map, rectangle_corners, frame_value)
        frame_value = frame_value+1

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(jpg_rgb_data)
    fig.add_subplot(1, 2, 2)
    plt.imshow(state_map)
    plt.show()


if __name__ == '__main__':
    dataset_dir = "/home/winter/Downloads/IndoorHDRDatasetPreview/100samplesDataset/"
    exr_files = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f)) and f.endswith(".exr")]
    for file in exr_files:
        # exr2jpg(join(dataset_dir, file), join(dataset_dir, file.replace(".exr", ".jpg")))
        plot_result(join(dataset_dir, file))
    # plot_result(join(dataset_dir, "AG8A9144-ae4159fa42.exr"))
