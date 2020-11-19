import random
import math
from DataPreprocess.Consts import *


def warp(pano, target_loc):
    pass


def row_col2theta_phi(row, col, width, height):
    theta = ((row+0.5)/height)*math.pi
    phi = (0.5 - (col+0.5)/width)*2.0*math.pi
    return theta, phi


def theta_phi2row_col(theta, phi, width, height):
    row = (theta/math.pi)*height-0.5
    col = (0.5-phi/(2.0*math.pi))*width-0.5
    row = max(0, min(HEIGHT-1,int(row)))
    col = max(0, min(WIDTH-1,int(col)))
    return row, col


def theta_phi2xyz(theta, phi):
    x = math.sin(theta)*math.sin(phi)
    y = math.cos(theta)
    z = math.sin(theta)*math.cos(phi)
    return np.array((x, y, z))


def crop(img_data):
    col_range = WIDTH - SIZE_CROP
    row_range = HEIGHT - SIZE_CROP
    left_up_row = random.randrange(778, row_range)
    left_up_col = random.randrange(0, col_range)
    crop_img = img_data[left_up_row:left_up_row+SIZE_CROP, left_up_col:left_up_col+SIZE_CROP, :]
    return crop_img


def crop_rec(img):
    col_range = WIDTH - SIZE_CROP
    row_range = HEIGHT - SIZE_CROP
    left_up_row = random.randrange(1000, row_range)
    left_up_col = random.randrange(0, col_range)

    crop_img = img[left_up_row:left_up_row+SIZE_CROP, left_up_col:left_up_col+SIZE_CROP, :]

    theta, phi = row_col2theta_phi(left_up_row, left_up_col, WIDTH, HEIGHT)
    cropped = np.zeros([SIZE_CROP, SIZE_CROP, 3])
    for row in range(SIZE_CROP):
        for col in range(SIZE_CROP):
            delta_theta = row/HEIGHT * math.pi
            delta_phi = col/WIDTH * 2.0 * math.pi
            row_init, col_init = theta_phi2row_col(theta+delta_theta, phi-delta_phi, WIDTH, HEIGHT)
            cropped[row, col, :] = img[row_init, col_init, :]
    return cropped, crop_img
