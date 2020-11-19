
# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
from os import listdir
from os.path import isfile, join
from DataPreprocess.Consts import *
import imageio as im


class NFOV():
    def __init__(self, height=SIZE_CROP, width=SIZE_CROP):
        self.FOV = [0.8, 0.8]
        # self.FOV = [0.38, 0.38]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI_2, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI_2, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])

        max_v = np.amax(nfov)
        min_v = np.amin(nfov)
        nfov = (nfov-min_v)/(max_v-min_v)

        import matplotlib.pyplot as plt
        plt.imshow(nfov)
        plt.show()
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)


def crop_center2row_col(center_point):
    c1 = center_point[0]
    c2 = center_point[1]
    col = ((c1+0.5)/2.0) * WIDTH
    row = c2*HEIGHT
    row = max(0, min(HEIGHT-1,int(row)))
    col = max(0, min(WIDTH-1,int(col)))
    return row, col


# test the class
if __name__ == '__main__':
    # jpg_files = [f for f in listdir(hdr_jpgs_dir) if isfile(join(hdr_jpgs_dir, f)) and f.endswith(".jpg")]
    # file_index = random.randrange(0, len(jpg_files))
    # print(jpg_files[file_index])
    # img = im.imread(hdr_jpgs_dir+jpg_files[file_index])
    img = im.imread("../Files/pano.jpg")
    nfov = NFOV()
    c1 = 0.5  # range [0, 2)
    c2 = 0.5  # range [0, 1], greater than 1: upward-down
    for i in range(10):
        c1 = np.random.uniform(low=0.0, high=2.0)
        c2 = np.random.normal(loc=CROP_DISTRIB_MU, scale=CROP_DISTRIB_SIGMA)
        print(c1, c2)
        center_point = np.array([c1, c2])  # camera center point (valid range [0,2])
        center_row, center_col = crop_center2row_col(center_point)
        print("row, col of center: ", center_row, center_col)
        print("theta, phi of center: ", row_col2theta_phi(center_row, center_col, WIDTH, HEIGHT))
        nfov.toNFOV(img, center_point)
    # center_point = np.array([c1, c2])  # camera center point (valid range [0,2])
    # nfov.toNFOV(img, center_point)
