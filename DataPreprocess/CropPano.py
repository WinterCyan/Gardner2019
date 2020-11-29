
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
from DataPreprocess.WarpUtils import *
from DataPreprocess.ProcessEXR import *
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

        # max_v = np.amax(nfov)
        # min_v = np.amin(nfov)
        # nfov = (nfov-min_v)/(max_v-min_v)

        # import matplotlib.pyplot as plt
        # plt.imshow(nfov)
        # plt.show()
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


nfov = NFOV()
def get_cropped_and_param(hdr_file_name, count=8):
    # origin_params = text_param2list_param(read_result(light_param_file, hdr_file_name))
    cropped_imgs = []
    cropped_thetas = []
    cropped_phis = []
    img = im.imread(fusion_hdr_jpgs_dir+hdr_file_name.replace(".exr", ".jpg"))
    offset = np.random.uniform(low=0.0, high=0.25)
    for i in range(count):
        # c1 = np.random.uniform(low=0.0, high=2.0)
        c1 = 0.25*(i-1)
        c2 = min(0.6, np.random.normal(loc=CROP_DISTRIB_MU, scale=CROP_DISTRIB_SIGMA))
        # c1 = 0.5108134127257686
        # c2 = 0.65
        center_point = np.array([c1, c2])  # camera center point (valid range [0,2])
        center_row, center_col = crop_center2row_col(center_point)
        crop_theta, crop_phi = row_col2theta_phi(center_row, center_col, WIDTH, HEIGHT)
        single_cropped = nfov.toNFOV(img, center_point)
        cropped_imgs.append(single_cropped)
        cropped_thetas.append(crop_theta)
        cropped_phis.append(crop_phi)
    # got imgs, thetas, phis; then render results;
    cropped_params = []
    for i in range(count):
        crop_img = cropped_imgs[i]
        plt.imsave("../Files/crop_img_"+i.__str__()+".jpg", crop_img)
        crop_theta = cropped_thetas[i]
        crop_phi = cropped_phis[i]
        param = text_param2list_param(read_result(light_param_file, hdr_file_name))
        for light_param in param:
            l = light_param[0]
            l_theta, l_phi = xyz2theta_phi(l[0], l[1], l[2])
            final_theta = l_theta + math.pi/2.0 - crop_theta
            final_phi = l_phi - crop_phi
            rotate_l = theta_phi2xyz(final_theta, final_phi)
            light_param[0] = rotate_l
            # print(param)
            # render_sg(param, hdr_file.replace(".exr", "_"+i.__str__()+".exr"), sg_dir="../Files/")
            # print('---------------------------')
        cropped_params.append(param)
    return {"imgs":cropped_imgs, "params":cropped_params, "thetas":cropped_thetas, "phis":cropped_phis}


if __name__ == '__main__':
    # hdr_file = "AG8A7692-79e4b5baea.exr"
    pfm_files = [f for f in listdir(depth_files_dir) if isfile(join(depth_files_dir, f)) and f.endswith(".pfm")]
    idx = random.randrange(0, len(pfm_files))
    hdr_file_name = pfm_files[idx].replace("-depth.pfm", ".exr")
    print(hdr_file_name)
    crop_results = get_cropped_and_param(hdr_file_name)
    imgs = crop_results["imgs"]
    params = crop_results["params"]
    for i in range(len(params)):
        render_sg(params[i], hdr_file_name, i.__str__()+"_sg.jpg", sg_dir="../Files/")
