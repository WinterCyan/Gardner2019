import numpy as np
from numpy import array
from DataPreprocess.WarpUtils import *
from matplotlib import pyplot as plt
from DataPreprocess.Consts import *


# map_of_u = np.zeros((HEIGHT, WIDTH, 3))
map_of_u = np.load('map_of_u.npy')
# print("executed map_of_u")
# for row in range(HEIGHT):
#     for col in range(WIDTH):
#         theta, phi = row_col2theta_phi(row, col, HEIGHT, WIDTH)
#         u = theta_phi2xyz(theta, phi)
#         map_of_u[row, col, :] = u
# np.save('map_of_u.npy', map_of_u)

def render_sg(param):
    pano = np.zeros((HEIGHT, WIDTH, 3))
    for light_param in param:
        l, s, c = light_param
        l_dot_u = np.dot(map_of_u, l)
        expo =  (l_dot_u - 1) / (s/(4 * math.pi))
        single_channel_weight = np.exp(expo)
        repeated__weight = np.repeat(single_channel_weight[:, :, np.newaxis], 3, axis=2)
        single_light_pano = np.multiply(c, repeated__weight)
        pano = pano + single_light_pano
    return pano


if __name__ == '__main__':
    param = [[array([-0.13448813,  0.97001897, -0.20242565]), 0.1354832053234527, array([0.20750341, 0.12834334, 0.05568736])], [array([-0.65908867,  0.4200916 , -0.62379899]), 0.08573862545842387, array([0.12924565, 0.07920828, 0.02991809])]]

    pano = render_sg(param)
    max_v = np.amax(pano)
    min_v = np.amin(pano)
    pano_correct = (pano-min_v)/(max_v-min_v)
    plt.imsave("light_pano.jpg", pano_correct)
    max_v = np.amax(map_of_u)
    min_v = np.amin(map_of_u)
    map_correct = (map_of_u-min_v)/(max_v-min_v)
    plt.imsave("map_of_u.jpg", map_correct)

