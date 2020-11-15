import numpy as np
from numpy import array
from DataPreprocess.WarpUtils import *
from matplotlib import pyplot as plt
from DataPreprocess.Consts import *
from os.path import isfile


map_of_u = np.zeros((HEIGHT, WIDTH, 3))
if (isfile('map_of_u.npy')):
    print("loaded map_of_u")
    map_of_u = np.load('map_of_u.npy')
else:
    print("executed map_of_u calculation")
    for row in range(HEIGHT):
        for col in range(WIDTH):
            theta, phi = row_col2theta_phi(row, col, WIDTH, HEIGHT)
            u = theta_phi2xyz(theta, phi)
            map_of_u[row, col, :] = u
    np.save('map_of_u.npy', map_of_u)


def render_sg(param):
    pano = np.zeros((HEIGHT, WIDTH, 3))
    for light_param in param:
        l, s, c = light_param
        l_dot_u = np.dot(map_of_u, l)
        expo =  (l_dot_u - 1.0) / (s/(4 * math.pi))
        single_channel_weight = np.exp(expo)
        repeated__weight = np.repeat(single_channel_weight[:, :, np.newaxis], 3, axis=2)
        single_light_pano = np.multiply(c, repeated__weight)
        pano = pano + single_light_pano
    return pano


if __name__ == '__main__':
    param = [[array([0.        , 0.97615965, 0.21705377]), 3.3595790645430497, array([0.39468768, 0.34555568, 0.17350615])], [array([0.        , 0.96751224, 0.25282419]), 3.3713074614217966, array([0.15601374, 0.13731149, 0.06627019])]]
    pano = render_sg(param)
    max_v = np.amax(pano)
    min_v = np.amin(pano)
    pano_corrected = (pano-min_v)/(max_v-min_v)
    plt.imsave("../Files/light_pano.jpg", pano_corrected)

