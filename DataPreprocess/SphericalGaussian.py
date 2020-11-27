from matplotlib import pyplot as plt
from DataPreprocess.Consts import *
from os.path import isfile
from DataPreprocess.WarpUtils import *


map_of_u = np.zeros((HEIGHT, WIDTH, 3))
if (isfile('../Files/map_of_u.npy')):
    print("loaded map_of_u")
    map_of_u = np.load('../Files/map_of_u.npy')
else:
    print("executed map_of_u calculation")
    for row in range(HEIGHT):
        for col in range(WIDTH):
            theta, phi = row_col2theta_phi(row, col, WIDTH, HEIGHT)
            u = theta_phi2xyz(theta, phi)
            map_of_u[row, col, :] = u
    np.save('../Files/map_of_u.npy', map_of_u)


def render_sg(param, hdr_file_name, sg_file_name, sg_dir=light_sg_renderings_dir):
    pano = np.zeros((HEIGHT, WIDTH, 3))
    for light_param in param:
        l, s, c, _ = light_param
        l_dot_u = np.dot(map_of_u, l)
        expo =  (l_dot_u - 1.0) / (s/(4 * math.pi))
        single_channel_weight = np.exp(expo)
        repeated__weight = np.repeat(single_channel_weight[:, :, np.newaxis], 3, axis=2)
        single_light_pano = np.multiply(c, repeated__weight)
        pano = pano + single_light_pano
    max_v = np.amax(pano)
    min_v = np.amin(pano)
    pano_corrected = (pano-min_v)/(max_v-min_v)
    plt.imsave(sg_dir+sg_file_name, pano_corrected)
