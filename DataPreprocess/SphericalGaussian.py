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


map_of_theta_phi = np.zeros((HEIGHT, WIDTH, 2))
if (isfile('../Files/map_of_theta_phi.npy')):
    print("loaded map_of_theta_phi")
    map_of_theta_phi = np.load('../Files/map_of_theta_phi.npy')
else:
    print("execute map_of_theta_phi calculation")
    u_x = map_of_u[:, :, 0]
    u_y = map_of_u[:, :, 1]
    u_z = map_of_u[:, :, 2]
    theta, phi = xyz2theta_phi(u_x, u_y, u_z)
    expand_theta = np.expand_dims(theta, axis=2)
    expand_phi = np.expand_dims(phi, axis=2)
    map_of_theta_phi = np.concatenate((expand_theta, expand_phi), axis=2)
    np.save('../Files/map_of_theta_phi.npy', map_of_theta_phi)


def render_sg(param, sg_file_name, sg_dir=light_sg_renderings_dir, save_sg=True):
    pano = np.zeros((HEIGHT, WIDTH, 3))
    for light_param in param:
        l, s, c, _ = light_param
        l_dot_u = np.dot(map_of_u, l)
        expo =  (l_dot_u - 1.0) / (s/(4 * np.pi))
        single_channel_weight = np.exp(expo)
        repeated_weight = np.repeat(single_channel_weight[:, :, np.newaxis], 3, axis=2)
        single_light_pano = np.multiply(c, repeated_weight)
        pano = pano + single_light_pano
    if save_sg:
        max_v = np.amax(pano)
        min_v = np.amin(pano)
        pano_corrected = (pano-min_v)/(max_v-min_v)
        plt.imsave(sg_dir+sg_file_name, pano_corrected)
    return pano
