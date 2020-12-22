from matplotlib import pyplot as plt
from DataPreprocess.Consts import *
from os.path import isfile
from DataPreprocess.WarpUtils import *
import time
import torch


map_of_u = np.zeros((HEIGHT, WIDTH, 3))
if (isfile('../Files/map_of_u.npy')):
    print("loaded map_of_u")
    map_of_u = torch.Tensor(np.load('../Files/map_of_u.npy')).permute(2,0,1).to(device)
else:
    print("executed map_of_u calculation")
    for row in range(HEIGHT):
        for col in range(WIDTH):
            theta, phi = row_col2theta_phi(row, col, WIDTH, HEIGHT)
            u = theta_phi2xyz(theta, phi)
            map_of_u[row, col, :] = u
    np.save('../Files/map_of_u.npy', map_of_u)
    map_of_u = torch.Tensor(np.load('../Files/map_of_u.npy')).permute(2,0,1)


map_of_u_resize = np.zeros((RESIZE_H, RESIZE_W, 3))
if (isfile('../Files/map_of_u_resize.npy')):
    print("loaded map_of_u_resize")
    map_of_u_resize = torch.Tensor(np.load('../Files/map_of_u_resize.npy')).permute(2,0,1).to(device)
else:
    print("executed map_of_u_resize calculation")
    for row in range(RESIZE_H):
        for col in range(RESIZE_W):
            theta, phi = row_col2theta_phi(row, col, RESIZE_W, RESIZE_H)
            u = theta_phi2xyz(theta, phi)
            map_of_u_resize[row, col, :] = u
    np.save('../Files/map_of_u_resize.npy', map_of_u_resize)
    map_of_u_resize = torch.Tensor(np.load('../Files/map_of_u_resize.npy')).permute(2,0,1).to(device)


if (isfile('../Files/map_of_theta_phi.npy')):
    print("loaded map_of_theta_phi")
    map_of_theta_phi = torch.Tensor(np.load('../Files/map_of_theta_phi.npy')).permute(2,0,1)
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
    map_of_theta_phi = torch.Tensor(np.load('../Files/map_of_theta_phi.npy')).permute(2,0,1)


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


def render_sg_tensor(ls, ss, cs):  # [l1,l2,l3], [s1,s2,s3], [c1,c2,c3]
    t1 = time.time()
    pano = torch.zeros((3, RESIZE_H, RESIZE_W)).to(device)
    # TODO: calculate as whole
    # or TODO: calculate batch as whole
    for i in range(LIGHT_N):
        l = ls[i].unsqueeze(-1).unsqueeze(-1).to(device)  # [3,1,1]
        s = ss[i].to(device).to(device)
        c = cs[i].unsqueeze(-1).unsqueeze(-1).to(device)  # [3,1,1]
        l_dot_u = torch.mul(map_of_u_resize, l).sum(0, keepdim=True)  # [1,H,W]
        expo = (l_dot_u - 1.0) / (s / (4 * np.pi))  # [1,H,W]
        single_channel_weight = torch.exp(expo)  # [1,H,W]
        repeat_weight = single_channel_weight.repeat(3,1,1)  # [3,H,W]
        single_light_pano = torch.mul(repeat_weight, c)  # [3,H,W]
        pano = pano + single_light_pano
    # np_pano = pano.permute(1,2,0).detach().numpy()
    # max_v = np.amax(np_pano)
    # min_v = np.amin(np_pano)
    # pano_corrected = (np_pano - min_v) / (max_v - min_v)
    # plt.imsave("../Files/temppano.jpg", pano_corrected)
    # print("saved.")
    # print("max of sg pano: ", torch.max(pano))
    t2 = time.time()
    print("render sg time: ", t2-t1)
    return pano

#
# def render_sg_batch(param_batch, batch_size=BATCH_SIZE):
#     pano_batch = torch.zeros((batch_size, 3, HEIGHT, WIDTH)).to(device)
#     # TODO: run on GPU
#     for batch_idx in range (batch_size):
#         single_img_param = [param[batch_idx] for param in param_batch]  # shape: [3,9,3,9,3]
#         param = []
#         for i in range (LIGHT_N):
#             light_param = []
#             light_param.append(single_img_param[0][i])  # d
#             light_param.append(single_img_param[1][0*i:3*i])  # l
#             light_param.append(single_img_param[2][i])  # s
#             light_param.append(single_img_param[3][0*i:3*i])  # c
#             # light_param.append(single_img_param[4][i])
#             param.append(light_param)
#         print(param)

