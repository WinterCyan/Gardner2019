import random
from DataPreprocess.Consts import *


def warp_hdr(hdr_data, delta_theta, delta_phi):
    map_of_theta_phi = np.load('../Files/map_of_theta_phi.npy')  # map_of_theta_phi: [H, W, 2]
    map_of_theta = map_of_theta_phi[:,:,0]
    map_of_phi = map_of_theta_phi[:,:,1]
    delta_theta_phi_array = np.array([delta_theta-np.pi/2.0, delta_phi])
    delta_theta_phi_array = np.expand_dims(np.expand_dims(delta_theta_phi_array, axis=0), axis=0)
    delta_theta_phi = np.repeat(delta_theta_phi_array, HEIGHT, axis=0)
    delta_theta_phi = np.repeat(delta_theta_phi, WIDTH, axis=1)  # delta_theta_phi: [H, W, 2]
    warp_theta_phi = np.add(map_of_theta_phi, delta_theta_phi)
    warp_theta = warp_theta_phi[:,:,0]
    warp_phi = warp_theta_phi[:,:,1]
    warp_row, warp_col = theta_phi2row_col_array(warp_theta_phi[:,:,0], warp_theta_phi[:,:,1], WIDTH, HEIGHT)
    warp_row = np.expand_dims(warp_row, axis=2)
    warp_col = np.expand_dims(warp_col, axis=2)
    warp_row_col = np.concatenate((warp_row, warp_col), axis=2)
    warp_hdr_data = np.zeros((HEIGHT, WIDTH, 3))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            warp_hdr_data[row, col, :] = hdr_data[warp_row_col[row, col, 0], warp_row_col[row, col, 1], :]
    return warp_hdr_data


def row_col2theta_phi(row, col, width, height):
    theta = ((row+0.5)/height)*np.pi
    phi = (0.5 - (col+0.5)/width)*2.0*np.pi
    return theta, phi


# def theta_phi2row_col(theta, phi, width, height):
#     row = (theta/np.pi)*height-0.5
#     col = (0.5-phi/(2.0*np.pi))*width-0.5
#     row = max(0, min(HEIGHT-1,int(row)))
#     col = max(0, min(WIDTH-1,int(col)))
#     return row, col


def theta_phi2row_col_array(theta, phi, width, height):
    row = (theta/np.pi)*height-0.5
    col = (0.5-phi/(2.0*np.pi))*width-0.5
    row = row.astype(int)
    col = col.astype(int)
    print("before:", np.amax(row))
    print("before:", np.amax(col))
    print("before:", np.amin(row))
    print("before:", np.amin(col))  # before: min col = 0 ???
    row = np.clip(row, 0, height-1)
    # col = np.clip(col, 1-width, width-1)
    # row = row%HEIGHT
    col = col%width
    print(np.amax(row))
    print(np.amax(col))
    return row, col


def theta_phi2xyz(theta, phi):
    x = np.sin(theta)*np.sin(phi)
    y = np.cos(theta)
    z = np.sin(theta)*np.cos(phi)
    return np.array((x, y, z))


def xyz2theta_phi(x, y, z):
    theta = np.arccos(y)
    # phi = np.arcsin(x/np.sin(theta))
    phi = np.arctan2(x, z)
    return theta, phi
