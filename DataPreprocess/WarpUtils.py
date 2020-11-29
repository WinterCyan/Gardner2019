import random
from DataPreprocess.Consts import *


def warp_hdr(hdr_data, delta_theta, delta_phi):
    map_of_theta_phi = np.load('../Files/map_of_theta_phi.npy')  # map_of_theta_phi: [H, W, 2]
    delta_theta_phi_array = np.array([delta_theta, delta_phi])
    delta_theta_phi_array = np.expand_dims(np.expand_dims(delta_theta_phi_array, axis=0), axis=0)
    delta_theta_phi = np.repeat(delta_theta_phi_array, HEIGHT, axis=0)
    delta_theta_phi = np.repeat(delta_theta_phi, WIDTH, axis=1)  # delta_theta_phi: [H, W, 2]
    warp_theta_phi = np.add(map_of_theta_phi, delta_theta_phi)
    warp_row, warp_col = theta_phi2row_col_array(warp_theta_phi[:,:,0], warp_theta_phi[:,:,1], WIDTH, HEIGHT)
    print(np.amax(warp_row))
    print(np.amax(warp_col))
    warp_row = np.expand_dims(warp_row, axis=2)
    warp_col = np.expand_dims(warp_col, axis=2)
    warp_row_col = np.concatenate((warp_row, warp_col), axis=2)
    print(warp_row_col.shape)
    print(hdr_data.shape)
    warp_hdr_data = np.zeros((HEIGHT, WIDTH, 3))
    print(np.amax(warp_row_col[:,:,0]))
    print(np.amax(warp_row_col[:,:,1]))
    print(np.amin(warp_row_col[:,:,0]))
    print(np.amin(warp_row_col[:,:,1]))
    for row in range(HEIGHT):
        for col in range(WIDTH):
            # TODO: figure out row and col
            warp_hdr_data[row, col, :] = hdr_data[warp_row_col[row, col, 1], warp_row_col[row, col, 0], :]
    return warp_hdr_data


def row_col2theta_phi(row, col, width, height):
    theta = ((row+0.5)/height)*np.pi
    phi = (0.5 - (col+0.5)/width)*2.0*np.pi
    return theta, phi


def theta_phi2row_col(theta, phi, width, height):
    row = (theta/np.pi)*height-0.5
    col = (0.5-phi/(2.0*np.pi))*width-0.5
    row = max(0, min(HEIGHT-1,int(row)))
    col = max(0, min(WIDTH-1,int(col)))
    return row, col


def theta_phi2row_col_array(theta, phi, width, height):
    row = (theta/np.pi)*height-0.5
    col = (0.5-phi/(2.0*np.pi))*width-0.5
    row = row.astype(int)
    col = col.astype(int)
    # row = max(0, min(HEIGHT-1,row.astype(int)))
    # col = max(0, min(WIDTH-1,col.astype(int)))
    np.clip(row, 0, HEIGHT-1)
    np.clip(col, 0, WIDTH-1)
    return row, col


def theta_phi2xyz(theta, phi):
    x = np.sin(theta)*np.sin(phi)
    y = np.cos(theta)
    z = np.sin(theta)*np.cos(phi)
    return np.array((x, y, z))


def xyz2theta_phi(x, y, _):
    theta = np.arccos(y)
    phi = np.arcsin(x/np.sin(theta))
    return theta, phi
