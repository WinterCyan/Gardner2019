import random
import math
from DataPreprocess.Consts import *


def warp_hdr(hdr_data, theta, phi):
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


def xyz2theta_phi(x, y, _):
    theta = math.acos(y)
    phi = math.asin(x/math.sin(theta))
    return theta, phi
