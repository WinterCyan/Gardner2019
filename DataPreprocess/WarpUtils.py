import math
#  define pano's warp operation
#  input: pano with depth annotation & a pixel in target location
#  output: new pano


def warp(pano, target_loc):
    pass


def row_col2theta_phi(row, col, width, height):
    theta = ((row+0.5)/height)*math.pi
    phi = ((col+0.5)/width - 0.5)*2.0*math.pi
    return theta, phi


def theta_phi2xyz(theta, phi):
    x = math.sin(theta)*math.sin(phi)
    y = math.cos(theta)
    z = math.sin(theta)*math.cos(phi)
    return x, y, z
