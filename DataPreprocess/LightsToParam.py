import numpy as np
from DataPreprocess.RetrieveLights import *
from DataPreprocess.WarpUtils import *
#  light params: d, l, s, c
#  light region to param:
#       peak's loc -> l
#       region's average depth -> d
#       region's fitting eclipse -> s
#       region's average color -> c

#  regions: eclipse, fitted with horizontal rectangles

def lightregion_to_param(img):
    _, regions = retrieve_lights(img, count=3, percentage=0.8)


def get_fitted_ellipse(region, width, height):
    #  region: [N,2]
    region = np.asarray(region)  # convert to array
    #  to sphere coordinates
    region_size = region.shape[0]
    rows = region[:, 0]
    cols = region[:, 1]
    thetas, phis = row_col2theta_phi(rows, cols, width, height)
    thetas = thetas.reshape((region_size, 1))
    phis = phis.reshape((region_size, 1))
    new_region = np.concatenate((thetas, phis), axis=1)
    #  end
    arg_min_x, arg_min_y = np.argmin(new_region, axis=0)
    arg_max_x, arg_max_y = np.argmax(new_region, axis=0)
    top = new_region[arg_min_x][0]
    left = new_region[arg_min_y][1]
    bottom = new_region[arg_max_x][0]
    right = new_region[arg_max_y][1]
    return [left, right, bottom, top]


def draw_rectangle(state_map, corners):
    x_max, y_max, x_min, y_min = corners
    for col in range(x_min, x_max+1):
        state_map[col, y_min] = 3
        state_map[col, y_max] = 3
    for row in range(y_min, y_max+1):
        state_map[x_min, row] = 3
        state_map[x_max, row] = 3


if __name__ == "__main__":
    img = Image.open("C:\\Desktop\\drawlight.png", mode='r')
    img_array = np.asarray(img)[:, :, 0]

    # state_map: 2 for lights areas, 0 for others
    # regions: array of region, region is array of coords of light pixels
    state_map, regions = retrieve_lights(img_array, count=3, percentage=0.95)
    for region in regions:
        corners = get_fitted_ellipse(region, img_array.shape[1], img_array.shape[0])
        center = [(corners[2]+corners[3])/2.0, (corners[0]+corners[1])/2.0]  # phi, theta
        print("center:", center)
        s = (abs(corners[1]-corners[0]) + abs(corners[2]-corners[3]))/2.0
        print("s:", s)
        l = theta_phi2xyz(center[1], center[0])
        print("l:", l)
        # draw_rectangle(state_map, corners)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(state_map)
    plt.show()
