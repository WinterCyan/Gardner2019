import numpy as np
from DataPreprocess.WarpUtils import *
from PIL import Image
# from matplotlib import image
from matplotlib import pyplot as plt


WIDTH = 7768
HEIGHT = 3884


def find_peak(img, state_map):
    peak = 0
    p = []
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if img[row, col] > peak and state_map[row, col] == 0:
                peak = img[row, col]
                p = [row, col]
    return p, peak


def region_growth(img, state_map, percentage):
    border_list = []
    new_border_list = []
    seed, peak = find_peak(img, state_map)
    state_map[seed[0], seed[1]] = 1
    new_border_list.append(seed)
    changed = True
    region = []
    while changed:
        changed = False
        border_list = new_border_list
        new_border_list = []
        for p in border_list:
            has_new_light_neighbor = False
            x, y = p
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if x+i < 0 or x+j >= HEIGHT or y+j < 0 or y+j >= WIDTH:
                        continue
                    if (not(i == 0 and j == 0)) and state_map[x+i, y+j] == 0 and img[x+i, y+j] >= peak*percentage:
                        new_border_list.append([x+i, y+j])
                        state_map[x+i, y+j] = 1
                        state_map[x, y] = 2
                        region.append(p)
                        has_new_light_neighbor = True
                        changed = True
            if not has_new_light_neighbor:
                is_inner = True
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if x + i < 0 or x + j >= HEIGHT or y + j < 0 or y + j >= WIDTH:
                            continue
                        if state_map[p[0]+i, p[1]+j] == 0:
                            is_inner = False
                if not is_inner:
                    new_border_list.append(p)
                else:
                    state_map[p[0], p[1]] = 2
                    region.append(p)
    for p in border_list:
        state_map[p[0], p[1]] = 2
        region.append(p)
    return state_map, region


def get_fitted_ellipse(region):
    #  region: [N,2]
    region = np.asarray(region)  # convert to array
    #  in rectangle
    arg_min_x, arg_min_y = np.argmin(region, axis=0)
    arg_max_x, arg_max_y = np.argmax(region, axis=0)
    top_corner = region[arg_min_x][0]
    left_corner = region[arg_min_y][1]
    bottom_corner = region[arg_max_x][0]
    right_corner = region[arg_max_y][1]
    #  to sphere coordinates
    region_size = region.shape[0]
    rows = region[:, 0]
    cols = region[:, 1]
    thetas, phis = row_col2theta_phi(rows, cols, WIDTH, HEIGHT)
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
    return {"sphere":[left, right, bottom, top], "rectangle":[left_corner, right_corner, bottom_corner, top_corner]}


def rectangle_size(region):
    l, r, b, t = get_fitted_ellipse(region)["rectangle"]
    return (r-l+1)*(b-t+1)


def draw_rectangle(state_map, corners, frame_value):
    left_corner, right_corner, bottom_corner, top_corner = corners
    for col in range(left_corner, right_corner+1):
        state_map[bottom_corner, col] = frame_value
        state_map[top_corner, col] = frame_value
    for row in range(top_corner, bottom_corner+1):
        state_map[row, left_corner] = frame_value
        state_map[row, right_corner] = frame_value


def overlap_elimination(regions):
    sizes = [rectangle_size(region) for region in regions]  # region area
    size_order = np.argsort(sizes)  # small -> big
    min_region = regions[size_order[0]]
    mid_region = regions[size_order[1]]
    max_region = regions[size_order[2]]
    min_rectangle_corners = get_fitted_ellipse(min_region)["rectangle"]
    mid_rectangle_corners = get_fitted_ellipse(mid_region)["rectangle"]
    max_rectangle_corners = get_fitted_ellipse(max_region)["rectangle"]

    def inside(small_corners, big_corners):
        ls, rs, bs, ts = small_corners
        lb, rb, bb, tb = big_corners
        if ls>=lb and rs<=rb and bs<=bb and ts>=tb:
            return True
        else: return False

    if inside(min_rectangle_corners, mid_rectangle_corners):
        mid_region.extend(min_region)
        regions.remove(min_region)
    if inside(min_rectangle_corners, max_rectangle_corners):  # remove smallest one
        max_region.extend(min_region)
        regions.remove(min_region)
    if inside(mid_rectangle_corners, max_rectangle_corners):  # remove the second smallest one, in the case that 3 regions are overlapped together
        max_region.extend(mid_region)
        regions.remove(mid_region)
    return regions


def retrieve_lights(img, count, percentage):
    state_map = np.zeros(shape=img.shape, dtype=int)
    regions = []
    for i in range(count):
        out_state_map, region = region_growth(img, state_map, percentage)
        state_map = out_state_map
        regions.append(region)
    simplest_regions = overlap_elimination(regions)
    return state_map, simplest_regions


def get_parametric_lights(rgb_img_data, regions):
    param=[]
    for region in regions:
        single_param = []
        corners = get_fitted_ellipse(region)["sphere"]
        center = [(corners[2]+corners[3])/2.0, (corners[0]+corners[1])/2.0]  # phi, theta
        s = (abs(corners[1]-corners[0]) + abs(corners[2]-corners[3]))/2.0
        l = theta_phi2xyz(center[1], center[0])
        r = 0
        g = 0
        b = 0
        for p in region:
            r = r + rgb_img_data[p[0], p[1], 0]
            g = g + rgb_img_data[p[0], p[1], 1]
            b = b + rgb_img_data[p[0], p[1], 2]
        c = np.array([r, g, b])/len(region)
        single_param.extend([l, s, c])  # single_param: [[l],[s],[c]]
        param.append(single_param)
    return param  # [[array, float, array], [array, float, array], [array, float, array]]

