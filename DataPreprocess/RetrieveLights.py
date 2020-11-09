import numpy as np
from DataPreprocess.LightsToParam import *
from PIL import Image
# from matplotlib import image
from matplotlib import pyplot as plt


def find_peak(img, state_map):
    peak = 0
    p = []
    width = img.shape[0]
    height = img.shape[1]
    for row in range(width):
        for col in range(height):
            if img[row, col] > peak and state_map[row, col] == 0:
                peak = img[row, col]
                p = [row, col]
    return p, peak


def region_growth(img, state_map, percentage):
    border_list = []
    new_border_list = []
    height, width = img.shape
    seed, peak = find_peak(img, state_map)
    print(peak)
    # state_map = np.zeros(shape=img.shape, dtype=int)
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
                    if x+i < 0 or x+j >= height or y+j < 0 or y+j >= width:
                        # print(x+i, y+j)
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
                        if x + i < 0 or x + j >= height or y + j < 0 or y + j >= width:
                            # print(x+i, y+j)
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


def overlap_elimination(regions, width, height):
    sizes = [len(region) for region in regions]  # region area
    size_order = np.argsort(sizes)  # small -> big
    min_region = regions[size_order[0]]
    mid_region = regions[size_order[1]]
    max_region = regions[size_order[2]]
    min_rectangle_corners = get_fitted_ellipse(min_region, width, height)
    mid_rectangle_corners = get_fitted_ellipse(mid_region, width, height)
    max_rectangle_corners = get_fitted_ellipse(max_region, width, height)

    def inside(small_corners, big_corners):
        ls, rs, bs, ts = small_corners
        lb, rb, bb, tb = big_corners
        if ls>=lb and rs<=rb and bs<=bb and ts>=tb: return True
        else: return False

    if inside(min_rectangle_corners, mid_rectangle_corners) or inside(min_rectangle_corners, max_rectangle_corners):  # remove smallest one
        regions.remove(min_region)
    if inside(mid_rectangle_corners, max_rectangle_corners):  # remove the second smallest one, in the case that 3 regions are overlapped together
        regions.remove(mid_region)
    return regions


def retrieve_lights(img, count, percentage):
    state_map = np.zeros(shape=img.shape, dtype=int)
    regions = []
    for i in range(count):
        out_state_map, region = region_growth(img, state_map, percentage)
        state_map = out_state_map
        regions.append(region)
    simplest_regions = overlap_elimination(regions, img.shape[1], img.shape[0])  # eliminate overlapped regions
    print("region number: ", len(simplest_regions))
    return state_map, simplest_regions


