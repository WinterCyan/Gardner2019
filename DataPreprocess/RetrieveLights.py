import numpy as np
from PIL import Image
# from matplotlib import image
from matplotlib import pyplot as plt


def find_peak(img):
    peak = 0
    p = []
    width = img.shape[0]
    height = img.shape[1]
    for row in range(width):
        for col in range(height):
            if img[row, col] > peak:
                peak = img[row, col]
                p = [row, col]
    return p, peak


def region_growth(img):
    border_list = []
    new_border_list = []
    init_seed, peak = find_peak(img)
    state_map = np.zeros(shape=img.shape, dtype=int)
    state_map[init_seed[0], init_seed[1]] = 1
    new_border_list.append(init_seed)
    changed = True
    while changed:
        changed = False
        border_list = new_border_list
        new_border_list = []
        for p in border_list:
            has_new_light_neighbor = False
            x, y = p
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (not(i == 0 and j == 0)) and state_map[x+i, y+j] == 0 and img[x+i, y+j] >= peak*0.95:
                        new_border_list.append([x+i, y+j])
                        state_map[x+i, y+j] = 1
                        state_map[x, y] = 2
                        has_new_light_neighbor = True
                        changed = True
            if not has_new_light_neighbor:
                is_inner = True
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if state_map[p[0]+i, p[1]+j] == 0:
                            is_inner = False
                if not is_inner:
                    new_border_list.append(p)
                else:
                    state_map[p[0], p[1]] = 2
    return border_list, state_map


if __name__ == "__main__":
    img = Image.open("C:\\Desktop\\l2.jpg", mode='r')
    img_array = np.asarray(img)[:, :, 0]

    border_list, state_map = region_growth(img_array)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(state_map)
    plt.show()
