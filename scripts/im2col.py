import numpy as np
import math


def gen_sliding_windows(W: int, H: int, BW: int, BH: int, step: int) -> np.ndarray:
    """
    W: Matrix width
    H: Matrix heigth
    BW: Block (windows) Width
    BH:Block (windows) Heigth
    step: Step btweeen windows (stride)
    """
    x = [x for x in range(0, 1 + W - BW, step)]
    y = [y for y in range(0, 1 + H - BH, step)]

    x_hat = [i + BW for i in x]
    y_hat = [i + BH for i in y]

    cx, cy = np.meshgrid(x, y)
    cx_hat, cy_hat = np.meshgrid(x_hat, y_hat)

    cord = np.stack((cx, cy), axis=2)
    cord_hat = np.stack((cx_hat, cy_hat), axis=2)

    windows = np.stack((cord, cord_hat), axis=2)
    rows, colums, t, t = windows.shape
    windows_coords = windows.reshape((rows * colums, t, t))

    return windows_coords


def gen_windows(X: int, Y: int, win_shape=(3, 3)):

    n_x_slidigns = math.ceil(X / win_shape[0])
    n_y_slidigns = math.ceil(Y / win_shape[1])
    coords = []
    x_coords = list(range(0, (win_shape[0] * n_x_slidigns) + 1, win_shape[0]))
    y_coords = list(range(0, (win_shape[1] * n_y_slidigns) + 1, win_shape[1]))

    for j in range(len(y_coords) - 1):
        for i in range(len(x_coords) - 1):
            init_x = x_coords[i]
            init_y = y_coords[j]
            end_x = x_coords[i + 1]
            end_y = y_coords[j + 1]
            coord = ((init_x, init_y), (end_x, end_y))
            coords.append(coord)

    return coords


def make_blocks(img, window_height, window_width):
    windows = gen_windows(
        img.shape[0], img.shape[1], (window_height, window_width))
    blocks = []
    coords = []
    for win in windows:
        init, end = win
        x0, x1 = init[0], end[0]
        y0, y1 = init[1], end[1]
        block = img[x0:x1, y0:y1]
        coords.append(win)
        blocks.append(block)

    return windows, blocks,  coords


def revert_block(blocks, windows, img_size):
    r = np.zeros(img_size)

    for block, win in zip(blocks, windows):
        init, end = win
        x0, x1 = init[0], end[0]
        y0, y1 = init[1], end[1]
        r[x0:x1, y0:y1] = block
    return r


# img = np.array(
#     [
#         [1, 2, 3, 4, 5, 6],
#         [7, 8, 9, 10, 11, 12],
#         [1, 2, 3, 4, 5, 6],
#         [7, 8, 9, 10, 11, 12],
#         [1, 2, 3, 4, 5, 6],
#         [7, 8, 9, 10, 11, 12],
#     ]
# )
# sz = img.shape
# print(sz)
# # gen_sliding_windows(sz[0], sz[1], 3, 3, 1)


# blocks, coords = make_blocks(img)

# fim = revert_block(blocks, coords, sz)
# print(fim.shape)
# print(img)
# print(fim)
