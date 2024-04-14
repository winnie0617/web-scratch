import math
import matplotlib.pyplot as plt
import time


def boxes_to_patch_idx_multitarget(box, num_cols):
    """ box is a tensor. Returns a list """
    # pos_idxs = set()
    l, b, w, h = box[0], box[1], box[2], box[3]
    # unscaled 2d idx -> scaled 2d idx
    x1, x2 = l//2, (l+w)//2
    y1, y2 = b//2, (b+h)//2
    # scaled 2d idx -> patch 2d idx
    x1, x2 = math.floor(x1/16), math.ceil(x2/16)
    y1, y2 = math.floor(y1/16), math.ceil(y2/16)
    # 2d -> 1d
    return [num_cols*r + c for c in range(x1, x2) for r in range(y1, y2)]

def boxes_to_patch_idx(box, num_cols, patch_width, patch_height):
    """ returns the patch closest to the center of the element """
    # pos_idxs = set()
    l, b, w, h = box[0], box[1], box[2], box[3]
    # scaled 2d coordinate -> patch 2d coordinate
    x1, x2 = l/patch_width, (l+w)/patch_width
    y1, y2 = b/patch_height, (b+h)/patch_height
    # patch 2d coordinate -> 1d idx
    c = math.floor((x1+x2)/2)
    r = math.floor((y1+y2)/2)
    # if x2 - x1 >= 2: # element at least contains 1 whole patch
    # else: # element within 2 patches
    return [num_cols*r + c]

def patch_idx_to_click(patch_idx, num_cols, patch_width, patch_height):
    """ (x, y), default to clicking the centre of the patch"""
    r, c = patch_idx // num_cols, patch_idx % num_cols
    return patch_width * (c+0.5), patch_height * (r+0.5)

def patch_idx_to_patch_box(patch_idx, num_cols, patch_width, patch_height):
    """ (x, y), default to clicking the centre of the patch"""
    r, c = patch_idx // num_cols, patch_idx % num_cols
    return patch_width * c, patch_width, patch_height * r, patch_height
    

def plot_image(pixel_values, patch_width=None, patch_height=None, predicted=None, target=None, save=False):
    plt.figure(figsize=(12, 40))
    plt.imshow(pixel_values.cpu()[0].permute((1,2,0)))
    if patch_width:
        # plot patches
        num_cols = pixel_values.shape[-1] // patch_width
        num_rows = pixel_values.shape[-2] // patch_width
        plt.vlines([i*patch_width for i in range(1, num_cols)], 0, num_rows*patch_width, colors='b', linestyles='dashed', alpha=0.5)
        plt.hlines([i*patch_height for i in range(1, num_rows)], 0, num_cols*patch_height, colors='b', linestyles='dashed', alpha=0.5)
        if predicted:
            x, y = patch_idx_to_click(predicted, num_cols, patch_width, patch_height)
            plt.plot(x, y, 'ro')
        if target:
            x, y = patch_idx_to_click(target, num_cols, patch_width, patch_height)
            # draw a green circle
            plt.plot(x, y, 'go')
    # save image with timestamp
    if save:
        plt.savefig(f'result_screenshot/{time.time()}.png')
    else:
        plt.show()