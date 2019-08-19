import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from midatasets.preprocessing import mat2gray


def blend2d(image, labelmap, alpha, label=1):
    image = np.stack((image,) * 3, axis=-1)
    labelmap = np.stack((labelmap,) * 3, axis=-1)
    labelmap[:, :, 1:2] = 0
    return alpha * labelmap + \
           np.multiply((1 - alpha) * mat2gray(image), mat2gray(labelmap == label)) \
           + np.multiply(mat2gray(image), 1 - mat2gray(labelmap == label))


def create_sliceview(image, step=3, dim=0):
    """
    Displays slices from the 3D image in a grid along a given dimension
    :param image: input image
    :param step: step between images
    :param dim: dimension along which to extract slices
    :return:
    """
    slices = []
    for i in range(3):
        slices.append(slice(0, image.shape[i]))

    assert (dim <= len(image.shape))

    dz = image.shape[dim] // step
    cols = int(round(math.sqrt(dz)))
    rows = int(dz // cols)
    print(rows)
    print(cols)
    out = []
    k = 0
    for i in range(rows):
        row_images = []
        for j in range(cols):
            ind = k * step
            k += 1
            slicesc = list(slices)
            slicesc[dim] = ind
            row_images.append(image[slicesc])
        if out == []:
            out = np.concatenate(row_images, axis=1)

        else:
            out = np.concatenate([out, np.concatenate(row_images, axis=1)], axis=0)
    return out


def display_slices(image, step=3, dim=0, save_path=None):
    """
    Displays slices from the 3D image in a grid along a given dimension
    :param image: input image
    :param step: step between images
    :param dim: dimension along which to extract slices
    :return:
    """
    slices = []
    for i in range(3):
        slices.append(slice(0, image.shape[i]))

    assert (dim <= len(image.shape))

    dz = image.shape[dim] // step
    cols = int(round(math.sqrt(dz)))
    rows = int(dz // cols)
    print(rows)
    print(cols)
    fig, ax = plt.subplots(rows, cols, figsize=[2 * cols, 2 * rows])
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.025, hspace=0.05)
    for i in range(rows * cols):
        slicesc = list(slices)

        ind = i * step
        slicesc[dim] = ind
        x, y = i // cols, i % cols
        # ax[x, y].set_title('slice %d' % ind)
        ax[x, y].imshow(image[slicesc], cmap='gray')
        ax[x, y].axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    if save_path:
        plt.savefig(save_path)
    plt.show()
