from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def whitening(image):
    image = image.astype(np.float32)
    return (image - np.mean(image)) / (np.std(image) + np.finfo(float).eps)


def mat2gray(im):
    im = im.astype(np.float32)
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def normalise_zero_one(image, vmin=None, vmax=None):
    if vmax is None:
        vmax = np.max(image)
    if vmin is None:
        vmin = np.min(image)

    image = image.astype(np.float32)
    return np.clip((image - vmin) / (vmax - vmin + np.finfo(float).eps), 0, 1)


def normalise_one_one(image, vmin=None, vmax=None):
    if vmax is None:
        vmax = np.max(image)
    if vmin is None:
        vmin = np.min(image)

    return 2 * normalise_zero_one(image, vmin, vmax) - 1


def normalise_range(image, nrange=(-1, 1), vmin=None, vmax=None):
    u = nrange[0]
    v = nrange[1]
    return (v - u) * normalise_zero_one(image, vmin, vmax) - (v - u) / 2


def clip_outliers(image, percentile_lower=0.5, percentile_upper=99.5):
    cut_off_lower = np.percentile(image, percentile_lower)
    cut_off_upper = np.percentile(image, percentile_upper)

    return np.clip(image, cut_off_lower, cut_off_upper)


def pad_image_to_size(image, img_size=(64, 64, 64), loc=(2, 2, 2), **kwargs):
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    to_padding = [[0, 0] for _ in range(rank)]

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i] < img_size[i]:
            if loc[i] == 0:
                to_padding[i][0] = (img_size[i] - image.shape[i])
                to_padding[i][1] = 0
            elif loc[i] == 1:
                to_padding[i][0] = 0
                to_padding[i][1] = (img_size[i] - image.shape[i])
            else:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2 + (img_size[i] - image.shape[i]) % 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            to_padding[i][0] = 0
            to_padding[i][1] = 0

    return np.pad(image, to_padding, **kwargs)


def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), loc=(2, 2, 2), **kwargs):
    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i] < img_size[i]:
            if loc[i] == 0:
                to_padding[i][0] = (img_size[i] - image.shape[i])
                to_padding[i][1] = 0
            elif loc[i] == 1:
                to_padding[i][0] = 0
                to_padding[i][1] = (img_size[i] - image.shape[i])
            else:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


def sitk_resample(sitk_image, min_spacing, interpolation=sitk.sitkLinear):
    resampleSliceFilter = sitk.ResampleImageFilter()

    # Resample slice to isotropic
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    new_spacing = [min_spacing, min_spacing, min_spacing]
    new_size = [int(round(original_size[0] * (original_spacing[0] / min_spacing))),
                int(round(original_size[1] * (original_spacing[1] / min_spacing))),
                int(round(original_size[2] * (original_spacing[2] / min_spacing)))]

    resampleSliceFilter.SetOutputSpacing(new_spacing)
    resampleSliceFilter.SetSize(new_size)
    resampleSliceFilter.SetInterpolator(interpolation)
    resampleSliceFilter.SetOutputDirection(sitk_image.GetDirection())
    resampleSliceFilter.SetOutputOrigin(sitk_image.GetOrigin())
    resampleSliceFilter.SetTransform(sitk.Transform())
    resampleSliceFilter.SetDefaultPixelValue(sitk_image.GetPixelIDValue())

    return resampleSliceFilter.Execute(sitk_image)


def extract_alldims_mid_slices_at_label(image, labelmap, label=None, offset=0, is_tight=False):
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)[0]
    else:
        slices = []
        for i in range(3):
            slices.append(slice(0, image.shape[i]))
    images = []
    labelmaps = []
    mids = []
    for i in range(3):

        mid = math.floor((slices[i].start + slices[i].stop) / 2) + offset
        mids.append(mid)
        slicesc = list(slices)
        slicesc[i] = mid
        if not is_tight:
            slicesc = [slicesc[j] if j == i else slice(None) for j in range(3)]

        images.append(image[slicesc[0], slicesc[1], slicesc[2]])
        if label is not None:
            labelmaps.append(labelmap[slicesc[0], slicesc[1], slicesc[2]])

    return (images, labelmaps)


def extract_vol_at_label(image, labelmap, label=None, vol_size=[32, 32, 32], offset=[0, 0, 0], is_rand=False):
    ndims = len(image.shape)
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)
        if len(slices) > 0:
            slices = slices[0]
        else:
            slices = [slice(0, image.shape[i]) for i in range(ndims)]
            # print('No valid label!')
    else:
        slices = []
        for i in range(ndims):
            slices.append(slice(0, image.shape[i]))

    slicesc = []
    padding = []
    do_pad = False
    for i in range(ndims):
        r = int(vol_size[i] / 2)
        if label is not None or is_rand == False:
            mid = math.floor((slices[i].start + slices[i].stop) / 2) + offset[i]
        else:
            mid = np.random.randint(r, image.shape[i] - r - 1)

        s = mid - r
        e = mid + r

        # computing padding in case of out of bounds
        pad_s = 0
        pad_e = 0
        if s < 0:
            pad_s = abs(s)
            s = 0
            e += pad_s
        if e > image.shape[i]:
            pad_e = e - image.shape[i]

        if e - s < vol_size[i]:
            pad_e += vol_size[i] - (e - s)
            e += vol_size[i] - (e - s)
        elif e - s > vol_size[i]:
            e += e - (e - s - vol_size[i])

        slicesc.append(slice(s, e))

        assert (e - s == vol_size[i])

        do_pad = do_pad or pad_e + pad_s > 0

        padding.append((pad_s, pad_e))

    # if do_pad:
    image = np.pad(image, padding, mode='constant')
    labelmap = np.pad(labelmap, padding, mode='constant')

    simage = image[tuple(slicesc)]
    slabelmap = labelmap[tuple(slicesc)]

    assert (all(simage.shape[i] == vol_size[i] for i in range(ndims)))

    return simage, slabelmap


def extract_vol_at_label_along_skel(image, labelmap, label=None, vol_size=(32, 32, 32), offset=(0, 0, 0),
                                    is_rand=False):
    ndims = len(image.shape)
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)[0]
    else:
        slices = []
        for i in range(ndims):
            slices.append(slice(0, image.shape[i]))

    slicesc = []
    padding = []
    do_pad = False
    # centre = ndimage.measurements.center_of_mass(labelmap == label)
    skel = skeletonize(labelmap == label)
    pos = np.argwhere(skel == 1)
    v_max = 0
    j_max = np.random.randint(0, pos.shape[0])
    r = int(vol_size[0] / 2)
    if is_rand:
        for j in range(0, pos.shape[0], 2):
            slicer = [slice(pos[j][0] - r, pos[j][0] + r), slice(pos[j][1] - r, pos[j][1] + r)]
            lmap = labelmap[slicer]
            v = np.sum(lmap)
            if v > v_max:
                v_max = v
                j_max = j

    for i in range(ndims):
        mid = math.floor((slices[i].start + slices[i].stop) / 2) + offset[i]

        mid = pos[j_max][i]

        r = int(vol_size[i] / 2)
        s = mid - r
        e = mid + r

        # computing padding in case of out of bounds
        pad_s = 0
        pad_e = 0
        if s < 0:
            pad_s = abs(s)
            s = 0
            e += pad_s
        if e > image.shape[i]:
            pad_e = e - image.shape[i]

        if e - s < vol_size[i]:
            pad_e += vol_size[i] - (e - s)
            e += vol_size[i] - (e - s)
        elif e - s > vol_size[i]:
            e += e - (e - s - vol_size[i])

        slicesc.append(slice(s, e))

        assert (e - s == vol_size[i])

        do_pad = do_pad or pad_e + pad_s > 0

        padding.append((pad_s, pad_e))

    # if do_pad:
    image = np.pad(image, padding, mode='edge')
    labelmap = np.pad(labelmap, padding, mode='edge')

    simage = image[slicesc]
    slabelmap = labelmap[slicesc]

    assert (all(simage.shape[i] == vol_size[i] for i in range(ndims)))
    # print(simage.shape)

    return simage, slabelmap


def extract_mid_slice_at_label(image, labelmap, label=None, offset=0, is_tight=False, dim=0):
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)[0]
    else:
        slices = []
        for i in range(3):
            slices.append(slice(0, image.shape[i]))
    simage = None
    slabelmap = None
    assert (dim <= len(image.shape))

    i = dim

    mid = math.floor((slices[i].start + slices[i].stop) / 2) + offset
    slicesc = list(slices)
    slicesc[i] = mid
    if not is_tight:
        slicesc = [slicesc[j] if j == i else slice(None) for j in range(3)]

    simage = image[slicesc[0], slicesc[1], slicesc[2]]
    if label is not None:
        slabelmap = labelmap[slicesc[0], slicesc[1], slicesc[2]]

    return simage, slabelmap, mid


def extract_all_slices_at_label(image, labelmap, label=None, step=2, dim=0, is_tight=False):
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)[0]
    else:
        slices = []
        for i in range(3):
            slices.append(slice(0, image.shape[i]))

    images = []
    labelmaps = []
    for i in range(slices[dim].start, slices[dim].stop, step):
        slicesc = list(slices)
        slicesc[dim] = i
        if not is_tight:
            slicesc = [slicesc[j] if j == dim else slice(None) for j in range(3)]

        images.append(image[slicesc[0], slicesc[1], slicesc[2]])
        labelmaps.append(labelmap[slicesc[0], slicesc[1], slicesc[2]] == label)

    return images, labelmaps


def extract_max_area_slice_at_label(image, labelmap, label=1, offset=0, is_tight=False, dim=0):
    if label is not None:
        slices = ndimage.find_objects(labelmap == label)[0]
    else:
        slices = []
        for i in range(3):
            slices.append(slice(0, image.shape[i]))

    assert (dim <= len(image.shape))

    i = dim

    max_area = 0
    max_slice = -1
    for si in range(slices[i].start, slices[i].stop):
        slicesc = list(slices)
        slicesc[i] = si

        slabelmap = labelmap[slicesc[0], slicesc[1], slicesc[2]]
        area = np.sum(slabelmap == label)
        if area > max_area:
            max_area = area
            max_slice = si

    slicesc = list(slices)
    slicesc[i] = max_slice
    if not is_tight:
        slicesc = [slicesc[j] if j == i else slice(None) for j in range(3)]
    simage = image[slicesc[0], slicesc[1], slicesc[2]]
    slabelmap = labelmap[slicesc[0], slicesc[1], slicesc[2]]

    return simage, slabelmap, max_slice


def extract_class_balanced_example_array(image, label, example_size=(1, 64, 64), n_examples=1, classes=2,
                                         class_weights=None):
    """
        Extract training examples from an image (and corresponding label) subject to class balancing.
        Returns an image example array and the corresponding label array.
        Adapted from https://github.com/DLTK/DLTK/blob/master/dltk/io/augmentation.py

        Parameters
        ----------
        image: np.ndarray
            image to extract class-balanced patches from
        label: np.ndarray
            labels to use for balancing the classes
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total
        classes : int or list or tuple
            number of classes or list of classes to extract

        Returns
        -------
        ex_imgs, ex_lbls
            class-balanced patches extracted from bigger images with shape [batch, example_size..., image_channels]
    """

    assert image.shape == label.shape, 'Image and label shape must match'
    assert image.ndim == len(example_size), 'Example size doesnt fit image size'
    assert all([i_s >= e_s for i_s, e_s in zip(image.shape, example_size)]), \
        'Image must be bigger than example shape'
    rank = len(example_size)

    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)

    assert n_examples >= n_classes, 'n_examples need to be bigger than n_classes'

    if class_weights is None:
        n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(n_examples / n_classes))
    else:
        assert len(class_weights) == n_classes, 'class_weights must match number of classes'
        class_weights = np.array(class_weights)
        n_ex_per_class = np.round((class_weights / class_weights.sum()) * n_examples).astype(int)

    # compute an example radius as we are extracting centered around locations
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0), np.ceil(np.array(example_size) / 2.0))),
                      dtype=np.int)

    class_ex_imgs = []
    class_ex_lbls = []
    min_ratio = 1.
    for c_idx, c in enumerate(classes):
        # get valid, random center locations belonging to that class
        idx = np.argwhere(label == c)

        ex_imgs = []
        ex_lbls = []

        if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
            class_ex_imgs.append([])
            class_ex_lbls.append([])
            continue

        # extract random locations
        r_idx_idx = np.random.choice(len(idx), size=min(n_ex_per_class[c_idx], len(idx)), replace=False).astype(int)
        r_idx = idx[r_idx_idx]

        # add a random shift them to avoid learning a centre bias - IS THIS REALLY TRUE?
        r_shift = np.array([list(a) for a in zip(
            *[np.random.randint(-ex_rad[i][0] // 2, ex_rad[i][1] // 2, size=len(r_idx_idx)) for i in range(rank)]
        )]).astype(int)

        r_idx += r_shift

        # shift them to valid locations if necessary
        r_idx = np.array([np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                                        ex_rad[dim][0]) for dim in range(rank)]) for r in r_idx])

        for i in range(len(r_idx)):
            # extract class-balanced examples from the original image
            slicer = tuple(
                [slice(r_idx[i][dim] - ex_rad[dim][0], r_idx[i][dim] + ex_rad[dim][1]) for dim in range(rank)])
            ex_img = image[slicer][np.newaxis, :]

            ex_lbl = label[slicer][np.newaxis, :]

            # concatenate and return the examples
            ex_imgs = np.concatenate((ex_imgs, ex_img), axis=0) if (len(ex_imgs) != 0) else ex_img
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) if (len(ex_lbls) != 0) else ex_lbl

        class_ex_imgs.append(ex_imgs)
        class_ex_lbls.append(ex_lbls)

        ratio = n_ex_per_class[c_idx] / len(ex_imgs)
        min_ratio = ratio if ratio < min_ratio else min_ratio

    indices = np.floor(n_ex_per_class * min_ratio).astype(int)

    ex_imgs = np.concatenate([cimg[:idxs] for cimg, idxs in zip(class_ex_imgs, indices) if len(cimg) > 0], axis=0)
    ex_lbls = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices) if len(clbl) > 0], axis=0)

    # print('returning {} samples with classes:'.format(len(ex_imgs)))
    # print(' - '.join(['{}: {} samples'.format(i, len(cimg[:idxs])) for i, (cimg, idxs) in
    #                  enumerate(zip(class_ex_imgs, indices))]))

    # print('returning {} {}'.format(ex_imgs.shape, ex_lbls.shape))

    return ex_imgs, ex_lbls


def extract_random_example_array(image_list, example_size=(1, 64, 64), n_examples=1):
    """
        Randomly extract training examples from image.
        Returns an image example array and the corresponding label array.
        Adapted from https://github.com/DLTK/DLTK/blob/master/dltk/io/augmentation.py

        Parameters
        ----------
        image_list: np.ndarray or list or tuple
            image(s) to extract random patches from
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total

        Returns
        -------
        examples
            random patches extracted from bigger images with same type as image_list with of shape
            [batch, example_size..., image_channels]
    """
    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size) or image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (i.ndim - 1 == image_list[0].ndim or i.ndim == image_list[0].ndim or i.ndim + 1 == image_list[
                0].ndim), \
                'Example size doesn''t fit image size'
            # assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
            #     'Image shapes must match'

    rank = len(example_size)

    # extract random examples from image and label
    valid_loc_range = [image_list[0].shape[i] - example_size[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
               if valid_loc_range[dim] > 0 else np.zeros(n_examples, dtype=int) for dim in range(rank)]

    examples = [[]] * len(image_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim]) for dim in range(rank)]

        for j in range(len(image_list)):
            ex_img = image_list[j][slicer][np.newaxis]
            # concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_img), axis=0) if (len(examples[j]) != 0) else ex_img

    if was_singular:
        return examples[0]
    return examples


def extract_random_nibabel(images, sample_shape=(1, 64, 64), n_samples=1):
    if n_samples < 0:
        raise Exception('n_samples should be greater than 0')
    #     if not all([i_s >= e_s for i_s, e_s in zip(images[0].shape, sample_shape)]):
    #         raise Exception('Image must be bigger than sample_shape')

    was_singular = False
    if not isinstance(images, list):
        images = [images]
        was_singular = True

    #     if  not (images[0].ndim - 1 == len(sample_shape) or images[0].ndim == len(sample_shape)):
    #         raise Exception('Example size does not match sampled dimensions')

    # if not any([(img.ndim - 1 == images[0].ndim or img.ndim == images[0].ndim or img.ndim + 1 == images[
    #             0].ndim) for img in images]):
    #     raise Exception('image dimension mismatch')

    rank = len(sample_shape)

    valid_loc_range = [images[0].shape[i] - sample_shape[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_samples)
               if valid_loc_range[dim] > 0 else np.zeros(n_samples, dtype=int) for dim in range(rank)]
    print('here')
    examples = [[]] * len(images)
    for i in range(n_samples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + sample_shape[dim]) for dim in range(rank)]

        for j in range(len(images)):
            ex_img = images[j][slicer[0], slicer[1], slicer[2]]
            # concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_img), axis=0) if (len(examples[j]) != 0) else ex_img

    if was_singular:
        return examples[0]
    return examples
