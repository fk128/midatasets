import os
from collections import defaultdict
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom as dicom
from skimage.draw import polygon
import logging

from midatasets import configs

logger = logging.getLogger(__name__)

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def read_rtstruct(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].RefdROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_labelmap_from_rtstruct(contours, slices, image):
    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    o_r = int(slices[0].ImageOrientationPatient[4])
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    o_c = int(slices[0].ImageOrientationPatient[0])
    spacing_c = slices[0].PixelSpacing[0]

    labelmap = np.zeros_like(image, dtype=np.uint8)

    for con in contours:
        num = int(con['number'])
        for i, c in enumerate(con['contours']):

            nodes = np.array(c).reshape((-1, 3))
            #             print(np.abs(np.diff(nodes[:, 2])))
            #             assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_new = [round(elem, 1) for elem in z]
            try:
                z_index = z.index(nodes[0, 2])
            except ValueError:
                z_index = z_new.index(nodes[0, 2])

            r = o_r * (nodes[:, 1] - pos_r) / spacing_r
            c = o_c * (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            labelmap[rr, cc, z_index] = num

    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return labelmap, colors


def read_dcm_image(dcms):
    slices = [dicom.read_file(dcm) for dcm in dcms]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    return image, slices


def safe_sitk_dicom_read(img_list, *args, **kwargs):
    dir_name = os.path.dirname(img_list[0])
    s_img_list = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dir_name)
    return sitk.ReadImage(s_img_list, *args, **kwargs)


def read_tag_file(filename, img_size=(512, 512)):
    """
    read sliceomatic tag file
    :param filename:
    :param img_size:
    :return:
    """

    v = np.fromfile(filename, dtype=np.uint8)
    imh = img_size[0]
    imw = img_size[1]
    img = v[-imh * imw:].reshape(imh, imw)
    return img


def export_train_test_split(reader, out_dir='.', type='csv', ratio=0.66, seed=42, cv=False, n_splits=3):
    name = reader.name

    if type == 'csv':
        train_x, train_y, test_x, test_y, names_x, names_y = \
            reader.get_train_test_split_labelled_images_list(ratio, is_paths=True, seed=seed)

        df = pd.DataFrame()
        df['image'] = train_x
        df['labelmap'] = train_y
        df['name'] = names_x
        df.set_index('name', inplace=True)
        df.to_csv(os.path.join(out_dir, name + '_train_image_labelmap_list.csv'))

        df = pd.DataFrame()
        df['image'] = test_x
        df['labelmap'] = test_y
        df['name'] = names_y
        df.set_index('name', inplace=True)
        df.to_csv(os.path.join(out_dir, name + '_test_image_labelmap_list.csv'))

    elif type == 'txt':
        if cv:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits)
            for i, (train_idx, test_idx) in enumerate(kf.split(reader.image_list)):
                train_x = np.array(reader.image_list)[train_idx]
                train_y = np.array(reader.labelmap_list)[train_idx]
                test_x = np.array(reader.image_list)[test_idx]
                test_y = np.array(reader.labelmap_list)[test_idx]
                with open(
                        os.path.join(out_dir, name + '_train_imagelist_f' + str(i + 1) + 'of' + str(n_splits) + '.txt'),
                        'w+') as file:
                    for img_path in train_x:
                        file.write(img_path + '\n')

                with open(
                        os.path.join(out_dir, name + '_train_labellist_f' + str(i + 1) + 'of' + str(n_splits) + '.txt'),
                        'w+') as file:
                    for img_path in train_y:
                        file.write(img_path + '\n')

                with open(
                        os.path.join(out_dir, name + '_test_imagelist_f' + str(i + 1) + 'of' + str(n_splits) + '.txt'),
                        'w+') as file:
                    for img_path in test_x:
                        file.write(img_path + '\n')

                with open(
                        os.path.join(out_dir, name + '_test_labellist_f' + str(i + 1) + 'of' + str(n_splits) + '.txt'),
                        'w+') as file:
                    for img_path in test_y:
                        file.write(img_path + '\n')

        else:
            train_x, train_y, test_x, test_y, names_x, names_y = \
                reader.get_train_test_split_labelled_images_list(ratio, is_paths=True, seed=seed)
            with open(os.path.join(out_dir, name + '_train_imagelist.txt'), 'w+') as file:
                for img_path in train_x:
                    file.write(img_path + '\n')

            with open(os.path.join(out_dir, name + '_train_labellist.txt'), 'w+') as file:
                for img_path in train_y:
                    file.write(img_path + '\n')

            with open(os.path.join(out_dir, name + '_test_imagelist.txt'), 'w+') as file:
                for img_path in test_x:
                    file.write(img_path + '\n')

            with open(os.path.join(out_dir, name + '_test_labellist.txt'), 'w+') as file:
                for img_path in test_y:
                    file.write(img_path + '\n')

    elif type == 'csv_all':
        df = pd.DataFrame()
        df['image'] = reader.image_list
        df['labelmap'] = reader.labelmap_list
        df['name'] = reader.get_image_names()
        df.set_index('name', inplace=True)
        df.to_csv(os.path.join(out_dir, name + '_image_labelmap_list.csv'))


def get_spacing_dirname(spacing):
    if spacing is None:
        return None
    if type(spacing) in [int, float]:
        if isinstance(spacing, float) and spacing.is_integer():
            spacing = int(spacing)
        spacing = [spacing]

    if sum(spacing) <= 0:
        spacing_dirname = configs.get('native_images_dir')
    elif len(spacing) == 1:
        spacing_dirname = configs.get('subsampled_images_dir_prefix') + str(spacing[0]) + 'mm'
    else:
        spacing_str = ''
        for s in spacing:
            spacing_str += str(s) + '-'
        spacing_str = spacing_str[:-1]
        spacing_dirname = configs.get('subsampled_images_dir_prefix') + spacing_str + 'mm'

    return spacing_dirname


def grouped_files(files_iter, ext, dataset_path, key='path'):
    """
    group files by spacing/name/image_type
    :param files_iter:
    :param ext:
    :param dataset_path:
    :param key:
    :return:
    """
    files = defaultdict(dict)
    for basefile_path in files_iter:
        file_path = basefile_path[key]
        prefix = str(Path(file_path).relative_to(dataset_path))

        if len(prefix.split('/')) == 3:
            label = None
            image_type, spacing, filename = prefix.split('/')
        elif len(prefix.split('/')) == 4:
            image_type, label, spacing, filename = prefix.split('/')
        else:
            logger.error(f'Failed to match pattern for grouping {prefix}')
            continue

        name = filename.replace(ext, '')
        image_type_dir = configs.get('remap_dirs', {}).get(image_type, image_type)
        image_key = f'{image_type_dir}-{label}' if label else image_type_dir
        if spacing not in files:
            files[spacing] = defaultdict(dict)

        for existing_name in files[spacing].keys():
            if existing_name in name:  # check if subset of existing name
                name = existing_name
                break

        files[spacing][name][image_key] = {k: str(v) for k, v in basefile_path.items()}
    return {k: dict(v) for k, v in files.items()}
