from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pathlib
from random import sample

import SimpleITK as sitk
import boto3
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import midatasets.preprocessing
import midatasets.visualise as vis
from midatasets import configs
from midatasets.preprocessing import sitk_resample, extract_vol_at_label
from midatasets.utils import printProgressBar


class MIReader(object):
    """Medical Image Reader

    A class to interface with locally stored medical image dataset in nifti format

    """

    def __init__(self,
                 spacing,
                 name='reader',
                 is_cropped=False,
                 crop_size=64,
                 dir_path=None,
                 subpath=None,
                 ext='.nii.gz',
                 label=None,
                 image_label=None,
                 images_only=False,
                 labels=None,
                 label_names=None,
                 aws_s3_bucket=None,
                 aws_profile=None,
                 aws_s3_prefix=None,
                 fail_on_error=False
                 ):

        self.name = name
        self.dir_path = dir_path
        self.subpath = subpath
        self.image_list = []
        self.labelmap_list = []
        self.boundingboxmap_list = []
        self.labels = labels
        self.label_names = label_names
        self.do_preprocessing = False
        self.spacing = spacing
        self.is_cropped = is_cropped
        self.crop_size = crop_size
        self.ext = ext
        self.label = label
        self.image_label = image_label
        self.images_only = images_only
        self.dataframe = pd.DataFrame()
        self.dataframe.index.name = 'name'
        self.aws_s3_bucket = aws_s3_bucket
        self.aws_profile = aws_profile
        self.aws_s3_prefix = aws_s3_prefix
        try:
            self.setup()
        except FileNotFoundError:
            if fail_on_error:
                raise FileNotFoundError('No files found. try calling .download()')
            else:
                print('No files found. try calling .download()')

    @classmethod
    def from_dict(cls, **data):
        return cls(**data)

    def __getitem__(self, index):
        return dict(self.dataframe.reset_index().iloc[index])

    def __len__(self):
        """
        Return number of samples
        Returns
        -------
        int
            number of samples
        """
        return len(self.dataframe)

    def get_root_path(self):
        return configs.get('root_path')

    def download(self, max_images=None):
        boto3.setup_default_session(profile_name=self.aws_profile)
        _, base_dir, images_sub_dir = self.get_imagetype_path('images', split=True)
        _, _, labelmaps_sub_dir = self.get_imagetype_path('labelmaps', split=True)
        local_dir = self.dir_path

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.aws_s3_bucket)

        for prefix in [os.path.join(self.aws_s3_prefix, images_sub_dir),
                       os.path.join(self.aws_s3_prefix, labelmaps_sub_dir)]:
            count = 0
            for obj in bucket.objects.filter(Prefix=prefix):
                if max_images and count > max_images:
                    break
                count += 1
                target = os.path.join(local_dir, os.path.relpath(obj.key, self.aws_s3_prefix))
                if os.path.exists(target):
                    print(f'[already exists] {target}')
                    continue
                if not os.path.exists(os.path.dirname(target)):
                    pathlib.Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)
                if obj.key[-1] == '/':
                    continue
                print(f'[Downloading] {target}')
                bucket.download_file(obj.key, target)

        self.setup()

    def setup(self):
        if self.dir_path is not None:
            if self.image_label is not None:
                image_dir_name = os.path.join(configs.get('images_dir'), self.image_label)
            else:
                image_dir_name = configs.get('images_dir')

            if self.label is not None:
                labelmap_dir_name = os.path.join(configs.get('labelmaps_dir'), self.label)
            else:
                labelmap_dir_name = configs.get('labelmaps_dir')

            root = self.get_imagetype_path(image_dir_name, is_cropped=self.is_cropped)
            label_root = self.get_imagetype_path(labelmap_dir_name, is_cropped=self.is_cropped)

            for image_filename in os.listdir(root):
                name = image_filename.replace(self.ext, '')
                # name = name.split('_')[0]

                if self.is_cropped:
                    name = name.replace('_' + configs.get('images_crop_prefix') + str(self.crop_size), '')
                self.dataframe.loc[name, 'image_path'] = os.path.join(root, image_filename)
                labelmap_path = glob.glob(os.path.join(label_root, name + '*' + self.ext))
                # try using prefix only to see if there is a match
                # if len(labelmap_path) == 0:
                #     name = name.split('_')[0]
                #     labelmap_path = glob.glob(os.path.join(label_root, name + '*' + self.ext))

                if len(labelmap_path) == 1:
                    self.dataframe.loc[name, 'labelmap_path'] = labelmap_path[0]
            if not self.images_only:
                self.dataframe.dropna(inplace=True)
            else:
                self.dataframe = self.dataframe[['image_path']].dropna()

    @property
    def num_images(self):
        return len(self.image_list)

    @classmethod
    def _load_image_from_disk(cls, img_path):
        # return nib.load(img_path).get_data()
        img = sitk.ReadImage(img_path)
        return cls.get_array_from_sitk_image(img)

    @classmethod
    def get_array_from_sitk_image(cls, img):
        def validate(v):
            if v == 0:
                return 1
        x = validate(int(img.GetDirection()[0]))
        y = validate(int(img.GetDirection()[4]))
        z = validate(int(img.GetDirection()[8]))
        return sitk.GetArrayFromImage(img)[::x, ::y, ::z]

    def _preprocess(self, image):
        raise NotImplementedError()

    def get_image_list(self, is_shuffled=False):
        if is_shuffled:
            return list(self.dataframe['image_path'].sample(frac=1).values)
        else:
            return list(self.dataframe['image_path'].values)

    def get_labelmap_list(self, is_shuffled=False):
        if is_shuffled:
            return list(self.dataframe['labelmap_path'].sample(frac=1).values)
        else:
            return list(self.dataframe['labelmap_path'].values)

    def get_labelled_images_list(self, num=-1, is_shuffled=False):

        lst = []
        for name, row in self.dataframe[['image_path', 'labelmap_path']].iterrows():
            lst.append([row['image_path'], row['labelmap_path']])

        if is_shuffled:
            return sample(lst, num)
        else:
            return lst[0:num]

    def get_imagetype_path(self, images_type,
                           spacing=None,
                           is_cropped=False,
                           crop_suffix='_crop',
                           crop_size=64,
                           split=False):
        if spacing is None:
            spacing = self.spacing

        if type(spacing) in [int, float]:
            if isinstance(spacing, float) and spacing.is_integer():
                spacing = int(spacing)
            spacing = [spacing]

        suffix = ''
        if is_cropped:
            suffix += crop_suffix + '_' + str(crop_size)
        if sum(spacing) <= 0:
            subpath = os.path.join(images_type + suffix, configs.get('native_images_dir'))
        else:
            if len(spacing) == 1:
                subpath = os.path.join(images_type + suffix,
                                       configs.get('subsampled_images_dir_prefix') + str(
                                           spacing[0]) + 'mm')
            else:
                spacing_str = ''
                for s in spacing:
                    spacing_str += str(s) + '-'
                spacing_str = spacing_str[:-1]
                subpath = os.path.join(images_type + suffix,
                                       configs.get('subsampled_images_dir_prefix') + spacing_str + 'mm')
        if split:
            return self.dir_path.split(self.subpath)[0], self.subpath, subpath
        else:
            return os.path.join(self.dir_path, subpath)

    def get_image_name(self, img_idx):
        return self.dataframe.index[img_idx]

    def get_image_names(self):
        return list(self.dataframe.index)

    def load_image(self, img_idx):

        if type(img_idx) is int:
            image_path = self.dataframe.iloc[img_idx]['image_path']
            if self.do_preprocessing:
                return self._preprocess(self._load_image_from_disk(image_path))
            else:
                return self._load_image_from_disk(image_path)
        else:
            return self._load_image_by_name(img_idx)

    def load_image_and_resample(self, img_idx, new_spacing):
        image_path = self.dataframe.iloc[img_idx]['image_path']
        sitk_image = sitk.ReadImage(image_path)
        sitk_image = sitk_resample(sitk_image, new_spacing)

        x = int(sitk_image.GetDirection()[0])
        y = int(sitk_image.GetDirection()[4])
        z = int(sitk_image.GetDirection()[8])
        return sitk.GetArrayFromImage(sitk_image)[::x, ::y, ::z]

    def _load_image_by_name(self, name):
        try:
            path = self.dataframe.loc[name, 'image_path']
        except:
            raise Exception(name + ' does not exist in dataset')

        if self.do_preprocessing:
            return self._preprocess(self._load_image_from_disk(path))
        else:
            return self._load_image_from_disk(path)

    def load_labelmap(self, img_idx):
        if type(img_idx) is int:
            labelmap_path = self.dataframe.iloc[img_idx]['labelmap_path']
            return self._load_image_from_disk(labelmap_path)
        else:
            return self._load_labelmap_by_name(img_idx)

    def load_labelmap_and_resample(self, img_idx, new_spacing):
        labelmap_path = self.dataframe.iloc[img_idx]['labelmap_path']
        sitk_image = sitk.ReadImage(labelmap_path)
        sitk_image = sitk_resample(sitk_image, new_spacing, sitk.sitkNearestNeighbor)

        x = int(sitk_image.GetDirection()[0])
        y = int(sitk_image.GetDirection()[4])
        z = int(sitk_image.GetDirection()[8])
        return sitk.GetArrayFromImage(sitk_image)[::x, ::y, ::z]

    def _load_labelmap_by_name(self, name):
        try:
            path = self.dataframe.loc[name, 'labelmap_path']
        except:
            raise Exception(name + ' does not exist in dataset')

        if self.do_preprocessing:
            return self._preprocess(self._load_image_from_disk(path))
        else:
            return self._load_image_from_disk(path)

    def load_boundingboxmap(self, img_idx):
        path = self.dataframe.iloc[img_idx]['boundingbox_path']
        return self._load_image_from_disk(path)

    def load_sitk_image(self, img_idx):
        image_path = self.dataframe.iloc[img_idx]['image_path']
        return sitk.ReadImage(image_path)

    def load_sitk_labelmap(self, img_idx):
        labelmap_path = self.dataframe.iloc[img_idx]['labelmap_path']
        return sitk.ReadImage(labelmap_path)

    def has_labelmap(self):
        return 'labelmap_path' in self.dataframe.columns

    def load_metadata(self, img_idx):
        image_path = self.dataframe.iloc[img_idx]['image_path']
        reader = sitk.ImageFileReader()

        reader.SetFileName(image_path)
        reader.LoadPrivateTagsOn()

        reader.ReadImageInformation()
        data = {}
        for k in reader.GetMetaDataKeys():
            v = reader.GetMetaData(k)
            data[k] = v
        data['spacing'] = reader.GetSpacing()
        return data

    def extract_random_subvolume(self, img_idx, subvol_size, num):

        return midatasets.preprocessing.extract_random_example_array([self.load_image(img_idx),
                                                                      self.load_labelmap(img_idx)],
                                                                     example_size=subvol_size,
                                                                     n_examples=num)

    def extract_random_class_balanced_subvolume(self, img_idx, subvol_size=(64, 64, 64), num=2, class_weights=[1, 1]):

        return midatasets.preprocessing.extract_class_balanced_example_array(self.load_image(img_idx),
                                                                             self.load_labelmap(img_idx),
                                                                             example_size=subvol_size,
                                                                             n_examples=num,
                                                                             classes=len(self.labels),
                                                                             class_weights=class_weights)

    def extract_all_slices(self, img_idx, label=None, step=2, dim=0, is_tight=False):
        I = self.load_image(img_idx)
        L = self.load_labelmap(img_idx)
        return midatasets.preprocessing.extract_all_slices_at_label(I, L, label, step, dim, is_tight)

    def extract_mid_slices(self, img_idx, label=None, offset=0, is_tight=False):
        I = self.load_image(img_idx)
        if label is None:
            L = []
        else:
            L = self.load_labelmap(img_idx)
        return midatasets.preprocessing.extract_alldims_mid_slices_at_label(I, L, label, offset, is_tight)

    def export_2d_slices(self, out_path=None, label=1, step=5):
        allimages = []
        alllabelmaps = []
        if out_path is None:
            out_path = self.dir_path

        for img_idx in range(self.num_images):
            printProgressBar(img_idx, self.num_images - 1, prefix='Progress:', suffix='Complete', length=50)
            name = self.get_image_name(img_idx)
            (images, labelmaps) = self.extract_all_slices(img_idx, label=label, step=step)
            allimages += images
            alllabelmaps += labelmaps

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        s = self.spacing
        try:
            s = s[0]
        except:
            pass
        np.savez_compressed(os.path.join(out_path, self.name + '_label' + str(label) +
                                         '_spacing' + str(s) + '_2dslices.npz'),
                            images=allimages, labelmaps=alllabelmaps)

    def load2d_slices(self, label):
        s = self.spacing
        try:
            s = s[0]
        except:
            pass
        path = os.path.join(self.dir_path, self.name + '_label' + str(label) +
                            '_spacing' + str(s) + '_2dslices.npz')
        if os.path.exists(path):
            slices = np.load(path)
        else:
            print('{} does not exist. Extracting...'.format(path))
            self.export_2d_slices(self.dir_path, label)
            slices = np.load(path)
        return slices[configs.get('images_dir')], slices[configs.get('labelmaps_dir')]

    def prune_image_list(self, keep_image_names):
        image_list = list(self.image_list)
        labelmap_list = list(self.labelmap_list)
        for image, labelmap in zip(self.image_list, self.labelmap_list):
            img_name = os.path.basename(image.replace('.nii.gz', ''))
            if img_name not in keep_image_names:
                image_list.remove(image)
                labelmap_list.remove(labelmap)

        self.image_list = image_list
        self.labelmap_list = labelmap_list

    def resample_image_and_save(self, img_idx, spacing, overwrite=False):
        name = self.get_image_name(img_idx)
        output_path = self.get_imagetype_path('images', spacing=spacing)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, name + '.nii.gz')
        if os.path.exists(output_path) and not overwrite:
            print('already exists')
            return
        sitk_image = self.load_sitk_image(img_idx)
        print('resampling from', sitk_image.GetSpacing(), 'to', spacing)
        sitk_image = sitk_resample(sitk_image, spacing)
        sitk.WriteImage(sitk_image, output_path)

    def resample_labelmap_and_save(self, img_idx, spacing, overwrite=False):
        name = self.get_image_name(img_idx)
        output_path = self.get_imagetype_path('labelmaps', spacing=spacing)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, name + '.nii.gz')
        if os.path.exists(output_path) and not overwrite:
            print('already exists')
            return
        sitk_image = self.load_sitk_labelmap(img_idx)
        print('resampling from', sitk_image.GetSpacing(), 'to', spacing)
        sitk_image = sitk_resample(sitk_image, spacing, sitk.sitkNearestNeighbor)
        sitk.WriteImage(sitk_image, output_path)

    def generate_resampled(self, spacing, parallel=True, num_workers=-1):

        def resample(img_idx, spacing):
            self.resample_image_and_save(img_idx=img_idx, spacing=spacing)
            if self.has_labelmap():
                self.resample_labelmap_and_save(img_idx=img_idx, spacing=spacing)

        if parallel:
            Parallel(n_jobs=num_workers)(delayed(resample)(i, spacing) for i in range(len(self)))
        else:
             for i in range(len(self)):
                 print(i)
                 resample(i, spacing)



    def extract_crop(self, i, label=None, vol_size=(64, 64, 64)):

        def get_output(oname):
            name_suffix = oname + str(vol_size[0])
            output_image = self.get_imagetype_path(name_suffix)
            if not os.path.exists(output_image):
                os.makedirs(output_image)
            return output_image, name_suffix

        name = self.get_image_name(i)
        print(name)
        output_image, image_name_suffix = get_output(configs.get('images_crop_prefix'))
        output_labelmap, labelmap_name_suffix = get_output(configs.get('labelmaps_crop_prefix'))

        lmap = self.load_labelmap(i)
        if label is not None:
            labels = [label]
        else:
            labels = list(np.unique(lmap).flatten())
            labels.remove(0)

        sitk_image = self.load_sitk_image(i)
        spacing = sitk_image.GetSpacing()
        img = self.get_array_from_sitk_image(sitk_image)
        for l in labels:
            image, labelmap = extract_vol_at_label(img, self.load_labelmap(i), label=l,
                                                   vol_size=vol_size)
            labelmap = (labelmap == l).astype(np.uint8)

            image = sitk.GetImageFromArray(image)
            labelmap = sitk.GetImageFromArray(labelmap)
            image.SetSpacing(spacing)
            labelmap.SetSpacing(spacing)

            suffix = '_' + str(l)

            sitk.WriteImage(image, os.path.join(output_image, name + suffix + '_' + image_name_suffix
                                                + '.nii.gz'))
            sitk.WriteImage(labelmap, os.path.join(output_labelmap, name + suffix + '_' + labelmap_name_suffix
                                                   + '.nii.gz'))

    def extract_crops(self, vol_size=(64, 64, 64), label=None, parallel=False):

        if not parallel:
            for i in range(self.num_images):
                printProgressBar(i + 1, self.num_images)
                self.extract_crop(i, label, vol_size)

        else:
            from joblib import Parallel, delayed
            Parallel(n_jobs=6)(delayed(self.extract_crop)(i, label, vol_size) for i in range(self.num_images))

    def load_image_crop(self, img_idx, vol_size=(64, 64, 64), label=1):
        name = self.get_image_name(img_idx)
        name_suffix = configs.get('images_crop_prefix') + str(vol_size[0])
        output = self.get_imagetype_path(name_suffix)
        path = os.path.join(output, name + '_' + str(label) + '_' + name_suffix
                            + '.nii.gz')
        return self._load_image_from_disk(path)

    def load_labelmap_crop(self, img_idx, vol_size=(64, 64, 64), label=1):
        name = self.get_image_name(img_idx)
        name_suffix = configs.get('labelmaps_crop_prefix') + str(vol_size[0])
        output = self.get_imagetype_path(name_suffix)
        path = os.path.join(output, name + '_' + str(label) + '_' + name_suffix
                            + '.nii.gz')
        return self._load_image_from_disk(path)

    def view_slices(self, img_idx, label=None, step=3, dim=0):
        if label is None:
            image = self.load_image(img_idx)
        else:
            image = self.load_labelmap(img_idx)
        vis.display_slices(image, step=step, dim=dim)
