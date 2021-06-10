# -*- coding: utf-8 -*-

__version__ = '0.3.11'

import os

import yaml

configs = dict(root_path='/media/datasets',
               root_s3_prefix='datasets/',
               images_dir='images',
               labelmaps_dir='labelmaps',
               native_images_dir='native',
               subsampled_images_dir_prefix='subsampled',
               images_crop_prefix='images_crop_',
               labelmaps_crop_prefix='labelmaps_crop_',
               aws_s3_bucket=None,
               aws_s3_profile=None,
               remap_dirs={'images': 'image', 'labelmaps': 'labelmap'}
               )


def load_config(path='~/.midatasets.yaml'):
    global configs
    try:
        with open(os.path.expanduser(path)) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            configs.update(data)
            configs['root_path'] = os.path.expandvars(configs['root_path'])
    except:
        pass


load_config()
