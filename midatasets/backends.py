import fnmatch
import logging
import os
from pathlib import Path
from typing import Callable, Union, Optional, Tuple

import boto3

from midatasets import configs
from midatasets.utils import get_spacing_dirname, grouped_files

logger = logging.getLogger(__name__)


class StorageBackend:
    def __init__(self, *args, **kwargs):
        pass

    def list_datasets(self):
        raise NotImplementedError

    def list_files(self, dataset_name: str, spacing: Optional[float] = None, ext: str = '.nii.gz',
                   grouped: bool = False):
        raise NotImplementedError

    def download(self, src_prefix: str, dest_path: str,
                 spacing: Optional[float] = 0,
                 max_images: Optional[int] = None,
                 ext: str = '.nii.gz', dryrun: bool = False,
                 include: Optional[Tuple[str, ...]] = None):
        raise NotImplementedError


class S3Backend(StorageBackend):
    def __init__(self, bucket, prefix='/', profile=None):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix
        self.profile = profile
        self.client = boto3.client('s3')

        if 'AWS_SECRET_ACCESS_KEY' not in os.environ and 'AWS_ACCESS_KEY_ID' not in os.environ:
            boto3.setup_default_session(profile_name=self.profile)

    def list_datasets(self):
        """

        :return: dict {dataset_name: prefix}
        """
        result = self.client.list_objects(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/')
        return {o.get('Prefix').split('/')[-2]: o.get('Prefix') for o in result.get('CommonPrefixes')}

    def list_files(self, dataset_name, spacing=None, ext='.nii.gz', grouped=False):
        if dataset_name not in self.list_datasets().keys():
            raise FileNotFoundError(f'`{dataset_name}` does not exist')
        src_prefix = str(Path(self.prefix) / dataset_name) + '/'
        result = self.client.list_objects(Bucket=self.bucket, Prefix=src_prefix, Delimiter='/')
        try:
            image_type_prefixes = [o.get('Prefix') for o in result.get('CommonPrefixes')]
        except Exception as e:
            logger.exception(e)
            raise

        image_types = [o.replace(src_prefix, '').replace('/', '') for o in image_type_prefixes]
        try:
            image_types.remove(configs['images_dir'])
            image_types = [configs['images_dir']] + image_types  # make sure images key is first
        except:
            logger.exception(f"Missing {configs['images_dir']} dir")

        pattern = f'*/{get_spacing_dirname(spacing)}/*' if spacing is not None else '*'
        files = []
        for image_type in image_types:
            prefix = Path(src_prefix)
            prefix = prefix / image_type
            prefix = str(prefix)
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                result = page.get('Contents', None)
                if not result:
                    raise FileNotFoundError(f's3://{self.bucket}/{prefix} not found')
                result = [r['Key'] for r in result]
                # filter only matching spacing
                if spacing is not None:
                    result = fnmatch.filter(result, pattern)
                files += [{'path': r} for r in result]

        if len(files) == 0:
            raise Exception(f'No files found at s3://{self.bucket}/{src_prefix}')

        if grouped:
            return grouped_files(files, ext, dataset_path=src_prefix)
        else:
            return files

    def download(self, dataset_name, dest_root_path,
                 spacing=0,
                 max_images=None,
                 ext='.nii.gz',
                 dryrun=False,
                 include=None):
        src_prefix = str(Path(self.prefix) / dataset_name) + '/'
        dest_path = Path(dest_root_path) / dataset_name
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)
        files = self.list_files(dataset_name=dataset_name, spacing=spacing, ext=ext, grouped=True)
        files = files[next(iter(files))]
        count = 0
        for name, file_prefixes in files.items():
            if max_images and count >= max_images:
                break
            count += 1
            for k, file_prefix in file_prefixes.items():
                if include and k not in include:
                    continue
                file_prefix = file_prefix['path']
                target = os.path.join(dest_path, os.path.relpath(file_prefix, src_prefix))
                if os.path.exists(target):
                    logger.info(f'[already exists] {target}')
                    continue
                if not os.path.exists(os.path.dirname(target)):
                    Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)

                logger.info(f'[Downloading] {file_prefix} -> {target}')
                if not dryrun:
                    bucket.download_file(file_prefix, target)


class LocalStorageBackend(StorageBackend):

    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.image_type_dirs = set()

    def list_datasets(self):
        return {p.stem: p for p in Path(self.root_path).iterdir()}

    def list_files(self, dataset_name, spacing=None, ext='.nii.gz', grouped=False):
        dataset_path = Path(self.root_path) / dataset_name
        image_type_paths = {image_type_path.name: image_type_path for image_type_path in dataset_path.iterdir()}
        image_types = list(image_type_paths.keys())
        try:
            image_types.remove(configs['images_dir'])
            image_types = [configs['images_dir']] + image_types  # make sure images key is first
        except:
            logger.exception(f"Missing {configs['images_dir']} dir")

        files = []
        pattern = f'*/{get_spacing_dirname(spacing)}/*' if spacing is not None else '*'
        for image_type in image_types:
            prefix = dataset_path
            prefix = prefix / image_type
            files_iter = (dataset_path / prefix).rglob(f'*' + ext)
            files_iter = [str(f) for f in files_iter]
            # filter only matching spacing
            if spacing is not None:
                files_iter = fnmatch.filter(files_iter, pattern)

            files += files_iter

        files = [{'path': f} for f in files]
        if grouped:
            return grouped_files(files, ext, dataset_path=dataset_path)
        else:
            return files


BACKENDS = {'s3': S3Backend, 'local': LocalStorageBackend}


def get_backend(name: Union[str, Callable]) -> Callable:
    if callable(name):
        return name
    else:
        return BACKENDS[name]
