import fnmatch
import logging
import os
from pathlib import Path
from typing import Callable, Union, Optional, Tuple

import boto3
from midatasets import configs
from midatasets.utils import get_spacing_dirname, grouped_files

logger = logging.getLogger(__name__)


class DatasetStorageBackendBase:

    def __init__(self, *args, **kwargs):
        pass

    def list_dirs(self, sub_path: Optional[str] = None):
        raise NotImplementedError

    def list_files(self, spacing: Optional[Union[float, int]] = None, ext: Tuple[str] = ('.nii.gz',),
                   grouped: bool = False):
        raise NotImplementedError

    def download(self, dest_path: str,
                 src_prefix: Optional[str] = None,
                 spacing: Optional[Union[float, int]] = 0,
                 max_images: Optional[int] = None,
                 ext: str = '.nii.gz', dryrun: bool = False,
                 include: Optional[Tuple[str, ...]] = None):
        raise NotImplementedError


class DatasetS3Backend(DatasetStorageBackendBase):
    def __init__(self, bucket: str, prefix: str, profile=None):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix
        self.profile = profile
        self.client = boto3.client('s3')

        if 'AWS_SECRET_ACCESS_KEY' not in os.environ and 'AWS_ACCESS_KEY_ID' not in os.environ:
            boto3.setup_default_session(profile_name=self.profile)

    def list_dirs(self, sub_path: Optional[str] = None):
        prefix = self.prefix if sub_path is None else os.path.join(self.prefix, sub_path)
        prefix = prefix if prefix.endswith('/') else prefix + '/'
        result = self.client.list_objects(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
        return {o.get('Prefix').split('/')[-2]: o.get('Prefix') for o in result.get('CommonPrefixes')}

    def list_files(self, spacing: Optional[Union[float, int]] = None, ext: Union[Tuple[str], str] = ('.nii.gz',),
                   grouped: bool = False):
        if not isinstance(ext, tuple):
            ext = (ext,)

        src_prefix = str(Path(self.prefix)) + '/'
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
            raise Exception(f'No files found at s3://{self.bucket}/{src_prefix} at spacing {spacing}')

        if grouped:
            return grouped_files(files, ext, dataset_path=src_prefix)
        else:
            return files

    def download(self, dest_path,
                 src_prefix: Optional[str] = None,
                 spacing: Optional[Union[float, int]] = 0,
                 max_images=None,
                 ext: Tuple[str] = ('.nii.gz',),
                 dryrun=False,
                 include=None):
        if src_prefix is None:
            src_prefix = str(Path(self.prefix)) + '/'
        dest_path = Path(dest_path)
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)
        files = self.list_files(spacing=spacing, ext=ext, grouped=True)
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


class DatasetLocalBackend(DatasetStorageBackendBase):

    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.image_type_dirs = set()

    def list_dirs(self, sub_path: Optional[str] = None):
        path = self.root_path if sub_path is None else os.path.join(self.root_path, sub_path)
        return {p.stem: p for p in Path(path).iterdir()}

    def list_files(self, spacing: Optional[Union[float, int]] = None, ext: Tuple[str] = ('.nii.gz',), grouped=False):
        if not isinstance(ext, tuple):
            ext = (ext,)
        dataset_path = Path(self.root_path)
        image_type_paths = self.list_dirs()
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
            files_iter = [str(f) for e in ext for f in (dataset_path / prefix).rglob(f'*' + e)]

            # filter only matching spacing
            if spacing is not None:
                files_iter = fnmatch.filter(files_iter, pattern)

            files += files_iter

        files = [{'path': f} for f in files]
        if grouped:
            return grouped_files(files, ext, dataset_path=dataset_path)
        else:
            return files


BACKENDS = {'s3': DatasetS3Backend, 'local': DatasetLocalBackend}


def get_backend(name: Union[str, Callable[..., DatasetStorageBackendBase]]) -> Callable[..., DatasetStorageBackendBase]:
    if callable(name):
        return name
    else:
        return BACKENDS[name]
