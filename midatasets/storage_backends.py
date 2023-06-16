import fnmatch
import itertools
import os
from pathlib import Path
from typing import Callable, Union, Optional, Tuple, List

import boto3
import botocore
from loguru import logger

from midatasets import configs
from midatasets.utils import get_spacing_dirname, grouped_files


class DatasetStorageBackendBase:
    def __init__(self, *args, **kwargs):
        pass

    def list_dirs(self, sub_path: Optional[str] = None):
        raise NotImplementedError

    def list_files_at_dir(
        self,
        sub_path: Optional[str] = None,
        pattern: Optional[str] = None,
        ext: Union[Tuple[str], str] = (".nii.gz",),
        skip: int = 0,
        limit: int = -1,
    ):
        raise NotImplementedError

    def list_files(
        self,
        spacing: Optional[Union[float, int]] = None,
        ext: Tuple[str] = (".nii.gz",),
        data_types: Optional[List[str]] = None,
        grouped: bool = False,
        skip: int = 0,
        limit: int = -1,
        primary_key: str = "image",
    ):
        raise NotImplementedError

    def get_base_dir(self):
        raise NotImplementedError

    def get_data_types(self, data_types: Optional[List[str]] = None):
        datatype_to_dirname = {v["name"]: v["dirname"] for v in configs.data_types}
        dirname_to_datatype = {v["dirname"]: v["name"] for v in configs.data_types}
        existing_data_type_dirs = list(self.list_dirs().keys())

        existing_data_types = []
        for dirname in existing_data_type_dirs:
            if dirname in dirname_to_datatype:
                existing_data_types.append(dirname_to_datatype[dirname])
            else:
                logger.warning(
                    f"Ignoring `{dirname}` as it is not a valid data type from {datatype_to_dirname.keys()}"
                )

        data_types = data_types or existing_data_types
        data_types = data_types[:]

        if configs.primary_type in data_types:
            data_types.remove(configs.primary_type)
            data_types = [
                configs.primary_type
            ] + data_types  # make sure images key is first
        return [
            {"name": name, "dirname": datatype_to_dirname[name]} for name in data_types
        ]

    def download(
        self,
        dest_path: str,
        src_prefix: Optional[str] = None,
        spacing: Optional[Union[float, int]] = 0,
        max_images: Optional[int] = None,
        ext: str = ".nii.gz",
        dryrun: bool = False,
        include: Optional[Tuple[str, ...]] = None,
        names: Optional[List[str]] = None,
    ):
        raise NotImplementedError

    def upload(
        self,
        path: str,
        subprefix: str,
        spacing: Optional[Union[float, int]] = 0,
        overwrite: bool = False,
    ):
        raise NotImplementedError


class DatasetS3Backend(DatasetStorageBackendBase):
    client = None

    def __init__(self, bucket: str, prefix: str, profile=None, **kwargs):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix
        self.profile = profile
        self.client = kwargs.get("client", self.get_boto_client())
        self.root_path = f"s3://{os.path.join(self.bucket, self.prefix)}"

        if (
            "AWS_SECRET_ACCESS_KEY" not in os.environ
            and "AWS_ACCESS_KEY_ID" not in os.environ
        ):
            boto3.setup_default_session(profile_name=self.profile)

    def get_boto_client(self):
        if self.__class__.client is None:
            self.__class__.client = boto3.session.Session().client(
                "s3",
                endpoint_url=configs.aws_endpoint_url,
                config=botocore.config.Config(
                    retries={"max_attempts": 10, "mode": "standard"}
                ),
            )
        return self.__class__.client

    def get_base_dir(self):
        return str(self.root_path)

    def list_dirs(self, sub_path: Optional[str] = None):
        prefix = (
            self.prefix if sub_path is None else os.path.join(self.prefix, sub_path)
        )
        prefix = prefix if prefix.endswith("/") else prefix + "/"
        result = self.client.list_objects(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/"
        )
        return {
            o.get("Prefix").split("/")[-2]: o.get("Prefix")
            for o in result.get("CommonPrefixes")
        }

    def list_files_at_dir(
        self,
        sub_path: Optional[str] = None,
        pattern: Optional[str] = None,
        ext: Union[Tuple[str], str] = (".nii.gz",),
        recursive: bool = False,
        skip: int = 0,
        limit: int = -1,
    ):
        prefix = self.prefix
        if sub_path:
            prefix = os.path.join(self.prefix, sub_path)
        if not prefix.endswith("/"):
            prefix += "/"
        paginator = self.client.get_paginator("list_objects_v2")
        results = []
        if limit > -1:
            pages = itertools.islice(
                paginator.paginate(
                    Bucket=self.bucket,
                    Prefix=prefix,
                    PaginationConfig={"PageSize": limit},
                ),
                skip,
                skip + 1,
            )
        else:
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        for page in pages:
            result = page.get("Contents", None)
            if result:
                result = {r["Key"]: r for r in result}
                if pattern:
                    result = {
                        k: result[k] for k in fnmatch.filter(result.keys(), pattern)
                    }
                for k, v in result.items():
                    if ext and not v["Key"].endswith(ext):
                        continue

                    results.append(
                        {
                            "path": f"s3://{self.bucket}/{k}",
                            "last_modified": v["LastModified"],
                            "size": v["Size"],
                        }
                    )
        return results

    def list_files(
        self,
        spacing: Optional[Union[float, int]] = None,
        data_types: Optional[List[str]] = None,
        ext: Union[Tuple[str, ...], str] = (".nii.gz",),
        grouped: bool = False,
        skip: int = 0,
        limit: int = -1,
        primary_key: str = "image",
    ):
        if not isinstance(ext, tuple):
            ext = (ext,)

        data_types = self.get_data_types(data_types)

        pattern = f"*/{get_spacing_dirname(spacing)}/*" if spacing is not None else "*"
        files = {}
        for data_type in data_types:

            files[data_type["name"]] = self.list_files_at_dir(
                sub_path=data_type["dirname"],
                pattern=pattern,
                ext=ext,
                skip=skip,
                limit=limit,
            )

        if grouped:
            files = grouped_files(files, root_prefix=self.root_path)

        if limit > -1:
            total = sum(
                1
                for _ in boto3.resource("s3", endpoint_url=configs.aws_endpoint_url)
                .Bucket(self.bucket)
                .objects.filter(Prefix=f"{self.prefix}/{primary_key}")
            )
            return {"total": total, "data": files, "limit": limit, "skip": skip}
        else:
            return files

    @staticmethod
    def _is_in_names(path, names):
        for name in names:
            if name in path:
                return True
        return False

    def download(
        self,
        dest_path,
        src_prefix: Optional[str] = None,
        spacing: Optional[Union[float, int]] = 0,
        max_images=None,
        ext: Tuple[str, ...] = (".nii.gz",),
        dryrun=False,
        include=None,
        names: Optional[List[str]] = None,
    ):
        if names:
            names = set(names)

        if src_prefix is None:
            src_prefix = str(Path(self.prefix)) + "/"
        dest_path = Path(dest_path)
        s3 = boto3.resource("s3", endpoint_url=configs.aws_endpoint_url)
        bucket = s3.Bucket(self.bucket)
        files = self.list_files(spacing=spacing, ext=ext, grouped=True)
        if not files:
            logger.info("No files found to download")
            return
        files = files[next(iter(files))]
        count = 0
        for name, file_paths in files.items():
            if max_images and count >= max_images:
                break
            count += 1
            for k, file_path in file_paths.items():
                if include and k not in include:
                    continue
                if names and not self._is_in_names(file_path["path"], names):
                    continue

                file_prefix = file_path["path"].replace(f"s3://{self.bucket}/", "")
                target = os.path.join(
                    dest_path, os.path.relpath(file_prefix, src_prefix)
                )
                if os.path.exists(target):
                    logger.info(f"[already exists] {target}")
                    continue
                if not os.path.exists(os.path.dirname(target)):
                    Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)

                logger.info(f"[Downloading] {file_prefix} -> {target}")
                if not dryrun:
                    bucket.download_file(file_prefix, target)

    def upload(
        self,
        path: str,
        subprefix: str,
        spacing: Optional[Union[float, int]] = 0,
        overwrite: bool = False,
    ):

        spacing_dir = get_spacing_dirname(spacing)
        name = Path(path).name
        prefix = f"{self.prefix}/{subprefix}/{spacing_dir}/{name}"
        if not overwrite:
            try:
                self.client.head_object(Bucket=self.bucket, Key=prefix)
                logger.info(
                    f"s3://{self.bucket}/{prefix} Exists. Skipping. Pass `overwrite=True` if you want to overwrite."
                )
                return
            except:
                pass

        self.client.upload_file(str(path), self.bucket, prefix)
        logger.info(f"Uploaded to s3://{self.bucket}/{prefix}")


class DatasetLocalBackend(DatasetStorageBackendBase):
    def __init__(self, root_path=None, **kwargs):
        super().__init__()
        self.root_path = root_path or kwargs.get("dir_path", None)
        Path(self.root_path).mkdir(exist_ok=True, parents=True)
        if self.root_path is None:
            raise Exception("Missing root_path or dir_path")
        self.image_type_dirs = set()

    def list_dirs(self, sub_path: Optional[str] = None):
        path = (
            self.root_path
            if sub_path is None
            else os.path.join(self.root_path, sub_path)
        )
        return {p.stem: p for p in Path(path).iterdir()}

    def get_base_dir(self):
        return str(Path(self.root_path))

    def list_files_at_dir(
        self,
        sub_path: Optional[str] = None,
        pattern: Optional[str] = None,
        ext: Union[Tuple[str], str] = (".nii.gz",),
        recursive: bool = False,
        skip: int = 0,
        limit: int = -1,
    ):
        path = Path(self.root_path)
        if sub_path:
            path /= sub_path

        files = (
            [str(f) for f in path.glob(f"*")]
            if not recursive
            else [str(f) for f in path.rglob(f"*")]
        )

        # filter only matching spacing
        if pattern:
            files = fnmatch.filter(files, pattern)

        return [{"path": f} for f in files if f.endswith(ext)]

    def list_files(
        self,
        spacing: Optional[Union[float, int]] = None,
        data_types: Optional[List[str]] = None,
        ext: Tuple[str] = (".nii.gz",),
            path: Optional[str] = None,
        grouped=False,
        skip: int = 0,
        limit: int = -1,
        primary_key: str = "image",
    ):
        if not isinstance(ext, tuple):
            ext = (ext,)
        dataset_path = Path(path) if path is not None else Path(self.root_path)

        data_types = self.get_data_types(data_types)

        files = {}
        pattern = f"*/{get_spacing_dirname(spacing)}/*" if spacing is not None else "*"
        for data_type in data_types:
            files_iter = [
                str(f)
                for e in ext
                for f in (dataset_path / data_type["dirname"]).rglob(f"*" + e)
            ]

            # filter only matching spacing
            if spacing is not None:
                files_iter = fnmatch.filter(files_iter, pattern)
            files[data_type["name"]] = [{"path": f} for f in files_iter]

        if grouped:
            return grouped_files(files, root_prefix=str(dataset_path))
        else:
            return files


BACKENDS = {"s3": DatasetS3Backend, "local": DatasetLocalBackend}


def get_backend(
    name: Union[str, Callable[..., DatasetStorageBackendBase]]
) -> Callable[..., DatasetStorageBackendBase]:
    if callable(name):
        return name
    else:
        return BACKENDS[name]
