import os
from pathlib import Path
from typing import Optional

import boto3
import nibabel as nib
from loguru import logger

from midatasets import configs
from midatasets.s3 import check_exists_s3, upload_file
from midatasets.utils import get_key_dirname, get_extension


class S3Object:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        key: Optional[str] = None,
        local_path: Optional[str] = None,
        base_dir: str = "/tmp",
        **kwargs,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.base_dir = base_dir
        self.key = key
        self._name = None
        self._ext = None
        self._local_path = local_path

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, s3_path={self.s3_path}, local_path={self.local_path})"

    @classmethod
    def from_s3_path(
        cls,
        s3_path: str,
        base_dir: str = "/tmp",
        key: Optional[str] = None,
        local_path: Optional[str] = None,
        **kwargs,
    ):
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts)
        return cls(
            bucket=bucket,
            prefix=prefix,
            base_dir=base_dir,
            key=key,
            local_path=local_path,
            **kwargs,
        )

    @classmethod
    def from_local_path(
        cls,
        local_path: str,
        base_dir: str = "/tmp",
        key: Optional[str] = None,
        **kwargs,
    ):
        if base_dir not in local_path:
            raise Exception(f"base_dir {base_dir} not part of {local_path}")
        return cls(
            bucket=kwargs.pop("bucket") if "bucket" in kwargs else "local",
            prefix=kwargs.pop("prefix") if "prefix" in kwargs else str(Path(local_path).relative_to(base_dir)),
            base_dir=base_dir,
            key=key,
            local_path=local_path,
            **kwargs,
        )

    @property
    def local_path(self):
        """
        use provided local path; otherwise, use from prefix
        """
        return self._local_path or str(Path(f"{self.base_dir}/{self.prefix}"))

    @property
    def s3_path(self):
        return f"s3://{self.bucket}/{self.prefix}"

    @property
    def extension(self):
        if self._ext is None:
            self._ext = get_extension(self.prefix)

        return self._ext

    @property
    def name(self):
        if self._name is None:
            path = Path(self.prefix)
            self._name = path.name.replace(self.extension, "")
        return self._name

    @property
    def basename(self):
        return Path(self.prefix).name

    def download(self, overwrite: bool = False):
        s3 = boto3.resource("s3", endpoint_url=configs.aws_endpoint_url)
        bucket = s3.Bucket(self.bucket)
        target = Path(self.local_path)
        if target.exists() and not overwrite:
            logger.info(f"[already exists] {target}, skipping download.")
            return
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Downloading] {self.s3_path} -> {target}")
        bucket.download_file(self.prefix, str(target))

    def upload(self, overwrite: bool = False):
        if not overwrite and check_exists_s3(self.bucket, self.prefix):
            logger.info(f"[Upload] {self.s3_path} exists -- skipping")
            return
        upload_file(self.local_path, bucket=self.bucket, prefix=self.prefix)
        logger.info(f"[Uploaded] {self.s3_path}")

    def exists_local(self):
        return os.path.exists(self.local_path)

    def exists_remote(self):
        return check_exists_s3(self.bucket, self.prefix)

    def delete(self):
        try:
            os.remove(self.local_path)
            logger.info(f"[Removed] {self.local_path}")
        except Exception as e:
            logger.warning(e)


class MObject(S3Object):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        key: str,
        local_path: Optional[str] = None,
        base_dir: str = "/tmp",
        validate_key: bool = True,
    ):
        super().__init__(
            bucket=bucket,
            prefix=prefix,
            local_path=local_path,
            base_dir=base_dir,
            key=key,
        )

        if validate_key:
            self.validate()

    def validate(self):
        if self.key_dir not in self.prefix:
            raise KeyError(f"`{self.key_dir}` not part of `{self.prefix}`")

    @property
    def key_dir(self):
        return get_key_dirname(self.key)

    @property
    def base_prefix(self):
        if self.key_dir in self.prefix:
            return self.prefix.split(f"/{self.key_dir}")[0]
        else:
            return self.prefix.rsplit("/",1)[0]

    @property
    def subprefix(self):
        return str(Path(self.prefix).relative_to(self.base_prefix))

    def derive(self, new_key: str, local_path: Optional[str] = None, prefix: Optional[str] = None):
        """
        Derive an object with a new key
        :param new_key:
        :param local_path:
        :return:
        """
        if prefix is None:
            if self.key_dir not in self.prefix:
                raise Exception(f"{self.key_dir} not part of {self.prefix}")
            prefix = self.prefix.replace(self.key_dir, get_key_dirname(new_key))
        if local_path is not None:
            prefix = prefix.replace(self.extension, get_extension(local_path))

        return self.__class__(
            bucket=self.bucket,
            prefix=prefix,
            key=new_key,
            base_dir=self.base_dir,
            validate_key=False,
            local_path=local_path,
        )


class MImage(MObject):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        key: str,
        local_path: Optional[str] = None,
        base_dir: str = "/tmp",
        validate_key: bool = False,
    ):
        super().__init__(
            bucket=bucket,
            prefix=prefix,
            key=key,
            local_path=local_path,
            base_dir=base_dir,
            validate_key=validate_key,
        )
        self._shape = None
        self._affine = None

    def _load_metadata(self):
        native_img = nib.load(self.local_path)
        self._shape = native_img.shape
        self._affine = native_img.affine
        del native_img

    @property
    def shape(self):
        if self._shape is None:
            self._load_metadata()
        return self._shape

    @property
    def affine(self):
        if self._affine is None:
            self._load_metadata()
        return self._affine

    @property
    def resolution_dir(self):
        return Path(self.prefix).parent.name
