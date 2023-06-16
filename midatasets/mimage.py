import os
from pathlib import Path
from typing import Optional

import nibabel as nib
from loguru import logger

from midatasets.s3 import S3Boto3
from midatasets.utils import get_key_dirname, get_extension


class S3Object:
    """
    A class to represent an s3 object to make it easier to upload and download
    """
    def __init__(
        self,
        bucket: str,
        prefix: str,
        key: Optional[str] = None,
        local_path: Optional[str] = None,
        base_dir: str = "/tmp",
        s3_client: Optional[S3Boto3] = None,
        **kwargs,
    ):
        """

        Args:
            bucket: s3 bucket
            prefix: s3 prefix
            key: a key that represents the objects
            local_path: a specific local path to use. If None it is derived from prefix and base_dir
            base_dir: a base directory to use locally
            s3_client: the s3 client
            **kwargs:
        """
        self.bucket = bucket
        self.prefix = prefix
        self.base_dir = base_dir
        self.key = key
        self._name = None
        self._ext = None
        self._local_path = local_path
        self.s3_client = s3_client or S3Boto3()

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
    def s3_path(self) -> str:
        """

        Returns: full s3 path

        """
        return f"s3://{self.bucket}/{self.prefix}"

    @property
    def extension(self) -> str:
        """

        Returns: file extension

        """
        if self._ext is None:
            self._ext = get_extension(self.prefix)

        return self._ext

    @property
    def name(self):
        """

        Returns: name of object

        """
        if self._name is None:
            path = Path(self.prefix)
            self._name = path.name.replace(self.extension, "")
        return self._name

    @property
    def basename(self):
        """

        Returns: basename from prefix

        """
        return Path(self.prefix).name

    def download(self, overwrite: bool = False):
        """
        Download an object from s3
        Args:
            overwrite: Whether to overwrite if it already exists

        Returns: None

        """

        self.s3_client.download_file(bucket=self.bucket,
                                     prefix=self.prefix,
                                     target=self.local_path, overwrite=overwrite)

    def upload(self, overwrite: bool = False) -> None:
        """
        Upload an object to s3
        Args:
            overwrite: Whether to overwrite if it already exists

        Returns: None

        """
        self.s3_client.upload_file(file_name=self.local_path, bucket=self.bucket,prefix=self.prefix, overwrite=overwrite)

    def exists_local(self):
        """
        Check if object exists locally
        Returns: True if exists

        """
        return os.path.exists(self.local_path)

    def exists_remote(self):
        """
        Check if object exists on s3
        Returns: True if exists

        """
        return self.s3_client.check_exists(self.bucket, self.prefix)

    def delete(self):
        """
        Delete an object locally
        Returns:

        """
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
        """

        Args:
            bucket: s3 bucket
            prefix: s3 prefix
            key: a key that represents the objects
            local_path: a specific local path to use. If None it is derived from prefix and base_dir
            base_dir: a base directory to use locally
            validate_key: whether to check if the key is part of the prefix
        """
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
        """

        Returns: the key directory

        """
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
    """
    MImage
    added functionality for loading image metadata if available
    """
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
        """

        Returns: affine metadata

        """
        if self._affine is None:
            self._load_metadata()
        return self._affine

    @property
    def resolution_dir(self):
        return Path(self.prefix).parent.name
