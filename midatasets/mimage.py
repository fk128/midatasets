from pathlib import Path
from typing import Optional

from loguru import logger

from s3obj import S3Object
from midatasets.utils import get_key_dirname, get_extension


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
        try:
            import nibabel as nib
            native_img = nib.load(self.local_path)
            self._shape = native_img.shape
            self._affine = native_img.affine
        except Exception as e:
            logger.exception("Failed to load metadata")

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
