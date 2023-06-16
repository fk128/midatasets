from pathlib import Path
from typing import List, Dict
from typing import Union

from midatasets.mimage import MImage
from midatasets.clients import DatasetClientBase
from midatasets.schemas import Dataset, Image
from midatasets.processing import resample_mimage_parallel
from midatasets.utils import get_spacing_dirname, get_key_dirname


class MIDataset:
    """
    Dataset class
    """
    def __init__(self,
                 dataset_id: Union[int, str],
                 client: DatasetClientBase,
                 base_dir="/tmp",
                 default_bucket: str = "local"):
        self.client = client
        self.dataset_id = dataset_id
        self._images = None
        self._info = None
        self.base_dir = base_dir
        self.bucket = default_bucket

    @property
    def info(self) -> Dataset:
        if self._info is None:
            self._info = self.client.get_dataset(self.dataset_id)
        return self._info

    @property
    def dataset_base_dir(self):
        return f"{self.base_dir}/datasets/{self.dataset_id}"

    def get_labels(self, key: str):
        return self.info.label_mappings.get(key, None)

    @property
    def images(self) -> List[Image]:
        if self._images is None:
            self._images = self.client.get_images(self.dataset_id)
        return self._images

    def download(self, keys=("image",), overwrite=False):
        for images in self.iterate_keys(keys=keys):
            for k, image in images.items():
                image.download(overwrite=overwrite)

    def get_resampled_mimage(self, image: MImage, target_spacing):
        prefix = image.prefix.replace(
            image.resolution_dir, get_spacing_dirname(target_spacing)
        )
        local_path = image.local_path.replace(
            image.resolution_dir, get_spacing_dirname(target_spacing)
        )
        return MImage(
            bucket=image.bucket,
            prefix=prefix,
            key=image.key,
            base_dir=image.base_dir,
            validate_key=False,
            local_path=local_path,
        )
    
    def _create_mimage(self,  path:str, key:str, name: str):
        if path.startswith("s3://"):
            return MImage.from_s3_path(
                path,
                base_dir=self.base_dir,
                validate_key=False,
                key=key,
                local_path=f"{self.dataset_base_dir}/{get_key_dirname(key)}/{get_spacing_dirname(0)}/{name}.nii.gz",
            )
        else:
            return MImage.from_local_path(
                base_dir=self.base_dir,
                validate_key=False,
                key=key,
                local_path=path,
                bucket=self.bucket
            )

    def iterate_key(self, key: str, spacing: Union[float, int] = 0):
        """
        iterate over objects with a given key
        Args:
            key: the key to iterate over
            spacing:

        Returns:

        Examples:
            Iterate over keys to download images
            >>> dataset = MIDataset(dataset_id="1")
            >>> for obj in dataset.iterate_key(key="image"):
            >>>         obj.download()

        """
        remap_keys = self.info.label_mappings.get("_remap_keys", {})
        for image in self.images:
            for artifact in image.artifacts:
                if key == remap_keys.get(
                        artifact.key, artifact.key
                ) and artifact.path.endswith(".nii.gz"):
                    obj = self._create_mimage(artifact.path, key=key, name=image.name)
                    if spacing != 0:
                        obj = self.get_resampled_mimage(
                            obj, target_spacing=spacing
                        )
                    yield obj

    def iterate_keys(self, keys=("image",), spacing: Union[float, int] = 0, allow_missing: bool = False) -> Dict[str, MImage]:
        """
        iterate over multiple keys
        Args:
            keys: the list of keys to iterate over
            spacing: the specific spacing of the images
            allow_missing: if True then also includes the images that don't have all the keys

        Returns:

        Examples:
            Iterate over keys to download images
            >>> dataset = MIDataset(dataset_id="1")
            >>> for artifacts in dataset.iterate_keys(keys=["image", "labelmap/lungmask"]):
            >>>     for k, obj in artifacts.items():
            >>>         obj.download()

        """
        remap_keys = self.info.label_mappings.get("_remap_keys", {})
        for image in self.images:
            artifacts = {}
            for artifact in image.artifacts:
                key = remap_keys.get(artifact.key, artifact.key)
                if key in keys and artifact.path.endswith(".nii.gz"):

                    artifacts[key] = self._create_mimage(artifact.path, key=key, name=image.name)
                    if spacing != 0:
                        artifacts[key] = self.get_resampled_mimage(
                            artifacts[key], target_spacing=spacing
                        )
            if allow_missing or len(artifacts) == len(keys):
                yield artifacts

    def resample(self, target_spacing, keys=("image",)):
        """
        resample images to a specific spacing
        Args:
            target_spacing: the target spacing to resample to
            keys:  the keys to resample

        Returns:

        Examples:
            Download from s3 and resample

            >>> dataset = MIDataset(dataset_id="lung")
            >>> dataset.download(keys=["image", "labelmap"])
            >>> dataset.resample(target_spacing=1, keys=["image"])
        """
        images = []
        for image in self.iterate_keys(keys=keys, spacing=0):
            images.extend(image.values())

        resample_mimage_parallel(images, target_spacing=target_spacing)


    def get_dir(self, key: str, spacing: Union[int, float] = 0) -> Path:
        """
        Get the directory of a given key at a given spacing
        Args:
            key: the key of the image
            spacing: the spacing

        Returns:

        """
        return Path(f"{self.dataset_base_dir}/{get_key_dirname(key)}/{get_spacing_dirname(spacing)}")