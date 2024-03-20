from pathlib import Path
from typing import Union, List

from joblib import Parallel, delayed
from loguru import logger
from midatasets.mimage import MImage
from midatasets.preprocessing import sitk_resample
from midatasets.utils import get_spacing_dirname
import SimpleITK as sitk


def resample_mimage(
    image: MImage, target_spacing: Union[float, int], overwrite: bool = False
):
    """
    resample an MImage to a target spacing
    Args:
        image:
        target_spacing:
        overwrite:

    Returns:

    """
    output_path = image.local_path.replace(
        image.resolution_dir, get_spacing_dirname(target_spacing)
    )
    prefix = image.prefix.replace(
        image.resolution_dir, get_spacing_dirname(target_spacing)
    )
    if not overwrite and Path(output_path).exists():
        return MImage(
            bucket=image.bucket,
            prefix=prefix,
            key=image.key,
            base_dir=image.base_dir,
            local_path=output_path,
            validate_key=False,
        )
    sitk_image = sitk.ReadImage(image.local_path)
    interpolation = (
        sitk.sitkLinear if "image" in image.key else sitk.sitkNearestNeighbor
    )
    interpolation_str = (
        "sitk.sitkLinear" if "image" in image.key else "sitk.sitkNearestNeighbor"
    )
    logger.info(
        f"[{image.key}/{Path(output_path).name}] resampling from {sitk_image.GetSpacing()} "
        f"to {target_spacing} using {interpolation_str}"
    )
    sitk_image: sitk.Image = sitk_resample(
        sitk_image, target_spacing, interpolation=interpolation
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_image, output_path)
    return MImage(
        bucket=image.bucket,
        prefix=prefix,
        key=image.key,
        base_dir=image.base_dir,
        validate_key=False,
        local_path=output_path,
    )


def resample_mimage_parallel(
    images: List[MImage],
    target_spacing: Union[float, int],
    overwrite: bool = False,
    n_jobs: int = -1,
):
    """
    resample MImages in parallel
    Args:
        images:
        target_spacing:
        overwrite:
        n_jobs:

    Returns:

    """
    Parallel(n_jobs=n_jobs)(
        delayed(resample_mimage)(image, target_spacing, overwrite) for image in images
    )
