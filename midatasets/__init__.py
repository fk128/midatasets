# -*- coding: utf-8 -*-

__version__ = "0.16.0"

import os
from typing import Optional, Dict, List

from pydantic import BaseSettings, validator


class Configs(BaseSettings):
    root_path: str = "/data/datasets"
    root_s3_prefix: str = "datasets/"
    native_images_dir: str = "native"
    images_dir: str = "images"
    subsampled_dir_prefix: str = "subsampled"
    images_crop_prefix: str = "images_crop_"
    labelmaps_crop_prefix: str = "labelmaps_crop_"
    aws_s3_bucket: Optional[str] = None
    aws_s3_profile: Optional[str] = None
    remap_dirs: Dict = {"images": "image", "labelmaps": "labelmap"}
    primary_type: str = "image"
    data_types: List[Dict] = [
        {"dirname": "images", "name": "image"},
        {"dirname": "labelmaps", "name": "labelmap"},
        {"dirname": "previews", "name": "preview"},
        {"dirname": "outputs", "name": "output"},
        {"dirname": "metadata", "name": "metadata"},
    ]
    database: str = "yaml"

    class Config:
        extra = "ignore"
        env_prefix = "midatasets_"

    @validator("root_path")
    def name_must_contain_space(cls, v):
        return os.path.expandvars(v)


configs = Configs()
