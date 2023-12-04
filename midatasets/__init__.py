# -*- coding: utf-8 -*-

__version__ = "0.25.2"

import os
from typing import Optional, Dict, List, Set

from pydantic import BaseSettings, validator
from midatasets.schemas import BaseType


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
    aws_endpoint_url: Optional[str] = None
    primary_type: str = "image"
    extensions: Set[str] = {
        ".jpg",
        ".jpeg",
        ".nii",
        ".gz",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".nrrd",
        ".vtk",
        ".gif",
        ".dcm"
    }
    data_types: List[Dict] = [
        {"dirname": "images", "name": "image"},
        {"dirname": "labelmaps", "name": "labelmap"},
        {"dirname": "previews", "name": "preview"},
        {"dirname": "outputs", "name": "output"},
        {"dirname": "metadata", "name": "metadata"},
    ]
    base_types: List[BaseType] = [
        BaseType(name="image", dirname="images"),
        BaseType(name="labelmap", dirname="labelmaps"),
        BaseType(name="preview", dirname="previews"),
        BaseType(name="output", dirname="outputs"),
        BaseType(name="metadata", dirname="metadata"),
    ]
    database: str = "yaml"

    class Config:
        extra = "ignore"
        env_prefix = "midatasets_"

    @validator("root_path")
    def expand_vars(cls, v):
        return os.path.expandvars(v)


configs = Configs()

from .midataset import MIDataset
from .mimage import MImage, MObject
from s3obj import S3Object