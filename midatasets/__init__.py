# -*- coding: utf-8 -*-

__version__ = "0.9.0"

import os
from typing import Optional, Dict, List

import yaml
from loguru import logger
from pydantic import BaseSettings, validator


class ConfigsBase(BaseSettings):
    configs_path: str = "~/.midatasets.yaml"

    class Config:
        env_prefix = "midatasets_"


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
    data_types: List[Dict] = [{"dirname": "images", "name": "image"},
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


def load_configs(path="~/.midatasets.yaml", raise_error: bool = False):
    import smart_open

    configs = Configs().dict()
    try:
        with smart_open.open(os.path.expanduser(path)) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            if "configs" in data:
                data = data["configs"]
            configs.update(data)
    except:
        if path != "~/.midatasets.yaml":
            logger.error(f"Failed to load {path}")
        if raise_error:
            raise
    return Configs(**configs)


_configs: Optional[Dict] = None


def get_configs():
    global _configs
    if _configs is None:
        _configs = load_configs(ConfigsBase().configs_path)
    return _configs


def update_configs(configs):
    global _configs
    if _configs:
        _configs.update(configs)


configs = get_configs()
