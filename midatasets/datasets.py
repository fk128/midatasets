import os
from typing import Optional, Dict, Union

from loguru import logger
from midatasets import configs
from midatasets.MIReader import MIReader
from midatasets.databases import MIDatasetDBBase, MIDatasetDBTypes, MIDatasetModel

_midataset_store = None


def get_db(db: Optional[Union[MIDatasetDBBase, str]] = None) -> MIDatasetDBBase:
    if isinstance(db, MIDatasetDBBase):
        return db
    elif isinstance(db, str):
        return MIDatasetDBTypes[db].value()
    else:
        return MIDatasetDBTypes[configs.get("database", "yaml")].value()


class MIDatasetStore:
    def __init__(self, db: Optional[Union[MIDatasetDBBase, str]] = None):
        self._db: MIDatasetDBBase = get_db(db)

    def get_info_all(self, selector: Optional[Dict] = None, names_only: bool = False):
        datasets = self._db.find_all(selector=selector)
        if names_only:
            return [d["name"] for d in datasets]
        else:
            return datasets

    def get_info(self, name: str):
        return self._db.find(selector={"name": name})

    def create(self, dataset: MIDatasetModel):
        return self._db.create(item=dataset)

    def delete(self, name: str):
        res = self._db.delete({"name": name})
        if res > 0:
            logger.info(f"deleted {name}")
        else:
            logger.error(f"{name} not found for deletion")
        return res

    def update(self, name: str, dataset: MIDatasetModel):
        return self._db.update(selector={"name": name}, item=dataset)

    def load(self, name: str, **kwargs) -> MIReader:
        dataset = self.get_info(name)
        dataset["dir_path"] = os.path.join(
            configs.get("root_path"), dataset.get("subpath", None) or dataset["name"]
        )
        dataset.update(kwargs)
        return MIReader.from_dict(**dataset)


def get_midataset_store():
    global _midataset_store
    if _midataset_store is None:
        _midataset_store = MIDatasetStore()
    return _midataset_store


def set_midataset_store(db):
    global _midataset_store
    _midataset_store = db


def _load_dataset_from_db(name, **kwargs) -> MIReader:
    dataset = get_midataset_store().get_info(name)
    dataset["dir_path"] = os.path.join(
        configs.get("root_path"), dataset.get("subpath", None) or dataset["name"]
    )
    dataset.update(kwargs)
    return MIReader.from_dict(**dataset)


def load_dataset(name, spacing, dataset_path=None, **kwargs) -> MIReader:
    if dataset_path:
        return MIReader(name=name, spacing=spacing, dir_path=dataset_path, **kwargs)
    else:
        return _load_dataset_from_db(name, spacing=spacing, **kwargs)
