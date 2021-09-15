import os
from typing import Optional, Dict, Union

from midatasets import configs
from midatasets.MIReader import MIReader
from midatasets.databases import MIDatasetDBBase, MIDatasetDBTypes, MIDatasetModel

_midatasetdb = None


def get_db(db: Optional[Union[MIDatasetDBBase, str]] = None) -> MIDatasetDBBase:
    if isinstance(db, MIDatasetDBBase):
        return db
    elif isinstance(db, str):
        return MIDatasetDBTypes[db].value()
    else:
        return MIDatasetDBTypes[configs.get("database", "yaml")].value()


class MIDatasets:
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
        return self._db.delete({"name": name})

    def update(self, name: str, dataset: MIDatasetModel):
        return self._db.update(selector={"name": name}, item=dataset)

    def load(self, name: str, **kwargs) -> MIReader:
        dataset = self.get_info(name)
        dataset["dir_path"] = os.path.join(
            configs.get("root_path"), dataset.get("subpath", None) or dataset["name"]
        )
        dataset.update(kwargs)
        return MIReader.from_dict(**dataset)


def get_midatasetdb():
    global _midatasetdb
    if _midatasetdb is None:
        _midatasetdb = MIDatasets()
    return _midatasetdb


def _load_dataset_from_db(name, **kwargs) -> MIReader:
    dataset = get_midatasetdb().get_info(name)
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
