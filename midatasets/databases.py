import json
import os
from enum import Enum
from typing import Optional, Dict, List

import yaml
from bson import ObjectId
from loguru import logger
from pydantic import BaseModel, Field, BaseSettings
from pymongo import MongoClient
from smart_open import smart_open


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class MIDatasetModel(BaseModel):
    # id: Optional[PyObjectId] = Field(alias="_id")
    name: str
    labels: Optional[List]
    aws_s3_bucket: str
    aws_s3_prefix: str

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class MIDatasetDBBase:
    def find_all(self, selector: Optional[Dict] = None):
        raise NotImplementedError

    def find(self, selector: Dict):
        raise NotImplementedError

    def create(self, item: BaseModel):
        raise NotImplementedError

    def update(self, selector: Dict, item: BaseModel):
        raise NotImplementedError

    def delete(self, selector: Dict):
        raise NotImplementedError


class MIDatasetDBBaseMongoDb(MIDatasetDBBase):
    class Config(BaseSettings):
        host: str = None
        db_name: str = "midatasets"
        collection_name: str = "datasets"
        primary_key: str = "name"

        class Config:
            env_prefix = "midatasets_mongo_"

    def __init__(
        self,
        host: str = None,
        db_name: str = None,
        collection_name: str = None,
        primary_key: str = None,
    ):
        config = self.Config()
        self.db_name = db_name or config.db_name
        self.collection_name = collection_name or config.collection_name
        self.primary_key = primary_key or config.primary_key
        self.client = MongoClient(host=host)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.collection.create_index(self.primary_key, unique=True)

    def find_all(self, selector=None):
        if selector is None:
            selector = {}
        return list(self.collection.find(selector))

    def find(self, selector):
        return self.collection.find_one(selector)

    def create(self, item: BaseModel):
        return self.collection.insert_one(json.loads(item.json()))

    def update(self, selector, item: BaseModel):
        return self.collection.replace_one(
            selector, json.loads(item.json())
        ).modified_count

    def delete(self, selector):
        return self.collection.delete_one(selector).deleted_count


class MIDatasetDBDict(MIDatasetDBBase):
    def __init__(
        self,
        data: Optional[Dict] = None,
        collection_name: str = "datasets",
        primary_key: str = "name",
    ):
        super().__init__()
        self.collection_name = collection_name
        self.data = data or {self.collection_name: []}
        self.primary_key = primary_key
        self._load()

    def _load(self):
        pass

    def _save(self):
        pass

    def find_all(self, selector=None):
        if not selector:
            return self.data[self.collection_name]
        else:
            return [
                d
                for d in self.data[self.collection_name]
                if all([d[k] == v for k, v in selector.items()])
            ]

    def find(self, selector):
        for d in self.data[self.collection_name]:
            if all([d[k] == v for k, v in selector.items()]):
                return d
        return None

    def create(self, item: BaseModel):
        if self.find({"name": item.name}) is None:
            self.data[self.collection_name].append(json.loads(item.json()))
            self._save()
        else:
            raise KeyError(f"name {item.name} already exists")

    def update(self, selector, item: BaseModel):
        update_count = 0
        for i, d in enumerate(self.data[self.collection_name]):
            if all([d[k] == v for k, v in selector.items()]):
                self.data[self.collection_name][i] = json.loads(item.json())
                update_count += 1
        if update_count > 0:
            self._save()
        return update_count

    def delete(self, selector):
        del_count = 0
        for i, d in enumerate(self.data[self.collection_name][:]):
            if all([d[k] == v for k, v in selector.items()]):
                del self.data[self.collection_name][i]
                del_count += 1
        if del_count > 0:
            self._save()
        return del_count


class CompositeDB(MIDatasetDBBase):
    def __init__(self, dbs: List[MIDatasetDBBase]):
        self.dbs = dbs

    def find_all(self, selector=None):
        result = []
        for db in self.dbs:
            result.extend(db.find_all(selector))
        return result

    def find(self, selector):
        result = None
        for db in self.dbs:
            _result = db.find(selector)

            if result is not None and _result is not None:
                logger.warning(
                    f"Overriding dataset {result['name']} definition using one from {db.__class__}"
                )
            if _result is not None:
                result = _result

        return result

    def create(self, item: BaseModel):
        raise NotImplementedError("Ambiguous composite for create")

    def update(self, selector, item: BaseModel):
        raise NotImplementedError("Ambiguous composite for update")


def delete(self, selector):
    return self.collection.delete_one(selector).deleted_count


class MIDatasetDBBaseYaml(MIDatasetDBDict):
    class Config(BaseSettings):
        path: str = "~/.midatasets.yaml"
        collection_name: str = "datasets"
        primary_key: str = "name"

        class Config:
            env_prefix = "midatasets_yaml_"

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        primary_key: Optional[str] = None,
    ):

        config = self.Config()
        self.path = path or config.path
        self.path = os.path.expanduser(self.path)
        collection_name = collection_name or config.collection_name
        primary_key = primary_key or config.primary_key
        super().__init__(
            data=None, collection_name=collection_name, primary_key=primary_key
        )

    def _load(self):
        try:
            with smart_open(self.path) as f:
                self.data = yaml.safe_load(f)
        except:
            logger.error(f"No yaml db found at {self.path}")

    def _save(self):
        with smart_open(self.path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)


class MIDatasetDBTypes(MIDatasetDBBase, Enum):
    composite = CompositeDB
    yaml = MIDatasetDBBaseYaml
    mongo = MIDatasetDBBaseMongoDb
