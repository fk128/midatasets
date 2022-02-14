import json
import os
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List

import boto3
import yaml
from botocore.exceptions import ClientError

# from bson import ObjectId
from loguru import logger
from pydantic import BaseModel, BaseSettings, Field

try:
    from pymongo import MongoClient
except:
    MongoClient = None
from smart_open import smart_open


# class PyObjectId(ObjectId):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v):
#         if not ObjectId.is_valid(v):
#             raise ValueError("Invalid objectid")
#         return ObjectId(v)
#
#     @classmethod
#     def __modify_schema__(cls, field_schema):
#         field_schema.update(type="string")


class MIDatasetModel(BaseModel):
    # id: Optional[PyObjectId] = Field(alias="_id")
    name: str
    label_mappings: Optional[Dict]
    aws_s3_bucket: str
    aws_s3_prefix: str
    description: str = ""
    created_time: datetime = Field(default_factory=datetime.now)
    modified_time: datetime = Field(default_factory=datetime.now)

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        # json_encoders = {ObjectId: str}


class DBBase:
    class Config(BaseSettings):
        db_name: str = "db"
        table_name: str = "table"
        primary_key: str = "name"
        path: str = ""
        host: Optional[str] = None

    def find_all(self, selector: Optional[Dict] = None):
        raise NotImplementedError

    def find(self, selector: Dict):
        raise NotImplementedError

    def create(self, item: BaseModel):
        raise NotImplementedError

    def update(self, selector: Dict, item: Dict):
        raise NotImplementedError

    def delete(self, selector: Dict):
        raise NotImplementedError


if MongoClient:

    class DBMongodb(DBBase):
        def __init__(
            self,
            host: str = None,
            db_name: str = None,
            collection_name: str = None,
            primary_key: str = None,
        ):
            config = self.Config()
            self.db_name = db_name or config.db_name
            self.collection_name = collection_name or config.table_name
            self.primary_key = primary_key or config.primary_key
            self.client = MongoClient(host=host or config.host)
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

        def update(self, selector, item: Dict):
            return self.collection.update_one(selector, {"$set": item}).modified_count

        def delete(self, selector):
            return self.collection.delete_one(selector).deleted_count


class DBDict(DBBase):
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
        if self.find({self.primary_key: getattr(item, self.primary_key)}) is None:
            self.data[self.collection_name].append(json.loads(item.json()))
            self._save()
        else:
            raise KeyError(f"name {getattr(item, self.primary_key)} already exists")

    def update(self, selector, item: Dict):
        update_count = 0
        for i, d in enumerate(self.data[self.collection_name]):
            if all([d[k] == v for k, v in selector.items()]):
                self.data[self.collection_name][i].update(item)
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


class DBComposite(DBBase):
    def __init__(self, dbs: List[DBBase]):
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
        logger.info(f"Created using {self.dbs[0].__class__}")
        return self.dbs[0].create(item)

    def update(self, selector, item: Dict):
        logger.info(f"Updated using {self.dbs[0].__class__}")
        return self.dbs[0].update(selector, item)

    def delete(self, selector):
        logger.info(f"Deleted using {self.dbs[0].__class__}")
        return self.dbs[0].delete(selector)


class DBYaml(DBDict):
    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        primary_key: Optional[str] = None,
    ):

        config = self.Config()
        self.path = path or config.path
        self.path = os.path.expanduser(self.path)
        collection_name = collection_name or config.table_name
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


class DBDynamodb(DBBase):
    def __init__(
        self,
        table_name: str = None,
        primary_key: str = None,
    ):
        config = self.Config()
        self.table_name = table_name or config.table_name
        self.primary_key = primary_key or config.primary_key
        self.client = boto3.resource("dynamodb")
        self.table = self.client.Table(self.table_name)

    def find_all(self, selector: Optional[Dict] = None):
        response = self.table.scan()
        data = response["Items"]

        while "LastEvaluatedKey" in response:
            response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            data.extend(response["Items"])

        if selector:
            return [d for d in data if all([d[k] == v for k, v in selector.items()])]

        return data

    def find(self, selector: Dict):
        try:
            response = self.table.get_item(Key=selector)
        except ClientError as e:
            logger.error(e.response["Error"]["Message"])
        else:
            if "Item" in response:
                return response["Item"]
            else:
                raise Exception(f"{selector} does not exist")

    def create(self, item: BaseModel):
        response = self.table.put_item(Item=json.loads(item.json()))
        return response

    def update(self, selector: Dict, item: Dict):
        expression, names, values = self._get_update_params(item)
        return self.table.update_item(
            Key={self.primary_key: selector.get(self.primary_key)},
            UpdateExpression=expression,
            ExpressionAttributeValues=dict(values),
            ExpressionAttributeNames=dict(names),
        )

    def _get_update_params(self, body):
        update_expression = ["set "]
        update_values = dict()
        update_names = dict()

        for key, val in body.items():
            update_expression.append(f" #{key} = :v{key},")
            update_values[f":v{key}"] = val
            update_names[f"#{key}"] = key

        return "".join(update_expression)[:-1], update_names, update_values

    def delete(self, selector: Dict):
        response = self.table.delete_item(Key=selector)
        return response


class MIDatasetDBDynamodb(DBDynamodb):
    class Config(BaseSettings):
        table_name: str = "datasets"
        primary_key: str = "name"

        class Config:
            env_prefix = "midatasets_dynamodb_"


class MIDatasetDBYaml(DBYaml):
    class Config(BaseSettings):
        path: str = "~/.midatasets.yaml"
        table_name: str = "datasets"
        primary_key: str = "name"

        class Config:
            env_prefix = "midatasets_yaml_"


if MongoClient:

    class MIDatasetMongodb(DBMongodb):
        class Config(BaseSettings):
            host: str = None
            db_name: str = "midatasets"
            collection_name: str = "datasets"
            primary_key: str = "name"

            class Config:
                env_prefix = "midatasets_mongo_"


else:
    MIDatasetMongodb = None


class MIDatasetDBComposite(DBComposite):
    pass


class MIDatasetDBTypes(DBBase, Enum):
    composite = MIDatasetDBComposite
    yaml = MIDatasetDBYaml
    mongo = MIDatasetMongodb
    dynamodb = MIDatasetDBDynamodb
