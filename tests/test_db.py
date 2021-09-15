import os

import pytest
from midatasets.databases import (
    MIDatasetDBBaseYaml,
    MIDatasetModel,
    CompositeDB,
    MIDatasetDBBaseMongoDb,
    MIDatasetDBDynamoDB,
)
from midatasets.datasets import MIDatasetStore
from moto import mock_dynamodb2


def test_datasets(tmp_path):
    os.environ["MIDATASETS_YAML_PATH"] = f"{tmp_path}/midatasets.yaml"
    datasets = MIDatasetStore(db="yaml")
    datasets.create(MIDatasetModel(name="foo", aws_s3_prefix="bar", aws_s3_bucket="f"))
    assert len(datasets.get_info_all()) == 1


def test_datasets_composite(tmp_path):
    os.environ["MIDATASETS_YAML_PATH"] = f"{tmp_path}/midatasets.yaml"
    datasets = MIDatasetStore(db="yaml")
    datasets.create(MIDatasetModel(name="foo", aws_s3_prefix="bar", aws_s3_bucket="f"))
    assert len(datasets.get_info_all()) == 1


@pytest.mark.skip(msg="TODO: mock mongo")
def test_mongodb():
    db = MIDatasetDBBaseMongoDb()
    print(db.find_all())


def test_yaml_crud(tmp_path):
    db = MIDatasetDBBaseYaml(path=f"/{tmp_path}/midatasets.yaml")
    m = MIDatasetModel(name="foo", aws_s3_bucket="v", aws_s3_prefix="s")

    db.create(m)

    res = db.find({"name": "foo"})
    assert res != None
    m2 = MIDatasetModel(name="bar", aws_s3_bucket="v", aws_s3_prefix="s")
    db.update({"name": "foo"}, m2)
    res = db.find({"name": "bar"})

    assert res != None
    db.delete({"name": "bar"})
    assert len(db.find_all()) == 0


def test_composite(tmp_path):
    db1 = MIDatasetDBBaseYaml(path=f"{tmp_path}/midatasets1.yaml")
    db2 = MIDatasetDBBaseYaml(path=f"{tmp_path}/midatasets2.yaml")

    for i in range(10):
        m = MIDatasetModel(name=str(i), aws_s3_bucket="v", aws_s3_prefix="s")
        if i % 2 == 0:
            db1.create(m)
        else:
            db2.create(m)

    cdb = CompositeDB([db1, db2])
    assert len(cdb.find_all()) == 10


@mock_dynamodb2
def test_dynamodb():
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-2"
    import boto3

    dynamodb = boto3.resource("dynamodb", "eu-west-2")

    dynamodb.create_table(
        TableName="test",
        KeySchema=[
            {"AttributeName": "name", "KeyType": "HASH"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "name", "AttributeType": "S"},
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )

    db = MIDatasetDBDynamoDB(table_name="test")

    for i in range(10):
        m = MIDatasetModel(name=str(i), aws_s3_bucket="a", aws_s3_prefix="s")
        db.create(m)

    db.delete({"name": "1"})
    assert len(db.find_all()) == 10 - 1
