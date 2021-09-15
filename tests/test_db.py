import os

import pytest
from midatasets.databases import (
    MIDatasetDBBaseYaml,
    MIDatasetModel,
    CompositeDB, MIDatasetDBBaseMongoDb,
)
from midatasets.datasets import MIDatasetStore


def test_datasets(tmp_path):
    os.environ['MIDATASETS_YAML_PATH'] = f"{tmp_path}/midatasets.yaml"
    datasets = MIDatasetStore(db="yaml")
    datasets.create(
        MIDatasetModel(name="foo", aws_s3_prefix="bar", aws_s3_bucket="f")
    )
    assert len(datasets.get_info_all()) == 1



def test_datasets_composite(tmp_path):
    os.environ['MIDATASETS_YAML_PATH'] = f"{tmp_path}/midatasets.yaml"
    datasets = MIDatasetStore(db="yaml")
    datasets.create(
        MIDatasetModel(name="foo", aws_s3_prefix="bar", aws_s3_bucket="f")
    )
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
