from moto import mock_s3

from midatasets.clients import LocalClient, S3Client

from midatasets.utils import create_dummy_dataset, create_dummy_s3_dataset


def test_local_datasets(tmpdir):
    for name in range(10):
        create_dummy_dataset(name=f"dataset_{name}", labels=["l1"], root_path=tmpdir)
    client = LocalClient(root_dir=tmpdir)
    assert len(client.get_datasets()) == 10

@mock_s3
def test_local_datasets(tmpdir):
    for name in range(10):
        create_dummy_s3_dataset(name=f"dataset_{name}", labels=["l1"], bucket_name="test", prefix="datasets")
    client = S3Client(bucket="test", prefix="datasets")
    assert len(client.get_datasets()) == 10