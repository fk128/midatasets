from moto import mock_s3

from midatasets.clients import LocalClient, S3Client
from midatasets.midataset import MIDataset
from midatasets.utils import create_dummy_dataset, create_dummy_s3_dataset


def test_with_local_client(tmpdir):
    name = "test"
    create_dummy_dataset(name=name, labels=["l1"], root_path=tmpdir)
    client = LocalClient(root_dir=str(tmpdir))
    dataset = MIDataset(dataset_id=name, client=client, base_dir=str(tmpdir))
    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        assert len(a) == 2



@mock_s3
def test_with_s3_client(tmpdir):
    name = "test"
    create_dummy_s3_dataset(name=name, labels=["l1"], bucket_name="testbucket")
    client = S3Client(bucket="testbucket", prefix="datasets")
    dataset = MIDataset(dataset_id=name, client=client, base_dir=tmpdir)
    assert len(client.get_datasets()) == 1
    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        assert len(a) == 2
    dataset.download()

    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        a["image"].upload(overwrite=True)