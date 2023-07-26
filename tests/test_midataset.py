from moto import mock_s3

from midatasets.clients import LocalDatasetClient, S3DatasetClient
from midatasets.midataset import MIDataset
from midatasets.utils import create_dummy_dataset, create_dummy_s3_dataset


def test_with_local_client(tmpdir):
    name = "test"
    create_dummy_dataset(name=name, labels=["l1"], root_path=tmpdir)
    client = LocalDatasetClient(root_dir=str(tmpdir))
    dataset = MIDataset(dataset_id=name, client=client, base_dir=str(tmpdir))
    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        assert len(a) == 2


@mock_s3
def test_with_s3_client(tmpdir):
    name = "test"
    create_dummy_s3_dataset(name=name, labels=["l1"], bucket_name="testbucket")
    client = S3DatasetClient(bucket="testbucket", prefix="datasets")
    dataset = MIDataset(dataset_id=name, client=client, base_dir=tmpdir)
    assert len(client.get_datasets()) == 1
    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        assert len(a) == 2
    dataset.download()

    for a in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
        a["image"].upload(overwrite=True)


@mock_s3
def test_with_s3_client_nrrd(tmpdir):
    name = "test"
    create_dummy_s3_dataset(name=name, labels=["l1"], bucket_name="testbucket", labelmap_ext=".nrrd")
    client = S3DatasetClient(bucket="testbucket", prefix="datasets")
    dataset = MIDataset(dataset_id=name, client=client, base_dir=tmpdir)
    assert len(client.get_datasets()) == 1
    keys = ["image", "labelmap/l1"]
    exts = (".nii.gz", ".nrrd")
    for a in dataset.iterate_keys(keys=keys, extensions=exts):
        assert len(a) == 2
    dataset.download(keys=keys, extensions=exts)

    for a in dataset.iterate_keys(keys=keys, extensions=exts):
        for key in keys:
            a[key].upload(overwrite=True)
