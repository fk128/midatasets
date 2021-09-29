from pathlib import Path

import boto3
from midatasets import storage_backends
from midatasets.MIReader import MIReader
from midatasets.utils import get_spacing_dirname
from moto import mock_s3


def test_local_backend_sublabels(tmpdir):
    p = Path(tmpdir)
    labels = ["l1", "l2"]
    for dataset_name in ["foo", "bar"]:
        for l in labels:
            (p / dataset_name / "labelmaps" / l / "native").mkdir(
                exist_ok=True, parents=True
            )
        (p / dataset_name / "images" / "native").mkdir(exist_ok=True, parents=True)
        for i in range(10):
            for l in labels:
                (
                    p
                    / dataset_name
                    / "labelmaps"
                    / l
                    / "native"
                    / f"image_{i}_seg.nii.gz"
                ).touch()
            (p / dataset_name / "images" / "native" / f"image_{i}.nii.gz").touch()

    dataset = MIReader(
        dir_path=str(Path(tmpdir) / "foo"), spacing=0, remote_backend=None
    )

    assert list(dataset.dataframe.columns) == [
        "image",
        "labelmap/l2",
        "labelmap/l1",
    ]
    assert len(dataset.dataframe) == 10


def test_local_backend(tmpdir):
    p = Path(tmpdir)
    for dataset_name in ["foo", "bar"]:
        (p / dataset_name / "labelmaps" / "native").mkdir(exist_ok=True, parents=True)
        (p / dataset_name / "images" / "native").mkdir(exist_ok=True, parents=True)
        for i in range(10):
            (
                p / dataset_name / "labelmaps" / "native" / f"image_{i}_seg.nii.gz"
            ).touch()
            (p / dataset_name / "images" / "native" / f"image_{i}.nii.gz").touch()

    dataset = MIReader(
        dir_path=str(Path(tmpdir) / "foo"), spacing=0, remote_backend=None
    )
    assert len(dataset.dataframe) == 10


def test_local_backend_duplicate(tmpdir):
    p = Path(tmpdir)
    for dataset_name in ["foo", "bar"]:
        (p / dataset_name / "labelmaps" / "native").mkdir(exist_ok=True, parents=True)
        (p / dataset_name / "images" / "native").mkdir(exist_ok=True, parents=True)
        for i in range(10):
            (
                p / dataset_name / "labelmaps" / "native" / f"image_{i}_seg.nii.gz"
            ).touch()
            (p / dataset_name / "images" / "native" / f"image_{i}.nii.gz").touch()
        for i in range(10):
            (
                p / dataset_name / "labelmaps" / "native" / f"image_{i}_seg.nii.gz"
            ).touch()
            (p / dataset_name / "images" / "native" / f"image_{i}_2.nii.gz").touch()

    dataset = MIReader(
        dir_path=str(Path(tmpdir) / "foo"), spacing=0, remote_backend=None
    )
    assert len(dataset.dataframe) == 2 * 10


@mock_s3
def test_s3_backend(tmpdir):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="mybucket")
    s3 = boto3.client("s3", region_name="us-east-1")
    datasets = {}

    for dataset_name in ["foo", "bar"]:
        datasets[dataset_name] = f"datasets/{dataset_name}/"
        for i in range(10):
            key = f"datasets/{dataset_name}/labelmaps/native/img_{i}.nii.gz"
            s3.put_object(Bucket="mybucket", Key=key, Body="")
            key = f"datasets/{dataset_name}/images/native/img_{i}.nii.gz"
            s3.put_object(Bucket="mybucket", Key=key, Body="")
            key = f"datasets/{dataset_name}/previews/native/img_{i}.jpg"
            s3.put_object(Bucket="mybucket", Key=key, Body="")

        backend = storage_backends.DatasetS3Backend(bucket="mybucket", prefix=f"datasets/{dataset_name}")


        dest_path = f"{tmpdir}/datasets/{dataset_name}"
        backend.download(dest_path=dest_path, ext=(".jpg", ".nii.gz"))
        assert len(list(Path(dest_path).rglob("*.gz"))) == 2 * 10
        assert len(list(Path(dest_path).rglob("*.jpg"))) == 1 * 10


@mock_s3
def test_s3_backend_invalid(tmpdir):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="mybucket")
    s3 = boto3.client("s3", region_name="us-east-1")
    datasets = {}

    for dataset_name in ["foo", "bar"]:
        datasets[dataset_name] = f"datasets/{dataset_name}/"
        for i in range(10):
            key = f"datasets/{dataset_name}/invalid/native/img_{i}.nii.gz"
            s3.put_object(Bucket="mybucket", Key=key, Body="")

        backend = storage_backends.DatasetS3Backend(bucket="mybucket", prefix=f"datasets/{dataset_name}")
        assert len(backend.list_files()) == 0
        dest_path = f"{tmpdir}/datasets/{dataset_name}"
        backend.download(dest_path=dest_path, ext=(".jpg", ".gz"))
        assert len(list(Path(dest_path).rglob("*.gz"))) == 0


@mock_s3
def test_s3_backend_sublabel(tmpdir):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="mybucket")
    s3 = boto3.client("s3", region_name="us-east-1")
    datasets = {}
    labels = ["l1", "l2", "l3"]
    for dataset_name in ["foo", "bar"]:
        datasets[dataset_name] = f"datasets/{dataset_name}/"
        for spacing in ["native", "subsampled1mm"]:
            for i in range(10):
                for l in labels:
                    key = f"datasets/{dataset_name}/labelmaps/{l}/{spacing}/img_{i}_seg.nii.gz"
                    s3.put_object(Bucket="mybucket", Key=key, Body="")
                key = f"datasets/{dataset_name}/images/{spacing}/img_{i}.nii.gz"
                s3.put_object(Bucket="mybucket", Key=key, Body="")


        backend = storage_backends.DatasetS3Backend(bucket="mybucket", prefix=f"datasets/{dataset_name}")
        dest_path = f"{tmpdir}/datasets/{dataset_name}"

        for spacing in [0, 1]:
            spacing_dir = get_spacing_dirname(spacing)
            backend.download(dest_path=dest_path)
            assert len(list(Path(dest_path).rglob(f"{spacing_dir}/*.gz"))) == 4 * 10
            backend.download(dest_path=dest_path, spacing=1)
        assert len(list(Path(dest_path).rglob(f"{spacing_dir}/*.gz"))) == 4 * 10
