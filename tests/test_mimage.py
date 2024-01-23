from moto import mock_s3

from midatasets.mimage import MImage
from midatasets.utils import create_dummy_s3_dataset


def test_mimage_s3_path():
    for image_data in [
        dict(key="labelmap/airways", resolution="native", key_dir="labelmaps/airways"),
        dict(key="image", resolution="native", key_dir="images"),
        dict(
            key="labelmap/sublabel1/sublabel2",
            resolution="native",
            key_dir="labelmaps/sublabel1/sublabel2",
        ),
    ]:
        path = f"s3://midatasets/datasets/test/{image_data['key_dir']}/{image_data['resolution']}/image.nii.gz"
        image = MImage.from_s3_path(s3_path=path, key=image_data["key"])
        assert image.key_dir == image_data["key_dir"]


def test_mimage_local_path():
    for image_data in [
        dict(key="labelmap/airways", resolution="native", key_dir="labelmaps/airways"),
        dict(key="image", resolution="native", key_dir="images"),
        dict(
            key="labelmap/sublabel1/sublabel2",
            resolution="native",
            key_dir="labelmaps/sublabel1/sublabel2",
        ),
    ]:
        path = f"/tmp/midatasets/datasets/test/{image_data['key_dir']}/{image_data['resolution']}/image.nii.gz"
        image = MImage.from_local_path(local_path=path, key=image_data["key"])
        assert image.key_dir == image_data["key_dir"]


def test_mimage_derive():
    path = f"/tmp/midatasets/datasets/test/images/native/image.nii.gz"
    image = MImage.from_local_path(local_path=path, key="image", bucket="test")
    new_path = f"/tmp/labelmap.nrrd"
    new_image = image.derive(new_key="labelmap/test", local_path=new_path)
    assert new_image.s3_path.endswith(".nrrd")

@mock_s3
def test_header():
    paths = create_dummy_s3_dataset(name="dataset", labels=["l1"], bucket_name="test", prefix="datasets")
    assert MImage(bucket=paths[0]["bucket"], prefix=paths[0]["prefix"], key="labelmap/l1").header is not None
