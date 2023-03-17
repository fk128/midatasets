from midatasets.mimage import MImage


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
