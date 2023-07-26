import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

try:
    import SimpleITK as sitk
except ImportError as e:
    sitk = e
try:
    import numpy as np
except ImportError as e:
    np = e
try:
    import pydicom as dicom
except ImportError as e:
    pydicom = e
import pandas as pd

from loguru import logger

from midatasets import configs


def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()


def get_extension(path: str):
    return "".join([s for s in Path(path).suffixes if s in configs.extensions])


def read_rtstruct(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour["color"] = structure.ROIContourSequence[i].ROIDisplayColor
        contour["number"] = structure.ROIContourSequence[i].RefdROINumber
        contour["name"] = structure.StructureSetROISequence[i].ROIName
        assert contour["number"] == structure.StructureSetROISequence[i].ROINumber
        contour["contours"] = [
            s.ContourData for s in structure.ROIContourSequence[i].ContourSequence
        ]
        contours.append(contour)
    return contours


def get_labelmap_from_rtstruct(contours, slices, image):
    from skimage.draw import polygon

    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    o_r = int(slices[0].ImageOrientationPatient[4])
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    o_c = int(slices[0].ImageOrientationPatient[0])
    spacing_c = slices[0].PixelSpacing[0]

    labelmap = np.zeros_like(image, dtype=np.uint8)

    for con in contours:
        num = int(con["number"])
        for i, c in enumerate(con["contours"]):

            nodes = np.array(c).reshape((-1, 3))
            #             print(np.abs(np.diff(nodes[:, 2])))
            #             assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_new = [round(elem, 1) for elem in z]
            try:
                z_index = z.index(nodes[0, 2])
            except ValueError:
                z_index = z_new.index(nodes[0, 2])

            r = o_r * (nodes[:, 1] - pos_r) / spacing_r
            c = o_c * (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            labelmap[rr, cc, z_index] = num

    colors = tuple(np.array([con["color"] for con in contours]) / 255.0)
    return labelmap, colors


def read_dcm_image(dcms):
    slices = [dicom.read_file(dcm) for dcm in dcms]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    return image, slices


def safe_sitk_dicom_read(img_list, *args, **kwargs):
    dir_name = os.path.dirname(img_list[0])
    s_img_list = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dir_name)
    return sitk.ReadImage(s_img_list, *args, **kwargs)


def read_tag_file(filename, img_size=(512, 512)):
    """
    read sliceomatic tag file
    :param filename:
    :param img_size:
    :return:
    """

    v = np.fromfile(filename, dtype=np.uint8)
    imh = img_size[0]
    imw = img_size[1]
    img = v[-imh * imw :].reshape(imh, imw)
    return img


def export_train_test_split(
    reader, out_dir=".", type="csv", ratio=0.66, seed=42, cv=False, n_splits=3
):
    name = reader.name

    if type == "csv":
        (
            train_x,
            train_y,
            test_x,
            test_y,
            names_x,
            names_y,
        ) = reader.get_train_test_split_labelled_images_list(
            ratio, is_paths=True, seed=seed
        )

        df = pd.DataFrame()
        df["image"] = train_x
        df["labelmap"] = train_y
        df["name"] = names_x
        df.set_index("name", inplace=True)
        df.to_csv(os.path.join(out_dir, name + "_train_image_labelmap_list.csv"))

        df = pd.DataFrame()
        df["image"] = test_x
        df["labelmap"] = test_y
        df["name"] = names_y
        df.set_index("name", inplace=True)
        df.to_csv(os.path.join(out_dir, name + "_test_image_labelmap_list.csv"))

    elif type == "txt":
        if cv:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_splits)
            for i, (train_idx, test_idx) in enumerate(kf.split(reader.image_list)):
                train_x = np.array(reader.image_list)[train_idx]
                train_y = np.array(reader.labelmap_list)[train_idx]
                test_x = np.array(reader.image_list)[test_idx]
                test_y = np.array(reader.labelmap_list)[test_idx]
                with open(
                    os.path.join(
                        out_dir,
                        name
                        + "_train_imagelist_f"
                        + str(i + 1)
                        + "of"
                        + str(n_splits)
                        + ".txt",
                    ),
                    "w+",
                ) as file:
                    for img_path in train_x:
                        file.write(img_path + "\n")

                with open(
                    os.path.join(
                        out_dir,
                        name
                        + "_train_labellist_f"
                        + str(i + 1)
                        + "of"
                        + str(n_splits)
                        + ".txt",
                    ),
                    "w+",
                ) as file:
                    for img_path in train_y:
                        file.write(img_path + "\n")

                with open(
                    os.path.join(
                        out_dir,
                        name
                        + "_test_imagelist_f"
                        + str(i + 1)
                        + "of"
                        + str(n_splits)
                        + ".txt",
                    ),
                    "w+",
                ) as file:
                    for img_path in test_x:
                        file.write(img_path + "\n")

                with open(
                    os.path.join(
                        out_dir,
                        name
                        + "_test_labellist_f"
                        + str(i + 1)
                        + "of"
                        + str(n_splits)
                        + ".txt",
                    ),
                    "w+",
                ) as file:
                    for img_path in test_y:
                        file.write(img_path + "\n")

        else:
            (
                train_x,
                train_y,
                test_x,
                test_y,
                names_x,
                names_y,
            ) = reader.get_train_test_split_labelled_images_list(
                ratio, is_paths=True, seed=seed
            )
            with open(
                os.path.join(out_dir, name + "_train_imagelist.txt"), "w+"
            ) as file:
                for img_path in train_x:
                    file.write(img_path + "\n")

            with open(
                os.path.join(out_dir, name + "_train_labellist.txt"), "w+"
            ) as file:
                for img_path in train_y:
                    file.write(img_path + "\n")

            with open(
                os.path.join(out_dir, name + "_test_imagelist.txt"), "w+"
            ) as file:
                for img_path in test_x:
                    file.write(img_path + "\n")

            with open(
                os.path.join(out_dir, name + "_test_labellist.txt"), "w+"
            ) as file:
                for img_path in test_y:
                    file.write(img_path + "\n")

    elif type == "csv_all":
        df = pd.DataFrame()
        df["image"] = reader.image_list
        df["labelmap"] = reader.labelmap_list
        df["name"] = reader.get_image_names()
        df.set_index("name", inplace=True)
        df.to_csv(os.path.join(out_dir, name + "_image_labelmap_list.csv"))


def get_spacing_dirname(spacing):
    if spacing is None:
        return None
    if type(spacing) in [int, float]:
        if isinstance(spacing, float) and spacing.is_integer():
            spacing = int(spacing)
        spacing = [spacing]

    if sum(spacing) <= 0:
        spacing_dirname = configs.native_images_dir
    elif len(spacing) == 1:
        spacing_dirname = configs.subsampled_dir_prefix + str(spacing[0]) + "mm"
    else:
        spacing_str = ""
        for s in spacing:
            spacing_str += str(s) + "-"
        spacing_str = spacing_str[:-1]
        spacing_dirname = configs.subsampled_dir_prefix + spacing_str + "mm"

    return spacing_dirname


def get_key_dirname(key: str) -> str:
    """
    pluralise base type of key
    :param key:
    :return: dirname
    """
    parts = key.split("/")
    if not parts[0].endswith("s"):
        parts[0] = parts[0] + "s"
    return "/".join(parts)


def strip_extension(path):
    path = Path(path)
    remove = []
    for e in path.suffixes:
        if e in {".jpg", ".jpeg", ".nii", ".gz", ".json", ".yaml", ".csv", ".nrrd"}:
            remove.append(e)

    return str(path).rstrip("".join(path.suffixes))


def parse_filepaths(filepaths: List, root_prefix: str):
    # # find common suffix
    #
    if len(filepaths) > 1:
        suffix = os.path.commonprefix([c["path"][::-1] for c in filepaths])[::-1]
    dirname_to_datatype = {v["dirname"]: v["name"] for v in configs.data_types}

    parsed_filepaths = []
    for file in filepaths:
        prefix = str(Path(file["path"]).relative_to(root_prefix))

        try:
            base, spacing, filename = prefix.rsplit("/", 2)
            base = base.split("/", 1)
            if len(base) > 1:
                data_type_dirname, label = base
            else:
                data_type_dirname, label = base[0], None
        except:
            logger.error(f"Failed to parse path {prefix}")
            continue

        if data_type_dirname not in dirname_to_datatype:
            logger.error(
                f"Invalid data_type {data_type_dirname} from acceptable {dirname_to_datatype.keys()}"
            )
            continue
        data_type = dirname_to_datatype[data_type_dirname]

        if len(filepaths) > 1:
            filename = filename.replace(suffix, "")
        filename = strip_extension(filename)

        image_key = f"{data_type}/{label}" if label else data_type
        parsed_filepaths.append(
            {
                "spacing": spacing,
                "path": file["path"],
                "filename": filename,
                "key": image_key,
                "prefix": prefix,
                "last_modified": file.get("last_modified", None),
                "data_type": data_type,
            }
        )
    return parsed_filepaths


def find_longest_matching_name(name, filenames):
    longest_name = ""
    for existing_name in filenames:
        if existing_name in name and len(existing_name) > len(
            longest_name
        ):  # check if subset of existing name
            longest_name = existing_name
    if len(longest_name) > 0:
        name = longest_name
    return name


def grouped_by_name(files_iter: Dict[str, List], root_prefix: str) -> Dict:
    """
    group files by spacing/name/image_type
    :param files_iter:
    :param ext:
    :param dataset_path:
    :param key:
    :return:
    """

    files = defaultdict(dict)
    for data_type, file_list in files_iter.items():
        file_list = parse_filepaths(file_list, root_prefix=root_prefix)
        for file in file_list:
            spacing = file["spacing"]
            name = file["filename"]
            image_key = file["key"]
            if spacing not in files:
                files[spacing] = defaultdict(dict)

            if data_type != configs.primary_type:
                name = find_longest_matching_name(name, filenames=files[spacing].keys())

            files[spacing][name][image_key] = file
    return {k: dict(v) for k, v in files.items()}


def grouped_by_key(files_iter: Dict[str, List], root_prefix: str) -> Dict:
    files = defaultdict(list)
    for data_type, file_list in files_iter.items():
        file_list = parse_filepaths(file_list, root_prefix=root_prefix)
        for file in file_list:
            image_key = file.pop("key")
            files[image_key].append(file)
    return dict(files)


def grouped_files(
    files_iter: Dict[str, List], root_prefix: str, by: str = "name"
) -> Dict:
    if by == "name":
        return grouped_by_name(files_iter, root_prefix)
    elif by == "key":
        return grouped_by_key(files_iter, root_prefix)
    else:
        raise NotImplementedError


def create_dummy_dataset(name: str, labels: List[str], root_path: str, num: int = 10):
    p = Path(root_path)

    for l in labels:
        (p / name / "labelmaps" / l / "native").mkdir(exist_ok=True, parents=True)
    (p / name / "images" / "native").mkdir(exist_ok=True, parents=True)
    for i in range(num):
        for l in labels:
            (p / name / "labelmaps" / l / "native" / f"image_{i}_seg.nii.gz").touch()
        (p / name / "images" / "native" / f"image_{i}.nii.gz").touch()
    return Path(root_path) / name


def create_dummy_s3_dataset(
    name: str,
    labels: List[str],
    bucket_name: str,
    prefix: str = "datasets",
    region_name: str = "us-east-1",
    num: int = 10,
    image_ext: str = ".nii.gz",
    labelmap_ext: str = ".nii.gz",
):
    import boto3

    conn = boto3.resource("s3", region_name=region_name)
    conn.create_bucket(Bucket=bucket_name)
    s3 = boto3.client("s3", region_name=region_name)

    for i in range(num):
        for l in labels:
            key = f"{prefix}/{name}/labelmaps/{l}/native/img_{i}{labelmap_ext}"
            s3.put_object(Bucket=bucket_name, Key=key, Body="")
        key = f"{prefix}/{name}/images/native/img_{i}{image_ext}"
        s3.put_object(Bucket=bucket_name, Key=key, Body="")
