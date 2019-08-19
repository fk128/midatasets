# MIDatasets #


python library to interact with nifti medical image datasets available locally on disc.

### Setup

```
cd midatasets && pip install -e .
```


### Configurations

The default configuration are
```
[DEFAULT]
root_path=/media/Datasets
images_dir=images
labelmaps_dir=labelmaps
native_images_dir=native
subsampled_images_dir_prefix=subsampled
images_crop_prefix=images_crop_
labelmaps_crop_prefix=labelmaps_crop_
```

which can be overridden by defining a file at `~/.midatasets.cfg`.
The main variable you need to change is `root_path`, which should point
to the root directory path where you store your images.

For a given dataset, it assumes that the folder structure is:

- images: `<dataset_name>/images/native/`
- labelmaps (segmentations): `<dataset_name>/labelmaps/native/`


For example, a dataset of Lung CT images would have a directory structure at
`<root_path>/lung`:

```
images/
 |_ native
    |_ lung001.nii.gz
    |_ lung002.nii.gz
    |_ ...
 |_ subsampled1mm
    |_ lung001.nii.gz
    |_ lung002.nii.gz
    |_ ...
labelmaps/
 |_ native
    |_ lung001_seg.nii.gz
    |_ lung002_seg.nii.gz
    |_ ...  
 |_ subsampled1mm
    |_ lung001_seg.nii.gz
    |_ lung002_seg.nii.gz
    |_ ...
```

To match a labelmap/segmentation with its associated image, the image and its labelmap
need to have the same prefix.

Images that haven't been resampled should be placed in the `native` folder. Any resampled
images should go in the corresponding folders, e.g. 

- 1mm isotropic: `subsampled1mm`,
- 1mm in-plane isotropic and 4mm slice thickness: `subsampled1-1-4mm`

### Adding datasets

New datasets are loaded via entry points as plugins. An example of how to
define and add additional datasets can be found in `example_dataset`.

Running `cd example_dataset && pip install -e .`  would make `LiverReader` and `LungReader`
available as imports from `midatasets.datasets`

```python
from midataset.datasets import LungReader

reader = LungReader(spacing=0)

```

`spacing=0` loads images from the `native` folder, `spacing=1`, from  `subsampled1mm`, and `spacing=[1,1,4]`, 
from `subsampled1-1-4mm`.