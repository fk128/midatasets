# MIDatasets #


python library to interact with a local nifti medical image datasets.

### Setup

```
cd midatasets && pip install -e .
```


### Configurations

The default configuration are

```yaml
root_path: /media/Datasets
images_dir: images
labelmaps_dir: labelmaps
native_images_dir: native
subsampled_images_dir_prefix: subsampled
images_crop_prefix: images_crop_
labelmaps_crop_prefix: labelmaps_crop_
```

which can be overridden by defining a file at  `~/.midatasets.yaml`.
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

To add a new dataset, simply create a yaml in your home directory `~/.midatasets.yaml` file and add a new dataset entry


```yaml
root_path: /workdir/datasets/
datasets:
  - name: lung
    labels: [0,1]
    label_names: ['background', 'lung']
    subpath: lung
    aws_s3_bucket: midatasets-bucket
    aws_s3_prefix: datasets/lung
    aws_profile: myprofile

```


```python
from midataset.datasets import load_dataset

dataset = load_dataset('lung', spacing=0)

# download from s3
#dataset.download()

dataset.generate_resampled(spacing=2)
```



`spacing=0` loads images from the `native` folder, `spacing=1`, from  `subsampled1mm`, and `spacing=[1,1,4]`, 
from `subsampled1-1-4mm`.