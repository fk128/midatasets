# Home 


python library to interact with local or remote nifti medical images.

### Setup

```
pip install -e .
```

### Dir structure

For a given dataset, it assumes that the folder structure is:

- images: `<dataset_name>/images/native/`
- labelmaps (segmentations): `<dataset_name>/labelmaps/native/`


For example, a dataset of Lung CT images would have a directory structure at
`<root_path>/lung`:

```
images/
 |_ native/
    |_ lung001.nii.gz
    |_ lung002.nii.gz
    |_ ...
 |_ subsampled1mm/
    |_ lung001.nii.gz
    |_ lung002.nii.gz
    |_ ...
labelmaps/
 |_ l1
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


## Example


### Using local dataset

```python
from midatasets import MIDataset
from midatasets.clients import LocalDatasetClient

client = LocalDatasetClient(root_dir="/data/datasets")

dataset = MIDataset(dataset_id="lung", client=client)
for images in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
    print(images)
```

### Using dataset on S3

```python
from midatasets import MIDataset
from midatasets.clients import S3DatasetClient

client = S3DatasetClient(bucket="test", prefix="datasets")

dataset = MIDataset(dataset_id="lung", client=client)

dataset.download()
## or download specific keys
dataset.download(keys=["image"])

for images in dataset.iterate_keys(keys=["image", "labelmap/l1"], spacing=0):
    print(images)
    ## or download specific image
    images["image"].download()
```

