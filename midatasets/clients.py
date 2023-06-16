import json
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from uuid import UUID, uuid3

from loguru import logger

from midatasets.schemas import Dataset, Artifact, Image
from midatasets.storage_backends import DatasetLocalBackend, DatasetS3Backend
from midatasets.utils import get_spacing_dirname


class LOCAL_NAMESPACE:
    bytes = b'midataset'


class DatasetClientBase:
    """
    Base class for dataset client
    """

    def get_datasets(self) -> List[Dataset]:
        """
        Get list of datasets
        Returns:

        """
        raise NotImplementedError

    def get_dataset(self, id: Union[int, UUID, str]) -> Dataset:
        """
        Get dataset by id
        Args:
            id: dataset id

        Returns:

        """
        raise NotImplementedError

    def get_images(self, dataset_id: Union[int, UUID, str], skip:int =0, limit:int=2000) -> List[Image]:
        """
        Get images for dataset
        Args:
            dataset_id: dataset id
            skip: skip
            limit: limit

        Returns:

        """
        raise NotImplementedError

    def _generate_uuid(self, string: str):
        """
        Generate uuid from string
        Args:
            string:  string to generate uuid from

        Returns:

        """
        return (uuid3(LOCAL_NAMESPACE, str(string)))

class APIDatasetClient(DatasetClientBase):
    """
    Use REST API as source of datasets
    """
    def __init__(self, host:Optional[str] =None, access_token: Optional[str]=None):

        if access_token is None:
            logger.error("Missing access token")

        self.host = host
        self.access_token = access_token

    def _build_request(self, path, method="GET", body=None, query=None):
        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = urljoin(self.host, path)
        if query is not None:
            url += f"?{urlencode(query)}"

        if body:
            body = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return Request(url, data=body, headers=headers, method=method)

    def _send_request(self, path, method="GET", body=None):
        request = self._build_request(path, method, body)
        with urlopen(request, timeout=120) as response:
            return response.read()

    @property
    def datasets_prefix(self):
        return "/datasets"

    def get_datasets(self) -> List[Dataset]:
        """
        Get list of datasets
        Returns:

        """
        res = self._send_request(self.datasets_prefix)
        return [Dataset(**d) for d in json.loads(res.decode("utf-8"))]

    def get_dataset(self, id: int) -> Dataset:
        """
        Get dataset by id
        Args:
            id:

        Returns:

        """
        res = self._send_request(f"{self.datasets_prefix}/{id}")
        return Dataset(**json.loads(res.decode("utf-8")))

    def get_images(self, dataset_id: int, skip=0, limit=2000) -> List[Image]:
        """
        Get images for dataset"""
        res = self._send_request(
            f"{self.datasets_prefix}/{dataset_id}/images?skip={skip}&limit={limit}"
        )
        res = json.loads(res.decode("utf-8"))

        return [Image(**d) for d in res.get("data", [])]




class LocalDatasetClient(DatasetClientBase):
    """
    Use local directory as source of datasets

    Examples:
        >>> client = LocalDatasetClient(root_dir="/data")
        >>> client.get_datasets()
        >>> client.get_dataset(id=1)
        >>> client.get_images(dataset_id=1)
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._datasets = None
        self._images = None
    def get_datasets(self) -> List[Dataset]:
        datasets = []
        for path in Path(self.root_dir).iterdir():
            datasets.append(Dataset(name=path.name, path=str(path), id=str(path.name)))
        self._datasets = {str(dataset.id): dataset for dataset in datasets}
        return datasets

    @property
    def datasets_cached(self):
        if self._datasets is None:
            self.get_datasets()
        return self._datasets

    def get_dataset(self, id: int) -> Dataset:
        try:
            return self.datasets_cached[id]
        except:
            raise KeyError(f"Dataset {id} not found")

    def _generate_uuid(self, string: str):
        return (uuid3(LOCAL_NAMESPACE, str(string)))
    def get_images(self, dataset_id: Union[int, UUID, str], skip=0, limit=2000) -> List[Image]:
        dataset = self.get_dataset(dataset_id)
        backend = DatasetLocalBackend(root_path=dataset.path)
        files = backend.list_files(grouped=True, spacing=0)
        images = []
        files = files.get("native")
        if files is None:
            return []
        for name, file in files.items():
            artifacts = []
            for key, data in file.items():
                artifacts.append(Artifact(key=key, path=data["path"], id=self._generate_uuid(data["path"])))
            images.append(Image(name=name, id=self._generate_uuid(name), artifacts=artifacts))
        return images


class S3DatasetClient(DatasetClientBase):
    """
    Use s3 as source of datasets
    """

    def __init__(self, bucket: str, prefix: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix
        self._datasets = None
        self._images = None
    def get_datasets(self) -> List[Dataset]:
        """
        Get list of datasets from s3 bucket
        Returns:

        """
        backend = DatasetS3Backend(bucket=self.bucket, prefix=self.prefix)
        datasets = []
        for name, prefix in backend.list_dirs().items():
            datasets.append(Dataset(name=name, path=f"s3://{self.bucket}/{prefix}", id=str(name)))
        self._datasets = {str(dataset.id): dataset for dataset in datasets}
        return datasets
    @property
    def datasets_cached(self):
        if self._datasets is None:
            self.get_datasets()
        return self._datasets

    def get_dataset(self, id: int) -> Dataset:
        """
        Get dataset by id
        Args:
            id:

        Returns:

        """
        return self.datasets_cached[id]


    def get_images(self, dataset_id: Union[int, UUID, str], skip=0, limit=2000) -> List[Image]:
        """
        Get images from dataset
        Args:
            dataset_id:
            skip:
            limit:

        Returns:

        """
        dataset = self.get_dataset(dataset_id)
        backend = DatasetS3Backend(bucket=self.bucket, prefix=dataset.path.split(f"s3://{self.bucket}/")[1])
        files = backend.list_files(grouped=True, spacing=0)
        images = []
        files = files.get(get_spacing_dirname(0))
        if files is None:
            return []
        for name, file in files.items():
            artifacts = []
            for key, data in file.items():
                artifacts.append(Artifact(key=key, path=data["path"], id=self._generate_uuid(data["path"])))
            images.append(Image(name=name, id=self._generate_uuid(name), artifacts=artifacts))
        return images