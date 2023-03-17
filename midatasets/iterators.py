from typing import List, Dict


from midatasets.mimage import MImage


class MImageIterator:
    def __init__(self, dataset, key: str, remote: bool = True):
        self.dataset = dataset
        self.key = key
        self.data = next(
            iter(self.dataset.list_files(remote=remote, grouped=True).values())
        )
        self.data = {name: value for name, value in self.data.items() if key in value}
        self.names = list(self.data.keys())

    def __getitem__(self, index) -> MImage:
        name = self.names[index]
        return MImage(
            prefix=f'{self.dataset.remote_prefix}/{self.data[name][self.key]["prefix"]}',
            bucket=self.dataset.remote_bucket,
            key=self.key,
            base_dir=self.dataset.dir_path.replace(self.dataset.remote_prefix, ""),
        )

    def __len__(self):
        return len(self.data)


class MImageMultiIterator:
    def __init__(self, dataset, keys: List[str], remote: bool = True):

        self.dataset = dataset
        self.keys = keys
        self.data = next(
            iter(self.dataset.list_files(remote=remote, grouped=True).values())
        )
        self.data = {
            name: value
            for name, value in self.data.items()
            if self._issubset(keys, value)
        }
        self.names = list(self.data.keys())

    def _issubset(self, keys: List[str], values: Dict):
        for key in keys:
            if key not in values:
                return False
        return True

    def __getitem__(self, index) -> Dict[str, MImage]:
        name = self.names[index]
        return {
            key: MImage(
                prefix=f'{self.dataset.remote_prefix}/{self.data[name][key]["prefix"]}',
                bucket=self.dataset.remote_bucket,
                key=key,
                base_dir=self.dataset.dir_path.replace(self.dataset.remote_prefix, ""),
            )
            for key in self.keys
        }

    def __len__(self):
        return len(self.data)
