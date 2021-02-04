import os

from midatasets import configs
from midatasets.MIReader import MIReader


def get_available_datasets(names_only=False):
    datasets = configs['datasets']
    if names_only:
        return [d['name'] for d in datasets]
    else:
        return datasets


def get_dataset_info(name):
    datasets = configs['datasets']
    match = [d for d in datasets if d['name'] == name]
    if len(match) == 0:
        raise FileNotFoundError
    elif len(match) > 1:
        raise Exception(f'more than one match found {match}')
    else:
        return match[0]


def load_dataset(name, **kwargs):
    dataset = get_dataset_info(name)
    dataset['dir_path'] = os.path.join(configs.get('root_path'), dataset['subpath'])
    dataset.update(kwargs)
    return MIReader.from_dict(**dataset)
