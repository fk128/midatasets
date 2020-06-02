import os

import yaml

from midatasets import configs
from midatasets.MIReader import MIReader



def get_available_datasets(names_only=False):
    with open(os.path.expanduser('~/.midatasets.yaml')) as f:
        datasets = yaml.load(f, Loader=yaml.FullLoader)
    if names_only:
        return [d['name'] for d in datasets['datasets']]
    else:
        return datasets['datasets']


def load_dataset(name, **kwargs):
    datasets = get_available_datasets()
    match = [d for d in datasets if d['name'] == name]
    if len(match) == 0:
        raise FileNotFoundError
    elif len(match) > 1:
        raise Exception(f'more than one match found {match}')
    else:
        match = match[0]
        match['dir_path'] = os.path.join(configs.get('root_path'), match.pop('subpath'))
        match.update(kwargs)
        return MIReader.from_dict(**match)
