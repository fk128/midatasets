# -*- coding: utf-8 -*-

__version__ = '0.1.0'

import os
from configparser import ConfigParser
import pkg_resources
import sys
import inspect


__config_parser = ConfigParser()
__default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'midatasets.cfg')

__config_parser.read([__default_config_path,
                                os.path.expanduser('~/.midatasets.cfg')])

configs = __config_parser['DEFAULT']


# look for plugins
plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('midatasets.datasets')
}

# add module to namespace
for name, module in plugins.items():
    sys.modules.setdefault('midatasets.' + name, module)

# get list of available classes
if plugins:
    _cls_members = inspect.getmembers(sys.modules['midatasets.datasets'], inspect.isclass)
    available_datasets = dict()
    for (name, cls) in _cls_members:
        try:
            available_datasets[cls.name] = cls
        except:
            available_datasets[name] = cls


    def get_dataset(name):
        if name in available_datasets.keys():
            return available_datasets[name]
        else:
            print('invalid dataset name from {}'.format(available_datasets.keys()))
            return None


    def get_available_datasets():
        return available_datasets.keys()

    setattr(plugins['datasets'], 'get_dataset', get_dataset)
    setattr(plugins['datasets'], 'get_available_datasets', get_available_datasets)

    __all__ = [plugins.values()]