# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import yaml

# Inits the logging system. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(name).
# from .utils import log


def merge(user, default):
    """Merges a user configuration with the default one."""

    if isinstance(user, dict) and isinstance(default, dict):
        for kk, vv in default.items():
            if kk not in user:
                user[kk] = vv
            else:
                user[kk] = merge(user[kk], vv)

    return user


NAME = 'robostrategy'


# Loads config (general configuration)
config_file = os.path.dirname(__file__) + '/etc/{0}.yml'.format(NAME)
try:
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
except AttributeError:
    config = yaml.load(open(config_file))

# If there is a custom configuration file, updates the defaults using it.
custom_config_fn = os.path.expanduser('~/.{0}/{0}.yml'.format(NAME))
if os.path.exists(custom_config_fn):
    try:
        config = merge(yaml.load(open(custom_config_fn),
                                 Loader=yaml.FullLoader), config)
    except AttributeError:
        config = merge(yaml.load(open(custom_config_fn), config))


__version__ = '1.3.1update'
