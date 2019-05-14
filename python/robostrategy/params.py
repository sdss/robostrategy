# @Author: Michael R. Blanton
# @Date: April 3, 2019
# @Filename: params.py
# @License: BSD 3-Clause
# @Copyright: Michael R. Blanton

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import configparser


# Class to define a singleton
class RobostrategyParamsSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(RobostrategyParamsSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RobostrategyParams(object, metaclass=RobostrategyParamsSingleton):
    def __init__(self, plan=None):
        self.reset(plan=plan)

    def reset(self, plan=None):
        cfgfile = os.path.join(os.getenv('ROBOSTRATEGY_DIR'), 'etc',
                               'robostrategy-{plan}.cfg'.format(plan=plan))
        self.cfg = configparser.ConfigParser(allow_no_value=True)
        self.cfg.optionxform = str
        self.cfg.read(cfgfile)
        self.cfg['DEFAULT'] = {'fgot_minimum': 0.5}
        self.cfg['DEFAULT'] = {'fgot_maximum': 1.5}
        self.cfg['DEFAULT'] = {'AllFields': ''}
        self.cfg['DEFAULT'] = {'Schedule': 'normal'}
