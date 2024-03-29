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
        cfgfile = os.path.join(os.getenv('RSCONFIG_DIR'), 'etc',
                               'robostrategy-{plan}.cfg'.format(plan=plan))
        self.cfg = configparser.ConfigParser(allow_no_value=True)
        self.cfg.optionxform = str
        self.cfg.read_dict({'Assignment': {'fgot_minimum': 0.5,
                                           'fgot_maximum': 1.5,
                                           'offset_min_skybrightness': 0.5},
                            'Allocation': {'AllFields': '',
                                           'Schedule': 'normal',
                                           'ExtraTimeFactor': 1.0,
                                           'fClearAPO': 0.5,
                                           'fClearLCO': 0.7},
                            'Fields': {'PaCenterLCO': 270.}})
        self.cfg.read(cfgfile)
