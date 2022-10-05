import os
import robostrategy
import roboscheduler
import kaiju
import mugatu
import coordio
import fps_calibrations


def rsheader():
    hdr = list()
    hdr.append({'name':'STRATVER',
                'value':robostrategy.__version__,
                'comment':'robostrategy version'})
    hdr.append({'name':'SCHEDVER',
                'value':roboscheduler.__version__,
                'comment':'roboscheduler version'})
    hdr.append({'name':'MUGATVER',
                'value':mugatu.__version__,
                'comment':'mugatu version'})
    hdr.append({'name':'COORDVER',
                'value':coordio.__version__,
                'comment':'coordio version'})
    hdr.append({'name':'FPSCAVER',
                'value':fps_calibrations.__version__,
                'comment':'fps_calibrations version'})
    hdr.append({'name':'KAIJUVER',
                'value':kaiju.__version__,
                'comment':'kaiju version'})
    hdr.append({'name':'WOKDIR',
                'value':os.getenv('WOKCALIB_DIR'),
                'comment':'wok directory in fps_calibrations'})
    return(hdr)
