import robostrategy
import roboscheduler
import kaiju
import mugatu


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
    hdr.append({'name':'KAIJUVER',
                'value':kaiju.__version__,
                'comment':'kaiju version'})
    return(hdr)
