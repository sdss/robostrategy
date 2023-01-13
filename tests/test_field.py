import pytest

import numpy as np
import robostrategy.field as field
import roboscheduler.cadence as cadence


def targets(f=None, nt=100, seed=100, rsid_start=0, ra=None, dec=None,
            delta_ra=0., delta_dec=0., 
            category='science', fiberType='BOSS', cadence='single_1x1'):
    t_dtype = np.dtype([('ra', np.float64),
                        ('dec', np.float64),
                        ('epoch', np.float64),
                        ('delta_ra', np.float64),
                        ('delta_dec', np.float64),
                        ('priority', np.int32),
                        ('category', np.unicode_, 30),
                        ('cadence', np.unicode_, 30),
                        ('catalogid', np.int64),
                        ('magnitude', np.int32, 10),
                        ('fiberType', 'U8'),
                        ('rsid', np.int64)])
    t = np.zeros(nt, dtype=t_dtype)
    np.random.seed(seed)
    if(ra is None):
        t['ra'] = 180. - 1.5 + 3.0 * np.random.random(nt)
    else:
        t['ra'] = ra
    if(dec is None):
        t['dec'] = 0. - 1.5 + 3.0 * np.random.random(nt)
    else:
        t['dec'] = dec
    t['magnitude'] = 17.
    t['epoch'] = 2000.
    t['delta_ra'] = delta_ra
    t['delta_dec'] = delta_dec
    t['priority'] = 1
    t['category'] = category
    t['cadence'] = cadence
    t['fiberType'] = fiberType
    t['catalogid'] = np.arange(nt, dtype=np.int64)
    t['rsid'] = np.arange(nt, dtype=np.int64) + rsid_start
    f.targets_fromarray(t)
    return


def add_cadence_single_nxm(n=1, m=1):
    clist = cadence.CadenceList()
    clist.add_cadence(name='single_{n}x{m}'.format(n=n, m=m),
                      nepochs=n,
                      skybrightness=[1.] * n,
                      delta=[-1.] * n,
                      delta_min=[-1.] * n,
                      delta_max=[-1.] * n,
                      nexp=[m] * n,
                      max_length=[0.] * n,
                      min_moon_sep=[15.] * n,
                      min_deltav_ks91=[-2.5] * n,
                      min_twilight_ang=[8] * n,
                      max_airmass=[2.] * n,
                      obsmode_pk=['bright_time'] * n)
    return

def test_field_init():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    assert f.field_cadence.nepochs == 1

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x1')
    assert f.field_cadence.nepochs == 2

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                          field_cadence='single_2x2')
    assert f.field_cadence.nepochs == 2

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    assert f.radius == 0.95

    f = field.Field(racen=180., deccen=0., pa=45, observatory='apo',
                    field_cadence='single_2x2')
    assert f.radius == 1.5
    assert len(f.robotgrids) == 4


def test_radec():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    x, y, z = f.radec2xyz(ra=np.array([180.5]),
                          dec=np.array([0.5]), fiberType=['APOGEE'])
    ra, dec = f.xy2radec(x=x, y=y, fiberType=['APOGEE'])
    assert np.abs(ra - 180.5) < 1.e-5
    assert np.abs(dec - 0.5) < 1.e-5
    return


def test_target_fromarray():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    targets(f)
    return


def test_target_offset():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    f = field.Field(racen=180., deccen=0., pa=0., observatory='apo',
                    field_cadence='single_1x1')
    targets(f, nt=50)
    targets(f, delta_ra=10., delta_dec=10., nt=50, rsid_start=50)
    
    xoff = f.targets['x'][0:50] - f.targets['x'][50:100]
    yoff = f.targets['y'][0:50] - f.targets['y'][50:100]

    xoffarcsec = xoff / 217.736 * 3600.
    yoffarcsec = yoff / 217.736 * 3600.

    ibad = np.where((xoffarcsec - 10.) > 0.1)[0]
    assert len(ibad) == 0

    ibad = np.where((yoffarcsec - 10.) > 0.1)[0]
    assert len(ibad) == 0

    return

def test_flags():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')

    targets(f)

    f.set_flag(rsid=10, flagname='NOT_INCADENCE')
    assert f.check_flag(rsid=10, flagname='NOT_INCADENCE')[0] == True
    assert f.check_flag(rsid=10, flagname='NOT_COVERED')[0] == False
    assert f.get_flag_names(f.assignments['rsflags'][f.rsid2indx[10]]) == ['NOT_INCADENCE']


def test_assign_robot_epoch():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500)

    rid = list(f.robotgrids[0].robotDict.keys())[10]
    tids = f.robotgrids[0].robotDict[rid].validTargetIDs
    tid = tids[0]

    f.assign_robot_epoch(rsid=tid, robotID=rid, epoch=0, nexp=1)
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=2)[0] == False
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=2)[0] == True
    assert f.assignments['robotID'][f.rsid2indx[tid], 0] == rid
    assert f.assignments['robotID'][f.rsid2indx[tid], 1] == -1
    assert f.assignments['robotID'][f.rsid2indx[tid], 2] == -1
    assert f.assignments['robotID'][f.rsid2indx[tid], 3] == -1
    rindx = f.robotID2indx[rid]
    assert f._robot2indx[rindx, 0] == f.rsid2indx[tid]
    assert f._robot2indx[rindx, 1] == -1
    assert f._robot2indx[rindx, 2] == -1
    assert f._robot2indx[rindx, 3] == -1

    f.assign_robot_epoch(rsid=tid, robotID=rid, epoch=1, nexp=2)
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=2)[0] == False
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=1)[0] == False
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=2)[0] == False
    assert f.assignments['robotID'][f.rsid2indx[tid], 2] == rid
    assert f.assignments['robotID'][f.rsid2indx[tid], 3] == rid
    rindx = f.robotID2indx[rid]
    assert f._robot2indx[rindx, 2] == f.rsid2indx[tid]
    assert f._robot2indx[rindx, 3] == f.rsid2indx[tid]

    f.unassign_epoch(rsid=tid, epoch=0)
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=2)[0] == True
    assert f.assignments['robotID'][f.rsid2indx[tid], 0] == -1
    assert f.assignments['robotID'][f.rsid2indx[tid], 1] == -1


def test_available_epochs():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500, seed=102)

    for rid in f.robotgrids[0].robotDict:
        tids = f.robotgrids[0].robotDict[rid].validTargetIDs
        if(len(tids) > 1):
            tid0 = tids[0]
            tid1 = tids[1]

            av = f.available_epochs(rsid=tid0, epochs=[0, 1], nexps=[2, 2])

            f.assign_robot_epoch(rsid=tid0, robotID=rid, epoch=0, nexp=1)

            print(f.assignments['robotID'][f.rsid2indx[tid0]])

            av = f.available_epochs(rsid=tid1, epochs=[0, 1], nexps=[2, 2])
            ar = av['availableRobotIDs']
            st = av['statuses']

            assert rid not in ar[0]
            assert rid in ar[1]
            for i, car in enumerate(ar[0]):
                assert st[1][i].assignable[0] == True
                assert st[1][i].assignable[1] == True
            for i, car in enumerate(ar[1]):
                assert st[1][i].assignable[0] == True
                assert st[1][i].assignable[1] == True

            rindx = f.robotID2indx[rid]
            assert f._robot2indx[rindx, 0] == f.rsid2indx[tid0]

            f.unassign(rsids=[tid0])
            f.unassign(rsids=[tid1])

            assert f._robot2indx[rindx, 0] == - 1


def test_assign_epochs():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500, seed=102)

    for rid in f.robotgrids[0].robotDict:
        tids = f.robotgrids[0].robotDict[rid].validTargetIDs
        if(len(tids) > 2):
            tid0 = tids[0]
            indx0 = f.rsid2indx[tid0]
            tid1 = tids[1]
            indx1 = f.rsid2indx[tid1]
            tid2 = tids[2]
            indx2 = f.rsid2indx[tid2]

            ok = f.assign_epochs(rsid=tid0, epochs=[0, 1], nexps=[1, 1])
            assert ok is True

            ok = f.assign_epochs(rsid=tid1, epochs=[0, 1], nexps=[1, 1])
            assert ok is True

            ok = f.assign_epochs(rsid=tid2, epochs=[0, 1], nexps=[1, 1])
            assert ((ok is False) |
                    ((f.assignments['robotID'][indx2, 0] !=
                      f.assignments['robotID'][indx1, 0]) &
                     (f.assignments['robotID'][indx2, 0] !=
                      f.assignments['robotID'][indx0, 0])))

            return
            
    assert 1 == 0
    return

def test_clear_assignments():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500, seed=102)

    f.assign_science()
    f.decollide_unassigned()

    f.clear_assignments()

    assert len(np.where(f._robot2indx.flatten() >= 0)[0]) == 0
    assert len(np.where(f.assignments['robotID'].flatten() >= 0)[0]) == 0
    for rg in f.robotgrids:
        for robotID in rg.robotDict:
            assert rg.robotDict[robotID].isAssigned() is False
    for c in f.calibrations:
        for cnum in f.calibrations[c]:
            assert cnum == 0


def test_append_targets():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=50)
    targets(f, nt=50, rsid_start=50, ra=f.targets['ra'],
            dec=f.targets['dec'] + 0.005)

    assert len(f.targets) == 100
    assert len(f.assignments) == 100
    assert f.assignments['robotID'].shape == (100, 4)
    assert len(f.robotgrids[0].targetDict.keys()) == 100
    for i, t in enumerate(f.targets):
        assert f.rsid2indx[t['rsid']] == i


def test_append_targets_after_assign():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=500)

    f.assign_science()

    assert f.validate() == 0
    isass = np.array([f.robotgrids[0].robotDict[x].isAssigned() for x in f.robotgrids[0].robotDict],
                     dtype=int)
    print(isass.sum())

    f.decollide_unassigned()

    assert f.validate() == 0

    targets(f, nt=500, rsid_start=500, ra=f.targets['ra'],
            dec=f.targets['dec'] + 0.005)

    isass = np.array([f.robotgrids[0].robotDict[x].isAssigned() for x in f.robotgrids[0].robotDict],
                     dtype=int)
    print(isass.sum())

    f.decollide_unassigned()

    assert f.validate() == 0

    assert len(f.targets) == 1000
    assert len(f.assignments) == 1000
    assert f.assignments['robotID'].shape == (1000, 4)
    assert len(f.robotgrids[0].targetDict.keys()) == 1000
    for i, t in enumerate(f.targets):
        assert f.rsid2indx[t['rsid']] == i

    for i in np.arange(500):
        assert f.assignments['satisfied'][i] == f.assignments['satisfied'][i + 500]
    return


def test_equiv():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=150)
    targets(f, nt=150, rsid_start=150, ra=f.targets['ra'],
            dec=f.targets['dec'])

    f.assign_science()
    f.decollide_unassigned()
    
    igot = np.where(f.assignments['robotID'] >= 0)
    ibad = np.where(f.assignments['robotID'][igot] !=
                    f.assignments['equivRobotID'][igot])[0]
    assert len(ibad) == 0
    
    for i in np.arange(150):
        for iexp in np.arange(4):
            if(f.assignments['robotID'][i, iexp] >= 0):
                assert f.assignments['robotID'][i, iexp] == f.assignments['equivRobotID'][i + 150, iexp]
            if(f.assignments['robotID'][i + 150, iexp] >= 0):
                assert f.assignments['robotID'][i + 150, iexp] == f.assignments['equivRobotID'][i, iexp]
        
    return

def test_satisfied_1():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=1)
    add_cadence_single_nxm(n=1, m=2)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=2)
    targets(f, nt=2, rsid_start=2, ra=f.targets['ra'][0:2], dec=f.targets['dec'][0:2])
    targets(f, nt=2, rsid_start=4, ra=f.targets['ra'][0:2], dec=f.targets['dec'][0:2])
    targets(f, nt=2, rsid_start=6, ra=f.targets['ra'][0:2], dec=f.targets['dec'][0:2])

    # Should be [109, 110, 127]
    rids = f.mastergrid.targetDict[0].validRobotIDs

    rid = rids[0]
    f.assign_robot_exposure(rsid=0, robotID=rid, iexp=0)

    assert f.assignments['robotID'][0, 0] == rid
    assert f.assignments['equivRobotID'][0, 0] == rid
    assert f.assignments['equivRobotID'][2, 0] == rid
    assert f.assignments['equivRobotID'][4, 0] == rid
    assert f.assignments['equivRobotID'][6, 0] == rid
    
    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] > 0

    f.unassign(rsids=[0])

    assert f.assignments['robotID'][0, 0] == -1
    assert f.assignments['equivRobotID'][0, 0] == -1
    assert f.assignments['equivRobotID'][2, 0] == -1
    assert f.assignments['equivRobotID'][4, 0] == -1
    assert f.assignments['equivRobotID'][6, 0] == -1

    assert f.assignments['satisfied'][0] == 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=4, robotID=rid, iexp=0)

    assert f.assignments['robotID'][4, 0] == rid
    assert f.assignments['equivRobotID'][0, 0] == rid
    assert f.assignments['equivRobotID'][2, 0] == rid
    assert f.assignments['equivRobotID'][4, 0] == rid
    assert f.assignments['equivRobotID'][6, 0] == rid
    
    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] > 0

    f.unassign(rsids=[4])

    assert f.assignments['robotID'][0, 0] == -1
    assert f.assignments['equivRobotID'][0, 0] == -1
    assert f.assignments['equivRobotID'][2, 0] == -1
    assert f.assignments['equivRobotID'][4, 0] == -1
    assert f.assignments['equivRobotID'][6, 0] == -1

    assert f.assignments['satisfied'][0] == 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    return

def test_satisfied_2():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=1, m=2)
    add_cadence_single_nxm(n=2, m=1)
    clist.cadences['single_2x1'].delta = [0., 30.]
    clist.cadences['single_2x1'].delta_min = [0., 20.]
    clist.cadences['single_2x1'].delta_max = [0., 40.]
    add_cadence_single_nxm(n=2, m=2)
    clist.cadences['single_2x2'].delta = [0., 30.]
    clist.cadences['single_2x2'].delta_min = [0., 20.]
    clist.cadences['single_2x2'].delta_max = [0., 40.]

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=2, cadence='single_1x1')
    targets(f, nt=2, rsid_start=2, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_1x2')
    targets(f, nt=2, rsid_start=4, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_2x1')
    targets(f, nt=2, rsid_start=6, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_2x2')

    # Should be [109, 110, 127]
    rids = f.mastergrid.targetDict[0].validRobotIDs

    rid = rids[0]
    f.assign_robot_exposure(rsid=2, robotID=rid, iexp=0)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=4, robotID=rid, iexp=1)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=6, robotID=rid, iexp=2)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=0, robotID=rid, iexp=3)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] > 0

    assert f.validate() == 0

    f.unassign(rsids=[2])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[4])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[6])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[0])

    assert f.assignments['satisfied'][0] == 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    assert f.validate() == 0

    return

def test_satisfied_3():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=1, m=2)
    add_cadence_single_nxm(n=2, m=1)
    clist.cadences['single_2x1'].delta = [0., 30.]
    clist.cadences['single_2x1'].delta_min = [0., 20.]
    clist.cadences['single_2x1'].delta_max = [0., 40.]
    add_cadence_single_nxm(n=2, m=2)
    clist.cadences['single_2x2'].delta = [0., 30.]
    clist.cadences['single_2x2'].delta_min = [0., 20.]
    clist.cadences['single_2x2'].delta_max = [0., 40.]

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    targets(f, nt=2, cadence='single_1x1', seed=102)
    targets(f, nt=2, rsid_start=2, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_1x2')
    targets(f, nt=2, rsid_start=4, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_2x1')
    targets(f, nt=2, rsid_start=6, ra=f.targets['ra'][0:2],
            dec=f.targets['dec'][0:2], cadence='single_2x2')

    # Should be [109, 110, 127]
    rids = f.mastergrid.targetDict[0].validRobotIDs

    f.assign_robot_exposure(rsid=2, robotID=rids[0], iexp=0)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    print(rids)
    f.assign_robot_exposure(rsid=4, robotID=rids[2], iexp=1)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=6, robotID=rids[1], iexp=2)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] == 0

    f.assign_robot_exposure(rsid=0, robotID=rids[2], iexp=3)

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] > 0

    assert f.validate() == 0

    f.unassign(rsids=[2])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] > 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[4])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] > 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[6])

    assert f.assignments['satisfied'][0] > 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    f.unassign(rsids=[0])

    assert f.assignments['satisfied'][0] == 0
    assert f.assignments['satisfied'][2] == 0
    assert f.assignments['satisfied'][4] == 0
    assert f.assignments['satisfied'][6] == 0

    assert f.validate() == 0

    return

def test_collisions():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    ntot = 1000
    targets(f, nt=ntot, seed=101)
    targets(f, nt=ntot, rsid_start=ntot, ra=f.targets['ra'],
            dec=f.targets['dec'] + 0.005)

    rid1 = list(f.robotgrids[0].robotDict.keys())[100]
    tids1 = f.robotgrids[0].robotDict[rid1].validTargetIDs
    tid1 = tids1[0]

    rids1 = np.array(f.robotgrids[0].targetDict[tids1[0]].validRobotIDs)
    tid2 = tid1 + ntot
    rid2 = rids1[rids1 != rid1][0]

    f.assign_robot_epoch(rsid=tid1, robotID=rid1, epoch=0, nexp=2)
    f.assign_robot_epoch(rsid=tid2, robotID=rid2, epoch=0, nexp=2)
    assert f.robotgrids[0].isCollided(rid1) is True
    assert f.robotgrids[0].isCollided(rid2) is True


def test_assign_science():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    ntot = 400
    targets(f, nt=ntot, seed=101)

    f.assign_science()
    f.decollide_unassigned()
    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0
    for rg in f.robotgrids:
        for robotID in rg.robotDict:
            assert rg.isCollided(robotID) is False


def test_lock():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=3, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_3x1')
    ntot = 2000
    targets(f, nt=ntot, seed=101)

    f._robot_locked[10, 0:2] = True
    f._robot_locked[30, 1:] = True
    f._robot_locked[400, 1] = True

    f.assign_science()
    f.decollide_unassigned()
    assert f.validate() == 0

    assert f._robot2indx[10, 0] == -1
    assert f._robot2indx[10, 1] == -1
    assert f._robot2indx[30, 1] == -1
    assert f._robot2indx[30, 2] == -1
    assert f._robot2indx[400, 1] == -1
    return


def test_force():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    ntot = 400
    targets(f, nt=ntot, seed=101)

    robotID = list(f.robotgrids[0].robotDict.keys())[10]
    assert 10 not in f.robotgrids[0].robotDict[robotID].validTargetIDs
    
    f.assign_robot_exposure(robotID=robotID, rsid=10, iexp=0, force=True)
    assert f.assignments['robotID'][f.rsid2indx[10]] == robotID
    return


def test_assign():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')

    ntot = 600
    targets(f, nt=ntot, seed=101)
    ntot = 100
    targets(f, nt=ntot, seed=102, category='boss_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=103, category='boss_sky',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=104, category='apogee_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=105, category='apogee_sky',
            rsid_start=f.targets['rsid'].max() + 1)

    f.assign()

    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0


def test_available_exposures():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=3, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_3x1')

    ntot = 1200
    targets(f, nt=ntot, seed=101)
    ntot = 100
    targets(f, nt=ntot, seed=102, category='boss_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=103, category='boss_sky',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=104, category='apogee_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=105, category='apogee_sky',
            rsid_start=f.targets['rsid'].max() + 1)

    f.assign()

    iun = np.where(f.targets['within'] &
                   (f.assignments['assigned'] == 0))[0]
    for i in iun:
        rsid = f.targets['rsid'][i]
        for robotID in f.robotgrids[0].targetDict[rsid].validRobotIDs:
            status = f.available_robot_exposures(robotID=robotID,
                                                 rsid=rsid)
            iassignable = status.assignable_exposures()
            for iexp in iassignable:
                assigned = (f.robotgrids[iexp].robotDict[robotID].isAssigned() > 0)
                if(assigned):
                    targetID = f.robotgrids[iexp].robotDict[robotID].assignedTargetID
                    spare = f._is_spare(rsid=targetID, iexps=iexp) > 0
                else:
                    spare = False
                collided, fc, gc, cs = f.robotgrids[iexp].wouldCollideWithAssigned(robotID, rsid)
                if(collided & (len(cs) > 0)):
                    for c in cs:
                        cspare = f._is_spare(rsid=c, iexps=iexp)
                else:
                    cspare = False
                ok = (((not assigned) | spare) & ((not collided) | cspare))
                assert ok
                f.unassign_assignable(status=status, iexp=iexp)
                f.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp)

    return


def test_assign_noallgrids():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1', allgrids=False)

    ntot = 600
    targets(f, nt=ntot, seed=101)
    ntot = 100
    targets(f, nt=ntot, seed=102, category='boss_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=103, category='boss_sky',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=104, category='apogee_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=105, category='apogee_sky',
            rsid_start=f.targets['rsid'].max() + 1)

    f.assign()

    assert f.validate() == 1  # the 1 problem being allgrids=False
    assert f.assignments['assigned'].sum() > 0


def test_clear():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')

    ntot = 600
    targets(f, nt=ntot, seed=101)
    ntot = 100
    targets(f, nt=ntot, seed=102, category='boss_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=103, category='boss_sky',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=104, category='apogee_standard',
            rsid_start=f.targets['rsid'].max() + 1)
    ntot = 100
    targets(f, nt=ntot, seed=105, category='apogee_sky',
            rsid_start=f.targets['rsid'].max() + 1)

    f.assign_science()
    f.decollide_unassigned()

    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0

    nassigned = f.assignments['assigned'].sum()
    assignments = np.copy(f.assignments)
    _robot2indx = np.copy(f._robot2indx)

    f.clear_field_cadence()
    assert f.field_cadence is None

    f.set_field_cadence('single_1x1')
    f.assign_science()
    f.decollide_unassigned()

    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0
    assert f.assignments['assigned'].sum() == nassigned
    for indx, aorig, anew in zip(range(len(assignments)),
                                 assignments, f.assignments):
        for n in aorig.dtype.names:
            assert aorig[n] == anew[n]


def test_assign_apogee():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    ntot = 400
    targets(f, nt=ntot, seed=101, fiberType='APOGEE')

    f.assign_science()
    f.decollide_unassigned()
    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0
    for r in f.robotgrids[0].robotDict:
        if(f.robotgrids[0].robotDict[r].assignedTargetID >= 0):
            assert f.robotgrids[0].robotDict[r].hasApogee


def test_count():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)
    add_cadence_single_nxm(n=2, m=1)
    add_cadence_single_nxm(n=2, m=2)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x1')
    ntot = 400
    targets(f, nt=ntot, seed=101)
    targets(f, nt=ntot, seed=101, rsid_start=ntot)

    f.assign_science()
    f.decollide_unassigned()
    for itarget, target in enumerate(f.targets[0:ntot]):
        igot1 = np.where(f.assignments['robotID'][itarget, :] >= 0)[0]
        igot2 = np.where(f.assignments['robotID'][itarget + ntot, :] >= 0)[0]
        assert len(igot1) + len(igot2) == f.assignments['nexps'][itarget]
        assert len(igot1) + len(igot2) == f.assignments['nepochs'][itarget]

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    ntot = 400
    targets(f, nt=ntot, seed=101, cadence='single_2x1')
    targets(f, nt=ntot, seed=101, rsid_start=ntot, cadence='single_2x1')

    f.assign_science()
    f.decollide_unassigned()
    for itarget, target in enumerate(f.targets[0:ntot]):
        igot1 = np.where(f.assignments['robotID'][itarget, :] >= 0)[0]
        igot2 = np.where(f.assignments['robotID'][itarget + ntot, :] >= 0)[0]
        epochs = set()
        for cgot in igot1:
            epochs.add(f.field_cadence.epochs[cgot])
        for cgot in igot2:
            epochs.add(f.field_cadence.epochs[cgot])
        nepochs = len(epochs)
        assert len(igot1) + len(igot2) == f.assignments['nexps'][itarget]
        assert nepochs == f.assignments['nepochs'][itarget]
            
    return

def test_assign_cp_model():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=1, m=1)

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    for c in f.required_calibrations:
        f.required_calibrations[c][:] = 0
        f.achievable_calibrations[c][:] = 0
    ntot = 400
    targets(f, nt=ntot, seed=101)

    f.assign_full_cp_model(rsids=f.targets['rsid'])
    f.decollide_unassigned()
    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0
