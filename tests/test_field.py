import pytest

import numpy as np
import robostrategy.field as field
import roboscheduler.cadence as cadence


def targets(f=None, nt=100, seed=100, rsid_start=0, ra=None, dec=None,
            category='science'):
    t_dtype = np.dtype([('ra', np.float64),
                        ('dec', np.float64),
                        ('priority', np.int32),
                        ('category', np.unicode_, 30),
                        ('cadence', np.unicode_, 30),
                        ('catalogid', np.int64),
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
    t['priority'] = 1
    t['category'] = category
    t['cadence'] = 'single_1x1'
    t['catalogid'] = np.arange(nt, dtype=np.int64)
    t['rsid'] = np.arange(nt, dtype=np.int64) + rsid_start
    f.targets_fromarray(t)
    return


def test_field_init():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    assert f.field_cadence.nepochs == 1

    clist.add_cadence(name='single_2x1', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[1] * 2,
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x1')
    assert f.field_cadence.nepochs == 2

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1., -1.],
                      delta_min=[-1., -1.],
                      delta_max=[-1., -1.],
                      nexp=[2, 2],
                      instrument='BOSS')
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

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    x, y = f.radec2xy(ra=180.5, dec=0.5)
    ra, dec = f.xy2radec(x=x, y=y)
    assert np.abs(ra - 180.5) < 1.e-7
    assert np.abs(dec - 0.5) < 1.e-7


def test_target_fromarray():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    targets(f)


def test_flags():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')

    targets(f)

    f.set_flag(rsid=10, flagname='CADENCE_INCONSISTENT')
    assert f.check_flag(rsid=10, flagname='CADENCE_INCONSISTENT')[0] == True
    assert f.check_flag(rsid=10, flagname='NOT_COVERED_BY_BOSS')[0] == False
    assert f.get_flag_names(f.assignments['rsflags'][f.rsid2indx[10]]) == ['CADENCE_INCONSISTENT']


def test_assign_robot_epoch():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2] * 2,
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500)

    rid = 10
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
    assert f._robot2indx[rid, 0] == f.rsid2indx[tid]
    assert f._robot2indx[rid, 1] == -1
    assert f._robot2indx[rid, 2] == -1
    assert f._robot2indx[rid, 3] == -1

    f.assign_robot_epoch(rsid=tid, robotID=rid, epoch=1, nexp=2)
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=2)[0] == False
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=1)[0] == False
    assert f.available_robot_epoch(robotID=rid, epoch=1, nexp=2)[0] == False
    assert f.assignments['robotID'][f.rsid2indx[tid], 2] == rid
    assert f.assignments['robotID'][f.rsid2indx[tid], 3] == rid
    assert f._robot2indx[rid, 2] == f.rsid2indx[tid]
    assert f._robot2indx[rid, 3] == f.rsid2indx[tid]

    f.unassign_epoch(rsid=tid, epoch=0)
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=1)[0] == True
    assert f.available_robot_epoch(robotID=rid, epoch=0, nexp=2)[0] == True
    assert f.assignments['robotID'][f.rsid2indx[tid], 0] == -1
    assert f.assignments['robotID'][f.rsid2indx[tid], 1] == -1


def test_available_epochs():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2, 2],
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')
    targets(f, nt=500, seed=102)

    for rid in f.robotgrids[0].robotDict:
        tids = f.robotgrids[0].robotDict[rid].validTargetIDs
        if(len(tids) > 1):
            tid0 = tids[0]
            tid1 = tids[1]

            f.assign_robot_epoch(rsid=tid0, robotID=rid, epoch=0, nexp=1)

            av = f.available_epochs(rsid=tid1, epochs=[0, 1], nexps=[2, 2])
            ar = av['availableRobotIDs']
            nf = av['nFrees']
            sc = av['spareCalibrations']
            assert rid not in ar[0]
            assert rid in ar[1]

            assert f._robot2indx[rid, 0] == f.rsid2indx[tid0]

            f.unassign(rsid=tid0)
            f.unassign(rsid=tid1)

            assert f._robot2indx[rid, 0] == - 1


def test_assign_epochs():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2, 2],
                      instrument='BOSS')
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

            assert f._robot2indx[rid, 0] == indx0
            assert f._robot2indx[rid, 1] == indx1
            assert f._robot2indx[rid, 2] == indx0
            assert f._robot2indx[rid, 3] == indx1

            return


def test_clear_assignments():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2, 2],
                      instrument='BOSS')
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

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2] * 2,
                      instrument='BOSS')
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


def test_collisions():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x2', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[2] * 2,
                      instrument='BOSS')
    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x2')

    ntot = 400
    targets(f, nt=ntot, seed=101)
    targets(f, nt=ntot, rsid_start=ntot, ra=f.targets['ra'],
            dec=f.targets['dec'] + 0.005)

    rid1 = 10
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

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='APOGEE')

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


def test_assign():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='APOGEE')

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


def test_clear():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='APOGEE')

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

def test_assign_boss_in_apogee():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='APOGEE')

    clist.add_cadence(name='single_1x1_boss', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1_boss')
    ntot = 400
    targets(f, nt=ntot, seed=101)

    f.assign_science()
    f.decollide_unassigned()
    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0
    for r in f.robotgrids[0].robotDict:
        if(f.robotgrids[0].robotDict[r].assignedTargetID >= 0):
            assert f.robotgrids[0].robotDict[r].hasApogee


def test_assign_cp_model():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_1x1')
    ntot = 400
    targets(f, nt=ntot, seed=101)

    f.assign_full_cp_model(rsids=f.targets['rsid'])
    f.decollide_unassigned()
    assert f.validate() == 0
    assert f.assignments['assigned'].sum() > 0