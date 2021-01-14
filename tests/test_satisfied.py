import pytest

import numpy as np
import robostrategy.field as field
import roboscheduler.cadence as cadence


def targets(f=None, nt=100, seed=100, rsid_start=0, ra=None, dec=None,
            category='science', cadence='single_1x1'):
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
    t['cadence'] = cadence
    t['catalogid'] = np.arange(nt, dtype=np.int64)
    t['rsid'] = np.arange(nt, dtype=np.int64) + rsid_start
    f.targets_fromarray(t)
    return


def test_field_satisfied_simple():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='single_1x1', nepochs=1,
                      skybrightness=[1.],
                      delta=[-1.],
                      delta_min=[-1.],
                      delta_max=[-1.],
                      nexp=[1],
                      instrument='BOSS')

    clist.add_cadence(name='single_2x1', nepochs=2,
                      skybrightness=[1.] * 2,
                      delta=[-1.] * 2,
                      delta_min=[-1.] * 2,
                      delta_max=[-1.] * 2,
                      nexp=[1] * 2,
                      instrument='BOSS')

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='single_2x1')

    ntot = 300
    targets(f, nt=ntot, seed=102)
    targets(f, nt=ntot, seed=102, rsid_start=ntot)

    f.assign_cadence(rsid=f.targets['rsid'][50])
    for t, a in zip(f.targets, f.assignments):
        if(t['catalogid'] == f.targets['catalogid'][50]):
            assert a['satisfied'] == 1
        else:
            assert a['satisfied'] == 0

    return


def test_field_satisfied_complex():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='bright_3x2',
                      nepochs=3,
                      skybrightness=[1., 1., 1.],
                      instrument='BOSS',
                      delta=[-1., -1., -1.],
                      delta_min=[-1., -1., -1.],
                      delta_max=[-1., -1., -1.],
                      nexp=[2, 2, 2])

    clist.add_cadence(name='bright_3x1',
                      nepochs=3,
                      skybrightness=[1., 1., 1.],
                      instrument='BOSS',
                      delta=[-1., -1., -1.],
                      delta_min=[-1., -1., -1.],
                      delta_max=[-1., -1., -1.],
                      nexp=[1, 1, 1])

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='bright_3x2')

    ntot = 300
    targets(f, nt=ntot, seed=102, cadence='bright_3x1')
    targets(f, nt=ntot, seed=102, rsid_start=f.targets['rsid'].max() + 1,
            cadence='bright_3x1')
    targets(f, nt=ntot, seed=102, rsid_start=f.targets['rsid'].max() + 1,
            cadence='bright_3x1')
    f.targets['cadence'][50] = 'bright_3x2'

    f.assign_cadence(rsid=f.targets['rsid'][50])
    assert f.assignments['satisfied'][50] == 1
    assert f.assignments['satisfied'][350] == 1
    assert f.assignments['satisfied'][650] == 1


def test_field_satisfied_assign():
    clist = cadence.CadenceList()
    clist.reset()

    clist.add_cadence(name='bright_3x2',
                      nepochs=3,
                      skybrightness=[1., 1., 1.],
                      instrument='BOSS',
                      delta=[-1., -1., -1.],
                      delta_min=[-1., -1., -1.],
                      delta_max=[-1., -1., -1.],
                      nexp=[2, 2, 2])

    clist.add_cadence(name='bright_3x1',
                      nepochs=3,
                      skybrightness=[1., 1., 1.],
                      instrument='BOSS',
                      delta=[-1., -1., -1.],
                      delta_min=[-1., -1., -1.],
                      delta_max=[-1., -1., -1.],
                      nexp=[1, 1, 1])

    f = field.Field(racen=180., deccen=0., pa=45, observatory='lco',
                    field_cadence='bright_3x2')

    ntot = 300
    targets(f, nt=ntot, seed=102, cadence='bright_3x1')
    targets(f, nt=ntot, seed=102, rsid_start=f.targets['rsid'].max() + 1,
            cadence='bright_3x1')

    f.assign_science()

    for cid in np.unique(f.targets['catalogid']):
        itargets = np.where(f.targets['catalogid'] == cid)[0]
        assert f.assignments['assigned'][itargets].sum() <= 1
        assert (f.assignments['satisfied'][itargets].sum() ==
                f.assignments['assigned'][itargets].sum() * len(itargets))
