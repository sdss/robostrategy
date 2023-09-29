import pytest

import numpy as np
import robostrategy.allocate as allocate
import roboscheduler.cadence as cadence


def add_cadence_single_nxm(n=1, m=1):
    clist = cadence.CadenceList()
    clist.add_cadence(name='single_{n}x{m}'.format(n=n, m=m),
                      nepochs=n,
                      skybrightness=[1.] * n,
                      delta=[-1.] +  [30.] * (n - 1),
                      delta_min=[-1.] +  [0.5] * (n - 1),
                      delta_max=[-1.] +  [1800.] * (n - 1),
                      nexp=[m] * n,
                      max_length=[0.] * n,
                      min_moon_sep=[15.] * n,
                      min_deltav_ks91=[-2.5] * n,
                      min_twilight_ang=[8] * n,
                      max_airmass=[2.] * n,
                      obsmode_pk=['bright_time'] * n)
    return

def add_cadence_mixed2_nxm(n=2, m=1):
    clist = cadence.CadenceList()
    clist.add_cadence(name='mixed2_{n}x{m}'.format(n=n, m=m),
                      nepochs=n,
                      skybrightness=[0.35, 0.35] + [1.] * (n - 2),
                      delta=[0., 3.] + [-1.] +  [30.] * (n - 3),
                      delta_min=[0., 0.5] + [-1.] +  [0.5] * (n - 3),
                      delta_max=[0., 1800.] + [-1.] +  [1800.] * (n - 3),
                      nexp=[m] * n,
                      max_length=[0.] * n,
                      min_moon_sep=[15.] * n,
                      min_deltav_ks91=[-1.5] * 2 + [-2.5] * (n - 2),
                      min_twilight_ang=[15] * 2 + [8.] * (n - 2),
                      max_airmass=[1.4] * 2 + [2.] * (n - 2),
                      obsmode_pk=['dark_plane'] * 2 + ['bright_time'] * (n - 2))
    return


def test_option_status():
    clist = cadence.CadenceList()
    clist.reset()

    add_cadence_single_nxm(n=5)
    add_cadence_single_nxm(n=7)
    add_cadence_mixed2_nxm(n=7)

    option_cadence = 'single_5x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([2], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([0], dtype=np.int32))

    option_cadence = 'single_7x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([0, 2, 3, 4, 5, 6], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is False

    option_cadence = 'single_5x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([2, 3, 4], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([0, 1, 2], dtype=np.int32))

    option_cadence = 'mixed2_7x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([2, 3, 4], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([2, 3, 4], dtype=np.int32))

    option_cadence = 'mixed2_7x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([0, 1, 2, 3], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([0, 1, 2, 3], dtype=np.int32))

    option_cadence = 'mixed2_7x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([0, 2, 3], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([0, 2, 3], dtype=np.int32))

    option_cadence = 'mixed2_7x1'
    current_cadence = 'single_5x1'
    current_exposures_done = np.array([0, 1, 2], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is True
    assert np.all(option_epochs_done == np.array([2, 3, 4], dtype=np.int32))

    option_cadence = 'single_5x1'
    current_cadence = 'mixed2_7x1'
    current_exposures_done = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is False

    option_cadence = 'single_5x1'
    current_cadence = 'single_7x1'
    current_exposures_done = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)

    ok, option_epochs_done = allocate.option_epochs_done(option_cadence=option_cadence,
                                                         current_cadence=current_cadence,
                                                         current_exposures_done=current_exposures_done)

    assert ok is False
