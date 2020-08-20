import pytest

import numpy as np
import robostrategy.design as design


def test_design_greedy():
    np.random.seed(1)
    ntarget = 200
    target = np.zeros(ntarget, dtype=design._target_array_dtype)
    target['ra'] = np.random.random(ntarget) * 3. + 180. - 1.50
    target['dec'] = np.random.random(ntarget) * 3. - 1.50
    target['fiberType'] = 'BOSS'
    target['catalogid'] = np.arange(ntarget) + 1
    target['category'] = 'SCIENCE'
    target['program'] = 'test'
    target['priority'] = 1

    d = design.DesignGreedy(racen=180., deccen=0., pa=0., observatory='apo')
    d.targets_fromarray(target)
    d.assign()

    ii = np.where(d.target_assignments >= 0)[0]
    assert len(ii) == 111


def test_design_optimize():
    np.random.seed(1)
    ntarget = 200
    target = np.zeros(ntarget, dtype=design._target_array_dtype)
    target['ra'] = np.random.random(ntarget) * 3. + 180. - 1.50
    target['dec'] = np.random.random(ntarget) * 3. - 1.50
    target['fiberType'] = 'BOSS'
    target['catalogid'] = np.arange(ntarget) + 1
    target['category'] = 'SCIENCE'
    target['program'] = 'test'
    target['priority'] = 1

    d = design.DesignOptimize(racen=180., deccen=0., pa=0., observatory='apo')
    d.targets_fromarray(target)
    d.assign()

    ii = np.where(d.target_assignments >= 0)[0]
    assert len(ii) == 117
