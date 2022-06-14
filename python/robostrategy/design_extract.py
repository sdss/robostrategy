import os
import numpy as np
import peewee
import sdssdb.peewee.sdss5db.targetdb as targetdb
import robostrategy.targets

from sdssdb.peewee.sdss5db import database
database.set_profile('operations')


target_dtype = robostrategy.targets.target_dtype


def get_design_targets(designid=None):
    """Pull targets from design from the targetdb 

    Parameters
    ----------

    designid : int
        Design ID to pull
"""
    # First look at all targets in this carton/version
    nt = (targetdb.Target.select(targetdb.Target.pk)
          .join(targetdb.CartonToTarget)
          .join(targetdb.Assignment)
          .where(targetdb.Assignment.design_id == designid)).count()
    
    print(" ... {nt} targets".format(nt=nt), flush=True)
    tmp_targets = np.zeros(nt, dtype=target_dtype)

    if(nt == 0):
        return

    ts = (targetdb.Target.select(targetdb.Target.ra,
                                 targetdb.Target.dec,
                                 targetdb.Target.pmra,
                                 targetdb.Target.pmdec,
                                 targetdb.Target.epoch,
                                 targetdb.Target.parallax,
                                 targetdb.Target.pk.alias('target_pk'),
                                 targetdb.Target.catalogid,
                                 targetdb.CartonToTarget.pk.alias('carton_to_target_pk'),
                                 targetdb.CartonToTarget.priority,
                                 targetdb.CartonToTarget.value,
                                 targetdb.CartonToTarget.lambda_eff,
                                 targetdb.CartonToTarget.delta_ra,
                                 targetdb.CartonToTarget.delta_dec,
                                 targetdb.Magnitude.g,
                                 targetdb.Magnitude.r,
                                 targetdb.Magnitude.i,
                                 targetdb.Magnitude.bp,
                                 targetdb.Magnitude.gaia_g,
                                 targetdb.Magnitude.rp,
                                 targetdb.Magnitude.h,
                                 targetdb.Magnitude.z,
                                 targetdb.Magnitude.j,
                                 targetdb.Magnitude.k,
                                 targetdb.Carton.carton,
                                 targetdb.Carton.pk.alias('carton_pk'),
                                 targetdb.Carton.program,
                                 targetdb.Mapper.label.alias('mapper'),
                                 targetdb.Category.label.alias('category'),
                                 targetdb.Cadence.label_root.alias('cadence'),
                                 targetdb.Instrument.label.alias('fiberType'),
                                 targetdb.Version.plan,
                                 targetdb.Version.tag)
          .join(targetdb.CartonToTarget)
          .join(targetdb.Assignment).switch(targetdb.CartonToTarget)
          .join(targetdb.Instrument, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Cadence, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Magnitude, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Carton)
          .join(targetdb.Mapper, peewee.JOIN.LEFT_OUTER).switch(targetdb.Carton)
          .join(targetdb.Version).switch(targetdb.Carton)
          .join(targetdb.Category).switch(targetdb.Target)
          .where(targetdb.Assignment.design_id == designid)).dicts()

    castn = dict()
    for n in tmp_targets.dtype.names:
        castn[n] = np.cast[type(tmp_targets[n][0])]
            
    problems = []
    for indx, t in enumerate(ts):
        for n in tmp_targets.dtype.names:
            if((n != 'rsid') & (n != 'stage') & (n != 'magnitude')):
                if(t[n] is not None):
                    tmp_targets[n][indx] = castn[n](t[n])
                else:
                    if(n not in problems):
                        print("problem with {n}".format(n=n))
                        problems.append(n)
            elif(n == 'magnitude'):
                tmp_targets['magnitude'][indx, 0] = np.float32(t['g'])
                tmp_targets['magnitude'][indx, 1] = np.float32(t['r'])
                tmp_targets['magnitude'][indx, 2] = np.float32(t['i'])
                tmp_targets['magnitude'][indx, 3] = np.float32(t['z'])
                tmp_targets['magnitude'][indx, 4] = np.float32(t['bp'])
                tmp_targets['magnitude'][indx, 5] = np.float32(t['gaia_g'])
                tmp_targets['magnitude'][indx, 6] = np.float32(t['rp'])
                tmp_targets['magnitude'][indx, 7] = np.float32(t['j'])
                tmp_targets['magnitude'][indx, 8] = np.float32(t['h'])
                tmp_targets['magnitude'][indx, 9] = np.float32(t['k'])

    tmp_targets['rsid'] = tmp_targets['carton_to_target_pk']

    return(tmp_targets)
