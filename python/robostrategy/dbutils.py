import os
import numpy as np
import peewee
import sdssdb.peewee.sdss5db.targetdb as targetdb
import sdssdb.peewee.sdss5db.opsdb as opsdb
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


def field_status(fieldid=None, field_pk=None, plan=None, observatory=None, other_info=False):
    """Extract field status from the db
    
    Parameters
    ----------

    fieldid : int
        field identifier

    field_pk : int
        field_pk

    plan : str
        which robostrategy plan

    observatory : str
        which observatory ('apo' or 'lco')

    other_info : bool
        whether to return other information (MJD)

    Returns
    -------

    status : str
        one of 'not started', 'started', 'done'

    designid : ndarray of np.int32
        which design IDs satisfy each exposure

    designid_status : ndarray of str
        one of 'not started', 'started', 'done'

    field_exposure : ndarray of np.int32
        exposure numbers in field

    other_info : ndarray
        structure with other info

    Notes
    -----

    If field_pk 

    Designs are returned in order of field_exposure
"""
    dinfo_joins = (targetdb.Field.select(targetdb.Field.field_id,
                                         targetdb.Field.racen,
                                         targetdb.Field.deccen,
                                         targetdb.Version.plan,
                                         targetdb.Version.pk.alias('version_pk'),
                                         targetdb.Cadence.label.alias('cadence'),
                                         targetdb.Observatory.label.alias('observatory'),
                                         targetdb.DesignToField.field_exposure,
                                         opsdb.DesignToStatus.mjd,
                                         opsdb.CompletionStatus.label.alias('status'),
                                         targetdb.Design.design_id)
                   .join(targetdb.Version).switch(targetdb.Field)
                   .join(targetdb.Cadence).switch(targetdb.Field)
                   .join(targetdb.Observatory).switch(targetdb.Field)
                   .join(targetdb.DesignToField, peewee.JOIN.LEFT_OUTER)
                   .join(targetdb.Design)
                   .join(opsdb.DesignToStatus)
                   .join(opsdb.CompletionStatus))

    if(plan is not None):
        if(field_pk is not None):
            dinfo = dinfo_joins.where((targetdb.Field.pk == int(field_pk)) &
                                      (targetdb.Observatory.label == observatory.upper()) &
                                      (targetdb.Version.plan == plan)).dicts()
        else:
            dinfo = dinfo_joins.where((targetdb.Field.field_id == int(fieldid)) &
                                      (targetdb.Observatory.label == observatory.upper()) &
                                      (targetdb.Version.plan == plan)).dicts()
    else:
        if(field_pk is not None):
            dinfo = dinfo_joins.where((targetdb.Field.pk == int(field_pk)) &
                                      (targetdb.Observatory.label == observatory.upper())).dicts()
        else:
            dinfo = dinfo_joins.where((targetdb.Field.field_id == int(fieldid)) &
                                      (targetdb.Observatory.label == observatory.upper())).dicts()

    designid = np.zeros(len(dinfo), dtype=np.int32)
    designid_status = np.array(['not started'] * len(dinfo))
    field_exposure = np.zeros(len(dinfo), dtype=np.int32)
    version_pk = np.zeros(len(dinfo), dtype=np.int32)
    mjd = np.zeros(len(dinfo), dtype=np.float64)
    design_plan = [''] * len(dinfo)
    for i, d in enumerate(dinfo):
        designid[i] = d['design_id']
        designid_status[i] = d['status']
        if(d['field_exposure'] is not None):
            field_exposure[i] = d['field_exposure']
        if(d['mjd'] is not None):
            mjd[i] = d['mjd']
        if(d['plan'] is not None):
            design_plan[i] = d['plan']
        if(d['version_pk'] is not None):
            version_pk[i] = d['version_pk']
    design_plan = np.array(design_plan)

    isort = np.argsort(field_exposure)
    designid = designid[isort]
    designid_status = designid_status[isort]
    field_exposure = field_exposure[isort]
    design_plan = design_plan[isort]
    mjd = mjd[isort]
    version_pk = version_pk[isort]

    if(plan is not None):
        if(np.all(designid_status == 'done')):
            status = 'done'
        elif(np.any(designid_status != 'not started')):
            status = 'started'
        else:
            status = 'not started'
    else:
        status = 'not started'
        ngot = 0
        for iexp in np.unique(field_exposure):
            ic = np.where(field_exposure == iexp)[0]
            if(np.any(designid_status[ic] == 'done')):
                status = 'started'
                ngot += 1
        if(ngot == len(np.unique(field_exposure))):
            status = 'done'
            
    if(other_info):
        other_dtype = np.dtype([('mjd', np.float64),
                                ('version_pk', np.int32),
                                ('plan', str, 40)])
        other = np.zeros(len(field_exposure), dtype=other_dtype)
        other['mjd'] = mjd
        other['version_pk'] = version_pk
        other['plan'] = design_plan
        
        return(status, designid, designid_status, field_exposure, other)
    else:
        return(status, designid, designid_status, field_exposure)


def design_status(design_id=None):
    """Extract field status from the db
    
    Parameters
    ----------

    design_id : int
        design identifier

    Returns
    -------

    status : str
        one of 'not started', 'started', 'done'

    mjd : np.float32
        an MJD if done, 0. otherwise
"""
    dinfo = (opsdb.DesignToStatus.select(opsdb.DesignToStatus.mjd,
                                         opsdb.CompletionStatus.label.alias('status'))
             .join(opsdb.CompletionStatus)
             .where(opsdb.DesignToStatus.design_id == design_id)).dicts()

    if(len(dinfo) == 0):
        status = ''
        mjd = 0.
    elif(len(dinfo) == 1):
        status = dinfo[0]['status']
        mjd = dinfo[0]['mjd']
    else:
        raise ValueError("More than one status entries for design_id={did}".format(did=design_id))

    return(status, mjd)
