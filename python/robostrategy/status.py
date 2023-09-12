import numpy as np
import peewee
import sdssdb.peewee.sdss5db.opsdb as opsdb
import sdssdb.peewee.sdss5db.targetdb as targetdb
import sdssdb.peewee.sdss5db.catalogdb as catalogdb

from sdssdb.peewee.sdss5db import database
database.set_profile('operations')

status_dtype = [('fieldid', np.int32),  # in field
                ('field_pk', np.int64),  # from field
                ('field_cadence', np.unicode_, 50),  # from field, cadence
                ('plan', np.unicode_, 40),  # in version
                ('first_plan', np.unicode_, 40),  # separate query
                ('observatory', np.unicode_, 5),  # from observatory 
                ('exposure', np.int32), # from design_to_field 
                ('field_exposure', np.int32),
                ('design_id', np.int32),  # from design
                ('status', np.int32),  # from assignment_status
                ('mjd', np.float32),
                ('assignment_status_pk', np.int64),
                ('holeid', np.unicode_, 20),  # from hole
                ('carton_to_target_pk', np.int64), # from assignment
                ('priority', np.int32),  # from carton_to_target
                ('value', np.float32),
                ('lambda_eff', np.float32),
                ('delta_ra', np.float64),
                ('delta_dec', np.float64),
                ('can_offset', bool),
                ('ra', np.float64),  # from target
                ('dec', np.float64),
                ('epoch', np.float32),
                ('pmra', np.float32),
                ('pmdec', np.float32),
                ('parallax', np.float32),
                ('catalogid', np.int64),
                ('catalogdb_plan', str, 20),
                ('target_pk', np.int64),
                ('magnitude', np.float32, 10), # from magnitude
                ('carton', np.unicode_, 60), # from carton
                ('carton_pk', np.int32),
                ('program', np.unicode_, 15), 
                ('mapper', np.unicode_, 3), # from mapper
                ('category', np.unicode_, 15), # from category
                ('cadence', np.unicode_, 22), # from carton_to_target, cadence
                ('fiberType', np.unicode_, 6)]  # from instrument

status_field_dtype = [('fieldid', np.int32),
                      ('field_pk', np.int64),
                      ('field_exposure', np.int32),
                      ('design_id', np.int32),
                      ('status', str, 20)]


def get_status_by_fieldid(plan=None, fieldid=None):
    """Read in cartons

    Parameters
    ----------

    plan : str
        plan to extract information for

    fieldid : np.int32
        field ID to get information for

    Returns
    -------

    status : ndarray
        array with information on individual statuses for this field

    status_field : ndarray
        array with information on design_id statuses for this field

    Notes
    -----

    This assumes that all the relevant information for the field is
    correctly associated with the field table entry associated with
    the specified plan.

    status_field has entry for each unique combination of fieldid, field_pk,
    field_exposure, and design_id. It is considered 'done' if ANY target
    has the status 'done'.
"""

    # Get the field information first
    field_info = (targetdb.Field.select(targetdb.Field.pk.alias('field_pk'),
                                        targetdb.Field.field_id.alias('fieldid'),
                                        targetdb.Cadence.label_root.alias('cadence'),
                                        targetdb.Observatory.label.alias('observatory'),
                                        targetdb.Version.plan)
                  .join(targetdb.Observatory).switch(targetdb.Field)
                  .join(targetdb.Version).switch(targetdb.Field)
                  .join(targetdb.Cadence).switch(targetdb.Field)
                  .join(opsdb.FieldToPriority, peewee.JOIN.LEFT_OUTER)
                  .join(opsdb.FieldPriority, peewee.JOIN.LEFT_OUTER)
                  .where((targetdb.Field.field_id == fieldid) &
                         (targetdb.Version.plan == plan) &
                         ((opsdb.FieldPriority.label != 'disabled') |
                          (opsdb.FieldPriority.label.is_null(True))))).dicts()

    field_cadence_dict = dict()
    for f in field_info:
        print(f)
        field_cadence_dict[f['field_pk']] = f['cadence']

    if(True):
        q_status = (targetdb.AssignmentStatus.select(targetdb.AssignmentStatus.pk.alias('assignment_status_pk'),
                                                     targetdb.AssignmentStatus.status,
                                                     targetdb.AssignmentStatus.mjd,
                                                     targetdb.Field.pk.alias('field_pk'),
                                                     targetdb.Field.field_id.alias('fieldid'),
                                                     targetdb.Observatory.label.alias('observatory'),
                                                     targetdb.Version.plan,
                                                     catalogdb.Version.plan.alias('catalogdb_plan'),
                                                     targetdb.DesignToField.field_exposure,
                                                     targetdb.DesignToField.exposure,
                                                     targetdb.Design.design_id,
                                                     targetdb.Hole.holeid,
                                                     targetdb.Target.ra,
                                                     targetdb.Target.dec,
                                                     targetdb.Target.pmra,
                                                     targetdb.Target.pmdec,
                                                     targetdb.Target.epoch,
                                                     targetdb.Target.parallax,
                                                     targetdb.Target.pk.alias('target_pk'),
                                                     targetdb.Target.catalogid,
                                                     targetdb.Assignment.carton_to_target.alias('carton_to_target_pk'),
                                                     targetdb.CartonToTarget.priority,
                                                     targetdb.CartonToTarget.value,
                                                     targetdb.CartonToTarget.lambda_eff,
                                                     targetdb.CartonToTarget.delta_ra,
                                                     targetdb.CartonToTarget.delta_dec,
                                                     targetdb.CartonToTarget.can_offset,
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
                                                     targetdb.Instrument.label.alias('fiberType'))
                    .join(targetdb.Assignment)
                    .join(targetdb.Design)
                    .join(targetdb.DesignToField)
                    .join(targetdb.Field)
                    .join(targetdb.Observatory).switch(targetdb.Field)
                    .join(opsdb.FieldToPriority, peewee.JOIN.LEFT_OUTER)
                    .join(opsdb.FieldPriority, peewee.JOIN.LEFT_OUTER).switch(targetdb.Field)
                    .join(targetdb.Version).switch(targetdb.Assignment)
                    .join(targetdb.Hole).switch(targetdb.Assignment)
                    .join(targetdb.Instrument, peewee.JOIN.LEFT_OUTER).switch(targetdb.Assignment)
                    .join(targetdb.CartonToTarget)
                    .join(targetdb.Target, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
                    .join(targetdb.Cadence, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
                    .join(targetdb.Magnitude, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
                    .join(targetdb.Carton, peewee.JOIN.LEFT_OUTER)
                    .join(targetdb.Mapper, peewee.JOIN.LEFT_OUTER).switch(targetdb.Carton)
                    .join(targetdb.Category).switch(targetdb.Target)
                    .join(catalogdb.Catalog, on=(catalogdb.Catalog.catalogid == targetdb.Target.catalogid))
                    .join(catalogdb.Version)
                    .where((targetdb.Field.field_id == fieldid) &
                           (targetdb.Version.plan == plan) &
                           ((opsdb.FieldPriority.label != 'disabled') |
                            (opsdb.FieldPriority.label.is_null(True))))).dicts()

    sql_string, sql_params = q_status.sql()
    for sql_param in sql_params:
        if(type(sql_param) == str):
            sql_string = sql_string.replace("%s", "'" + sql_param + "'", 1)
        else:
            sql_string = sql_string.replace("%s", str(sql_param), 1)

    statuses = q_status.dicts()
    status_array = np.zeros(len(statuses), dtype=status_dtype)

    if(len(status_array) == 0):
        print("No status information for fieldid={fid}".format(fid=fieldid),
              flush=True)
        return(None, None)

    castn = dict()
    for n in status_array.dtype.names:
        castn[n] = np.cast[type(status_array[n][0])]
            
    problems = []
    for indx, s in enumerate(statuses):
        for n in status_array.dtype.names:
            if(n == 'magnitude'):
                status_array['magnitude'][indx, 0] = np.float32(s['g'])
                status_array['magnitude'][indx, 1] = np.float32(s['r'])
                status_array['magnitude'][indx, 2] = np.float32(s['i'])
                status_array['magnitude'][indx, 3] = np.float32(s['z'])
                status_array['magnitude'][indx, 4] = np.float32(s['bp'])
                status_array['magnitude'][indx, 5] = np.float32(s['gaia_g'])
                status_array['magnitude'][indx, 6] = np.float32(s['rp'])
                status_array['magnitude'][indx, 7] = np.float32(s['j'])
                status_array['magnitude'][indx, 8] = np.float32(s['h'])
                status_array['magnitude'][indx, 9] = np.float32(s['k'])
            elif(n == 'field_cadence'):
                status_array[n][indx] = field_cadence_dict[s['field_pk']]
            elif(n == 'first_plan'):
                pass
            else:
                if(s[n] is not None):
                    status_array[n][indx] = castn[n](s[n])
                else:
                    if(n not in problems):
                        problems.append(n)

    design_ids = np.unique(status_array['design_id'])
    first_plan = dict()
    for design_id in design_ids:
        plans_dict = (targetdb.Design.select(targetdb.Version.plan, targetdb.Version.pk)
                      .join(targetdb.DesignToField)
                      .join(targetdb.Field)
                      .join(targetdb.Version)
                      .where(targetdb.Design.design_id == design_id)).dicts()
        pks = []
        plans = []
        for p in plans_dict:
            pks.append(p['pk'])
            plans.append(p['plan'])
        pks = np.array(pks, dtype=np.int32)
        ilow = np.argmin(pks)
        first_plan[design_id] = plans[ilow]

    for i, s in enumerate(status_array):
        status_array['first_plan'] = first_plan[s['design_id']]

    status_designs_set = set()
    for s in status_array:
        thing = (s['fieldid'], s['field_pk'], s['field_exposure'], s['design_id'])
        status_designs_set.add(thing)
    status_field = np.zeros(0, dtype=status_field_dtype)
    for s in status_designs_set:
        tmp_status_field = np.zeros(1, dtype=status_field_dtype)
        tmp_status_field['fieldid'] = s[0]
        tmp_status_field['field_pk'] = s[1]
        tmp_status_field['field_exposure'] = s[2]
        tmp_status_field['design_id'] = s[3]
        tmp_status_field['status'] = 'not started'
        igd = np.where((status_array['fieldid'] == tmp_status_field['fieldid']) &
                       (status_array['field_pk'] == tmp_status_field['field_pk']) &
                       (status_array['field_exposure'] == tmp_status_field['field_exposure']) &
                       (status_array['design_id'] == tmp_status_field['design_id']) &
                       (status_array['status'] > 0))[0]
        if(len(igd) > 0):
            tmp_status_field['status'] = 'done'
        status_field = np.append(status_field, tmp_status_field)

    return(status_array, status_field)
