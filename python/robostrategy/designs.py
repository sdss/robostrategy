import numpy as np
import peewee
import sdssdb.peewee.sdss5db.targetdb as targetdb
import sdssdb.peewee.sdss5db.opsdb as opsdb

from sdssdb.peewee.sdss5db import database
database.set_profile('operations')

design_dtype = [('design_id', np.int64),
                ('field_id', np.int32),  # from field
                ('exposure', np.int32),
                ('design_mode', np.unicode_, 40),
                ('mugatu_version', np.unicode_, 40),
                ('run_on', np.unicode_, 40),
                ('racen', np.float64),  # from field
                ('deccen', np.float64),  # from field
                ('position_angle', np.float32),  # from field
                ('cadence', np.unicode_, 30),  # from cadence
                ('mjd', np.int32),  # from design_to_status
                ('completion_status', np.unicode_, 20)]


def get_designs(plan=None, observatory=None):
    """Pull designs from the targetdb

    Parameters
    ----------

    plan : str
        plan to retrieve

    observatory : str
        observatory to retrieve

    Returns
    -------

    designs : ndarray
        array with design information
"""

    # First find the number of designs
    ndesigns = (targetdb.Design.select(targetdb.Design.design_id)
                .join(targetdb.Field)
                .join(targetdb.Version).switch(targetdb.Field)
                .join(targetdb.Observatory)
                .where((targetdb.Version.plan == plan) &
                       (targetdb.Observatory.label == observatory.upper()))).count()

    designs = np.zeros(ndesigns, dtype=design_dtype)

    if(ndesigns == 0):
        return(designs)

    ddicts = (targetdb.Design.select(targetdb.Design.design_id,
                                     targetdb.Design.exposure,
                                     targetdb.Design.design_mode_label.alias('design_mode'),
                                     targetdb.Design.mugatu_version,
                                     targetdb.Design.run_on,
                                     targetdb.Field.racen,
                                     targetdb.Field.deccen,
                                     targetdb.Field.position_angle,
                                     targetdb.Field.field_id,
                                     targetdb.Cadence.label.alias('cadence'),
                                     opsdb.DesignToStatus.mjd,
                                     opsdb.CompletionStatus.label.alias('completion_status'))
              .join(targetdb.Field)
              .join(targetdb.Version).switch(targetdb.Field)
              .join(targetdb.Observatory).switch(targetdb.Field)
              .join(targetdb.Cadence).switch(targetdb.Design)
              .join(opsdb.DesignToStatus, peewee.JOIN.LEFT_OUTER)
              .join(opsdb.CompletionStatus, peewee.JOIN.LEFT_OUTER)
              .where((targetdb.Version.plan == plan) &
                     (targetdb.Observatory.label == observatory.upper()))).dicts()
    
    castn = dict()
    for n in designs.dtype.names:
        castn[n] = np.cast[type(designs[n][0])]

    for indx, d in enumerate(ddicts):
        for n in designs.dtype.names:
            if(d[n] is not None):
                designs[n][indx] = castn[n](d[n])

    return(designs)
