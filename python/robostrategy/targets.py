import os
import numpy as np
import peewee
import astropy.io.ascii
import sdssdb.peewee.sdss5db.targetdb as targetdb

from sdssdb.peewee.sdss5db import database
database.set_profile('operations')

target_dtype = [('stage', np.unicode_, 6),
                ('rsid', np.int64), # set equal to carton_to_target_pk
                ('carton_to_target_pk', np.int64), # from carton_to_target
                ('priority', np.int32),
                ('value', np.float32),
                ('lambda_eff', np.float32),
                ('delta_ra', np.float64),
                ('delta_dec', np.float64),
                ('ra', np.float64),  # from target
                ('dec', np.float64),
                ('epoch', np.float32),
                ('pmra', np.float32),
                ('pmdec', np.float32),
                ('parallax', np.float32),
                ('catalogid', np.int64),
                ('target_pk', np.int64),
                ('magnitude', np.float32, 7), # from magnitude
                ('carton', np.unicode_, 50), # from carton
                ('carton_pk', np.int32),
                ('program', np.unicode_, 15),
                ('mapper', np.unicode_, 3), # from mapper
                ('category', np.unicode_, 15), # from category
                ('cadence', np.unicode_, 22), # from cadence
                ('fiberType', np.unicode_, 6),  # from instrument
                ('plan', np.unicode_, 8),  # from version
                ('tag', np.unicode_, 8)]


def read_cartons(version=None, filename=None):
    """Read in cartons

    Parameters
    ----------

    version : str
        version of carton file

    filename : str
        explicit file name of carton file

    Returns
    -------

    cartons : Table
        table with carton information


    Notes
    -----

    Reads file as fixed_width, |-delimited file with astropy.io.ascii

    If filename is specified, reads in that file.

    If not, and version is specified, reads in $RSCONFIG_DIR/etc/cartons-[version].txt
"""
    if((version is None) and (filename is None)):
        print("Must specify either version or filename!")
        return

    if(filename is None):
        filename = os.path.join(os.getenv('RSCONFIG_DIR'),
                                'etc', 'cartons-{version}.txt')
        filename = filename.format(version=version)

    cartons = astropy.io.ascii.read(filename, format='fixed_width',
                                    delimiter='|')
    return(cartons)


def get_targets(carton=None, version=None, justcount=False, c2c=None):
    """Pull targets from the targetdb

    Parameters
    ----------

    cartons : str
        label of carton to pull

    version : str
        plan of carton to pull

    justcount : bool
        if True, just return the count (default False)

    c2c : config
        if not None, maps cartons to fiber type and cadences (default None)
"""
    if(justcount):
        print("Counting carton {p}, version {v}".format(p=carton,
                                                        v=version))
    else:
        print("Extracting carton {p}, version {v}".format(p=carton,
                                                          v=version))

    # First look at all targets in this carton/version
    ntall = (targetdb.Target.select(targetdb.Target.pk)
             .join(targetdb.CartonToTarget)
             .join(targetdb.Carton)
             .join(targetdb.Version)
             .where((targetdb.Carton.carton == carton) &
                    (targetdb.Version.plan == version))).count()

    if(justcount):
        print(" ... {ntall} targets".format(ntall=ntall), flush=True)
        return(ntall)

    # Now look at those with a cadence, instrument, and magnitude not null
    nt = (targetdb.Target.select(targetdb.Target.pk)
          .join(targetdb.CartonToTarget)
          .join(targetdb.Instrument, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Cadence, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Magnitude, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
          .join(targetdb.Carton)
          .join(targetdb.Version)
          .where((targetdb.Carton.carton == carton) &
                 (targetdb.Version.plan == version))).count()

    if(nt != ntall):
        print("WARNING: only {nt} of {ntall} targets in carton {carton} have cadence, instrument, and magnitude non-null".format(nt=nt, ntall=ntall, carton=carton))

    print(" ... {nt} targets".format(nt=nt), flush=True)
    tmp_targets = None
    if(nt > 0):
        tmp_targets = np.zeros(nt, dtype=target_dtype)

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
              .join(targetdb.Instrument, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
              .join(targetdb.Cadence, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
              .join(targetdb.Magnitude, peewee.JOIN.LEFT_OUTER).switch(targetdb.CartonToTarget)
              .join(targetdb.Carton)
              .join(targetdb.Mapper, peewee.JOIN.LEFT_OUTER).switch(targetdb.Carton)
              .join(targetdb.Version).switch(targetdb.Carton)
              .join(targetdb.Category).switch(targetdb.Target)
              .where((targetdb.Carton.carton == carton) &
                     (targetdb.Version.plan == version))).dicts()

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
#                    tmp_targets['magnitude'][indx, 3] = np.float32(t['z'])
                    tmp_targets['magnitude'][indx, 4] = np.float32(t['bp'])
                    tmp_targets['magnitude'][indx, 5] = np.float32(t['gaia_g'])
                    tmp_targets['magnitude'][indx, 6] = np.float32(t['rp'])
#                    tmp_targets['magnitude'][indx, 7] = np.float32(t['j'])
                    tmp_targets['magnitude'][indx, 8] = np.float32(t['h'])
#                    tmp_targets['magnitude'][indx, 9] = np.float32(t['k'])

        tmp_targets['rsid'] = tmp_targets['carton_to_target_pk']

        if(c2c is not None):
            inofibertype = np.where(tmp_targets['fiberType'] == '')[0]
            if(len(inofibertype) > 0):
                msg = "WARNING: {n} targets in {c} with no fiberType".format(n=len(inofibertype), c=carton)
                if(carton in c2c['CartonToFiberType']):
                    fiberType = c2c.get('CartonToFiberType', carton)
                    print("{msg}, SETTING TO {fiberType}".format(msg=msg, fiberType=fiberType))
                    tmp_targets['fiberType'][inofibertype] = fiberType
                else:
                    print("{msg}, NOT FIXING".format(msg=msg))

            inocadence = np.where(tmp_targets['cadence'] == '')[0]
            if(len(inocadence) > 0):
                msg = "WARNING: {n} targets in {c} with no cadence".format(n=len(inocadence), c=carton)
                if(carton in c2c['CartonToCadence']):
                    cadence = c2c.get('CartonToCadence', carton)
                    print("{msg}, SETTING TO {cadence}".format(msg=msg, cadence=cadence))
                    tmp_targets['cadence'][inocadence] = cadence
                    if(cadence == 'dark_174x8'):
                        ii = np.where((np.abs(tmp_targets['ra'][inocadence] - 90.) < 6.) &
                                      (np.abs(tmp_targets['dec'][inocadence] + 66.56) < 2.))[0]
                        if(len(ii) > 0):
                            tmp_targets['cadence'][inocadence[ii]] = 'dark_100x8'
                else:
                    print("{msg}, NOT FIXING".format(msg=msg))

    return(tmp_targets)
