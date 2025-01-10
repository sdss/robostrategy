import os
import numpy as np
import peewee
import astropy.io.ascii
import sdssdb.peewee.sdss5db.targetdb as targetdb
import sdssdb.peewee.sdss5db.catalogdb as catalogdb

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
                ('can_offset', bool),
                ('ra', np.float64),  # from target
                ('dec', np.float64),
                ('epoch', np.float32),
                ('pmra', np.float32),
                ('pmdec', np.float32),
                ('parallax', np.float32),
                ('catalogid', np.int64),
                ('catalogdb_plan', str, 12),
                ('target_pk', np.int64),
                ('magnitude', np.float32, 10), # from magnitude
                ('carton', np.unicode_, 60), # from carton
                ('carton_pk', np.int32),
                ('program', np.unicode_, 15), 
                ('mapper', np.unicode_, 3), # from mapper
                ('category', np.unicode_, 15), # from category
                ('cadence', np.unicode_, 60), # from cadence
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
                                     targetdb.Instrument.label.alias('fiberType'),
                                     catalogdb.Version.plan.alias('catalogdb_plan'),
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
              .join(catalogdb.Catalog, on=(catalogdb.Catalog.catalogid == targetdb.Target.catalogid))
              .join(catalogdb.Version)
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
                    tmp_targets['magnitude'][indx, 3] = np.float32(t['z'])
                    tmp_targets['magnitude'][indx, 4] = np.float32(t['bp'])
                    tmp_targets['magnitude'][indx, 5] = np.float32(t['gaia_g'])
                    tmp_targets['magnitude'][indx, 6] = np.float32(t['rp'])
                    tmp_targets['magnitude'][indx, 7] = np.float32(t['j'])
                    tmp_targets['magnitude'][indx, 8] = np.float32(t['h'])
                    tmp_targets['magnitude'][indx, 9] = np.float32(t['k'])

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


def match_v1_to_v0p5(catalogids_v1=None, all=False, use_sdss_ids=True):
    """Find catalogids in v0.5 corresponding to v1
    
    Parameters
    ----------

    catalogids_v1 : ndarray of np.int64
        input catalogids in v1

    all : bool
        if set True, return all v0.5 catalogids (not just one)

    use_sdss_ids : bool
        assume sdss_id_stacked table exists

    Returns
    -------

    catalogids_v1 : ndarray of np.int64
        catalogids in v1

    catalogids_v0p5 : ndarray of np.int64
        catalogids in v0.5 (-1 if not found)

    sdss_ids : ndarray of np.int64
        corresponding sdss_ids (-1 if not found)

    Notes
    -----

    If all is False, then the two arrays are in the same
    order as the input list, and have the same length.

    If all is True, then only matches are included in the 
    output lists, and repeats are included

    Hard-coded between these two versions because the db
    has the version names hard-coded into tables
"""
    if(len(catalogids_v1) == 0):
        if(use_sdss_ids):
            return(np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64))
        else:
            return(np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64))
    
    # Construct query
    if(use_sdss_ids):
        sql_template = """SELECT catalogid25, catalogid31, sdss_id FROM catalogdb.sdss_id_stacked JOIN (VALUES {v}) AS ver31(catalogid) ON catalogdb.sdss_id_stacked.catalogid31 = ver31.catalogid;
"""
    else:
        sql_template = """SELECT catalogid1, catalogid2 FROM catalogdb.catalog_ver25_to_ver31_full_unique JOIN (VALUES {v}) AS ver31(catalogid) ON catalogdb.catalog_ver25_to_ver31_full_unique.catalogid2 = ver31.catalogid;
"""
    values = ""
    ucatalogids_v1 = np.unique(catalogids_v1)
    for value in ucatalogids_v1:
        values = values + "({v}),".format(v=value)
    values = values[0:-1]
    sql_command = sql_template.format(v=values)

    if(all is False):
        # Set up output
        out_catalogids_v1 = catalogids_v1
        out_catalogids_v0p5 = np.zeros(len(catalogids_v1), dtype=np.int64) - 1
        if(use_sdss_ids):
            out_sdss_ids = np.zeros(len(catalogids_v1), dtype=np.int64) - 1
            
        indxs = dict()
        for cid_v1 in ucatalogids_v1:
            indxs[cid_v1] = np.where(catalogids_v1 == cid_v1)[0]

        # Run query
        cursor = database.execute_sql(sql_command)
        for row in cursor.fetchall():
            if(row[0] == None):
                continue
            catalogid_v1 = row[1]
            catalogid_v0p5 = row[0]
            out_catalogids_v0p5[indxs[catalogid_v1]] = catalogid_v0p5
            if(use_sdss_ids):
                sdss_id = row[2]
                out_sdss_ids[indxs[catalogid_v1]] = sdss_id
    else:
        cursor = database.execute_sql(sql_command)
        out_catalogids_v1 = np.zeros(len(catalogids_v1), dtype=np.int64) - 1
        out_catalogids_v0p5 = np.zeros(len(catalogids_v1), dtype=np.int64) - 1
        if(use_sdss_ids):
            out_sdss_ids = np.zeros(len(catalogids_v1), dtype=np.int64) - 1
        i = 0
        for row in cursor.fetchall():
            out_catalogids_v1[i] = row[1]
            out_catalogids_v0p5[i] = row[0]
            if(use_sdss_ids):
                out_sdss_ids[i] = row[2]
            i = i + 1
            if(i >= len(out_catalogids_v1)):
                out_catalogids_v1 = np.append(out_catalogids_v1,
                                              np.zeros(len(out_catalogids_v1),
                                                       dtype=np.int64) - 1)
                out_catalogids_v0p5 = np.append(out_catalogids_v0p5,
                                                np.zeros(len(out_catalogids_v0p5),
                                                         dtype=np.int64) - 1)
                if(use_sdss_ids):
                    out_sdss_ids = np.append(out_sdss_ids,
                                             np.zeros(len(out_sdss_ids),
                                                      dtype=np.int64) - 1)
        out_catalogids_v1 = out_catalogids_v1[0:i]
        out_catalogids_v0p5 = out_catalogids_v0p5[0:i]
        out_sdss_ids = out_sdss_ids[0:i]
        
    if(use_sdss_ids):
        return(out_catalogids_v1, out_catalogids_v0p5, out_sdss_ids)
    else:
        return(out_catalogids_v1, out_catalogids_v0p5)


def catalogids_are_targets(catalogids=None):
    """Check if catalogids are in target table

    Parameters
    ----------

    catalogids : ndarray of np.int64
        catalogids 

    Returns
    -------

    istarget : ndarray of bool
        whether present
"""
    # Construct query
    sql_template = """SELECT targetdb.target.catalogid FROM targetdb.target
JOIN (VALUES {v}) AS input(catalogid) ON targetdb.target.catalogid = input.catalogid;
"""

    values = ""
    ucatalogids = np.unique(catalogids)
    for value in ucatalogids:
        values = values + "({v}),".format(v=value)
    values = values[0:-1]
    sql_command = sql_template.format(v=values)

    # Set up output
    istarget = np.zeros(len(catalogids), dtype=bool)
    indxs = dict()
    for cid in ucatalogids:
        indxs[cid] = np.where(catalogids == cid)[0]
        
    # Run query
    cursor = database.execute_sql(sql_command)
    for row in cursor.fetchall():
        catalogid = row[0]
        istarget[indxs[catalogid]] = True

    return(istarget)


def catalogids_to_target_ids(catalogids=None, input_catalog=None):
    """Return target_ids for input catalog for catalogid

    Parameters
    ----------

    catalogids : ndarray of np.int64
        catalogids 

    input_catalog : str
        name of input catalog (like 'tic_v8')

    Returns
    -------

    target_ids : ndarray of np.int64
        input catalog IDs
"""
    # Construct query
    sql_template = """SELECT catalogdb.catalog.catalogid, catalogdb.catalog_to_{s}.target_id FROM catalogdb.catalog
JOIN catalogdb.catalog_to_{s} ON catalogdb.catalog.catalogid = catalogdb.catalog_to_{s}.catalogid
JOIN (VALUES {v}) AS desired(catalogid) ON catalogdb.catalog.catalogid = desired.catalogid;
"""

    values = ""
    ucatalogids = np.unique(catalogids)
    for value in ucatalogids:
        values = values + "({v}),".format(v=value)
    values = values[0:-1]
    sql_command = sql_template.format(v=values, s=input_catalog)

    # Set up output
    target_ids = np.zeros(len(catalogids), dtype=np.int64) - 1
    indxs = dict()
    for cid in ucatalogids:
        indxs[cid] = np.where(catalogids == cid)[0]
        
    # Run query
    cursor = database.execute_sql(sql_command)
    for row in cursor.fetchall():
        catalogid = row[0]
        target_id = row[1]
        target_ids[indxs[catalogid]] = target_id

    return(target_ids)


def target_ids_to_catalogids(target_ids=None, input_catalog=None,
                             crossmatch=None):
    """Map target_id to a catalogids from a particular version

    Parameters
    ----------

    target_ids : ndarray of np.int64
        IDs from input catalog

    crossmatch : str
        cross match version

    input_catalog : str
        name of input catalog (like 'tic_v8')

    Returns
    -------

    catalogids : ndarray of np.int64
        catalogids 
"""
    # Construct query
    sql_template = """SELECT catalogdb.catalog_to_{s}.target_id, catalogdb.catalog.catalogid FROM catalogdb.catalog_to_{s}
JOIN (VALUES {v}) AS desired(target_id) ON catalogdb.catalog_to_{s}.target_id = desired.target_id
JOIN catalogdb.catalog ON catalogdb.catalog.catalogid = catalogdb.catalog_to_{s}.catalogid
JOIN catalogdb.version ON catalogdb.version.id = catalogdb.catalog.version_id
WHERE catalogdb.version.plan = '{c}';
"""

    values = ""
    utarget_ids = np.unique(target_ids)
    for value in utarget_ids:
        values = values + "({v}),".format(v=value)
    values = values[0:-1]
    sql_command = sql_template.format(v=values, c=crossmatch, s=input_catalog)

    # Set up output
    catalogids = np.zeros(len(target_ids), dtype=np.int64) - 1
    indxs = dict()
    for tid in utarget_ids:
        indxs[tid] = np.where(target_ids == tid)[0]
        
    # Run query
    cursor = database.execute_sql(sql_command)
    for row in cursor.fetchall():
        target_id = row[0]
        catalogid = row[1]
        catalogids[indxs[target_id]] = catalogid

    return(catalogids)
