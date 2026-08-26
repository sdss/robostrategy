import os
import time

from astropy.time import Time
import fitsio
import numpy as np
import yaml

from mugatu.fpsdesign import FPSDesign
import roboscheduler.cadence
import robostrategy.field as field
from robostrategy.targets import get_targets, target_dtype
from sdssdb.peewee.sdss5db import database


roboscheduler.cadence.CadenceList().fromdb(version='v1')


def query_field(racen, deccen, field_id=999999, 
                cartons=[], cache=False, cacheDir=None, 
                observatory=None):
    """
    Query the targets for a given field and list of cartons.
    
    Parameters
    ----------
    racen : float
        Right ascension of the field center in degrees.
    deccen : float
        Declination of the field center in degrees.
    field_id : int, optional
        Identifier for the field. Default is 999999.
    cartons : list of dict, optional
        List of cartons to query. Each carton should be a dictionary with keys 'carton' and 'plan'.
    cache : bool, optional
        Whether to cache the results to disk. Default is False.
    cacheDir : str, optional
        Directory to store cached results. Required if cache is True.
    observatory : str
        Observatory name, 'APO' or 'LCO'
    
    Returns
    -------
    targets : numpy structured array
        Array of targets for the specified field and cartons.

    """
    if observatory.upper() == "APO":
        radius = 1.49
        position_angle = 40.
    elif observatory.upper() == "LCO":
        radius = 0.95
        position_angle = 270.
    else:
        raise ValueError(f"Unknown observatory: {observatory}")

    assert len(cartons) > 0, "No cartons provided for query_field"
    if cache:
        assert cacheDir is not None, "cacheDir must be provided to cache files"
        if not os.path.exists(cacheDir):
            os.makedirs(cacheDir)
        elif os.path.isfile(f"{cacheDir}/{field_id}.fits"):
            print(f"Reading cached targets for field {field_id} from {cacheDir}/{field_id}.fits")
            return fitsio.read(f"{cacheDir}/{field_id}.fits")

    nt = 0
    for carton in cartons:
        name = carton['carton']
        print(f"Counting targets in carton {name}")
        tmp_targets_file = f"{cacheDir}/{field_id}_{name}.fits"
        if os.path.exists(tmp_targets_file) == False:
            nt = nt + get_targets(carton=carton['carton'], version=carton['plan'],
                                  racen=racen, deccen=deccen, radius=radius,
                                  justcount=True)
        else:
            print("Counting from carton file {c}".format(c=tmp_targets_file))
            try:
                tmp_targets = fitsio.read(tmp_targets_file)
                nt += len(tmp_targets)
            except OSError as e:
                print(f"Error reading {tmp_targets_file}: {e}")

    targets = np.zeros(nt, dtype=target_dtype)

    nt = 0
    for carton in cartons:
        name = carton['carton']
        tmp_targets_file = f"{cacheDir}/{field_id}_{name}.fits"

        if os.path.exists(tmp_targets_file) == False:
            tmp_targets = get_targets(carton=carton['carton'], version=carton['plan'],
                                    racen=racen, deccen=deccen, radius=radius)
            if cache:
                fitsio.write(tmp_targets_file, tmp_targets, clobber=True)
        else:
            try:
                tmp_targets = fitsio.read(tmp_targets_file)
            except OSError as e:
                print(f"Error reading {tmp_targets_file}: {e}")
                tmp_targets = None

        if tmp_targets is None:
            continue

        targets[nt:nt + len(tmp_targets)] = tmp_targets
        nt = nt + len(tmp_targets)

    if cache:
        fitsio.write(f"{cacheDir}/{field_id}.fits", targets, clobber=True)

    return targets


def assign_field(racen, deccen, targtab,
                 field_id=999999,
                 cacheDir=None,
                 observatory=None,
                 cadence="bright_1x1",
                 design_mode=None):
    """
    Assigns targets to a field and writes the design to a FITS file.
    
    Parameters
    ----------
    racen : float
        Right ascension of the field center in degrees.
    deccen : float
        Declination of the field center in degrees.
    targtab : numpy structured array
        Array of targets to assign.
    field_id : int, optional
        Identifier for the field. Default is 999999.
    cacheDir : str, optional
        Directory to store the output FITS file. If None, uses the ROBOSTRATEGY_DATA environment variable.
    observatory : str
        Observatory name, 'APO' or 'LCO'.
    cadence : str, optional
        Cadence string for the field. Default is "bright_1x1".
    design_mode : str, optional
        Design mode for the field. If None, uses the default design mode.
    
    Returns
    -------
    fitsoutfname : str
        Path to the output FITS file containing the assigned design.
    """

    if cacheDir is None:
        cacheDir = os.getenv("ROBOSTRATEGY_DATA")
    if not os.path.exists(cacheDir):
        os.makedirs(cacheDir)

    fitsoutfname = os.path.join(cacheDir, f"field_{field_id}.fits")

    obsTime = Time.now().jd
    
    print(f"Making designs for {field_id} at {racen:.5f} {deccen:+.5f}")
    start = time.time()

    if observatory == "APO":
        position_angle = 40.
    elif observatory == "LCO":
        position_angle = 270.

    fiberType = np.zeros(len(targtab), dtype=(np.unicode_, 6))
    fiberType[np.isin(targtab["lambda_eff"], [0., 5400.])] = "BOSS"
    fiberType[np.isin(targtab["lambda_eff"], [16000.])] = "APOGEE"

    targtab["fiberType"] = fiberType

    print("create field object...")
    f = field.Field(racen=racen, deccen=deccen, pa=position_angle,
                    field_cadence=cadence, observatory=observatory.lower(),
                    offset_min_skybrightness=0.0, observe_epoch="2026-07-01",
                    verbose=True,
                    #bright_neighbors=False, reset_bright=True,
    )
    if design_mode is not None:
        nepochs = int(cadence.split("_")[-1].split("x")[0])
        f.design_mode = np.array([design_mode for i in range(nepochs)])
    print("design_mode", f.design_mode)

    # assign targets
    print("Assigning targets...")
    start2 = time.time()
    f.targets_fromarray(targtab)
    # f.assign()
    f.assign_science_and_calibs()
    print(f"Took {time.time()-start2:.1f}s")
    print(f.assess())
    print(f"Done with RS, writing to {fitsoutfname}")
    f.tofits(filename=fitsoutfname)

    #### Mugatu Validation
    # create a mugatu.FPSDesign object that is specified as a manual design
    fps_design = FPSDesign(design_pk=-1,
                           obsTime=obsTime,
                           design_file=fitsoutfname,
                           manual_design=True,
                           exp=0,
                           offset_min_skybrightness=0.0)
    print("Mugatu validation...")
    try:
        fps_design.validate_design()
    except:
        return fitsoutfname
    print(len(fps_design.targets_unassigned),fps_design.targets_unassigned)
    print(len(fps_design.targets_collided),fps_design.targets_collided)

    print(f"Finished designs for {field_id} at {racen:.5f} {deccen:+.5f} in {time.time()-start:.1f}sec ({fitsoutfname})")
    return fitsoutfname


def create_field(racen, deccen, targtab,
                 field_id=999999,
                 cacheDir=None,
                 observatory=None,
                 cadence="bright_1x1",
                 design_mode=None,
                 cartons=[],
                 cache=False):
    """
    Create a field from scratch given cartons and field center.

    Parameters
    ----------
    racen : float
        Right ascension of the field center in degrees.
    deccen : float
        Declination of the field center in degrees.
    targtab : numpy structured array
        Array of targets to assign.
    field_id : int, optional
        Identifier for the field. Default is 999999.
    cacheDir : str, optional
        Directory to store the output FITS file. If None, uses the ROBOSTRATEGY_DATA environment variable.
    observatory : str
        Observatory name, 'APO' or 'LCO'.
    cadence : str, optional
        Cadence string for the field. Default is "bright_1x1".
    design_mode : str, optional
        Design mode for the field. If None, uses the default design mode.
    cartons : list of dict, optional
        List of cartons to query. Each carton should be a dictionary with keys 'carton' and 'plan'.
    cache : bool, optional
        Whether to cache the results to disk. Default is False.
    
    Returns
    -------
    fitsoutfname : str
        Path to the output FITS file containing the assigned design.
    """

    targtab = query_field(racen, deccen, field_id=field_id, 
                          cartons=cartons, cache=cache, cacheDir=cacheDir, 
                          observatory=observatory)

    # priorities, counts = np.unique(targtab['priority'], return_counts=True)
    # for pri, cnt in zip(priorities, counts):
    #     print(f"Priority {pri}: {cnt} targets")

    assign_field(racen, deccen, targtab,
                 field_id=field_id,
                 cacheDir=cacheDir,
                 observatory=observatory,
                 cadence=cadence,
                 design_mode=design_mode)


def fields_from_file(input_yaml):
    """
    Create fields from a YAML input file.

    Parameters
    ----------
    input_yaml : str
        Path to the input YAML file containing field definitions.

    Returns
    -------
    None
    """

    with open(input_yaml, 'r') as f:
        field_params = yaml.safe_load(f)

    fields = field_params.get('fields', [])
    cacheDir = field_params.get('cacheDir', None)
    cache = field_params.get('cache', False)
    cartons = field_params.get('cartons', [])
    observatory = field_params.get('observatory', None)

    for field in fields:
        racen = field['racen']
        deccen = field['deccen']
        field_id = field.get('field_id', 999999)
        if len(cartons) == 0:
            cartons = field.get('cartons', [])
        cadence = field.get('cadence', "bright_1x1")
        design_mode = field.get('design_mode', None)

        print(f"Creating field {field_id} at RA={racen}, Dec={deccen} from file {input_yaml}")

        create_field(racen, deccen, targtab=None,
                     field_id=field_id,
                     cacheDir=cacheDir,
                     observatory=observatory,
                     cadence=cadence,
                     design_mode=design_mode,
                     cartons=cartons,
                     cache=cache)
