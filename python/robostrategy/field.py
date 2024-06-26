#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import re
import random
import datetime
import jinja2
import numpy as np
import fitsio
import collections
import matplotlib.pyplot as plt
import ortools.sat.python.cp_model as cp_model
import astropy.coordinates
import astropy.units
import PyAstronomy.pyasl as pyasl
import roboscheduler
import roboscheduler.cadence
import kaiju
import kaiju.robotGrid
import mugatu.designmode
import robostrategy
import robostrategy.targets
import robostrategy.header
import robostrategy.obstime as obstime
import robostrategy.params as params
import robostrategy.standards
import coordio.time
import coordio.utils
import sdss_access.path


sdss_path = sdss_access.path.Path(release='sdss5', preserve_envvars=True)

# Default collision buffer
defaultCollisionBuffer = 2.

# Default epoch to assume the catalog table has
default_catalog_epoch = 2015.5


# intersection of lists
def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))


# Type for targets array
targets_dtype = robostrategy.targets.target_dtype
targets_dtype = targets_dtype + [('x', np.float64),
                                 ('y', np.float64),
                                 ('z', np.float64),
                                 ('zone', np.int32),
                                 ('within', np.int32)]
                                 

design_status_dtype = np.dtype([('fieldid', np.int32),
                                ('designid', np.int32),
                                ('status', np.compat.unicode, 20)])

# Dictionary defining meaning of flags
_flagdict = {'STAGE_IS_NONE':1,
             'NOT_SCIENCE':2,
             'NOT_INCADENCE': 4,
             'NOT_COVERED': 8,
             'NONE_ALLOWED': 16,
             'NO_AVAILABILITY': 32,
             'ALREADY_ASSIGNED': 64}

_expflagdict = {'FIXED':1,
                'SRD':2,
                'REASSIGN':4,
                'OPEN':8,
                'FILLER':16,
                'COMPLETE':32,
                'OTHER':64,
                'FORCED':128}

_offsetdict = {'TOO_FAINT':1,
               'NO_FLUX':2,
               'TOO_CLOSE_FOR_MODE':4,
               'TOO_DARK':8,
               'NO_CAN_OFFSET':16,
               'TOO_BRIGHT':32}

__all__ = ['Field', 'read_field', 'read_cadences', 'AssignmentStatus',
           'read_bright_stars', 'write_bright_stars']


"""Field module class.
"""

# Establish access to the CadenceList singleton
clist = roboscheduler.cadence.CadenceList(skybrightness_only=True)


def read_cadences(plan=None, observatory=None, unpickle=False,
                  stage='srd'):
    """Convenience function to read a run's cadence list

    Parameters
    ----------

    plan : str
        plan name

    observatory : str
        observatory name ('apo' or 'lco')

    stage : str
        stage desired (default 'srd')

    Returns 
    -------

    clist : CadenceList
        CadenceList object with caden#ces (singleton)

    Notes
    -----

    stage only matters if set to 'Final', in which 
    case the final set of cadences is read
"""
    cadences_file = sdss_path.full('rsCadences', plan=plan,
                                   observatory=observatory)
    if(stage.capitalize() == 'Final'):
        cadences_file = cadences_file.replace('rsCadences',
                                              'final/rsCadencesFinal')
    clist.fromfits(filename=cadences_file, unpickle=unpickle)
    return clist


def read_field(plan=None, observatory=None, fieldid=None,
               stage='', targets=False, speedy=False,
               verbose=False, unpickle=False, oldmag=False,
               reset_bright=False, offset_min_skybrightness=None):
    """Convenience function to read a field object

    Parameters
    ----------

    plan : str
        plan name

    observatory : str
        observatory name ('apo' or 'lco')

    fieldid : int
        field id

    oldmag : bool
        if True, read in file with [N, 7] magnitude array (default False)

    reset_bright : bool
        if True, ignore bright star list in file, reload from db (default False)

    stage : str
        stage of assignments ('', 'Open', 'Filler', 'Reassign', 'Complete', 'Final')

    targets : bool
        if True, read rsFieldTargets file, do not set cadence (default False)

    speedy : bool
        if True, return a FieldSpeedy object (default False)

    unpickle : bool
        if set, read in pickled cadence_consistency cache  (default False)

    verbose : bool
        if set, be verbose (default False)

    Returns
    -------

    field : Field object
        field object read in
"""
    untrim_cadence_version = None

    rsParams = params.RobostrategyParams(plan=plan)

    cadence_version = None
    if('CadenceVersions' in rsParams.cfg):
        if('version' in rsParams.cfg['CadenceVersions']):
            cadence_version = rsParams.cfg.get('CadenceVersions', 'version')

    cadences_file = sdss_path.full('rsCadences', plan=plan,
                                   observatory=observatory)

    base = 'rsFieldAssignments'
    if(targets):
        base = 'rsFieldTargets'

    field_file = sdss_path.full(base,
                                plan=plan, observatory=observatory,
                                fieldid=fieldid)
    if(stage.capitalize() == 'Reassign'):
        field_file = field_file.replace('rsFieldAssignments',
                                        'rsFieldReassignments')
    if(stage.capitalize() == 'Open'):
        field_file = field_file.replace(base, base + 'Open')
    if(stage.capitalize() == 'Filler'):
        field_file = field_file.replace(base, base + 'Filler')
    if(stage.capitalize() == 'Complete'):
        field_file = field_file.replace(base, base + 'Complete')
    if(stage.capitalize() == 'Final'):
        field_file = field_file.replace('targets/' + base,
                                        'final/' + base + 'Final')
        cadences_file = cadences_file.replace('rsCadences',
                                              'final/rsCadencesFinal')
        untrim_cadence_version = cadence_version

    clist.fromfits(filename=cadences_file, unpickle=unpickle)

    if(speedy):
        f = FieldSpeedy(filename=field_file, fieldid=fieldid,
                        verbose=verbose, oldmag=oldmag, reset_bright=reset_bright)
    else:
        f = Field(filename=field_file, fieldid=fieldid, verbose=verbose,
                  untrim_cadence_version=untrim_cadence_version, oldmag=oldmag,
                  reset_bright=reset_bright,
                  offset_min_skybrightness=offset_min_skybrightness)

    return(f)


def read_bright_stars(fits=None, include_extname=False):
    """Read in bright stars from FITS file

    Parameters
    ----------

    fits : fitsio.FITS object
        FITS file object for reading from

    include_extname : bool
        if True, include extname as part of the key

    Returns
    -------

    bsDict : OrderedDict of ndarrays
        dictionary containing arrays of bright stars as values

    Notes
    -----

    Structure of dictionary is that it has keys which are tuples
    (design_mode, fiberType, extname), and whose values are the 
    ndarray.

    If no_extname is set, then the keys are (design_mode, fiberType)
"""
    bsDict = collections.OrderedDict()
    nbs = 0
    while('bs{n}'.format(n=nbs) in fits.hdu_map):
        nbs = nbs + 1
    for ibs in range(nbs):
        extname = 'BS{ibs}'.format(ibs=ibs)
        bshdr = fits[extname].read_header()
        bs = fits[extname].read()
        design_mode = bshdr['DESMODE']
        fiberType = bshdr['FIBERTY']
        if(include_extname):
            key = (design_mode, fiberType, extname)
        else:
            key = (design_mode, fiberType)
        bsDict[key] = bs
    return(bsDict)


def write_bright_stars(filename=None, bright_stars=None, clobber=False):
    """Write out bright stars

    Parameters
    ----------

    filename : str
        name of output file

    bright_stars : Dict of ndarrays
        dictionary with (design_mode, fiberType) keys and bright star array values
    
    clobber : bool
        whether to clobber file or add HDU
"""
    nbs = 0
    doclobber = clobber
    for design_mode, fiberType in bright_stars.keys():
        hdr = robostrategy.header.rsheader()
        hdr.append({'name':'DESMODE',
                    'value':design_mode,
                    'comment':'Bright stars for this design mode'})
        hdr.append({'name':'FIBERTY',
                    'value':fiberType,
                    'comment':'Bright stars for this fiber type'})
        fitsio.write(filename, bright_stars[(design_mode, fiberType)],
                     header=hdr,
                     extname='BS{n}'.format(n=nbs),
                     clobber=doclobber)
        doclobber = False
        nbs = nbs + 1
    return


class AssignmentStatus(object):
    """Status of a prospective assignment for a set of exposures

    Parameters
    ----------

    rsid : np.int64
        prospective target

    robotID : np.int32
        prospective robotID

    iexps : ndarray of np.int32
        prospective exposure numbers

    Attributes
    ----------

    already : ndarray of bool
        [len(iexps)] does the rsid already have an equivalent observation this exposure?

    assignable : ndarray of bool
        [len(iexps)] is the fiber free to assign and uncollided in exposure? 
        (initialized to True)

    bright_neighbor_allowed : ndarray of bool
        [len(iexps)] is this free of a bright neighbor (initialized to True)

    collided : ndarray of bool
        [len(iexps)] is the fiber collided in exposure? (initialized to False)

    expindx : ndarray of np.int32
        mapping of iexp (exposure within field cadence) to index of iexps array

    iexps : ndarray of np.int32
        [len(iexps)] prospective exposure numbers

    robotID : np.int32
        prospective robotID

    rsid : np.int64
        prospective target

    spare : ndarray of bool
        [len(iexps)] is fiber already assigned a spare calibration target in exposure?
        (initialized to False)

    spare_colliders : list of ndarrays of np.int32
        [len(iexps)] list of arrays of spare calibration targets that assignment collides with
        (initialized to list of empty arrays)

    Notes
    -----

    These objects are used to track information about prospective
    assignments. They only make sense in the context of the Field
    class, which has several methods to manipulate these objects.

    bright_neighbor checks both the APOGEE and BOSS fibers on the 
    robot.

    There is a certain fragility to this code, since AssignmentStatus
    does not check (and cannot check) whether a given robotID can 
    reach a given rsid. So using it properly relies on Field only
    creating an AssignmentStatus in appropriate cases.
"""
    def __init__(self, rsid=None, robotID=None, iexps=None):
        if(rsid is not None):
            self.rsid = np.int64(rsid)
        else:
            self.rsid = None
        self.robotID = np.int32(robotID)
        self.iexps = iexps
        self.expindx = np.zeros(iexps.max() + 1, dtype=np.int32) - 1
        self.expindx[iexps] = np.arange(len(iexps), dtype=np.int32)
        self.assignable = np.ones(len(self.iexps), dtype=bool)
        self.already = np.zeros(len(self.iexps), dtype=bool)
        self.collided = np.zeros(len(self.iexps), dtype=bool)
        self.bright_neighbor_allowed = np.ones(len(self.iexps), dtype=bool)
        self.spare = np.zeros(len(self.iexps), dtype=bool)
        self.spare_colliders = [np.zeros(0, dtype=np.int64)] * len(self.iexps)
        self.currindx = None
        self.locked = None
        return

    def __str__(self):
        template = """Assignment Status: rsid = {rsid} robotID = {robotID}
  iexps = {iexps}
  expindx = {expindx}
  assignable = {assignable}
  already = {already}
  collided = {collided}
  bright_neighbor_allowed = {bright_neighbor_allowed}
  spare = {spare}
  spare_colliders = {spare_colliders}
  currindx = {currindx}
  locked = {locked}
  assignable_exposures = {assignable_exposures}
"""
        template = template.format(rsid=self.rsid, robotID=self.robotID,
                                   iexps=self.iexps, expindx=self.expindx,
                                   assignable=self.assignable,
                                   already=self.already, collided=self.collided,
                                   bright_neighbor_allowed=self.bright_neighbor_allowed,
                                   spare=self.spare,
                                   spare_colliders=self.spare_colliders,
                                   currindx=self.currindx, locked=self.locked,
                                   assignable_exposures=self.assignable_exposures())
        return(template)


    def assignable_exposures(self):
        """List of assignable exposures
        
        Returns
        -------
        
        iexps : ndarray of np.int32
            list of assignable exposures
"""
        return(self.iexps[np.where(self.assignable)[0]])


class Field(object):
    """Field class

    Parameters
    ----------

    filename : str
        if set, reads from file (ignores other inputs)

    fieldid : np.int32
        field ID number (default 1)

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    pa : np.float32
        position angle of field (deg E of N) (default 0)

    observatory : str
        observatory field observed from, 'apo' or 'lco' (default 'apo')
    
    field_cadence : str
        field cadence (default 'none')

    nocalib : bool
        if True, do not account for calibrations (default False)

    allgrids : bool
        if True, keep track of all robotgrids (default True); if False
        automatically sets nocollide to True

    reload_design_mode : bool
        if True, will reload design mode dictionary from targetdb (default False)

    input_design_mode : designModeDict
        used this as design mode dict

    nocollide : bool
        if True,  do not check collisions (default False)

    nooffset : bool
        if True, treat all targets as if can_offset is False
    
    bright_neighbors : bool
        if True, check bright neighbor conditions (default False)

    reset_bright : bool
        if True, doesn't read bright stars with fromfits() (default False)

    verbose : bool
        if True, issue a lot of output statements (default False)

    veryverbose : bool
        if True, really issue a lot of output statements (default False)

    Attributes
    ----------

    achievable_calibrations : OrderedDict
        dictionary of lists with number of achievable calibration
        sources for each exposure specified for 'sky_boss',
        'standard_boss', 'sky_apogee', 'standard_apogee' (i.e. equal to
        required_calibrations if they all can be achieved even without
        science targets, or the maximum possible if less than that).

    allgrids : bool
        if True, keep track of all robotgrids (default True); if False
        automatically sets nocollide to True

    assignments_dtype : dtype
        structure for assignments array (depends on field cadence)

    assignments : ndarray or None
        [len(targets)] array; set to None prior to set_field_cadence() call

    bright_neighbors : bool
        True if checking for bright neighbor

    bright_stars : OrderedDict
        dictionary with keys (designmode, instrument), and 
        values which are ndarrays with bright star information

    bright_stars_coords : OrderedDict
        dictionary with keys (design_mode, fiberType), and 
        values which are SkyCoord objects with bright stars coords

    bright_stars_rmax : OrderedDict
        dictionary with keys (design_mode, fiberType), and 
        values which are the maximum exclusion radius in 
        the corresponding bright_stars entry

    bright_neighbor_cache : Dict
        dictionary with keys (rsid, robotID, design_mode) caching
        whether the given combination is allowed

    calibrations : OrderedDict
        dictionary of lists with numbers of calibration sources assigned
        for each epoch for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    calibration_order : ndarray of str
        Ordering of calibrations to perform assignment; note that spares
        are preferentially assigned to the later listed; default is
        ['sky_apogee', 'sky_boss', 'standard_apogee', 'standard_boss']

    collisionBuffer : float
        collision buffer for kaiju (in mm) IGNORED

    deccen : np.float64
        boresight Dec, J2000 deg

    designModeDict : dict of DesignMode objects
        possible design modes

    design_mode : np.array of str
        keys to DesignModeDict for each epoch

    expflagdict : Dict
        dictionary of exposure flag values

    exposure_locked : ndarray of bool
        whether the exposure has been locked because it is done

    field_cadence : Cadence object
        cadence associated with field (set to None prior to set_field_cadence()

    flagdict : Dict
        dictionary of assignment flag values

    methods : Dict
        dictionary to set methods within assignment (not to be used)

    mastergrid : RobotGrid object
        robotGrid used to inquire about robots & targets (not for assignment)

    nocalib : bool
        if True, do not account for calibrations

    nocollide : bool
        if True,  do not check collisions

    observatory : str
        observatory field observed from ('apo' or 'lco')

    obstime : coordio Time object
        nominal time of observation to use for calculating x/y

    pa : np.float32
        position angle of field (deg E of N)

    preferred_robotids : ndarray of np.int32
        [ntargets, nexp_total] robotIDs to prefer for each target (or None)

    racen : np.float64
        boresight RA, J2000 deg

    radius : np.float32
        distance from racen, deccen to search for for targets (deg);
        set to 1.5 for observatory 'apo' and 0.95 for observatory 'lco'

    reset_bright : bool
        if True, will not load bright stars in fromfits()

    reload_design_mode : bool
        if True, will reload design mode dictionary from targetdb

    input_design_mode : designModeDict
        used this as design mode dict

    required_calibrations : OrderedDict
        dictionary with numbers of required calibration sources specified
        for each exposure, for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    required_calibrations_per_zone : OrderedDict
        dictionary with numbers of required calibrations per zone
        'sky_boss', 'standard_boss', 'sky_apogee', 'standard_apogee'

    robotIDs : ndarray of np.int32
        robotID values in order given by RobotGrid object's robotDict dictionary

    robotID2indx : Dict
        for each key robotID, the value is its 0-indexed position in robotDict
        (i.e. inverse of robotIDs)

    robotgrids : list of RobotGrid objects
        robotGrids associated with each exposure

    robotHasApogee : ndarray of bool
        whether each robotID has an APOGEE fiber

    rsid2indx : Dict
        dictionary linking rsid (key) to index of targets and assignments arrays.
        (values). E.g. targets['rsid'][f.rsid2indx[rsid]] == rsid

    stage : str
        current stage (used for setting bits)

    targets : ndarray
        array of targets

    target_duplicated : ndarray of np.int32
        [len(targets)] initially 0s; set in assign() if there are
        coordinated targets which have already been assigned

    verbose : bool
        if True, issue a lot of output statements

    veryverbose : bool
        if True, really issue a lot of output statements

    _calibration_index : ndarray of np.int32
        [len(targets)] indicates which type of calibration target this object
        is; 0 for a science target, and 1..4 for each of the required_calibration
        categories in order.

    _competing_targets : ndarray of np.int32
        [nrobots] count of how many targets are competing for a given
        robot; used only in certain methods of assignment.

    _equivindx : OrderedDict
        keys are tuples (catalogid, fiberType, lambda_eff, delta_ra, 
        delta_dec), values are ndarray of np.int32, with 0-indexed 
        positions in targets of all targets with those settings

    _equivkey : OrderedDict
        keys are 0-indexed positions in targets, values are tuples of
        that target's (catalogid, fiberType, lamdda_eff, delta_ra,
        delta_dec), to quickly reference _equivindx

    _has_spare_calib : 2D ndarray of bool
        [5, nexp_total] indicates whether there are spare calibrations
        in any category target in this exposure. The first axis is
        in the order 'science' and then the calibrations as specified
        in calibration_order

    _is_calibration : ndarray of bool
        [len(targets)] list of whether the target is a calibration target

    _ot : ObsTime object
        observing time object for convenience

    _robot_locked : ndarray of bool
        True if the robot is locked from use; used to block robots from use if necessary

    _robot2indx : ndarray of int32 or None
        [nrobots, nexp_total] array of indices into targets from robots

    _robotnexp : ndarray of int32 or None
        [nrobots, nepochs] array of number of exposures unused per epoch

    _robotnexp_max : ndarray of int32 or None
        [nrobots, nepochs] array of number of exposures unused by science
        fibers in each epoch

    _unique_catalogids : ndarray of np.int64
        list of unique catalogids for convenience

    Notes
    -----

    Before creating a field object, you will typically need to
    instantiate the singleton CadenceList through roboscheduler, and
    make sure it has the cadences in it that the field will need.

    Instantiating a field will create (or replace) cadences
    _field_single_1x1 and _field_single_12x1 in CadenceList. These
    dummy cadences are used to identify target cadences that are just
    strings of unrelated exposures.

"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=None,
                 fieldid=1, allgrids=True, nocalib=False, nocollide=False,
                 bright_neighbors=True, verbose=False, veryverbose=False,
                 trim_cadence_version=False, untrim_cadence_version=None,
                 noassign=False, oldmag=False, reload_design_mode=False,
                 input_design_mode=None, reset_bright=False,
                 offset_min_skybrightness=None, nooffset=False):
        self.calibration_order = np.array(['sky_apogee', 'sky_boss',
                                           'standard_boss', 'standard_apogee'])
        self._add_dummy_cadences()
        self.flagdict = _flagdict
        self.expflagdict = _expflagdict
        self.offset_min_skybrightness = offset_min_skybrightness
        self.design_status = None
        self.stage = None
        self.nooffset = nooffset
        self.preferred_robotids = None
        self.verbose = verbose
        self.oldmag = oldmag
        self.veryverbose = veryverbose
        self._trim_cadence_version = trim_cadence_version  # trims field cadence
        self._untrim_cadence_version = untrim_cadence_version  # adds version to target cadence
        self.fieldid = fieldid
        self.nocalib = nocalib
        self.nocollide = nocollide
        self.allgrids = allgrids
        self.reload_design_mode = reload_design_mode
        self.input_design_mode = input_design_mode
        self.reset_bright = reset_bright
        self.bright_neighbors = bright_neighbors
        if(self.bright_neighbors):
            self.bright_stars = collections.OrderedDict()
            self.bright_stars_coords = collections.OrderedDict()
            self.bright_stars_rmax = collections.OrderedDict()
            self.bright_neighbor_cache = dict()
        if(self.bright_neighbors):
            self.fmagloss = coordio.utils.Moffat2dInterp()
        else:
            self.fmagloss = None
        self.robotHasApogee = None
        self.collisionBuffer = collisionBuffer
        if(self.allgrids is False):
            self.nocollide = True
        if(self.nocollide):
            self.allgrids = False
        if(self.allgrids):
            self.robotgrids = []
        else:
            self.robotgrids = None
        self.assignments = None
        self._has_spare_calib = None
        self.rsid2indx = dict()
        self.targets = np.zeros(0, dtype=targets_dtype)
        self.target_duplicated = np.zeros(0, dtype=np.int32)
        self._is_calibration = np.zeros(0, dtype=bool)
        self._is_good_calibration = np.zeros(0, dtype=bool)
        self._calibration_index = np.zeros(1, dtype=bool)
        self._unique_catalogids = None
        if(filename is not None):
            if(self.verbose):
                print("fieldid {fid}: Reading from {f}".format(f=filename, fid=self.fieldid), flush=True)
            self.fromfits(filename=filename, noassign=noassign)
        else:
            self.racen = racen
            self.deccen = deccen
            self.pa = pa
            self.observatory = observatory
            self._ot = obstime.ObsTime(observatory=self.observatory)
            self.obstime = coordio.time.Time(self._ot.nominal(lst=self.racen))
            if(self.collisionBuffer is None):
                self.collisionBuffer = defaultCollisionBuffer
            self._set_masterGrid()
            self.robotIDs = np.array([x for x in self.mastergrid.robotDict.keys()],
                                     dtype=int)
            self.robotID2indx = dict()
            for indx, robotID in enumerate(self.robotIDs):
                self.robotID2indx[robotID] = indx
            self.designModeDict = mugatu.designmode.allDesignModes() 
            if(self.designModeDict is None):
                print("Using default design modes.")
                default_dm_file= os.path.join(os.getenv('ROBOSTRATEGY_DIR'),
                                              'data',
                                              'default_designmodes.fits')
                mugatu.designmode.allDesignModes(filename=default_dm_file)
            if(self.nocalib is False):
                self.required_calibrations = collections.OrderedDict()
                for n in self.calibration_order:
                    self.required_calibrations[n] = np.zeros(0, dtype=np.int32)
                self.required_calibrations_per_zone = np.zeros(len(self.calibration_order) + 1, dtype=int) - 1
                for i, n in enumerate(self.calibration_order):
                    self.required_calibrations_per_zone[i + 1] = -1
                self.calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                       0, robostrategy.standards.nzone),
                                                      dtype=np.int32)
                self.achievable_calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                                  0, robostrategy.standards.nzone),
                                                                 dtype=np.int32)
                self.calibrations = collections.OrderedDict()
                for n in self.calibration_order:
                    self.calibrations[n] = np.zeros(0, dtype=np.int32)
                self.achievable_calibrations = collections.OrderedDict()
                for n in self.calibration_order:
                    self.achievable_calibrations[n] = self.required_calibrations[n].copy()
            self.set_field_cadence(field_cadence)
        self._set_radius()
        self._competing_targets = None
        self.methods = dict()
        self.methods['assign_epochs'] = 'first'
        return

    def set_stage(self, stage=None):
        self.stage = stage
        if(self.verbose):
            print("fieldid {fid}: Setting stage {stage}".format(fid=self.fieldid, stage=stage), flush=True)
        return

    def query_bright_stars(self, design_mode=None,
                           fiberType=None):
        """Retrieve bright stars to avoid

        Parameters
        ----------

        design_mode : str
            name of design mode

        fiberType : str
            fiber type ('APOGEE' or 'BOSS')

        Returns
        -------

        bright_stars : ndarray
            array of bright stars, with columns 'ra', 'dec', 'mag',
            'catalogid', 'r_exclude'

        Notes
        -----

        r_exclude is the radius of the exclusion zone for each star
"""
        bright = 'bright' in design_mode
        if(bright):
            lunation = 'bright'
        else:
            lunation = 'dark'
        mag_lim = self._mag_lim(design_mode=design_mode,
                                fiberType=fiberType)
        mag_limits = self._mag_limits(design_mode=design_mode,
                                      fiberType=fiberType)

        db_query = mugatu.designmode.build_brigh_neigh_query('designmode',
                                                             fiberType,
                                                             mag_lim,
                                                             self.racen,
                                                             self.deccen)

        bright_stars_dtype = np.dtype([('catalog_ra', np.float64),
                                       ('catalog_dec', np.float64),
                                       ('ra', np.float64),
                                       ('dec', np.float64),
                                       ('pmra', np.float32),
                                       ('pmdec', np.float32),
                                       ('mag', np.float32),
                                       ('catalogid', np.int64),
                                       ('r_exclude', np.float32)])

        if len(db_query) > 0:
            if isinstance(db_query, tuple):
                ras, decs, mags, catalogids, pmra, pmdec = db_query
            else:
                ras, decs, mags, catalogids, pmra, pmdec = map(list, zip(*list(db_query.tuples())))

            r_exclude, dummy = coordio.utils.offset_definition(mags,
                                                               mag_limits,
                                                               lunation,
                                                               fiberType.capitalize(),
                                                               self.observatory.upper(),
                                                               safety_factor=0.,
                                                               fmagloss=self.fmagloss)
            
            bright_stars = np.zeros(len(ras), dtype=bright_stars_dtype)

            bright_stars['catalog_ra'] = ras
            bright_stars['catalog_dec'] = decs
            bright_stars['mag'] = mags
            bright_stars['pmra'] = pmra
            bright_stars['pmdec'] = pmdec
            badpm = ((np.isfinite(bright_stars['pmra']) == False) |
                     (np.isfinite(bright_stars['pmdec']) == False))
            bright_stars['pmra'][badpm] = 0.
            bright_stars['pmdec'][badpm] = 0.
            bright_stars['catalogid'] = catalogids
            bright_stars['r_exclude'] = r_exclude

            epoch = (np.zeros(len(bright_stars), dtype=np.float32) +
                     default_catalog_epoch)
            x, y, z = self.radec2xyz(ra=bright_stars['catalog_ra'],
                                     dec=bright_stars['catalog_dec'],
                                     epoch=epoch,
                                     pmra=bright_stars['pmra'],
                                     pmdec=bright_stars['pmdec'],
                                     fiberType=fiberType)
            (bright_stars['ra'], 
             bright_stars['dec']) = self.xy2radec(x=x, y=y, fiberType=fiberType)

        else:
            bright_stars = np.zeros(0, dtype=bright_stars_dtype)

        return(bright_stars)

    def set_bright_stars(self, design_mode=None,
                         fiberType=None,
                         bright_stars=None,
                         reset=False):
        """Records in attributes which bright stars to avoid

        Parameters
        ----------

        design_mode : str
            design mode

        fiberType : str
            fiber type ('APOGEE' or 'BOSS')

        bright_stars : ndarray
            bright stars, if setting explicitly (default None) 

        reset : bool
            force a reset if dictionary element already set

        Notes
        -----

        Adds an element to dictionaries bright_stars, 
        bright_stars_coords, and bright_stars_rmax, corresponding
        to this design_mode and fiberType

        If an element with the key (design_mode, fiberType) already
        exists, will only reset it if reset=True; otherwise it just 
        leaves it alone.

        If the input bright_stars is None, the method query_bright_stars() 
        is used to retrieve bright stars from targetdb and put into a 
        value in the bright_stars dictionary. If the input bright_stars is 
        not None, it will be used as that value instead.

        The SkyCoord version of the coordinates and the maximum r_exclude
        are stored in the bright_stars_coords and bright_stars_rmax
        dictionaries.
"""
        if(((design_mode, fiberType) in self.bright_stars.keys()) &
           (reset is False)):
            if(self.verbose):
                print("fieldid {fid}: Already got bright stars for {d}, {f}".format(d=design_mode, f=fiberType, fid=self.fieldid), flush=True)
            return

        if(self.verbose):
            print("fieldid {fid}: Getting bright stars for {d}, {f}".format(fid=self.fieldid, d=design_mode, f=fiberType), flush=True)

        if(bright_stars is None):
            print("fieldid {f}: Start bright star queries {t}".format(f=self.fieldid, t=datetime.datetime.today()), flush=True)
            bright_stars = self.query_bright_stars(design_mode=design_mode,
                                                   fiberType=fiberType)
            print("fieldid {f}: Finish bright star queries {t}".format(f=self.fieldid, t=datetime.datetime.today()), flush=True)

        if(self.verbose):
            print("fieldid {fid}: found {n} bright stars".format(fid=self.fieldid, n=len(bright_stars)), flush=True)

        if(len(bright_stars) > 0):
            bright_stars_coords = astropy.coordinates.SkyCoord(bright_stars['ra'],
                                                               bright_stars['dec'],
                                                               frame='icrs',
                                                               unit='deg')
            bright_stars_rmax = bright_stars['r_exclude'].max()
        else:
            bright_stars_coords = None
            bright_stars_rmax = None

        self.bright_stars[(design_mode, fiberType)] = bright_stars
        self.bright_stars_coords[(design_mode, fiberType)] = bright_stars_coords
        self.bright_stars_rmax[(design_mode, fiberType)] = bright_stars_rmax
        return

    def _bright_allowed_direct(self, design_mode=None, targets=None,
                               assignments=None):
        """Report which input targets are not too close to a bright neighbor

        Parameters
        ----------

        design_mode : str
            design mode to make determination for

        targets : ndarray
            some elements of the targets ndarray

        assignments : ndarray
            some elements of the assignments ndarray

        Returns
        -------

        bright_allowed : ndarray of bool
            for each element of targets, True if allowed, False otherwise

        Notes
        -----

        This bright allowance only checks the fiber used for the target.
        This method is appropriate to use to check targets before they 
        are assigned a specific robot. Once a specific robotID is under
        consideration, the other fiber on the robot needs to be checked
        too (with _bright_allowed_robot).
"""
        bright_allowed = np.ones(len(targets), dtype=bool)
        target_coords = astropy.coordinates.SkyCoord(assignments['fiber_ra'],
                                                     assignments['fiber_dec'],
                                                     frame='icrs',
                                                     unit='deg')
        for fiberType in ['APOGEE', 'BOSS']:
            bright = self.bright_stars[(design_mode, fiberType)]
            if(len(bright) > 0):
                rmax = self.bright_stars_rmax[(design_mode, fiberType)]
                bright_coords = self.bright_stars_coords[(design_mode, fiberType)]
                itype = np.where(targets['fiberType'] == fiberType)[0]
                ibright, itargets, d2d, d3d = astropy.coordinates.search_around_sky(bright_coords, target_coords[itype], rmax * astropy.units.arcsec)
                itooclose = np.where(d2d < bright['r_exclude'][ibright] *
                                     astropy.units.arcsec)[0]
                bright_allowed[itype[itargets[itooclose]]] = 0
        return(bright_allowed)

    def _bright_allowed_robot(self, rsid=None, robotID=None,
                              design_mode=None):
        """Reports if bright neighbor considerations allow an assignment

        Parameters
        ----------

        rsid : np.int64
            rsid of target in assignment
        
        robotID : int
            robotID of robot in assignment

        design_mode : str
            design mode to consider

        Returns
        -------

        allowed : bool
            True if the assignment is allowed by bright neighbor considerations
            and False if not
"""
        try:
            self.mastergrid.assignRobot2Target(robotID, rsid)
        except RuntimeError:
            print("assignRobot2Target failure", flush=True)
            print("robotID={r}".format(r=robotID), flush=True)
            print("rsid={r}".format(r=rsid), flush=True)
            irobot = self.robotID2indx[robotID]
            print("{rl}".format(rl=self._robot_locked[irobot, :]))
            sys.exit()

        x = dict()
        y = dict()
        
        x['BOSS'] = self.mastergrid.robotDict[robotID].bossWokXYZ[0]
        y['BOSS'] = self.mastergrid.robotDict[robotID].bossWokXYZ[1]
        x['APOGEE'] = self.mastergrid.robotDict[robotID].apWokXYZ[0]
        y['APOGEE'] = self.mastergrid.robotDict[robotID].apWokXYZ[1]

        for fiberType in ['APOGEE', 'BOSS']:
            bright = self.bright_stars[(design_mode, fiberType)]
            if(len(bright) > 0):
                ra_robo, dec_robo = self.xy2radec(x=np.array([x[fiberType],
                                                              x[fiberType]]),
                                                  y=np.array([y[fiberType],
                                                              y[fiberType]]),
                                                  fiberType=fiberType)
                sep = pyasl.getAngDist(ra_robo[0], dec_robo[0],
                                       bright['ra'],
                                       bright['dec']) * 3600.
                itooclose = np.where(sep < bright['r_exclude'])[0]
                if(len(itooclose) > 0):
                    self.mastergrid.unassignTarget(rsid)
                    return(False)
            
        self.mastergrid.unassignTarget(rsid)
        return(True)
        
    def _add_dummy_cadences(self): 
        """Adds some dummy cadences for singlebright and multibright"""
        clist.add_cadence(name='_field_single_1x1',
                          nepochs=1,
                          skybrightness=[1.],
                          delta=[-1.],
                          delta_min=[-1.],
                          delta_max=[-1.],
                          nexp=[1],
                          max_length=[9999999.],
                          min_moon_sep=15.,
                          min_deltav_ks91=-2.5,
                          min_twilight_ang=8.,
                          max_airmass=2.)
        clist.add_cadence(name='_field_single_12x1',
                          nepochs=12,
                          skybrightness=[1.] * 12,
                          delta=[-1.] * 12,
                          delta_min=[-1.] * 12,
                          delta_max=[-1.] * 12,
                          nexp=[1] * 12,
                          max_length=[9999999.] * 12,
                          min_moon_sep=[15.] * 12,
                          min_deltav_ks91=[-2.5] * 12,
                          min_twilight_ang=[8.] * 12,
                          max_airmass=[2.] * 12)
        clist.add_cadence(name='_field_dark_single_1x1',
                          nepochs=1,
                          skybrightness=[0.35],
                          delta=[-1.],
                          delta_min=[-1.],
                          delta_max=[-1.],
                          nexp=[1],
                          max_length=[9999999.],
                          min_moon_sep=[35.],
                          min_deltav_ks91=[-1.5],
                          min_twilight_ang=[15.],
                          max_airmass=[1.6])
        clist.add_cadence(name='_field_dark_single_12x1',
                          nepochs=12,
                          skybrightness=[0.35] * 12,
                          delta=[-1.] * 12,
                          delta_min=[-1.] * 12,
                          delta_max=[-1.] * 12,
                          nexp=[1] * 12,
                          max_length=[9999999.] * 12,
                          min_moon_sep=[35.] * 12,
                          min_deltav_ks91=[-1.5] * 12,
                          min_twilight_ang=[15.] * 12,
                          max_airmass=[1.6] * 12)
        return

    def fromfits(self, filename=None, noassign=False):
        """Read field from FITS file

        Parameters
        ----------

        filename : str
            name of file to read in

        noassign : bool
            if True, do not apply assignments in file (default False)

        Notes
        -----

        Expects header keywords:
       
          * FIELDID
          * RACEN
          * DECCEN
          * PA
          * OBS

        If NOCALIB is in header and the nocalib attribute is 
        False, nocalib will be set according to the keyword.

        If FCADENCE is not 'none', field cadence will be set

        Required calibrations are stored in RCNAME# and RCNUM#
        keywords. Each RCNAME# (RCNAME0, RCNAME1, etc) will have 
        the names of the calibration type. Each RCNUM# will have 
        a set of white-space-separated numbers with the number 
        of required calibrations for each exposure.

        IF ACNAME# and ACNUM# are set in header (and HDU named
        ASSIGN is present) these are interpreted as the "achievable 
        calibrations" and stored in achievable_calibrations.

        Expects HDUs named as follows:

         * TARGET : has the targets array (usable by targets_fromarray())
         * ASSIGN : if it exists, has the assignments array with assignments for each target and exposure
         * DESMODE : if it exists, has the definitions of design modes
         * BS# : bright stars for neighbor checks (with DESMODE & FIBERTY specified in header)

        This method does not copy the assignments table directly, it adds 
        the assignments using assign_robot_exposure(), so all columns in 
        that HDU other than 'robotID' and 'rsflags' are ignored.

        In the context of a robostrategy run, this method can read in
        an rsFieldTargets file (i.e. the input files to assignment)
        or an rsFieldAssignments file (i.e. the output files from
        assignment).
"""
        f = fitsio.FITS(filename)
        hdr = f[0].read_header()
        if('FIELDID' in hdr):
            self.fieldid = np.int32(hdr['FIELDID'])
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        if('BRIGHTN' in hdr):
            self.bright_neighbors = np.bool(hdr['BRIGHTN'])
        if(self.bright_neighbors):
            self.bright_stars = collections.OrderedDict()
            self.bright_stars_coords = collections.OrderedDict()
            self.bright_stars_rmax = collections.OrderedDict()
            self.bright_neighbor_cache = dict()
        if(self.collisionBuffer is None):
            self.collisionBuffer = hdr['CBUFFER']
        if(('NOCALIB' in hdr) & (self.nocalib == False)):
            self.nocalib = np.bool(hdr['NOCALIB'])
        if(('OFFMINSKY' in hdr) & (self.offset_min_skybrightness is None)):
            self.offset_min_skybrightness = np.float32(hdr['OFFMINSKY'])
            if(self.offset_min_skybrightness != self.offset_min_skybrightness):
                self.offset_min_skybrightness = None
        self._set_masterGrid()
        self.robotIDs = np.array([x for x in self.mastergrid.robotDict.keys()],
                                 dtype=int)
        self.robotID2indx = dict()
        for indx, robotID in enumerate(self.robotIDs):
            self.robotID2indx[robotID] = indx
        self._ot = obstime.ObsTime(observatory=self.observatory)
        self.obstime = coordio.time.Time(self._ot.nominal(lst=self.racen))
        field_cadence = hdr['FCADENCE']
        if(self._untrim_cadence_version is not None):
            if(field_cadence != 'none'):
                if(field_cadence.split('_')[-1] != self._untrim_cadence_version):
                    field_cadence = field_cadence + '_' + self._untrim_cadence_version
        if(self._trim_cadence_version):
            w = field_cadence.split('_')
            if(w[-1][0] == 'v'):
                field_cadence = '_'.join(w[:-1])
        if(self.nocalib is False):
            self.required_calibrations = collections.OrderedDict()
            for name in hdr:
                m = re.match('^RCNAME([0-9]*)$', name)
                if(m is not None):
                    num = 'RCNUM{d}'.format(d=m.group(1))
                    if(num in hdr):
                        if(hdr[num].strip() != ''):
                            self.required_calibrations[hdr[name]] = np.array([np.int32(np.float32(x)) for x in hdr[num].split()], dtype=np.int32)
                        else:
                            self.required_calibrations[hdr[name]] = np.zeros(0, dtype=np.int32)
            self.required_calibrations_per_zone = np.zeros(len(self.calibration_order) + 1, dtype=int) - 1
            for name in hdr:
                m = re.match('^CPZNAME([0-9]*)$', name)
                if(m is not None):
                    num = 'CPZNUM{d}'.format(d=m.group(1))
                    if(num in hdr):
                        ireq = list(self.calibration_order).index(hdr[name]) + 1
                        if(hdr[num].strip() != ''):
                            self.required_calibrations_per_zone[ireq] = np.int32(hdr[num])
                        else:
                            self.required_calibrations_per_zone[ireq] = -1
            self.calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                   0, robostrategy.standards.nzone),
                                                  dtype=np.int32)
            self.achievable_calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                              0, robostrategy.standards.nzone),
                                                             dtype=np.int32)
            self.calibrations = collections.OrderedDict()
            for n in self.calibration_order:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
            self.achievable_calibrations = collections.OrderedDict()
            for n in self.calibration_order:
                self.achievable_calibrations[n] = self.required_calibrations[n].copy()

        if(self.input_design_mode is not None):
            if(self.verbose):
                print("fieldid {fid}: Design mode from input".format(fid=self.fieldid), flush=True)
            self.designModeDict = self.input_design_mode
        elif(self.reload_design_mode):
            if(self.verbose):
                print("fieldid {fid}: Design mode from targetdb".format(fid=self.fieldid), flush=True)
            self.designModeDict = mugatu.designmode.allDesignModes() 
        else:
            try:
                if(self.verbose):
                    print("fieldid {fid}: Design mode from field file".format(fid=self.fieldid), flush=True)
                self.designModeDict = mugatu.designmode.allDesignModes(filename,
                                                                       ext='DESMODE')

            except:
                if(self.verbose):
                    print("fieldid {fid}: Design mode from defaults file", flush=True)
                default_dm_file= os.path.join(os.getenv('ROBOSTRATEGY_DIR'),
                                              'data',
                                              'default_designmodes.fits')
                self.designModeDict = mugatu.designmode.allDesignModes(default_dm_file)

        if((self.reset_bright is False) & (self.bright_neighbors is True)):
            bsDict = read_bright_stars(fits=f)
            for design_mode, fiberType in bsDict:
                bs = bsDict[design_mode, fiberType]
                self.set_bright_stars(design_mode=design_mode,
                                      fiberType=fiberType,
                                      bright_stars=bs)

        self.set_field_cadence(field_cadence)

        if('EXPLOCK' in hdr):
            self.exposure_locked = np.array([np.bool(x == 'True') for x in hdr['EXPLOCK'].split()])

        if('status' in f.hdu_map):
            design_status = f['status'].read()
            self.set_design_status(design_status=design_status)
        else:
            self.set_design_status()
            
        targets = f['TARGET'].read()
        self.targets_fromarray(target_array=targets)
        if(('assign' in f.hdu_map) & (noassign is False)):
            assignments = f['ASSIGN'].read()
        else:
            assignments = None
        if(assignments is not None):
            self.achievable_calibrations = collections.OrderedDict()
            for n in self.calibration_order:
                self.achievable_calibrations[n] = self.required_calibrations[n].copy()
            for name in hdr:
                m = re.match('^ACNAME([0-9]*)$', name)
                if(m is not None):
                    num = 'ACNUM{d}'.format(d=m.group(1))
                    if(num in hdr):
                        if(hdr[num].strip() != ''):
                            self.achievable_calibrations[hdr[name]] = np.array([np.int32(np.float32(x)) for x in hdr[num].split()], dtype=np.int32)
                        else:
                            self.achievable_calibrations[hdr[name]] = np.zeros(0, dtype=np.int32)

            if(self.field_cadence.nexp_total == 1):
                iassigned = np.where(assignments['robotID'] >= 1)
                for itarget in iassigned[0]:
                    self.assign_robot_exposure(robotID=assignments['robotID'][itarget],
                                               rsid=targets['rsid'][itarget],
                                               iexp=0, reset_satisfied=False,
                                               reset_has_spare=False,
                                               reset_count=False,
                                               set_expflag=False,
                                               force=self.exposure_locked[0])
            else:
                iassigned = np.where(assignments['robotID'] >= 1)
                for itarget, iexp in zip(iassigned[0], iassigned[1]):
                    self.assign_robot_exposure(robotID=assignments['robotID'][itarget, iexp],
                                               rsid=targets['rsid'][itarget],
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False,
                                               reset_count=False,
                                               set_expflag=False,
                                               force=self.exposure_locked[iexp])

            hasexpflag = ('expflag' in assignments.dtype.names)
            for assignment, target in zip(assignments, targets):
                indx = self.rsid2indx[target['rsid']]
                self.assignments['rsflags'][indx] = assignment['rsflags']
                if(hasexpflag):
                    self.assignments['expflag'][indx] = assignment['expflag']
            self._set_masterGrid()
            self._set_has_spare_calib()
            self._set_satisfied()
            self._set_satisfied(science=True)
            self._set_count(reset_equiv=False)
            self.decollide_unassigned()

        self.decollide_unassigned()
        
        return

    def clear_assignments(self):
        """Clear the assignments for this field

        Notes
        -----

        Uses unassign() to unassign every target.
"""
        if(self.assignments is not None):
            iassigned = np.where(self.assignments['assigned'])[0]
            self.assignments['expflag'] = 0
            self.assignments['rsflags'] = 0
            self.unassign(self.targets['rsid'][iassigned])
            self.assignments['assigned'] = 0
            self.assignments['satisfied'] = 0
        return

    def clear_field_cadence(self):
        """Resets the field cadence to 'none' and clears all the ancillary data

        Notes
        -----

        Calls the clear_assignments() method, deletes all the
        robotGrids in the robotgrids array, sets field_cadence to
        None, resets the calibration counts to zero, and resets all
        the internal tracking variables.

        Used in rs_assign to avoid having to read in the same field
        over and over when testing multiple cadences.
"""
        if(self.verbose):
            print("fieldid {fid}: Clearing field cadence".format(fid=self.fieldid), flush=True)
        if(self.assignments is not None):
            self.clear_assignments()

        if(self.allgrids):
            for i in range(self.field_cadence.nexp_total):
                self.robotgrids[i] = None
            self.robotgrids = []
        self.exposure_locked = None
        self._robot_locked = None
        self._robot2indx = None
        self._robotnexp = None
        self._robotnexp_max = None
        self.field_cadence = None
        self.assignments_dtype = None
        self.assignments = None
        self.design_mode = None
        if(self.nocalib is False):
            self.calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                   0,
                                                   robostrategy.standards.nzone), dtype=np.int32)
            self.achievable_calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                              0,
                                                              robostrategy.standards.nzone), dtype=np.int32)
            for n in self.calibration_order:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
                self.required_calibrations[n] = np.zeros(0, dtype=np.int32)
                self.achievable_calibrations[n] = self.required_calibrations[n].copy()

        if(self.verbose):
            print("fieldid {fid}:  (done clearing field cadence)".format(fid=self.fieldid), flush=True)

        return

    def _arrayify(self, quantity=None, dtype=np.float64):
        """Cast quantity as ndarray of numpy.float64"""
        try:
            length = len(quantity)
        except TypeError:
            length = 1
        return np.zeros(length, dtype=dtype) + quantity

    def _robotGrid(self):
        """Return a RobotGridAPO or RobotGridLCO instance

        Notes
        -----

        Sets all robots to home position.

        When first called, sets robotHasApogee attribute.
"""
        if(self.observatory == 'apo'):
            rg = kaiju.robotGrid.RobotGridAPO(stepSize=0.05)
        if(self.observatory == 'lco'):
            rg = kaiju.robotGrid.RobotGridLCO(stepSize=0.05)
        for k in rg.robotDict.keys():
            rg.homeRobot(k)
        if(self.robotHasApogee is None):
            self.robotHasApogee = np.array([rg.robotDict[x].hasApogee
                                            for x in rg.robotDict.keys()],
                                           dtype=bool)
        return(rg)

    def _set_radius(self):
        """Set radius limit in deg depending on observatory"""
        if(self.observatory == 'apo'):
            self.radius = 1.5
        if(self.observatory == 'lco'):
            self.radius = 0.95
        return

    def set_field_cadence(self, field_cadence='none'):
        """Set the field cadence, and set up robotgrids and assignments output

        Parameters
        ----------

        field_cadence : str
            Name of field cadence

        Notes
        -----

        Sets the field cadence. 

        If the object is instantiated with parameters including 
        field_cadence, this routine is called in the initialization. 
        If the object is instantiated with a file name, if the file 
        header has the FCADENCE keyword set to anything but 'none', 
        this routine will be called.

        The cadence must be one in the CadenceList singleton. Upon
        setting the field cadence with this routine, the robotgrids,
        assignments_dtype, assignments, calibrations, and
        field_cadence attributes are configured.

        You can reset the field cadence, but first you must call
        clear_field_cadence(). Obviously this deletes all the assignments.
"""
        if(self.allgrids):
            if(len(self.robotgrids) > 0):
                print("Cannot reset field_cadence")
                return
        if(field_cadence != 'none'):
            if(self.verbose):
                print("fieldid {fid}: Setting field cadence".format(fid=self.fieldid), flush=True)
            self.field_cadence = clist.cadences[field_cadence]
            if(self.allgrids):
                for i in range(self.field_cadence.nexp_total):
                    self.robotgrids.append(self._robotGrid())
            self.exposure_locked = np.zeros(self.field_cadence.nexp_total,
                                            dtype=bool)
            self._robot_locked = np.zeros((len(self.mastergrid.robotDict),
                                           self.field_cadence.nexp_total),
                                          dtype=bool)
            self._robot2indx = np.zeros((len(self.mastergrid.robotDict),
                                         self.field_cadence.nexp_total),
                                        dtype=np.int32) - 1
            self._robotnexp = np.zeros((len(self.mastergrid.robotDict),
                                        self.field_cadence.nepochs),
                                       dtype=np.int32)
            self._robotnexp_max = np.zeros((len(self.mastergrid.robotDict),
                                            self.field_cadence.nepochs),
                                           dtype=np.int32)
            for i, n in enumerate(self.field_cadence.nexp):
                self._robotnexp[:, i] = n
                self._robotnexp_max[:, i] = n
            self.assignments_dtype = np.dtype([('assigned', np.int32),
                                               ('satisfied', np.int32),
                                               ('science_satisfied', np.int32),
                                               ('extra', np.int32),
                                               ('nexps', np.int32),
                                               ('nepochs', np.int32),
                                               ('x', np.float64),
                                               ('y', np.float64),
                                               ('z', np.float64),
                                               ('fiber_ra', np.float64),
                                               ('fiber_dec', np.float64),
                                               ('delta_ra', np.float32),
                                               ('delta_dec', np.float32),
                                               ('incadence', bool),
                                               ('allowed', bool,
                                                (self.field_cadence.nepochs,)),
                                               ('mags_allowed', bool,
                                                (self.field_cadence.nepochs,)),
                                               ('bright_allowed', bool,
                                                (self.field_cadence.nepochs,)),
                                               ('offset_allowed', bool,
                                                (self.field_cadence.nepochs,)),
                                               ('offset_flag', np.int32,
                                                (self.field_cadence.nepochs,)),
                                               ('robotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('holeID', np.dtype("|U15"), (self.field_cadence.nexp_total)),
                                               ('equivRobotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('scienceRobotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('target_skybrightness', np.float32,
                                                (self.field_cadence.nexp_total,)),
                                               ('field_skybrightness', np.float32,
                                                (self.field_cadence.nexp_total,)),
                                               ('fiberType', np.unicode_, 10),
                                               ('rsflags', np.int32),
                                               ('expflag', np.int32,
                                                (self.field_cadence.nexp_total,))])
            self.assignments = np.zeros(0, dtype=self.assignments_dtype)

            try:
                obsmode_pk = self.field_cadence.obsmode_pk
            except AttributeError:
                obsmode_pk = np.array([''] * self.field_cadence.nexp_total)

            if(obsmode_pk[0] != ''):
                if(self.verbose):
                    print("fieldid {f}: obsmode_pk has been set".format(f=self.fieldid), flush=True)
                if((type(obsmode_pk) == list) |
                   (type(obsmode_pk) == np.ndarray)):
                    self.design_mode = np.array(obsmode_pk)
                else:
                    self.design_mode = np.array([obsmode_pk])
            else:
                if(self.verbose):
                    print("fieldid {f}: Using heuristics for obsmode_pk".format(f=self.fieldid), flush=True)
                self.design_mode = np.array([''] *
                                            self.field_cadence.nepochs)
                for epoch in np.arange(self.field_cadence.nepochs):
                    if(self.field_cadence.skybrightness[epoch] >= 0.5):
                        self.design_mode[epoch] = 'bright_time'
                    else:
                        if(('dark_100x8' in self.field_cadence.name) |
                           ('dark_174x8' in self.field_cadence.name)):
                            self.design_mode[epoch] = 'dark_rm'
                        elif(('dark_10x4' in self.field_cadence.name) |
                             ('dark_2x4' in self.field_cadence.name) |
                             ('dark_3x4' in self.field_cadence.name)):
                            self.design_mode[epoch] = 'dark_monit'
                        elif(('dark_1x1' in self.field_cadence.name) |
                             ('dark_1x2' in self.field_cadence.name) |
                             ('dark_2x1' in self.field_cadence.name) |
                             ('mixed2' in self.field_cadence.name)):
                            self.design_mode[epoch] = 'dark_plane'
                        else:
                            self.design_mode[epoch] = 'dark_faint'
                    
            if(self.nocalib is False):
                dms = self.design_mode[self.field_cadence.epochs]
                for c in self.calibration_order:
                    if(c == 'standard_boss'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_stds_min['BOSS'] for d in dms], dtype=np.int32)
                    elif(c == 'standard_apogee'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_stds_min['APOGEE'] for d in dms], dtype=np.int32)
                    elif(c == 'sky_boss'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_skies_min['BOSS'] for d in dms], dtype=np.int32)
                    elif(c == 'sky_apogee'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_skies_min['APOGEE'] for d in dms], dtype=np.int32)
                self.calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                       self.field_cadence.nexp_total,
                                                       robostrategy.standards.nzone),
                                                      dtype=np.int32)
                self.achievable_calibrations_per_zone = np.zeros((len(self.calibration_order) + 1,
                                                                  self.field_cadence.nexp_total,
                                                                  robostrategy.standards.nzone),
                                                                 dtype=np.int32)
                for c in self.calibration_order:
                    self.calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                                    dtype=np.int32)
                for c in self.calibration_order:
                    self.achievable_calibrations[c] = self.required_calibrations[c].copy()

            if(self.bright_neighbors):
                if(self.verbose):
                    print("fieldid {fieldid}: Find bright stars".format(fieldid=self.fieldid), flush=True)
                umode = np.unique(self.design_mode)
                for design_mode in umode:
                    for fiberType in ['APOGEE', 'BOSS']:
                        self.set_bright_stars(design_mode=design_mode,
                                              fiberType=fiberType)

            if(self.verbose):
                print("fieldid {fieldid}: Setup assignments".format(fieldid=self.fieldid), flush=True)
            self.assignments = self._setup_assignments_for_cadence(self.targets)

            self._set_masterGrid()
            
            if(self.nocalib is False):
                self._set_has_spare_calib()

            self.set_design_status()

            self._set_holeid()
            
            if(self.verbose):
                print("fieldid {fid}:   (done setting field cadence)".format(fid=self.fieldid), flush=True)
        else:
            self.field_cadence = None
            self.design_status = None
            if(self.allgrids):
                self.robotgrids = []
            else:
                self.robotgrids = None
            self.assignments_dtype = None
            self._has_spare_calib = None
        return

    def set_design_status(self, design_status=None):
        """Set design_status, with assumed status for each design
        
        Parameters
        ----------

        design_status : ndarray
            [nexp_total] array, with 'designid' and 'status' elements

        Notes
        -----

        If design_status is none, sets all status to default; designid=-1
        and status='not started'
"""
        if(design_status is None):
            if(self.field_cadence is not None):
                self.design_status = np.zeros(self.field_cadence.nexp_total,
                                              dtype=design_status_dtype)
                self.design_status['fieldid'] = self.fieldid
                self.design_status['designid'] = -1
                self.design_status['status'] = 'not started'
            else:
                self.design_status = None
            return
        if(len(design_status) != self.field_cadence.nexp_total):
            raise Exception('design_status must exist for each exposure')
        if('designid' not in design_status.dtype.names):
            raise Exception('"designid" must be specified by design_status')
        if('status' not in design_status.dtype.names):
            raise Exception('"status" must be specified by design_status')
        self.design_status = design_status
        return

    def set_design_status_from_status_field(self, status_field=None):
        """Set design_status based on get_status_by_field() output
        
        Parameters
        ----------

        status_field : ndarray
            array with 'designid', 'field_exposure', and 'status' elements

        Notes
        -----

        If status_field is None, sets all status to default; designid=-1
        and status='not started'

        Otherwise, status_field should contain any instance of a
        designid associated with this field.

        In that case sets design_status to an array with nexp_total elements
        with designid=-1 and status='not started' for incomplete designs
        and status='done' for done ones, and assigning the lowest designid.
"""
        if(status_field is None):
            self.set_design_status(design_status=None)
            return
        design_status = np.zeros(self.field_cadence.nexp_total, dtype=design_status_dtype)
        design_status['fieldid'] = self.fieldid
        for field_exposure in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
            icurr = np.where((status_field['field_exposure'] == field_exposure) &
                             (status_field['status'] == 'done'))[0]
            if(len(icurr) > 0):
                designid = status_field['design_id'][icurr].min()
                if(designid < 0):
                    raise Exception('A done designid should not be negative')
                design_status['designid'][field_exposure] = designid
                design_status['status'][field_exposure] = 'done'
            else:
                design_status['designid'][field_exposure] = -1
                design_status['status'][field_exposure] = 'not started'
                
        self.set_design_status(design_status=design_status)
        return

    def set_flag(self, rsid=None, flagname=None):
        """Set a bitmask flag for a target

        Parameters
        ----------

        rsid : np.int64
            IDs of the target-cadence

        flagname : str
            name of flag to set
"""
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)], dtype=int)
        self.assignments['rsflags'][indxs] = (self.assignments['rsflags'][indxs] | self.flagdict[flagname])
        return

    def check_flag(self, rsid=None, flagname=None):
        """Check a bitmask flag for a target

        Parameters
        ----------

        rsid : np.int64 or ndarray
            IDs of the target-cadence

        flagname : str
            name of flag to set

        Returns
        -------

        setornot : ndarray of bool
            True if flag is set, flag otherwise
"""
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)], dtype=int)
        setornot = ((self.assignments['rsflags'][indxs] & self.flagdict[flagname]) != 0)
        return(setornot)

    def get_flag_names(self, flagval=None):
        """Return names associated with flag

        Parameters
        ----------

        flagval : np.int32
            flag

        Returns
        -------

        flagnames : list
            strings corresponding to each set bit
"""
        flagnames = []
        for fn in self.flagdict:
            if(flagval & self.flagdict[fn]):
                flagnames.append(fn)
        return(flagnames)

    def set_expflag(self, rsid=None, iexp=None, flagname=None):
        """Set a bitmask flag for a target's exposure

        Parameters
        ----------

        rsid : np.int64
            IDs of the target-cadence

        iexp : np.int32
            exposure

        flagname : str
            name of flag to set
"""
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)], dtype=int)
        self.assignments['expflag'][indxs, iexp] = (self.assignments['expflag'][indxs, iexp] | self.expflagdict[flagname])
        return

    def check_expflag(self, rsid=None, iexp=None, flagname=None):
        """Check a bitmask flag for a target

        Parameters
        ----------

        rsid : np.int64 or ndarray
            ID of the target-cadence

        iexp : np.int32
            exposure

        flagname : str
            name of flag to set

        Returns
        -------

        setornot : ndarray of bool
            True if flag is set, flag otherwise
"""
        indx = self.rsid2indx[rsid]
        setornot = ((self.assignments['expflag'][indx, iexp] & self.expflagdict[flagname]) != 0)
        return(setornot)

    def get_expflag_names(self, flagval=None):
        """Return names associated with exposure flag

        Parameters
        ----------

        flagval : np.int32
            flag

        Returns
        -------

        flagnames : list
            strings corresponding to each set bit
"""
        flagnames = []
        for fn in self.expflagdict:
            if(flagval & self.expflagdict[fn]):
                flagnames.append(fn)
        return(flagnames)

    def _offset_radec(self, ra=None, dec=None, delta_ra=0., delta_dec=0.):
        """Offsets ra and dec according to specified amount
        
        Parameters
        ----------

        ra : np.float64 or ndarray of np.float64
        right ascension, deg

        dec : np.float64 or ndarray of np.float64
            declination, deg

        delta_ra : np.float64 or ndarray of np.float64
            right ascension direction offset, arcsec

        delta_dec : np.float64 or ndarray of np.float64
            declination direction offset, arcsec

        Returns
        -------

        offset_ra : np.float64 or ndarray of np.float64
            offset right ascension, deg

        offset_dec : np.float64 or ndarray of np.float64
            offset declination, deg

        Notes
        -----

        Assumes that delta_ra, delta_dec are in proper coordinates; i.e.
        an offset of delta_ra=1 arcsec represents the same angular separation 
        on the sky at any declination.

        Carefully offsets in the local directions of ra, dec based on
        the local tangent plane (i.e. does not just scale delta_ra by
        1/cos(dec))
"""
        deg2rad = np.pi / 180.
        arcsec2rad = np.pi / 180. / 3600.
        x = np.cos(dec * deg2rad) * np.cos(ra * deg2rad)
        y = np.cos(dec * deg2rad) * np.sin(ra * deg2rad)
        z = np.sin(dec * deg2rad)
        ra_x = - np.sin(ra * deg2rad)
        ra_y = np.cos(ra * deg2rad)
        ra_z = 0.
        dec_x = - np.sin(dec * deg2rad) * np.cos(ra * deg2rad)
        dec_y = - np.sin(dec * deg2rad) * np.sin(ra * deg2rad)
        dec_z = np.cos(dec * deg2rad)
        xoff = x + (ra_x * delta_ra + dec_x * delta_dec) * arcsec2rad
        yoff = y + (ra_y * delta_ra + dec_y * delta_dec) * arcsec2rad
        zoff = z + (ra_z * delta_ra + dec_z * delta_dec) * arcsec2rad
        offnorm = np.sqrt(xoff**2 + yoff**2 + zoff**2)
        xoff = xoff / offnorm
        yoff = yoff / offnorm
        zoff = zoff / offnorm
        decoff = np.arcsin(zoff) / deg2rad
        raoff = ((np.arctan2(yoff, xoff) / deg2rad) + 360.) % 360.
        return(raoff, decoff)

    # This is a bit fragile, since choice must match coordio.utils.offset_definition
    def _mag_lim(self, fiberType=None, design_mode=None):
        """Return the bright limit used by offset()"""
        dm = self.designModeDict[design_mode]
        bright = 'bright' in design_mode
        if fiberType == 'BOSS':
            # grab r_sdss limit for boss
            if(bright): 
                mag_lim = dm.bright_limit_targets['BOSS'][5][0]
            else:
                mag_lim = dm.bright_limit_targets['BOSS'][1][0]
        else:
            # grab h 2mass mag for limit
            mag_lim = dm.bright_limit_targets['APOGEE'][8][0]
        return(mag_lim)

    def _mag_limits(self, fiberType=None, design_mode=None):
        """Return the bright limits in all bands used by offset()"""
        dm = self.designModeDict[design_mode]
        if fiberType == 'BOSS':
            mag_limits = dm.bright_limit_targets['BOSS'][:, 0]
        else:
            mag_limits = dm.bright_limit_targets['APOGEE'][:, 0]
        return(mag_limits)

    def offset(self, targets=None, design_mode=None):
        """Returns appropriate offsets for each target given design mode

        Parameters
        ----------

        targets : ndarray
            target information
        
        design_mode : str
            key to designModeDict

        Returns
        -------

        delta_ra : ndarray of np.float64
            offset in RA (proper angular distance)

        delta_dec : ndarray of np.float64
            offset in Dec (proper angular distance)
"""
        if('bright' in design_mode):
            lunation = 'bright'
            skybrightness = 1.0
        if('dark' in design_mode):
            lunation = 'dark'
            skybrightness = 0.35
        mags = targets['magnitude'][:, :]
        boss = targets['fiberType'] == 'BOSS'
        apogee = targets['fiberType'] == 'APOGEE'

        delta_ra = np.zeros(len(targets), dtype=np.float64)
        delta_dec = np.zeros(len(targets), dtype=np.float64)
        offset_flag = np.zeros(len(targets), dtype=np.int32)

        iboss = np.where(boss)[0]
        if(len(iboss) > 0):
            mag_limits = self._mag_limits(design_mode=design_mode, fiberType='BOSS')
            tmp_delta_ra, tmp_delta_dec, tmp_offset_flag = coordio.utils.object_offset(mags[iboss, :],
                                                                                       mag_limits,
                                                                                       lunation,
                                                                                       'Boss',
                                                                                       self.observatory.upper(),
                                                                                       fmagloss=self.fmagloss,
                                                                                       can_offset=targets['can_offset'][iboss],
                                                                                       skybrightness=skybrightness,
                                                                                       offset_min_skybrightness=self.offset_min_skybrightness)
            delta_ra[iboss] = tmp_delta_ra
            delta_dec[iboss] = tmp_delta_dec
            offset_flag[iboss] = tmp_offset_flag

        iapogee = np.where(apogee)[0]
        if(len(iapogee) > 0):
            mag_limits = self._mag_limits(design_mode=design_mode, fiberType='APOGEE')
            tmp_delta_ra, tmp_delta_dec, tmp_offset_flag = coordio.utils.object_offset(mags[iapogee, :],
                                                                                       mag_limits,
                                                                                       lunation,
                                                                                       'Apogee',
                                                                                       self.observatory.upper(),
                                                                                       fmagloss=self.fmagloss,
                                                                                       can_offset=targets['can_offset'][iapogee],
                                                                                       skybrightness=skybrightness,
                                                                                       offset_min_skybrightness=self.offset_min_skybrightness)
            delta_ra[iapogee] = tmp_delta_ra
            delta_dec[iapogee] = tmp_delta_dec
            offset_flag[iapogee] = tmp_offset_flag

        return(delta_ra, delta_dec, offset_flag)

    def radec2xyz(self, ra=None, dec=None, epoch=None, pmra=None,
                  pmdec=None, delta_ra=0., delta_dec=0., fiberType=None):
        """Converts ra and dec to wok x, y, and z

        Parameters
        ----------

        ra : ndarray of np.float64
            right ascensions in J2000 deg

        dec : ndarray of np.float64
            declinations in J2000 deg

        epoch : ndarray of np.float32
            epoch of ra and dec in years (e.g. 2015.5)

        pmra : ndarray of np.float32
            RA proper motion in mas/year

        pmdec : ndarray of np.float32
            Dec proper motion in mas/year

        delta_ra : ndarray of np.float32
            RA offset to apply for fibers in arcsec (default 0)

        delta_dec : ndarray of np.float32
            Dec offset to apply for fibers in arcsec (default 0)

        fiberType : str, list of str, or ndarray of str
            fiber type ('APOGEE' or 'BOSS')

        Returns
        -------

        x : ndarray of np.float64
            X position in wok (mm)

        y : ndarray of np.float64
            Y position in wok (mm)

        z : ndarray of np.float64
            Z position in wok (mm)

        Notes
        -----

        delta_ra and delta_dec are proper angular distances.

        Z is just returned as a constant value (equal to 
        coordio.defaults.POSITIONER_HEIGHT).
"""
        if(isinstance(fiberType, str)):
            wavename = fiberType.capitalize()
        else:
            wavename = np.array([x.capitalize() for x in fiberType])
        if(epoch is not None):
            epoch_jd = np.zeros(len(epoch), dtype=np.float64)
            oneday = datetime.timedelta(days=1)
            for i, e in enumerate(epoch):
                epoch_year = int(e)
                epoch_frac = e - int(e)
                epoch_year_dt = datetime.datetime(epoch_year, 1, 1)
                epoch_dt = epoch_year_dt + oneday * epoch_frac * 365.25
                epoch_jd[i] = coordio.time.Time(epoch_dt).jd
        else:
            epoch_jd = None
        raoff, decoff = self._offset_radec(ra=ra, dec=dec, delta_ra=delta_ra,
                                           delta_dec=delta_dec)
        radVel = np.zeros(len(raoff), dtype=np.float64) + 1.e-4
        parallax = np.zeros(len(raoff), dtype=np.float64) + 1.e-4
        x, y, warn, ha, pa = coordio.utils.radec2wokxy(raoff, decoff, epoch_jd,
                                                       wavename,
                                                       self.racen, self.deccen,
                                                       self.pa,
                                                       self.observatory.upper(),
                                                       self.obstime.jd,
                                                       pmra=pmra,
                                                       pmdec=pmdec,
                                                       parallax=parallax,
                                                       radVel=radVel)
        z = coordio.defaults.POSITIONER_HEIGHT
        return(x, y, z)

    def xy2radec(self, x=None, y=None, fiberType=None):
        """X and Y back to RA, Dec, without proper motions or deltas

        Parameters
        ----------

        x : ndarray of np.float64
            X position in wok (mm)

        y : ndarray of np.float64
            Y position in wok (mm)

        fiberType : str, list of str, or ndarray of str
            fiber type ('APOGEE' or 'BOSS')

        Returns
        -------

        ra : ndarray of np.float64
            right ascensions in J2000 deg

        dec : ndarray of np.float64
            declinations in J2000 deg
"""
        if(isinstance(fiberType, str)):
            wavename = fiberType.capitalize()
        else:
            wavename = np.array([t.capitalize() for t in fiberType])
        xa = self._arrayify(x, dtype=np.float64)
        ya = self._arrayify(y, dtype=np.float64)
        ra, dec, warn = coordio.utils.wokxy2radec(xa, ya,
                                                  wavename,
                                                  self.racen, self.deccen,
                                                  self.pa,
                                                  self.observatory.upper(),
                                                  self.obstime.jd)
        return(ra, dec)

    def targets_fromfits(self, filename=None):
        """Read in targets from FITS file

        Parameters
        ----------

        filename : str
            file name to read from

        Notes
        -----

        Just reads extension 1. Then calls targets_fromarray()
"""
        t = fitsio.read(filename, ext=1)
        self.targets_fromarray(t)
        return

    def _mags_allowed(self, targets=None, designMode=None):
        """Report whether magnitude limits allow targets

        Parameters
        ----------

        targets : ndarray
            elements of targets array

        designMode : DesignMode object
            design mode to test against

        Returns
        -------

        allowed : ndarray of bool
            For each target, True if magnitude limits allow, False otherwise
"""
        fiberTypes = ['BOSS', 'APOGEE']
        categories = ['science', 'standard']
        target_category = np.array([x.split('_')[0]
                                    for x in targets['category']])
        target_allowed = np.ones(len(targets), dtype=bool)
        for fiberType in fiberTypes:
            for category in categories:
                icurr = np.where((targets['fiberType'] == fiberType) &
                                 (target_category == category))[0]
                mags = targets['magnitude'][icurr, :]
                if(category == 'science'):
                    limits = designMode.bright_limit_targets[fiberType]
                if(category == 'standard'):
                    limits = designMode.stds_mags[fiberType]
                ok = np.ones(len(icurr), dtype=bool)
                for i in np.arange(limits.shape[0], dtype=np.int32):
                    icheck = np.where((np.isnan(mags[:, i]) == False) &
                                      (mags[:, i] != 0.) &
                                      (mags[:, i] != 99.9) &
                                      (mags[:, i] != 999.) &
                                      (mags[:, i] != - 999.) &
                                      (mags[:, i] != - 9999.))[0]
                    if(limits[i, 0] != - 999.):
                        ok[icheck] = ok[icheck] & (mags[icheck, i] > limits[i, 0])
                    if(limits[i, 1] != - 999.):
                        ok[icheck] = ok[icheck] & (mags[icheck, i] < limits[i, 1])
                target_allowed[icurr] = ok
        return(target_allowed)

    def _targets_to_robotgrid(self, targets=None, assignments=None,
                              robotgrid=None):
        """Assign targets to a RobotGrid object

        Parameters
        ----------

        targets : ndarray
            targets array

        assignments : ndarray
            assignments array (delta_ra, delta_dec here)

        robotgrid : RobotGrid object
            robot grid to assign to
"""
        for indx, target in enumerate(targets):
            if(target['fiberType'] == 'APOGEE'):
                fiberType = kaiju.cKaiju.ApogeeFiber
            else:
                fiberType = kaiju.cKaiju.BossFiber
            if(assignments is not None):
                xyzWok = [assignments['x'][indx],
                          assignments['y'][indx],
                          assignments['z'][indx]]
            else:
                xyzWok = [target['x'], target['y'], target['z']]
            robotgrid.addTarget(targetID=target['rsid'],
                                xyzWok=xyzWok,
                                priority=np.float64(target['priority']),
                                fiberType=fiberType)
        return

    def _setup_assignments_for_cadence(self, targets=None,
                                       assignment_array=None):
        """Sets up the assignments array for a given cadence

        Parameters
        ----------

        targets : ndarray
            array of targets

        assignments : ndarray
            assignments array

        Returns
        -------

        assignments : ndarray
            adjusted assignments array

        Notes
        -----
        
        Using field cadence, appropriately sets:

          field_skybrightness
          mags_allowed (is it allowed by magnitude limits)
          bright_allowed (is it allowed by bright neighbor limits)
          offset_allowed (is it allowed because of its offset)
          allowed
          incadence
"""
        if(targets is None):
            return(None)

        # Set up outputs
        assignments = np.zeros(len(targets),
                               dtype=self.assignments_dtype)

        field_skybrightness = self.field_cadence.skybrightness[self.field_cadence.epochs]
        assignments['field_skybrightness'] = np.outer(np.ones(len(targets)),
                                                      field_skybrightness)

        # Determine if it is within the field cadence
        if(self.verbose):
            print("fieldid {fieldid}: Check cadences".format(fieldid=self.fieldid), flush=True)
        for itarget, target_cadence in enumerate(targets['cadence']):
            if(self.veryverbose):
                  print("fieldid {fieldid}: Checking {rsid} with cadence {c}".format(fieldid=self.fieldid, rsid=targets['rsid'][itarget], c=targets['cadence'][itarget]))
                  
            if(target_cadence in clist.cadences):
                ok, solns = clist.cadence_consistency(target_cadence,
                                                      self.field_cadence.name)
                assignments['incadence'][itarget] = ok
        
        if(self.verbose):
            print("fieldid {fieldid}: Setup allowed".format(fieldid=self.fieldid), flush=True)
        umode = np.unique(self.design_mode)

        # Now find minimum offset among all modes; use this offset; for
        # any modes where the offset is larger, exclude these. The idea
        # is that (a) probably we don't want to mix these offsets; (b)
        # we don't want to take an unnecessarily large offset when an
        # epoch will allow it to be smaller; (c) this makes life much
        # simpler.
        if(self.verbose):
            print("fieldid {fieldid}: Calculating offsets".format(fieldid=self.fieldid), flush=True)
        delta_ra_all = np.zeros((len(umode), len(targets)), dtype=np.float32)
        delta_dec_all = np.zeros((len(umode), len(targets)), dtype=np.float32)
        offset_flag_all = np.zeros((len(umode), len(targets)), dtype=np.int32)
        for imode, mode in enumerate(umode):
            tmp_delta_ra, tmp_delta_dec, tmp_offset_flag = self.offset(targets=targets,
                                                                       design_mode=mode)
            delta_ra_all[imode, :] = tmp_delta_ra
            delta_dec_all[imode, :] = tmp_delta_dec
            offset_flag_all[imode, :] = tmp_offset_flag
        delta_all = np.sqrt(delta_ra_all**2 + delta_dec_all**2)
        idelta = np.argmin(delta_all, axis=0)
        delta_ra = delta_ra_all[idelta, np.arange(len(targets), dtype=int)]
        delta_dec = delta_dec_all[idelta, np.arange(len(targets), dtype=int)]
        delta = np.sqrt(delta_ra**2 + delta_dec**2)
        offset_allowed = dict()
        offset_flag = dict()
        for imode, mode in enumerate(umode):
            offset_flag[mode] = offset_flag_all[imode, :]
            offset_allowed[mode] = (delta_all[imode, :] <= delta) & (offset_flag[mode] == 0)
            inot = np.where(offset_allowed[mode] == False)[0]
            offset_flag[mode][inot] = offset_flag[mode][inot] | _offsetdict['TOO_CLOSE_FOR_MODE']

        assignments['delta_ra'] = delta_ra
        assignments['delta_dec'] = delta_dec

        # Set offset allowed so we can double check in _bright_allowed_direct
        for epoch, mode in enumerate(self.design_mode):
            assignments['offset_allowed'][:, epoch] = offset_allowed[mode]
            assignments['offset_flag'][:, epoch] = offset_flag[mode]

        # Assign positions
        if(self.verbose):
            print("fieldid {fieldid}: Setting coords".format(fieldid=self.fieldid), flush=True)
        (assignments['x'],
         assignments['y'],
         assignments['z']) = self.radec2xyz(ra=targets['ra'],
                                            dec=targets['dec'],
                                            epoch=targets['epoch'],
                                            pmra=targets['pmra'],
                                            pmdec=targets['pmdec'],
                                            delta_ra=assignments['delta_ra'],
                                            delta_dec=assignments['delta_dec'],
                                            fiberType=targets['fiberType'])

        # Convert back to RA/Dec
        (assignments['fiber_ra'],
         assignments['fiber_dec']) = self.xy2radec(x=assignments['x'],
                                                   y=assignments['y'],
                                                   fiberType=targets['fiberType'])

        # Check for each mode whether each target is allowed
        if(self.verbose):
            print("fieldid {fieldid}: Checking allowed".format(fieldid=self.fieldid), flush=True)
        mags_allowed = dict()
        bright_allowed = dict()
        for mode in umode:
            dm = self.designModeDict[mode]
            mags_allowed[mode] = self._mags_allowed(designMode=dm,
                                                    targets=targets)
            if(self.bright_neighbors):
                bright_allowed[mode] = self._bright_allowed_direct(design_mode=mode,
                                                                   targets=targets,
                                                                   assignments=assignments)
            else:
                bright_allowed[mode] = np.ones(len(targets), dtype=bool)

        # Set allowed in assignments; note offset_allowed was already
        # set above.
        for epoch, mode in enumerate(self.design_mode):
            assignments['mags_allowed'][:, epoch] = mags_allowed[mode]
            assignments['bright_allowed'][:, epoch] = bright_allowed[mode]
            assignments['allowed'][:, epoch] = ((mags_allowed[mode] |
                                                 offset_allowed[mode]) &
                                                bright_allowed[mode])

        if(self.allgrids):
            if(self.verbose):
                print("fieldid {fieldid}: Setup all grids".format(fieldid=self.fieldid), flush=True)

            ng = len(self.robotgrids)
            for iexp, rg in enumerate(self.robotgrids):
                if(self.verbose):
                    print("fieldid {fid}:   grid {i}/{n}".format(fid=self.fieldid, i=iexp, n=ng), flush=True)
                epoch = self.field_cadence.epochs[iexp]
                self._targets_to_robotgrid(targets=targets,
                                           assignments=assignments,
                                           robotgrid=rg)

        if(self.verbose):
            print("fieldid {fieldid}: assign inputs".format(fieldid=self.fieldid), flush=True)
        if(assignment_array is None):
            assignments['fiberType'] = targets['fiberType']
            assignments['robotID'] = -1
            assignments['equivRobotID'] = -1
            assignments['scienceRobotID'] = -1
            assignments['target_skybrightness'] = -1.
        else:
            for n in self.assignments_dtype.names:
                listns = ['robotID', 'equivRobotID', 'scienceRobotID',
                          'target_skybrightness', 'field_skybrightness']
                if((n in listns) & (self.field_cadence.nexp_total == 1)):
                    assignments[n][:, 0] = assignment_array[n]
                else:
                    assignments[n] = assignment_array[n]
        return(assignments)

    def _set_masterGrid(self):
        """Reset the master grid and associated information"""
        self.mastergrid = self._robotGrid()
        self._targets_to_robotgrid(targets=self.targets,
                                   assignments=self.assignments,
                                   robotgrid=self.mastergrid)
        self.masterTargetDict = self.mastergrid.targetDict
        for itarget, rsid in enumerate(self.targets['rsid']):
            t = self.masterTargetDict[rsid]
            self.targets['within'][itarget] = len(t.validRobotIDs) > 0
        return

    def targets_fromarray(self, target_array=None, assignment_array=None):
        """Read in targets from ndarray

        Parameters
        ----------

        target_array : ndarray
            array with target information

        assignment_array : ndarray
            if not None, array with assignment information (default None)
"""
        # Read in
        targets = np.zeros(len(target_array), dtype=targets_dtype)
        for n in targets.dtype.names:
            if(n in target_array.dtype.names):
                if(self.oldmag & (n == 'magnitude')):
                    magmap = [0, 1, 2, 4, 5, 6, 8]
                    for imag, imagmap in enumerate(magmap):
                        targets[n][:, imagmap] = target_array[n][:, imag]
                    continue
                targets[n] = target_array[n]

        # Deal with use case where we need to reference a cadence version
        if(self._untrim_cadence_version is not None):
            for itarget, tc in enumerate(targets['cadence']):
                if(tc.split('_')[-1] != self._untrim_cadence_version):
                    targets['cadence'][itarget] = tc + '_' + self._untrim_cadence_version

        # Default values for priority, value, and stage
        if('value' not in target_array.dtype.names):
            targets['value'] = 1.
        if('priority' not in target_array.dtype.names):
            targets['priority'] = 1.
        if('stage' not in target_array.dtype.names):
            targets['stage'] = 'srd'

        # If nooffset is set, set can_offset to False
        if(self.nooffset):
            targets['can_offset'] = False

        # Convert ra/dec to x/y
        if(self.verbose):
            print("fieldid {f}: Convert targets coords to x/y".format(f=self.fieldid), flush=True)
        (targets['x'],
         targets['y'],
         targets['z']) = self.radec2xyz(ra=targets['ra'],
                                        dec=targets['dec'],
                                        epoch=targets['epoch'],
                                        pmra=targets['pmra'],
                                        pmdec=targets['pmdec'],
                                        delta_ra=targets['delta_ra'],
                                        delta_dec=targets['delta_dec'],
                                        fiberType=targets['fiberType'])

        # Set zone
        targets['zone'] = robostrategy.standards.standard_zone(x=targets['x'],
                                                               y=targets['y'])

        # Add targets to robotGrids
        if(self.verbose):
            print("fieldid {f}: Assign targets to robot grid".format(f=self.fieldid), flush=True)
        self._targets_to_robotgrid(targets=targets,
                                   robotgrid=self.mastergrid)

        # Determine if within
        if(self.verbose):
            print("fieldid {f}: Check whether targets are within grid".format(f=self.fieldid), flush=True)
        self.masterTargetDict = self.mastergrid.targetDict
        for itarget, rsid in enumerate(targets['rsid']):
            t = self.masterTargetDict[rsid]
            targets['within'][itarget] = len(t.validRobotIDs) > 0

        # Create internal look-up of whether it is a calibration target
        _is_calibration = np.zeros(len(targets), dtype=bool)
        _is_good_calibration = np.zeros(len(targets), dtype=bool)
        _calibration_index = np.zeros(len(targets), dtype=np.int32)

        if(self.nocalib is False):
            for icategory, category in enumerate(self.required_calibrations):
                icat = np.where(targets['category'] == category)[0]
                _is_calibration[icat] = True
                _is_good_calibration[icat] = True
                _calibration_index[icat] = icategory + 1
        else:
            inotsci = np.where(targets['category'] != 'science')[0]
            _is_calibration[inotsci] = True
            _is_good_calibration[inotsci] = True
            _calibration_index[inotsci] = 1

        if(self.verbose):
            print("fieldid {f}: Connect rsid and index".format(f=self.fieldid), flush=True)

        # Connect rsid with index of list
        for itarget, t in enumerate(targets):
            if(t['rsid'] in self.rsid2indx.keys()):
                print("Cannot replace identical rsid={rsid}. Will not add array.".format(rsid=t['rsid']))
                return
            else:
                self.rsid2indx[t['rsid']] = len(self.targets) + itarget

        # If field_cadence is set, set up potential outputs
        if(self.field_cadence is not None):
            if(self.verbose):
                print("fieldid {fieldid}: Setup assignments".format(fieldid=self.fieldid), flush=True)
            assignments = self._setup_assignments_for_cadence(targets,
                                                              assignment_array)
        else:
            assignments = None

        target_duplicated = np.zeros(len(targets), dtype=np.int32)

        if(self.verbose):
            print("fieldid {f}: Setup calibration tracking".format(f=self.fieldid), flush=True)

        self.targets = np.append(self.targets, targets)
        self.target_duplicated = np.append(self.target_duplicated,
                                           target_duplicated)
        self._is_calibration = np.append(self._is_calibration,
                                         _is_calibration)
        self._is_good_calibration = np.append(self._is_good_calibration,
                                              _is_good_calibration)
        self._calibration_index = np.append(self._calibration_index,
                                            _calibration_index)

        self._unique_catalogids = np.unique(self.targets['catalogid'])

        self.irancalib = np.arange(len(self.targets), dtype=int)
        np.random.seed(self.fieldid)
        np.random.shuffle(self.irancalib)

        # Set up lists of equivalent observation conditions, meaning
        # that for each target we can look up all of the other targets
        # whose catalog, fiberType, lambda_eff, delta_ra, delta_dec 
        # are the same
        if(self.verbose):
            print("fieldid {f}: Setup equiv dictionaries".format(f=self.fieldid), flush=True)
        self._equivindx = collections.OrderedDict()
        self._equivindx_science = collections.OrderedDict()
        self._equivkey = collections.OrderedDict()
        for itarget, target in enumerate(self.targets):
            ekey = (target['catalogid'], target['fiberType'],
                    target['lambda_eff'], target['delta_ra'],
                    target['delta_dec'])
            if(ekey not in self._equivindx):
                self._equivindx[ekey] = np.zeros(0, dtype=np.int32)
            if(ekey not in self._equivindx_science):
                self._equivindx_science[ekey] = np.zeros(0, dtype=np.int32)
            self._equivkey[itarget] = ekey
            self._equivindx[ekey] = np.append(self._equivindx[ekey],
                                              np.array([itarget], dtype=int))
            if(self.targets['category'][itarget] == 'science'):
                self._equivindx_science[ekey] = np.append(self._equivindx_science[ekey],
                                                          np.array([itarget],
                                                                   dtype=int))

        if(assignments is not None):
            if(self.verbose):
                print("fieldid {f}: Add assignments, set satisifed".format(f=self.fieldid), flush=True)
            self.assignments = np.append(self.assignments, assignments, axis=0)
            self._set_satisfied()
            self._set_satisfied(science=True)
            self._set_count(reset_equiv=False)

        if(self.verbose):
            print("fieldid {f}: Setup mastergrid".format(f=self.fieldid), flush=True)
        self._set_masterGrid()

        return

    def _set_holeid(self):
        if(self.field_cadence.nexp_total == 1):
            self.assignments['holeID'][:] = ''
        else:
            self.assignments['holeID'][:, :] = ''
        for i, assignment in enumerate(self.assignments):
            if(self.field_cadence.nexp_total == 1):
                if(assignment['robotID'][0] >= 1):
                    robotID = assignment['robotID'][0]
                    holeID = self.mastergrid.robotDict[robotID].holeID
                    self.assignments['holeID'][i] = holeID
            else:
                iexps = np.where(assignment['robotID'] >= 1)[0]
                for iexp in iexps:
                    robotID = assignment['robotID'][iexp]
                    holeID = self.mastergrid.robotDict[robotID].holeID
                    self.assignments['holeID'][i, iexp] = holeID
        return

    def tofits(self, filename=None):
        """Write field and assignments to FITS file

        Parameters
        ----------

        filename : str
            file name to write to

        Notes
        -----

        Writes out a file readable by fromfits(). Has header keywords
        defining field, and HDUs:

         * TARGET : has the targets array (usable by targets_fromarray())
         * ASSIGN : if it exists, has the assignments array with assignments for each target and exposure
         * DESMODE : if it exists, has the definitions of design modes
         * BS# : bright stars for neighbor checks (with DESMODE & FIBERTY specified in header)
         * ROBOTS - targets assigned for each robot
"""
        hdr = robostrategy.header.rsheader()
        hdr.append({'name':'FIELDID',
                    'value':self.fieldid,
                    'comment':'field identification number'})
        hdr.append({'name':'RACEN',
                    'value':self.racen,
                    'comment':'RA J2000 center of field (deg)'})
        hdr.append({'name':'DECCEN',
                    'value':self.deccen,
                    'comment':'Dec J2000 center of field (deg)'})
        hdr.append({'name':'OBS',
                    'value':self.observatory,
                    'comment':'observatory used for field'})
        hdr.append({'name':'PA',
                    'value':self.pa,
                    'comment':'position angle (deg E of N)'})
        if(self.offset_min_skybrightness is not None):
            hdr.append({'name':'OFFMINSKY',
                        'value':self.offset_min_skybrightness,
                        'comment':'minimum skybrightness for offset'})
        else:
            hdr.append({'name':'OFFMINSKY',
                        'value':-1.,
                        'comment':'minimum skybrightness for offset'})
        hdr.append({'name':'BRIGHTN',
                    'value':self.bright_neighbors,
                    'comment':'account for bright neighbor constraints'})
        if(self.field_cadence is not None):
            hdr.append({'name':'FCADENCE',
                        'value':self.field_cadence.name,
                        'comment':'field cadence'})
            hdr.append({'name':'NEXP',
                        'value':self.field_cadence.nexp_total,
                        'comment':'number of exposures in cadence'})
            dmodelist = ' '.join(list(self.design_mode[self.field_cadence.epochs]))
            hdr.append({'name':'DESMODE',
                        'value':dmodelist,
                        'comment':'list of design modes'})
            if(self.exposure_locked is not None):
                explocklist = ' '.join([str(x) for x in self.exposure_locked])
                hdr.append({'name':'EXPLOCK',
                            'value':explocklist,
                            'comment':'set to 1 for exposures already observed before this assignment'})
        else:
            hdr.append({'name':'FCADENCE',
                        'value':'none',
                        'comment':'field cadence'})
        hdr.append({'name':'CBUFFER',
                    'value':self.collisionBuffer,
                    'comment':'kaiju collision buffer'})
        hdr.append({'name':'NOCALIB',
                    'value':self.nocalib,
                    'comment':'True if this field ignores calibrations'})
        if(self.nocalib is False):
            for indx, rc in enumerate(self.calibration_order):
                name = 'RCNAME{indx}'.format(indx=indx)
                num = 'RCNUM{indx}'.format(indx=indx)
                hdr.append({'name':name,
                            'value':rc,
                            'comment':'calibration category'})
                ns = ' '.join([str(int(n)) for n in self.required_calibrations[rc]])
                hdr.append({'name':num,
                            'value':ns,
                            'comment':'number required per exposure'})
            for indx, ac in enumerate(self.calibration_order):
                name = 'ACNAME{indx}'.format(indx=indx)
                num = 'ACNUM{indx}'.format(indx=indx)
                hdr.append({'name':name,
                            'value':ac,
                            'comment':'calibration category'})
                ns = ' '.join([str(int(n)) for n in self.achievable_calibrations[ac]])
                hdr.append({'name':num,
                            'value':ns,
                            'comment':'number achievable per exposure'})
            for indx, zc in enumerate(self.calibration_order):
                name = 'CPZNAME{indx}'.format(indx=indx)
                num = 'CPZNUM{indx}'.format(indx=indx)
                hdr.append({'name':name,
                            'value':zc,
                            'comment':'calibration category'})
                ns = str(self.required_calibrations_per_zone[indx + 1])
                hdr.append({'name':num,
                            'value':ns,
                            'comment':'number required per zone'})

        fitsio.write(filename, None, header=hdr, clobber=True)
        fitsio.write(filename, self.targets, extname='TARGET')
        if(self.assignments is not None):
            self._set_holeid()
            self._set_satisfied(science=False)
            self._set_satisfied(science=True)
            self._set_count(reset_equiv=False)
            fitsio.write(filename, self.assignments, extname='ASSIGN')
        dmarr = None
        for i, d in enumerate(self.designModeDict):
            arr = self.designModeDict[d].toarray()
            if(dmarr is None):
                dmarr = np.zeros(len(self.designModeDict), dtype=arr.dtype)
            dmarr[i] = arr
        fitsio.write(filename, dmarr, extname='DESMODE')

        if(self.assignments is not None):
            robots_dtype = [('robotID', np.int32),
                            ('holeID', np.dtype("|U15")),
                            ('hasBoss', bool),
                            ('hasApogee', bool),
                            ('rsid', np.int64, self.field_cadence.nexp_total),
                            ('itarget', np.int32, self.field_cadence.nexp_total),
                            ('catalogid', np.int64, self.field_cadence.nexp_total),
                            ('fiberType', np.dtype("|U6"),
                             self.field_cadence.nexp_total)]
            robots = np.zeros(len(self.robotIDs),
                              dtype=robots_dtype) 
            for indx, robotID in enumerate(self.robotIDs):
                robots['robotID'][indx] = robotID
                robots['holeID'][indx] = self.mastergrid.robotDict[robotID].holeID
                robots['hasBoss'][indx] = self.mastergrid.robotDict[robotID].hasBoss
                robots['hasApogee'][indx] = self.mastergrid.robotDict[robotID].hasApogee
                if(self.field_cadence.nexp_total == 1):
                    robots['itarget'][indx] = self._robot2indx[indx, 0]
                    if(robots['itarget'][indx] == -1):
                        robots['rsid'][indx] = -1
                        robots['catalogid'][indx] = -1
                        robots['fiberType'][indx] = ''
                    else:
                        robots['rsid'][indx] = self.targets['rsid'][robots['itarget'][indx]]
                        robots['catalogid'][indx] = self.targets['catalogid'][robots['itarget'][indx]]
                        robots['fiberType'][indx] = self.targets['fiberType'][robots['itarget'][indx]]
                else:
                    for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
                        robots['itarget'][indx, iexp] = self._robot2indx[indx, iexp]
                        if(robots['itarget'][indx, iexp] == -1):
                            robots['rsid'][indx, iexp] = -1
                            robots['catalogid'][indx, iexp] = -1
                            robots['fiberType'][indx, iexp] = ''
                        else:
                            robots['rsid'][indx, iexp] = self.targets['rsid'][robots['itarget'][indx, iexp]]
                            robots['catalogid'][indx, iexp] = self.targets['catalogid'][robots['itarget'][indx, iexp]]
                            robots['fiberType'][indx, iexp] = self.targets['fiberType'][robots['itarget'][indx, iexp]]

            fitsio.write(filename, robots, extname='ROBOTS', clobber=False)

        r2t_dtype = [('robotID', np.int32),
                     ('holeID', np.dtype("|U15")),
                     ('rsid', np.int64)]
        r2t = np.zeros(0, dtype=r2t_dtype)
        for indx, robotID in enumerate(self.robotIDs):
            valid = self.mastergrid.robotDict[robotID].validTargetIDs
            if(len(valid) == 0):
                continue
            tmp_r2t = np.zeros(len(valid), dtype=r2t_dtype)
            tmp_r2t['robotID'] = robotID
            tmp_r2t['holeID'] = self.mastergrid.robotDict[robotID].holeID
            tmp_r2t['rsid'] = valid
            r2t = np.append(r2t, tmp_r2t)
        fitsio.write(filename, r2t, extname='VALIDTARGETS', clobber=False)

        if(self.bright_neighbors):
            if(len(self.bright_stars) > 0):
                write_bright_stars(filename=filename, bright_stars=self.bright_stars,
                                   clobber=False)

        if(self.design_status is not None):
            fitsio.write(filename, self.design_status, extname='STATUS',
                         clobber=False)
                    
        return

    def _set_has_spare_calib(self):
        """Set _has_spare for each exposure"""
        self._has_spare_calib = np.zeros((len(self.required_calibrations) + 1,
                                          self.field_cadence.nexp_total),
                                         dtype=np.int32)
        for icategory, category in enumerate(self.required_calibrations):
            self._has_spare_calib[icategory + 1, :] = (self.calibrations[category] -
                                                       self.achievable_calibrations[category])
        return

    def has_spare_calib(self, rsid=None, indx=None, iexps=None):
        """Reports whether this target can be spared in any exposures

        Parameters
        ----------

        rsid : np.int64
            rsid to consider

        indx : np.int32 or ndarray of np.int32
            target index to consider

        iexps : ndarray of np.int32, or np.int32
            exposures of field to check (default all field exposures)

        Returns
        -------

        isspare : ndarray of np.int32, or np.int32
            is spare in each exposure in iexps

        Notes
        -----

        If indx is set, overrides rsid.

        Either indx can be an array or iexps can be an array,
        but not both. rsid cannot be an array.
"""
        if(iexps is None):
            iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        if(indx is None):
            indx = self.rsid2indx[rsid]
        isspare = self._has_spare_calib[self._calibration_index[indx + 1], iexps] > 0
        if(np.any(isspare) == False):
            return(isspare)
        
        zone = self.targets['zone'][indx]
        ical = self._calibration_index[indx + 1]
        ninzone = self.calibrations_per_zone[ical, iexps, zone].flatten()
        isspare_zone = ((ninzone > self.required_calibrations_per_zone[ical]) &
                        (self._is_good_calibration[indx]))
        isspare = isspare & (isspare_zone |
                             (self._is_good_calibration[indx] == False))
        return(isspare)

    def set_assignment_status(self, status=None, isspare=None, check_spare=True):
        """Set parameters of status object

        Parameters
        ----------

        status : AssignmentStatus object
            object to set attributes of 

        isspare : ndarray of bool
            is rsid of this AssignmentStatus a spare calibration fiber
            in each exposure (default all False)
        
        check_spare : bool
            if True, checks whether spare calibrations can be bumped (default True)

        Notes
        -----

        For each exposure, sets the corresponding element in the attributes:
        
         * spare - is this robotID assigned to a spare calib fiber
         * assignable - is this robotID assignable to this rsid
         * collided - would this assignment cause a collision
         * bright_neighbor_allowed - is it allowed from a bright neighbor POV?
"""
        if(isspare is None):
            isspare = np.zeros(len(status.iexps), dtype=bool)
        robotindx = self.robotID2indx[status.robotID]
        if(self.nocalib is False):
            # Get indices of assigned targets to this robot
            # and make Boolean arrays of which are assigned and not
            if(status.rsid is not None):
                indx = self.rsid2indx[status.rsid]
                epochs = self.field_cadence.epochs
                allowed = self.assignments['allowed'][indx, epochs[status.iexps]]
            else:
                allowed = True
            status.currindx = self._robot2indx[robotindx, status.iexps]
            status.locked = self._robot_locked[robotindx, status.iexps]
            free = (status.currindx < 0) & (status.locked == False)
            has_spare =self.has_spare_calib(indx=status.currindx,
                                            iexps=status.iexps)
            fixed = (((self.assignments['expflag'][status.currindx, status.iexps] &
                       self.expflagdict['FIXED']) != 0) & (status.currindx >= 0))
            status.spare = (has_spare > 0) & (isspare == False) & (free == False) & (fixed == False)
            if(check_spare):
                status.assignable = (free | status.spare) & (allowed > 0) & (fixed == False) & (status.locked == False)
            else:
                status.assignable = (free) & (allowed > 0) & (fixed == False) & (status.locked == False)
        else:
            # Consider exposures for this epoch
            status.currindx = self._robot2indx[robotindx, status.iexps]
            status.locked = self._robot_locked[robotindx, status.iexps]
            if(status.rsid is not None):
                indx = self.rsid2indx[status.rsid]
                epochs = self.field_cadence.epochs
                allowed = self.assignments['allowed'][indx, epochs[status.iexps]]
            else:
                allowed = True
            status.assignable = (status.currindx < 0) & (allowed > 0) & (status.locked == False)

        if(status.rsid is not None):
            for iexp in status.assignable_exposures():
                self.set_collided_status(status=status, iexp=iexp,
                                         check_spare=check_spare)
                if(self.bright_neighbors):
                    self.set_bright_neighbor_status(status=status, iexp=iexp)
                    i = status.expindx[iexp]
                    status.assignable[i] = status.assignable[i] & status.bright_neighbor_allowed[i]

            # Set "already" to whether the exposure is already allocated
            # to this target (by any robot, including this one). But, if 
            # it is also "assignable" that means that the reason it is
            # already gotten is because of a spare calibration fiber. So
            # do not count this. 
            status.already = ((self.assignments['equivRobotID'][indx, status.iexps] >= 0) &
                              (status.assignable == False))
                              
        return

    def set_bright_neighbor_status(self, status=None, iexp=None):
        """Set the bright_neighbor_allowed attribute of status

        Parameters
        ----------

        status : AssignmentStatus object
            object to set attributes of 

        iexp : int
            exposure index (0-indexed within field cadence)

        Notes
        -----

        Sets bright_neighbor_allowed[status.expindx[iexp]] to True
        if there isn't a bright neighbor problem caused by either fiber
        if this assignment is made.
"""
        epoch = self.field_cadence.epochs[iexp]
        design_mode = self.design_mode[epoch]
        key = (status.rsid, status.robotID, design_mode)
        if(key not in self.bright_neighbor_cache):
            self.bright_neighbor_cache[key] = self._bright_allowed_robot(rsid=status.rsid, robotID=status.robotID, design_mode=design_mode)
        i = status.expindx[iexp]
        status.bright_neighbor_allowed[i] = self.bright_neighbor_cache[key]
        return

    def equiv_target(self, target=None):
        """Find rsids which are equivalent to an input target

        Parameters
        ----------

        target : single-element ndarray record array
            target input (with 'catalogid', 'lambda_eff', 'fiberType', 'delta_ra', 'delta_dec')

        Returns
        -------

        rsids : ndarray of np.int64
            equivalent rsids
"""
        ekey = (target['catalogid'], target['fiberType'],
                target['lambda_eff'], target['delta_ra'],
                target['delta_dec'])
        if(ekey in self._equivindx.keys()):
            itargets = self._equivindx[ekey]
            rsids = self.targets['rsid'][itargets]
            return(rsids)
        else:
            return(np.zeros(0, dtype=np.int64))

    def set_collided_status(self, status=None, iexp=None, check_spare=True):
        """Set the collded attribute of status

        Parameters
        ----------

        status : AssignmentStatus object
            object to set attributes of 

        iexp : int
            exposure index (0-indexed within field cadence)
        
        check_spare : bool
            if True, checks whether spare calibrations can be bumped (default True)

        Notes
        -----

        Sets collided[status.expindx[iexp]] to True if this assignment
        would cause a collision in the given exposure.

        If the collision involves a spare calibration fiber, account
        for this when deciding if the exposure is assignable, and 
        store collided object in spare_colliders.

        Updates assignable attribute (setting to False for collisions).
"""
        i = status.expindx[iexp]

        # not relevant if there is no rsid
        if(status.rsid is None):
            return

        # don't even check if not assignable anyway
        if(status.assignable[i] == False):
            return

        # if collisions are being ignored, just check if any
        # equivalent rsid is assigned to another robot
        if((not self.allgrids) | (self.nocollide)):
            indx = self.rsid2indx[status.rsid]
            allindxs = set(self._equivindx[self._equivkey[indx]])
            if(len(allindxs) > 1):
                allindxs.discard(indx)
                allindxs = np.array(list(allindxs), dtype=np.int32)
                if(self.assignments['robotID'][allindxs, iexp].max() >= 0):
                    status.collided[i] = True
                else:
                    status.collided[i] = False
            else:
                status.collided[i] = False
            status.assignable[i] = (status.assignable[i] and
                                    (status.collided[i] == False))
            return

        # check collisions
        rg = self.robotgrids[iexp]
        collided, fcollided, gcollided, colliders = rg.wouldCollideWithAssigned(status.robotID, status.rsid)
        colliders = np.array(colliders, dtype=np.int32)
        status.collided[i] = collided | fcollided | gcollided

        # If it is not collided, just return without changing assignable
        if(status.collided[i] == False):
            return

        # If it collides with a fiducial or GFA, can't be assigned
        if(fcollided or gcollided):
            status.assignable[i] = (status.assignable[i] and
                                    (status.collided[i] == False))
            return

        # If it collides with another robot and we aren't allowing
        # it to bump spares, then it can't be assigned
        if(check_spare == False):
            status.assignable[i] = (status.assignable[i] and
                                    (status.collided[i] == False))
            return

        # If it collides with another robot but would be a spare fiber,
        # then don't make it assignable
        isspare = self.has_spare_calib(rsid=status.rsid, iexps=iexp)
        if(isspare):
            status.assignable[i] = (status.assignable[i] and
                                    (status.collided[i] == False))
            return

        # At this point, the assignment causes a collision
        # and the target is not a spare calibration fiber
        # Check if the colliders are all spare calibrations
        colliderindxs = np.array([self.robotID2indx[x]
                                  for x in colliders], dtype=int)
        itargets = self._robot2indx[colliderindxs, iexp]
        has_spare = self.has_spare_calib(indx=itargets, iexps=iexp)
        collidernotfixed = (((self.assignments['expflag'][itargets, iexp] &
                              self.expflagdict['FIXED']) == 0) &
                            (itargets >= 0))
        status.spare_colliders[i] = self.targets['rsid'][itargets[has_spare & collidernotfixed]]

        # If they are not ALL spare, just set assignable
        if((has_spare & collidernotfixed).min() <= 0):
            status.assignable[i] = (status.assignable[i] and
                                    (status.collided[i] == False))
            return

        # If there is just one collider (which at this point
        # MUST be spare) and the current assigned fiber is also
        # not a spare then collision doesn't matter, just return
        if((len(colliderindxs) == 1) & (status.spare[i] == 0)):
            return
            
        # If there are colliders, and they are all spare calib
        # fibers, then set assignable based on whether they
        # can actually all be removed
        removable = self._are_colliders_removable(i=i, iexp=iexp,
                                                  status=status,
                                                  itargets=itargets)
        status.assignable[i] = (status.assignable[i] and removable)

        return

    def _are_colliders_removable(self, i=None, iexp=None,
                                 status=None, itargets=None):
        # If they are ALL spare, check if removing them all 
        # is possible
        toremove = dict()
        for c in self.calibration_order:
            toremove[c] = 0
        if(status.spare[i]):
            toremove[self.targets['category'][status.currindx[i]]] += 1
        for itarget in itargets:
            toremove[self.targets['category'][itarget]] += 1

        enough = True
        for c in self.calibration_order:
            excess = (self.calibrations[c][iexp] -
                      self.achievable_calibrations[c][iexp])
            if(toremove[c] > excess):
                enough = False
        return(enough)

    def unassign_assignable(self, status=None, iexp=None,
                            reset_satisfied=True, reset_has_spare=True,
                            reset_count=True):
        """Unassign spare calibrations to allow an assignment to happen

        Parameters
        ----------

        status : AssignmentStatus object
            assignment status object 

        iexp : int
            exposure in question (0-indexed within field cadence)

        reset_satisfied : bool
            reset satisfied parameter? (default True)

        reset_has_spare : bool
            reset spare calibration parameter? (default True)

        reset_count : bool
            reset exposure count parameter? (default True)
"""
        i = status.expindx[iexp]
        if(status.assignable[i] is False):
            return

        if(status.spare[i]):
            robotindx = self.robotID2indx[status.robotID]
            rsid = self.targets['rsid'][self._robot2indx[robotindx, iexp]]
            self.unassign_exposure(rsid=rsid, iexp=iexp,
                                   reset_assigned=True,
                                   reset_count=reset_count,
                                   reset_satisfied=reset_satisfied,
                                   reset_has_spare=reset_has_spare)

        for spare_collider in status.spare_colliders[i]:
            self.unassign_exposure(rsid=spare_collider, iexp=iexp,
                                   reset_assigned=True,
                                   reset_count=reset_count,
                                   reset_satisfied=reset_satisfied,
                                   reset_has_spare=reset_has_spare)

        return

    def available_robot_epoch(self, rsid=None,
                              robotID=None, epoch=None, nexp=None,
                              isspare=None):
        """Check if a robot-epoch has enough exposures

        Parameters
        ----------

        rsid : np.int64
            rsid (optional; will check for collisions)

        robotID : np.int64
            robotID to check

        epoch : int or np.int32
            epoch to check

        nexp : int or np.int32
            number of exposures needed

        isspare : bool
            True if this is a spare calibration target

        Returns
        -------

        available : bool
            is it available or not?

        status : list of AssignmentStatus
            which exposures in the epoch are free?

        Notes
        -----

        Checks if a robot is available at each exposure AND if
        assigning the robot to the given target would cause a
        collision.

        The robot is available if it is not assigned to any science
        target AND it is not assigned to a "spare" calibration
        target. A spare calibration target is one for which there are
        more than enough calibration targets of that type already.
"""
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
        status = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexps)

        # Checks obvious case that this epoch doesn't have enough exposures
        available = False
        cnexp = self.field_cadence.nexp[epoch]
        if(cnexp < nexp):
            status.assignable=np.zeros(len(iexps), dtype=bool)
            return available, status

        # Set assignent status
        self.set_assignment_status(status=status, isspare=isspare)

        # Count this epoch as available if there are enough free exposures
        nfree = status.assignable.sum()
        nalready = status.already.sum()

        available = (nfree + nalready) >= nexp

        return available, status

    def available_robot_exposures(self, rsid=None, robotID=None, isspare=False):
        """Return available robot exposures for an rsid

        Parameters
        ----------

        rsid : np.int64
            rsid

        robotID : np.int64
            robotID to check

        isspare : bool
            True if this is a spare calibration target (default False)

        Returns
        -------

        status : AssignmentStatus for object
            for each exposure, is it available or not?

        Notes
        -----

        Checks if a robot is available to assign at each exposure.
        The robot is available if it is not assigned to any target or
        if it is assigned to a "spare" calibration target, AND if
        assigning the target would not collide with any other robot or
        if it would collide, it would be with a "spare" calibration
        target. A spare calibration target is one for which there are
        more than enough calibration targets of that type already.

        The unassign_assignable() method can be used to unassign the
        target assignments that are standing in the way of the
        exposures deemed available.

        So to assign these spare exposures one would do something like
        the following::

         status = f.available_robot_exposures(robotID=robotID, rsid=rsid)
         iassignable = np.where(status.assignable)[0]
         for iexp in iassignable:
             f.unassign_assignable(status=status, iexp=iexp)
             f.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp)

"""
        iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        status = AssignmentStatus(rsid=rsid, robotID=robotID,
                                  iexps=iexps)
        self.set_assignment_status(status=status, isspare=isspare)
        return(status)

    def _is_spare(self, rsid=None, iexps=None):
        """Is this rsid a spare calibration in these exposures?
"""
        if(iexps is None):
            iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        return(self._has_spare_calib[self._calibration_index[self.rsid2indx[rsid] + 1], iexps] > 0)

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None, nexp=None,
                           reset_satisfied=True, reset_has_spare=True,
                           status=None, reset_count=True):
        """Assign an rsid to a particular robot-epoch

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign

        robotID : np.int64
            robotID to assign to

        epoch : int or np.int32
            epoch to assign to

        nexp : int or np.int32
            number of exposures needed

        status : AssignmentStatus  object
            status for each exposure 

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix based on this assignment
            (default True)

        reset_count : bool
            if True, reset the exposure and epoch counts
            (default True)

        Returns
        -------

        success : bool
            True if successful, False otherwise
"""
        # Only try to assign if you can. You should count on it being
        # assignable if any exposure in the epoch can be observed.
        iexpst = self.field_cadence.epoch_indx[epoch]
        if(rsid not in self.mastergrid.robotDict[robotID].validTargetIDs):
            return False

        # Get list of available exposures in the epoch
        if(status is None):
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
            isspare = self.has_spare_calib(rsid=rsid, iexps=iexps)
            status = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexps)
            self.set_assignment_status(status=status, isspare=isspare)

        assignable = status.assignable_exposures()

        # Bomb if there aren't enough available; note that this 
        # implicitly assumes that if the target is already observed
        # in an exposure, it cannot also be assignable in that exposure
        nassignable = len(assignable)
        nalready = status.already.sum()
        ntotal = nassignable + nalready
        if(ntotal < nexp):
            return False

        # Don't assign more than necessary
        ntoassign = nexp - nalready

        # Now actually assign (to first available exposures)
        for iexp in assignable[0:ntoassign]:
            self.unassign_assignable(status=status, iexp=iexp,
                                     reset_count=False,
                                     reset_satisfied=False,
                                     reset_has_spare=False)
            self.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp,
                                       reset_count=False,
                                       reset_satisfied=False,
                                       reset_has_spare=False)

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=[rsid], iexps=assignable[0:nexp])

        if(reset_satisfied):
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return True

    def _set_competing_targets(self, rsids=None):
        """Set number of competing targets for each robotID from this set

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsid values to count for each robot

        Notes
        -----

        Sets attribute _competing_targets to an array with number of competing targets.

"""
        self._competing_targets = np.zeros(len(self.mastergrid.robotDict), dtype=np.int32)
        for rsid in rsids:
            robotIDs = self.masterTargetDict[rsid].validRobotIDs
            robotindx = np.array([self.robotID2indx[r] for r in robotIDs],
                                 dtype=int)
            self._competing_targets[robotindx] += 1
        return

    def assign_robot_exposure(self, robotID=None, rsid=None, iexp=None,
                              reset_satisfied=True, reset_has_spare=True,
                              reset_count=True, set_expflag=True,
                              set_fixed=False, force=False):
        """Assign an rsid to a particular robot-exposure

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign

        robotID : np.int64
            robotID to assign to

        iexp : int or np.int32
            exposure to assign to

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix (default True)

        reset_count : bool
            if True, reset the 'nexp' and 'nepochs' columns (default True)

        set_expflag : bool
            if True, set 'expflag' according to current value of stage (default True)

        set_fixed : bool
            if True, set 'expflag's FIXED bit so this assignment is not removed (default False)

        force : bool
            if True, does everything but if the target cannot be reached, skips setting RobotGrid; use with EXTREME care (default False)

        Returns
        --------

        success : bool
            True if successful, False otherwise
"""
        itarget = self.rsid2indx[rsid]
        robotindx = self.robotID2indx[robotID]

        if(self._robot_locked[robotindx, iexp]):
            print("fieldid {fid}: WARNING, tried to assign locked robot rsid={rsid} iexp={iexp} robotID={robotID}, expflag={expflag}".format(rsid=rsid, iexp=iexp, robotID=robotID, fid=self.fieldid, expflag=self.assignments['expflag'][itarget, iexp]), flush=True)
            return False

        if(self.assignments['robotID'][itarget, iexp] >= 0):
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=True,
                                   reset_satisfied=True, reset_has_spare=True)

        if(self._robot2indx[robotindx, iexp] >= 0):
            rsid_unassign = self.targets['rsid'][self._robot2indx[robotindx,
                                                                  iexp]]
            self.unassign_exposure(rsid=rsid_unassign, iexp=iexp,
                                   reset_assigned=True, reset_satisfied=True,
                                   reset_has_spare=True)

        self.assignments['robotID'][itarget, iexp] = robotID
        self._robot2indx[robotindx, iexp] = itarget
        epoch = self.field_cadence.epochs[iexp]
        self._robotnexp[robotindx, epoch] = self._robotnexp[robotindx, epoch] - 1
        if(self.targets['category'][itarget] == 'science'):
            self._robotnexp_max[robotindx, epoch] = self._robotnexp_max[robotindx, epoch] - 1
        self.assignments['assigned'][itarget] = 1

        if(set_expflag):
            if(self.stage is None):
                self.set_expflag(rsid=rsid, iexp=iexp, flagname='OTHER')
            else:
                self.set_expflag(rsid=rsid, iexp=iexp, flagname=self.stage.upper())

        if(set_fixed):
            self.set_expflag(rsid=rsid, iexp=iexp, flagname='FIXED')

        # If this is a calibration target, update calibration target tracker
        if(self.nocalib is False):
            if(self._is_calibration[itarget]):
                category = self.targets['category'][itarget]
                self.calibrations[category][iexp] = self.calibrations[category][iexp] + 1
                if(self._is_good_calibration[itarget]):
                    ical = self._calibration_index[itarget]
                    zone = self.targets['zone'][itarget]
                    self.calibrations_per_zone[ical, iexp, zone] = self.calibrations_per_zone[ical, iexp, zone] + 1

        if(self.allgrids):
            rg = self.robotgrids[iexp]
            if(force):
                if(rsid in rg.robotDict[robotID].validTargetIDs):
                    rg.assignRobot2Target(robotID, rsid)
                else:
                    self.set_expflag(rsid=rsid, iexp=iexp, flagname='FORCED')
                    print("fieldid {fid}: WARNING, forcing unreachable robot assignment rsid={rsid} iexp={iexp} robotID={robotID}".format(rsid=rsid, iexp=iexp, robotID=robotID, fid=self.fieldid, flush=True))
            else:
                rg.assignRobot2Target(robotID, rsid)

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=[rsid], iexps=[iexp])

        if(reset_satisfied):
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return True

    def assign_exposures(self, rsid=None, iexps=None, check_spare=True,
                         reset_satisfied=True, reset_has_spare=True,
                         set_fixed=False, set_expflag=True):
        """Assign an rsid to particular exposures

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign

        iexps : ndarray of np.int32
            exposures to assign to

        set_fixed : bool
            set the FIXED flag to prevent this target being taken away (default False)

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix
            (default True)

        check_spare : bool
            if True, checks whether spare calibrations can be bumped (default True)

        set_expflag : bool
            if True, set expflag (default True)

        Returns
        -------

        success : ndarray of bool
            for each exposure, True if successful, False otherwise
"""
        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs, dtype=np.int32)
        validRobotIndxs = np.array([self.robotID2indx[x]
                                    for x in validRobotIDs], dtype=int)
        hasApogee = self.robotHasApogee[validRobotIndxs]
        validRobotIDs = validRobotIDs[np.argsort(hasApogee,
                                                 kind='stable')]
        done = np.zeros(len(iexps), dtype=bool)

        for robotID in validRobotIDs:
            cexps = iexps[np.where(done == False)[0]]
            if(len(cexps) == 0):
                break
            status = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=cexps)
            self.set_assignment_status(status=status, check_spare=check_spare)
            for iexp in status.assignable_exposures():
                self.unassign_assignable(status=status, iexp=iexp,
                                         reset_count=False,
                                         reset_satisfied=False,
                                         reset_has_spare=False)
                self.assign_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp,
                                           reset_count=False,
                                           reset_satisfied=False,
                                           reset_has_spare=False,
                                           set_fixed=set_fixed,
                                           set_expflag=set_expflag)
                iorig = np.where(iexps == iexp)[0]
                done[iorig] = True

        if(reset_satisfied):
            self._set_equiv(rsids=[rsid], iexps=iexps)
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return done

    def _set_assigned(self, itarget=None):
        """Set assigned flag
    
        Parameters
        ----------

        itarget : np.int32 or int
            0-indexed position of target in targets array

        Notes
        -----

        Sets 'assigned' in assigments array if any exposure has robotID set
        in the assignments array.
"""
        if(itarget is None):
            print("Must specify a target.")
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        return

    def unassign_exposure(self, rsid=None, iexp=None, reset_assigned=True,
                          reset_satisfied=True, reset_has_spare=True,
                          reset_count=True, respect_fixed=False):
        """Unassign an rsid from a particular exposure

        Parameters
        ----------

        rsid : np.int64
            rsid of target to unassign

        iexp : int or np.int32
            exposure to unassign from

        reset_assigned : bool
            if True, reset the 'assigned' flag after unassignment
            (default True)

        reset_satisfied : bool
            if True, reset the 'satisfied' flag after unassignment
            (default True)

        reset_count : bool
            if True, reset the exposures and epochs count
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix after unassignment
            (default True)

        respect_fixed : bool
            if True, refuse to unassign a fixed exposure for a target (default False)
"""
        itarget = self.rsid2indx[rsid]
        robotID = self.assignments['robotID'][itarget, iexp]
        category = self.targets['category'][itarget]
        zone = self.targets['zone'][itarget]

        if(self.assignments['expflag'][itarget, iexp] & self.expflagdict['FIXED']):
            if(respect_fixed is False):
                print("fieldid {fid}: WARNING, removing supposedly fixed assignment, rsid={rsid} iexp={iexp} robotID={robotID}, expflag={expflag}".format(rsid=rsid, iexp=iexp, robotID=robotID, fid=self.fieldid, expflag=self.assignments['expflag'][itarget, iexp]), flush=True)
            else:
                return

        if(robotID >= 1):
            robotindx = self.robotID2indx[robotID]
            if(self.allgrids):
                rg = self.robotgrids[iexp]
                rg.unassignTarget(rsid)
            self.assignments['robotID'][itarget, iexp] = -1
            self.assignments['expflag'][itarget, iexp] = 0
            self._robot2indx[robotindx, iexp] = -1
            epoch = self.field_cadence.epochs[iexp]
            self._robotnexp[robotindx, epoch] = self._robotnexp[robotindx, epoch] + 1
            if(self.targets['category'][itarget] == 'science'):
                self._robotnexp_max[robotindx, epoch] = self._robotnexp_max[robotindx, epoch] + 1
            if(self.nocalib is False):
                if(self._is_calibration[itarget]):
                    self.calibrations[category][iexp] = self.calibrations[category][iexp] - 1
                if(self._is_good_calibration[itarget]):
                    ical = self._calibration_index[itarget]
                    self.calibrations_per_zone[ical, iexp, zone] = self.calibrations_per_zone[ical, iexp, zone] - 1
        else:
            return

        if(reset_assigned == True):
            self._set_assigned(itarget=itarget)

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=[rsid], iexps=[iexp])

        if(reset_satisfied):
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return

    def unassign_epoch(self, rsid=None, epoch=None, reset_assigned=True,
                       reset_satisfied=True, reset_has_spare=True,
                       reset_count=True, respect_fixed=False):
        """Unassign an rsid from a particular epoch

        Parameters
        ----------

        rsid : np.int64
            rsid of target to unassign

        epoch : int or np.int32
            epoch to unassign from

        reset_assigned : bool
            if True, reset the 'assigned' flag after unassignment
            (default True)

        reset_satisfied : bool
            if True, reset the 'satisfied' flag after unassignment
            (default True)

        reset_count : bool
            if True, reset the epoch and exposure counts
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix after unassignment
            (default True)

        respect_fixed : bool
            if True, refuse to unassign a fixed exposure for a target (default False)

        Returns
        -------

        status : int
            0 if the target had been assigned and was successfully removed
"""
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        iexps = np.arange(iexpst, iexpnd)
        for iexp in iexps:
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=False,
                                   reset_satisfied=False, reset_has_spare=False,
                                   reset_count=False, respect_fixed=respect_fixed)

        if(reset_assigned):
            self._set_assigned(itarget=self.rsid2indx[rsid])

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=[rsid], iexps=iexps)

        if(reset_satisfied):
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return 0

    def unassign(self, rsids=None, reset_assigned=True, reset_satisfied=True,
                 reset_has_spare=True, reset_count=True, respect_fixed=False):
        """Unassign a set of rsids entirely

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids of targets to unassign

        reset_assigned : bool
            if True, resets assigned flag for this rsid (default True)

        reset_satisfied : bool
            if True, resets satified flag for this catalogid (default True)

        reset_count : bool
            if True, resets exposure and epoch count (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix after unassignment
            (default True)

        respect_fixed : bool
            if True, refuse to unassign a fixed exposure for a target (default False)
"""
        if(len(rsids) == 0):
            return

        for rsid in rsids:
            for epoch in range(self.field_cadence.nepochs):
                self.unassign_epoch(rsid=rsid, epoch=epoch, reset_assigned=False,
                                    reset_satisfied=False, reset_has_spare=False,
                                    respect_fixed=respect_fixed)

        if(reset_assigned):
            for rsid in rsids:
                self._set_assigned(itarget=self.rsid2indx[rsid])

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=rsids)

        if(reset_satisfied):
            self._set_satisfied(rsids=rsids, reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=rsids, reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return

    def _merge_epochs(self, epochs=None, nexps=None):
        """Merge epoch list to combine repeats

        Parameters
        ----------

        epochs : ndarray of np.int32
            epochs to assign to (default all)

        nexps : ndarray of np.int32
            number of exposures needed

        Returns
        -------

        epochs_merged : ndarray of np.int32
            new merged epochs

        nexps_merged : ndarray of np.int32
            number of exposures in merged epochs
"""
        epochs_merged, epochs_inverse = np.unique(epochs, return_inverse=True)
        nexps_merged = np.zeros(len(epochs_merged), dtype=np.int32)
        for i, nexp in zip(epochs_inverse, nexps):
            nexps_merged[i] = nexps_merged[i] + nexp

        return(epochs_merged, nexps_merged)

    def available_epochs(self, rsid=None, epochs=None, nexps=None,
                         first=False, strict=False):
        """Find robots available for each epoch

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign

        epochs : ndarray of np.int32
            epochs to assign to (default all)

        nexps : ndarray of np.int32
            number of exposures needed (default 1 per epoch)

        first : bool
            if set, just return the first available robot

        strict : bool
            if set, first check if epoch request is possible, and
            return nothing if the full request cannot be fulfilled

        Returns
        -------

        available : dictionary, with key value pairs below
            'available' : bool
                are ALL needed exposures in every listed epoch available

            'nAvailableRobotIDs' : ndarray of int32
                how many available robotIDs at each epoch

            'availableRobotIDs' : list of lists
                for each epoch, list of available robotIDs sorted by robotID

            'statuses' : list of list of AssignmentStatus
                for each epoch, and each available robotID, status
                regarding whether each exposure is "free"
"""
        if(epochs is None):
            epochs = np.arange(self.field_cadence.nepochs, dtype=np.int32)
        if(nexps is None):
            nexps = np.ones(len(epochs))

        nAvailableRobotIDs = np.zeros(len(epochs), dtype=np.int32)
        availableRobotIDs = [[]] * len(epochs)
        statuses = [[]] * len(epochs)

        bad = (self.assignments['allowed'][self.rsid2indx[rsid], epochs] == 0)
        if(bad.min() > 0):
            available = dict()
            available['available'] = False
            available['nAvailableRobotIDs'] = nAvailableRobotIDs
            available['availableRobotIDs'] = availableRobotIDs
            available['statuses'] = statuses
            return(available)

        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs, dtype=np.int32)
        validRobotIndxs = np.array([self.robotID2indx[x]
                                    for x in validRobotIDs], dtype=int)

        if(len(validRobotIDs) == 0):
            available = dict()
            available['available'] = False
            available['nAvailableRobotIDs'] = nAvailableRobotIDs
            available['availableRobotIDs'] = availableRobotIDs
            available['statuses'] = statuses
            return(available)

        # Prefer BOSS-only robots if they are available
        hasApogee = self.robotHasApogee[validRobotIndxs]
        validRobotIDs = validRobotIDs[hasApogee.argsort(kind='stable')]


        if(self.nocalib is False):
            isspare = self.has_spare_calib(rsid=rsid)
        else:
            isspare = np.zeros(self.field_cadence.nexp_total, dtype=bool)

        for iepoch, epoch in enumerate(epochs):
            nexp = nexps[iepoch]
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
            arlist = []
            slist = []

            # Move preferred robotID to front
            preferred_robotid = self.get_preferred_robotid(epoch=epoch,
                                                           rsid=rsid)
            if(preferred_robotid is not None):
                ipreferred = np.where(validRobotIDs == preferred_robotid)[0]
                if(len(ipreferred) > 0):
                    validRobotIDs = np.delete(validRobotIDs, ipreferred[0])
                    validRobotIDs = np.insert(validRobotIDs, 0,
                                              preferred_robotid)

            for robotID in validRobotIDs:
                ok, status = self.available_robot_epoch(rsid=rsid,
                                                        robotID=robotID,
                                                        epoch=epoch,
                                                        nexp=nexp,
                                                        isspare=isspare[iexps])

                if(ok):
                    arlist.append(robotID)
                    slist.append(status)
                    # If this robot was good, then let's just return it
                    if(first):
                        break

            availableRobotIDs[iepoch] = arlist
            nAvailableRobotIDs[iepoch] = len(arlist)
            statuses[iepoch] = slist

        available = dict()
        available['available'] = nAvailableRobotIDs.min() > 0
        available['nAvailableRobotIDs'] = nAvailableRobotIDs
        available['availableRobotIDs'] = availableRobotIDs
        available['statuses'] = statuses
        return(available)

    def assign_epochs(self, rsid=None, epochs=None, nexps=None, test_only=False):
        """Assign target to robots in a set of epochs

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign

        epochs : ndarray of np.int32
            epochs to assign to

        nexps : ndarray of np.int32
            number of exposures needed

        test_only : bool
            if set, only perform test and do not actually assign

        Returns
        -------

        success : bool
            True if successful, False otherwise
"""
        if(self.methods['assign_epochs'] == 'first'):
            first = True
        else:
            first = False

        available = self.available_epochs(rsid=rsid, epochs=epochs,
                                          nexps=nexps,
                                          strict=True, first=first)
        availableRobotIDs = available['availableRobotIDs']
        statuses = available['statuses']

        # Check if there are robots available
        nRobotIDs = np.array([len(x) for x in availableRobotIDs], dtype=int)
        if(nRobotIDs.min() < 1):
            if(self.veryverbose):
                print("rsid={r}: no robots available".format(r=rsid))
            return False

        if(test_only):
            return True

        # Assign to each epoch
        robotID = -1
        for iepoch, epoch in enumerate(epochs):
            currRobotIDs = np.array(availableRobotIDs[iepoch], dtype=np.int32)
            currRobotIndxs = np.array([self.robotID2indx[x]
                                       for x in currRobotIDs], dtype=int)
            if(self.methods['assign_epochs'] == 'first'):
                irobot = 0
            if(self.methods['assign_epochs'] == 'same'):
                irobot = np.where(robotID == currRobotIDs)[0]
                if(len(irobot) > 0):
                    irobot = irobot[0]
                else:
                    irobot = 0
            if(self.methods['assign_epochs'] == 'fewestcompeting'):
                irobot = np.argmin(self._competing_targets[currRobotIndxs])
            preferred_robotid = self.get_preferred_robotid(epoch=epoch,
                                                           rsid=rsid)
            if(preferred_robotid is not None):
                ipreferred = np.where(currRobotIDs == preferred_robotid)[0]
                if(len(ipreferred) > 0):
                    irobot = ipreferred[0]
                
            robotID = currRobotIDs[irobot]
            status = statuses[iepoch][irobot]
            nexp = nexps[iepoch]

            if(self.veryverbose):
                print("rsid={r}: assigning robotID {robotID}".format(r=rsid,
                                                                     robotID=robotID))

            self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch,
                                    nexp=nexp, status=status,
                                    reset_satisfied=False,
                                    reset_has_spare=False,
                                    reset_count=False)

        self._set_satisfied(rsids=[rsid])
        self._set_count(rsids=[rsid], reset_equiv=False)
        if(self.nocalib is False):
            self._set_has_spare_calib()

        return True

    def assign_cadence(self, rsid=None, test_only=False):
        """Assign target to robots according to its cadence

        Parameters
        ----------

        rsid : np.int64
            rsid of target to assign
        
        test_only : bool
            if True, only perform test, do not actually assign

        Returns
        -------

        success : bool
            True if successful, False otherwise
"""
        indx = self.rsid2indx[rsid]
        target_cadence = self.targets['cadence'][indx]

        ok, epochs_list, nexps_list = clist.cadence_consistency(target_cadence,
                                                                self.field_cadence.name,
                                                                return_solutions=True,
                                                                epoch_level=True,
                                                                merge_epochs=True)

        if(ok == False):
            if(self.veryverbose):
                print("rsid={r} does not fit field cadence".format(r=rsid))
            return False

        # Check for all potential epochs whether they can accomodate at
        # least the minimum number of exposures; if not we can eliminate
        # them.
        if(len(epochs_list) > 100):
            epochs = np.arange(self.field_cadence.nepochs, dtype=np.int32)
            nexps = (np.zeros(self.field_cadence.nepochs, dtype=np.int32) + 
                     clist.cadences[target_cadence].nexp.min())
        else:
            epochs = np.unique(np.array([e for es in epochs_list for e in es], dtype=int))
            nexps = np.zeros(len(epochs), dtype=np.int32) + np.array([ne for nes in nexps_list for ne in nes], dtype=np.int32).min()

        available = self.available_epochs(rsid, epochs=epochs, nexps=nexps,
                                          strict=False, first=True)
        ibad = np.where(available['nAvailableRobotIDs'] == 0)[0]
        epoch_bad = np.zeros(self.field_cadence.nepochs, dtype=bool)
        epoch_bad[epochs[ibad]] = True

        if(self.veryverbose):
            print("rsid={r}: note epoch_bad=".format(r=rsid) + str(epoch_bad)) 
        
        allowed = self.assignments['allowed'][indx, :]
        any_allowed = False
        for eindx, epochs in enumerate(epochs_list):
            all_allowed = allowed[epochs].min()
            if(all_allowed > 0):
                any_allowed = True
                if(epoch_bad[epochs].max() == False):
                    if(self.veryverbose):
                        print("rsid={r}: trying epochs: ".format(r=rsid) + str(epochs))
                    nexps = nexps_list[eindx]
                    if(self.assign_epochs(rsid=rsid, epochs=epochs, nexps=nexps,
                                          test_only=test_only)):
                        return True

        if(test_only):
            return False

        if(any_allowed is False):
            self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
        else:
            self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        if(self.veryverbose):
            print("rsid={r}: no epochs worked".format(r=rsid))
                
        return False

    def _set_equiv(self, rsids=None, iexps=None, science=False):
        """Set equivRobotID to reflect any compatible observations with this rsid

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids to update (default all currently assigned)

        iexps : ndarray of np.int32
            exposures to update (default all field exposures)

        science : bool
            if True, set for science (default False)

        Notes
        -----

        This finds ALL entries with the same:

            catalogid
            fiberType
            lambda_eff
            delta_ra
            delta_dec

        and sets the assignments['equivRobotID'] for all of them, if 
        any of them have assignments['robotID'] set

        If 'science' is True, this operation is performed only
        among science targets and assignments['scienceRobotID']
        is set instead.
"""
        if(rsids is None):
            rsids = self.targets['rsid']

        ekey = self._equivkey

        if(science):
            # For scienceRobotID, only count robots assigned to 
            # equivalent science targets
            counts_indx = self._equivindx_science
            update_indx = self._equivindx
            robotidname = 'scienceRobotID'
        else:
            counts_indx = self._equivindx
            update_indx = self._equivindx
            robotidname = 'equivRobotID'

        if(iexps is None):
            iexps = np.arange(self.field_cadence.nexp_total, dtype=int)
        else:
            iexps = np.int32(iexps)

        epochs = self.field_cadence.epochs

        for rsid in rsids:
            indx = self.rsid2indx[rsid]
            count = counts_indx[ekey[indx]]
            update = update_indx[ekey[indx]]

            if(len(count) == 0):
                continue

            if(len(count) > 1):
                for iexp in iexps:
                    robotIDs = self.assignments['robotID'][count, iexp]
                    robotIDs = robotIDs[robotIDs >= 0]
                    if(len(robotIDs) > 0):
                        if(len(robotIDs) > 1):
                            print("fieldid {fid}: Inconsistency: multiple equivalent rsids with robots assigned".format(fid=self.fieldid), flush=True)
                            return
                        robotID = robotIDs[0]
                    else:
                        robotID = -1

                    # Only update the equivRobotID or scienceRobotID
                    # for target entries for which this exposure is
                    # allowed
                    allowed = self.assignments['allowed'][update,
                                                          epochs[iexp]]
                    self.assignments[robotidname][update[allowed],
                                                  iexp] = robotID
            else:
                # Assumes update array is a superset of count, so if there is one counted,
                # it is the one to be updated. Only update for exposures
                # where this one is allowed. (Should be all of them,
                # since we think update[0] == count[0]).
                allowed = self.assignments['allowed'][update[0],
                                                      epochs[iexps]]
                self.assignments[robotidname][update[0], iexps[allowed]] = self.assignments['robotID'][count[0], iexps[allowed]]

        return
            
    def _set_satisfied(self, rsids=None, reset_equiv=True, science=False):
        """Set satisfied flag based on assignments

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids to set (defaults to apply to all targets)

        reset_equiv : bool
            whether to reset equivRobotID before assessing (default True)

        science : bool
            if True, set for science (default False)

        Notes
        -----

        'satisfied' means that the exposures obtained satisfy
        the cadence for an rsid and the right instrument.

        Uses equivRobotID to assess whether the conditions are
        satisfied.

        Only set reset_equiv=False if you have already just run
        _set_equiv() for these rsids (or all of them). Doing so 
        will save doing that twice.

        If 'science' is True, this operation is performed only
        among science targets and the attribute _scienceSatisfied 
        is set instead.
"""
        if(science):
            robotidname = 'scienceRobotID'
            satisfiedname = 'science_satisfied'
        else:
            robotidname = 'equivRobotID'
            satisfiedname = 'satisfied'

        if(reset_equiv):
            self._set_equiv(rsids=rsids, science=science)

        if(rsids is None):
            set_rsids = self.targets['rsid']
        else:
            set_rsids = set(rsids)
            for rsid in rsids:
                indx = self.rsid2indx[rsid]
                for eindx in self._equivindx[self._equivkey[indx]]:
                    set_rsids.add(self.targets['rsid'][eindx])
            set_rsids = np.array(list(set_rsids), dtype=np.int64)

        for rsid in set_rsids:
            indx = self.rsid2indx[rsid]
            iexp = np.where(self.assignments[robotidname][indx, :] >= 0)[0]
            target_cadence = self.targets['cadence'][indx]

            if(target_cadence != ''):

                # if the target cadence is really just a suite of single
                # bright exposures, just set on basis of number of
                # exposures
                if(clist.cadence_consistency(target_cadence,
                                             '_field_single_12x1',
                                             return_solutions=False)):
                    if(len(iexp) >=
                       clist.cadences[target_cadence].nexp_total):
                        sat = 1
                    else:
                        sat = 0
                else:
                    # if not, check consistency in detail
                    sat = clist.exposure_consistency(self.targets['cadence'][indx],
                                                     self.field_cadence.name, iexp)
            else:
                sat = 0

            self.assignments[satisfiedname][indx] = sat

        return

    def _set_count(self, rsids=None, reset_equiv=True):
        """Set exposure and epochs based on assignments

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids to set (defaults to apply to all targets)

        reset_equiv : bool
            whether to reset equivRobotID before assessing (default True)

        Notes
        -----

        Sets nexps, nepochs for each target, based on equivRobotID.

        Only set reset_equiv=False if you have already just run
        _set_equiv() for these rsids (or all of them). Doing so 
        will save doing that twice.
"""
        if(reset_equiv):
            self._set_equiv(rsids=rsids)

        if(rsids is None):
            set_rsids = self.targets['rsid']
            indxs = np.arange(len(self.targets), dtype=int)
        else:
            set_rsids = set(rsids)
            for rsid in rsids:
                indx = self.rsid2indx[rsid]
                for eindx in self._equivindx[self._equivkey[indx]]:
                    set_rsids.add(self.targets['rsid'][eindx])
            set_rsids = np.array(list(set_rsids), dtype=np.int64)
            indxs = np.array([self.rsid2indx[x] for x in set_rsids], dtype=int)

        self.assignments['nexps'][indxs] = (self.assignments['equivRobotID'][indxs, :] >= 0).sum(axis=1)
        self.assignments['nepochs'][indxs] = 0
        icheck = np.where(self.assignments['nexps'][indxs] > 0)
        for indx in indxs[icheck]:
            iexp = np.where(self.assignments['equivRobotID'][indx, :] >= 0)[0]
            epochs = np.unique(self.field_cadence.epochs[iexp])
            self.assignments['nepochs'][indx] = len(epochs)

        return

    def _assign_one_by_one(self, rsids=None, check_satisfied=True, test_only=False):
        """Assign a set of targets to robots

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        check_satisfied : bool
            if True, do not try to reassign targets that are already satisfied

        test_only : bool
            if True, only test success, do not actually assign

        Returns
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes
        -----

        Performs assigment in order rsids are given.
"""
        success = np.zeros(len(rsids), dtype=bool)
        for i, rsid in enumerate(rsids):
            # Perform the assignment
            if((check_satisfied == False) |
               (self.assignments['satisfied'][self.rsid2indx[rsid]] == 0)):
                success[i] = self.assign_cadence(rsid=rsid, test_only=test_only)
        return(success)

    def _unsatisfied(self, indxs):
        # Return which are unsatisfied, and include ones which are
        # satisfied but only because of a calibration target
        self._set_satisfied(rsids=self.targets['rsid'][indxs], science=True)
        unsatisfied = ((self.assignments['satisfied'][indxs] == 0) |
                       ((self.assignments['satisfied'][indxs] != self.assignments['science_satisfied'][indxs]) &
                        (self.targets['category'][indxs] == 'science')))
        return(unsatisfied)

    def assign_cadences(self, rsids=None, check_satisfied=True, test_only=False):
        """Assign a set of targets to robots

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        check_satisfied : bool
            if True, do not try to reassign targets that are already satisfied

        test_only : bool
            if True, only test success, do not actually assign

        Returns
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes
        -----

        Sorts cadences by priority for assignment.

        Then it identifies the targets that:

          - Are not assignable as "single bright", "multi bright", or
            "multi dark" (see below); these will be assigned with the
            method _assign_one_by_one()

          - Are assignable as "single bright" -- i.e. just need a 
            single bright exposures; these will be assigned with the 
            method _assign_singlebright()

          - Are assignable as "multi bright" -- i.e. just need a 
            bunch of bright exposures with no particular cadence 
            (up to 12); these will be assigned with the method 
            _assign_multibright()

          - Are assignable as "multi dark" -- i.e. just need a 
            bunch of bright or dark exposures with no particular
            cadence (up to 12); these will be assigned with the 
            method _assign_multidark()

        Within each priority level, it assigns the targets in those
        categories in that order.  The reason to separate them is that
        _assign_one_by_one() is the slowest method, whereas a bunch of
        shortcuts are possible if the exposures don't need a
        particular cadence.
"""
        success = np.zeros(len(rsids), dtype=bool)
        indxs = np.array([self.rsid2indx[r] for r in rsids], dtype=np.int32)

        # Find single bright, multibright, multidark cases
        cadences = np.unique(self.targets['cadence'][indxs])
        singlebright = np.zeros(len(self.targets), dtype=bool)
        multibright = np.zeros(len(self.targets), dtype=bool)
        multidark = np.zeros(len(self.targets), dtype=bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_1x1',
                                         return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                singlebright[indxs[icad]] = True
            elif(clist.cadence_consistency(cadence, '_field_single_12x1',
                                           return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                multibright[indxs[icad]] = True
            elif(clist.cadence_consistency(cadence, '_field_dark_single_12x1',
                                           return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                multidark[indxs[icad]] = True

        # Sort by priority
        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            ipriority = np.where(self.targets['priority'][indxs] == priority)[0]
            cindxs = indxs[ipriority]
            crsids = rsids[ipriority]

            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            # Choose the ones to assign 1-by-1
            iassign = np.where((singlebright[cindxs] == False) &
                               (multibright[cindxs] == False) &
                               (multidark[cindxs] == False) &
                               (self._unsatisfied(indxs=cindxs)))[0]

            # If we have already assigned some locked exposures, prioritize
            # those.
            ilocked = np.where(self.exposure_locked > 0)[0]
            if(len(ilocked) > 0):
                ngotten = np.zeros(len(iassign), dtype=np.int32)
                for i in ilocked:
                    ngotten = (ngotten +
                               (self.assignments['robotID'][iassign, i] >= 0))
                isort = np.flip(np.argsort(ngotten))
                iassign = iassign[isort]

            if(self.verbose):
                iall = np.where(self.assignments['satisfied'][cindxs] == 0)[0]

                outstr = "fieldid {fid}: Includes cadences ".format(fid=self.fieldid)
                pcads = np.unique(self.targets['cadence'][cindxs[iall]])
                for pcad in pcads:
                    outstr = outstr + pcad + " "
                print(outstr, flush=True)

                outstr = "fieldid {fid}: Includes cartons ".format(fid=self.fieldid)
                pcarts = np.unique(self.targets['carton'][cindxs[iall]])
                for pcart in pcarts:
                    outstr = outstr + pcart + " "
                print(outstr, flush=True)
            
            if(len(iassign) > 0):
                if(self.verbose):
                    print("fieldid {fid}:  - {n} assigning one-by-one".format(n=len(iassign), fid=self.fieldid), flush=True)
                    
                success[iassign] = self._assign_one_by_one(rsids=crsids[iassign],
                                                           check_satisfied=check_satisfied,
                                                           test_only=test_only)  
                    
                if(self.verbose):
                    print("fieldid {fid}:    (assigned {n})".format(n=success[iassign].sum(), fid=self.fieldid), flush=True)

            # Assign single bright, which we do in two cycles.
            # It is always affordable to run through the single bright
            # cases twice. Why does it matter? Because when they displace
            # calibration targets on the first cycle, that can change the
            # collision situation on the second round. This is a 1% effect.
            # A second cycle might be worth doing for one-by-one cases, but
            # it is more expensive in that case in terms of run-time.
            isinglebright = np.where(singlebright[cindxs])[0]
            if(len(isinglebright) > 0):
                for icycle in range(2):
                    isinglebright = np.where(singlebright[cindxs] &
                                             (self._unsatisfied(indxs=cindxs)))[0]
                    if(len(isinglebright) > 0):
                        if(self.verbose):
                            print("fieldid {fid}:  - {n} assigning as single bright (cycle {i})".format(n=len(isinglebright), i=icycle, fid=self.fieldid), flush=True)
                            
                    tmp_success = self._assign_singlebright(indxs=cindxs[isinglebright],
                                                            test_only=test_only)
                    if(test_only):
                        success[isinglebright] = tmp_success
                    else:
                        success[isinglebright] = (self._unsatisfied(indxs=cindxs[isinglebright]) == False)

                        if(self.verbose):
                            print("fieldid {fid}:    (assigned {n})".format(n=success[isinglebright].sum(), fid=self.fieldid), flush=True)

            # Assign multi-bright cases (one cycle)
            imultibright = np.where(multibright[cindxs])[0]
            if(len(imultibright) > 0):
                for icycle in range(1):
                    imultibright = np.where(multibright[cindxs] &
                                            (self._unsatisfied(indxs=cindxs)))[0]
                    if(len(imultibright) > 0):
                        if(self.verbose):
                            print("fieldid {fid}:  - {n} assigning as multi bright (cycle {i})".format(n=len(imultibright), i=icycle, fid=self.fieldid), flush=True)
                            tmp_success = self._assign_multibright(indxs=cindxs[imultibright], test_only=test_only)
                        if(test_only):
                            success[imultibright] = tmp_success
                        else:
                            success[imultibright] = (self._unsatisfied(indxs=cindxs[imultibright]) == False)

                        if(self.verbose):
                            print("fieldid {fid}:    (assigned {n})".format(n=success[imultibright].sum(), fid=self.fieldid), flush=True)

            # Assign multi-dark cases (one cycle)
            imultidark = np.where(multidark[cindxs])[0]
            if(len(imultidark) > 0):
                for icycle in range(2):
                    imultidark = np.where(multidark[cindxs] &
                                          (self._unsatisfied(indxs=cindxs)))[0]
                    if(len(imultidark) > 0):
                        if(self.verbose):
                            print("fieldid {fid}:  - {n} assigning as multi dark (cycle {i})".format(n=len(imultidark), i=icycle, fid=self.fieldid), flush=True)
                        tmp_success = self._assign_multidark(indxs=cindxs[imultidark], test_only=test_only)
                        if(test_only):
                            success[imultidark] = tmp_success
                        else:
                            success[imultidark] = (self._unsatisfied(indxs=cindxs[imultidark]) == False)

                        if(self.verbose):
                            print("fieldid {fid}:    (assigned {n})".format(n=success[imultidark].sum(), fid=self.fieldid), flush=True)

            self._competing_targets = None

        return(success)

    def _assign_singlebright(self, indxs=None, test_only=False):
        """Assigns 1x1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign

        test_only : bool
            just report success, do not actually assign

        Returns
        -------

        success : ndarray of bool
            returns true if assigned

        Note
        ----

        During a real assignment, success reported is ignored, because we 
        need to be more careful and NOT count successes which are from
        calibration targets, because those might get bumped if they are
        spare. Not to worry, the science observations will count toward
        the standard count in the end.
"""
        rsids = self.targets['rsid'][indxs]
        iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)

        tdict = self.mastergrid.targetDict

        inotsat = np.where(self._unsatisfied(indxs) == True)[0]
        succeed = np.zeros(len(rsids), dtype=bool)
        for cinotsat in inotsat:
            rsid = rsids[cinotsat]
            if(self.veryverbose):
                print("fieldid {fid}: singlebright {rsid}".format(fid=self.fieldid, rsid=rsid))
            indx = self.rsid2indx[rsid]
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            robotindx = np.array([self.robotID2indx[x] for x in robotIDs],
                                 dtype=int)
            hasApogee = self.robotHasApogee[robotindx]
            robotIDs = robotIDs[np.argsort(hasApogee, kind='stable')]
            notpreferred = (self.are_preferred_robotids(robotIDs=robotIDs, rsid=rsid) == False)
            robotIDs = robotIDs[np.argsort(notpreferred, kind='stable')]
            robotindx = None

            succeed[cinotsat] = False
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexps)
                self.set_assignment_status(status=s)
                cexps = s.assignable_exposures()
                if(len(cexps) > 0):
                    if(test_only):
                        succeed[cinotsat] = True
                        break
                    if(self.veryverbose):
                        print("fieldid {fid}: assigning {rsid} to robotID={robotID}, iexp={iexp}".format(fid=self.fieldid, rsid=rsid, robotID=robotID, iexp=cexps[0]))
                    self.unassign_assignable(status=s, iexp=cexps[0])
                    self.assign_robot_exposure(robotID=robotID,
                                               rsid=rsid,
                                               iexp=cexps[0],
                                               reset_count=False,
                                               reset_satisfied=False,
                                               reset_has_spare=True)
                    succeed[cinotsat] = True
                    break

            if((succeed[cinotsat] is False) & (test_only is False)):
                if(self.assignments['allowed'][indx].sum() == 0):
                    self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
                else:
                    self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        if(test_only):
            return(succeed)

        self._set_satisfied(rsids=rsids[inotsat])
        return(succeed)

    def _assign_multibright(self, indxs=None, test_only=False):
        """Assigns nx1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign

        test_only : bool
            just report success, do not actually assign

        Returns
        -------

        success : ndarray of bool
            returns true if assigned

        Note
        ----

        During a real assignment, success reported is ignored, because we 
        need to be more careful and NOT count successes which are from
        calibration targets, because those might get bumped if they are
        spare. Not to worry, the science observations will count toward
        the standard count in the end.
"""
        rsids = self.targets['rsid'][indxs]
        iexpsall = np.arange(self.field_cadence.nexp_total, dtype=np.int32)

        tdict = self.mastergrid.targetDict

        inotsat = np.where(self._unsatisfied(indxs) == True)[0]
        succeed = np.zeros(len(rsids), dtype=bool)
        for cinotsat in inotsat:
            rsid = rsids[cinotsat]
            indx = self.rsid2indx[rsid]
            if(self._unsatisfied([indx])[0] == False):
                succeed[cinotsat] = True
                continue
            nexp_cadence = clist.cadences[self.targets['cadence'][indx]].nexp_total
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            robotindx = np.array([self.robotID2indx[x]
                                  for x in robotIDs], dtype=int)
            hasApogee = self.robotHasApogee[robotindx]
            robotIDs = robotIDs[np.argsort(hasApogee, kind='stable')]
            notpreferred = (self.are_preferred_robotids(robotIDs=robotIDs, rsid=rsid) == False)
            robotIDs = robotIDs[np.argsort(notpreferred, kind='stable')]
            robotindx = None

            statusDict = dict()
            expRobotIDs = [[] for _ in range(self.field_cadence.nexp_total)]
            nExpRobotIDs = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)
            nExpAlready = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexpsall)
                self.set_assignment_status(status=s)
                statusDict[robotID] = s
                nExpAlready[s.already] = 1
                for iexp in s.assignable_exposures():
                    expRobotIDs[iexp].append(robotID)
                    nExpRobotIDs[iexp] = nExpRobotIDs[iexp] + 1

            # if number of exposures with at least one free robot is high
            # enough, go ahead
            nalready = nExpAlready.sum()
            nexp_need = nexp_cadence - nalready
            iexps = np.where(nExpRobotIDs > 0)[0]
            if(len(iexps) >= nexp_need):
                succeed[cinotsat] = True

                # Do not assign if this is just a test
                if(test_only):
                    continue

                for iexp in iexps[0:nexp_need]:
                    robotID = expRobotIDs[iexp][0]
                    status = statusDict[robotID]
                    self.unassign_assignable(status=status, iexp=iexp,
                                             reset_count=False,
                                             reset_satisfied=False,
                                             reset_has_spare=True)
                    self.assign_robot_exposure(robotID=robotID,
                                               rsid=rsid,
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False)

                self._set_satisfied(rsids=[rsid])
                if(self.nocalib is False):
                    self._set_has_spare_calib()

            else:
                if(self.assignments['allowed'][indx].sum() == 0):
                    self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
                else:
                    self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        return(succeed)

    def _assign_multidark(self, indxs=None, test_only=False):
        """Assigns nx1 dark targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign

        test_only : bool
            just report success, do not actually assign

        Returns
        -------

        success : ndarray of bool
            returns true if assigned

        Note
        ----

        During a real assignment, success reported is ignored, because we 
        need to be more careful and NOT count successes which are from
        calibration targets, because those might get bumped if they are
        spare. Not to worry, the science observations will count toward
        the standard count in the end.
"""
        rsids = self.targets['rsid'][indxs]
        ok, epochs_list = clist.cadence_consistency('_field_dark_single_1x1', self.field_cadence.name)
        iexpsall = np.array([self.field_cadence.epoch_indx[x[0]] +
                             np.arange(self.field_cadence.nexp[x[0]],
                                       dtype=int) for x in epochs_list],
                            dtype=int).flatten()
        tdict = self.mastergrid.targetDict

        inotsat = np.where(self._unsatisfied(indxs) == True)[0]
        succeed = np.zeros(len(rsids), dtype=bool)

        if(len(iexpsall) == 0):
            return(succeed)

        for cinotsat in inotsat:
            rsid = rsids[cinotsat]
            indx = self.rsid2indx[rsid]
            if(self._unsatisfied([indx])[0] == False):
                succeed[cinotsat] = True
                continue
            nexp_cadence = clist.cadences[self.targets['cadence'][indx]].nexp_total
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            robotindx = np.array([self.robotID2indx[x]
                                  for x in robotIDs], dtype=int)
            hasApogee = self.robotHasApogee[robotindx]
            robotIDs = robotIDs[np.argsort(hasApogee, kind='stable')]
            notpreferred = (self.are_preferred_robotids(robotIDs=robotIDs, rsid=rsid) == False)
            robotIDs = robotIDs[np.argsort(notpreferred, kind='stable')]
            robotindx = None

            statusDict = dict()
            expRobotIDs = [[] for _ in range(len(iexpsall))]
            nExpRobotIDs = np.zeros(len(iexpsall), dtype=np.int32)
            nExpAlready = np.zeros(len(iexpsall), dtype=np.int32)
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID,
                                     iexps=iexpsall)
                self.set_assignment_status(status=s)
                statusDict[robotID] = s
                nExpAlready[iexpsall[s.already]] = 1
                for iexp in s.assignable_exposures():
                    expRobotIDs[iexp].append(robotID)
                    nExpRobotIDs[iexp] = nExpRobotIDs[iexp] + 1

                iexps = np.where(nExpRobotIDs > 0)[0]
                if(len(iexps) >= nexp_cadence):
                    break

            # if number of exposures with at least one free robot is high
            # enough, go ahead
            nalready = nExpAlready.sum()
            nexp_need = nexp_cadence - nalready
            iexps = iexpsall[np.where(nExpRobotIDs > 0)[0]]
            if(len(iexps) >= nexp_need):
                succeed[cinotsat] = True

                # Do not assign if this is just a test
                if(test_only):
                    continue

                for iexp in iexps[0:nexp_need]:
                    robotID = expRobotIDs[iexp][0]
                    status = statusDict[robotID]
                    self.unassign_assignable(status=status, iexp=iexp,
                                             reset_count=False,
                                             reset_satisfied=False,
                                             reset_has_spare=True)
                    self.assign_robot_exposure(robotID=robotID,
                                               rsid=rsid,
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False)

                self._set_satisfied(rsids=[rsid])
                if(self.nocalib is False):
                    self._set_has_spare_calib()

            else:
                if(self.assignments['allowed'][indx].sum() == 0):
                    self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
                else:
                    self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        return(succeed)

    def _assign_cp_model(self, force=[], deny=[],
                         check_collisions=True,
                         calibrations=True, iexp=0):
        """Assigns using CP-SAT to optimize number of targets

        Parameters
        ----------

        force : list of np.int64
            list of rsids to force (default [])
        
        iexp : int
            relevant exposure, used for allowability constraints (default 0)

        check_collisions : bool
            if set, check for collisions (default True)

        calibrations : bool
            if set, guarantee calibration numbers (default True)

        Returns
        -------

        assignedRobotIDs : ndarray of np.int32
            [N] robots to assign to

        Notes
        -----

        Will only assign targets "allowed" in the exposure;
        "force" will not override this.

"""
        rsids = self.targets['rsid']

        # This don't work with offset
        rg = self.mastergrid
        for r in rg.robotDict:
            rg.unassignRobot(r)
            rg.homeRobot(r)

        # Initialize Model
        model = cp_model.CpModel()

        # Add variables; one for each robot-target pair
        # Make a dictionary to organize them as wwrt[robotID][rsid],
        # and one to organize them as wwtr[rsid][robotID], and
        # also a flattened list
        wwrt = dict()
        wwtr = dict()
        for robotID in rg.robotDict:
            r = rg.robotDict[robotID]
            for rsid in interlist(r.validTargetIDs, rsids):
                name = 'ww[{r}][{c}]'.format(r=robotID, c=rsid)
                if(rsid not in wwtr):
                    wwtr[rsid] = dict()
                if(robotID not in wwrt):
                    wwrt[robotID] = dict()
                wwrt[robotID][rsid] = model.NewBoolVar(name)
                wwtr[rsid][robotID] = wwrt[robotID][rsid]

        # List of all robot-target pairs
        ww_list = [wwrt[y][x] for y in wwrt for x in wwrt[y]]

        # List of robot-(science-target) pairs
        ww_science = []
        for robotID in wwrt:
            for rsid in wwrt[robotID]:
                indx = self.rsid2indx[rsid]
                if(self.targets['category'][indx] == 'science'):
                    ww_science.append(wwrt[robotID][rsid])

        # Constrain to use only one target per robot (0 if robot is locked)
        wwsum_robot = dict()
        for robotID in wwrt:
            robotindx = self.robotID2indx[robotID]
            rlist = [wwrt[robotID][c] for c in wwrt[robotID]]
            wwsum_robot[robotID] = cp_model.LinearExpr.Sum(rlist)
            if(self._robot_locked[robotindx, 0]):
                model.Add(wwsum_robot[robotID] == 0)
            else:
                model.Add(wwsum_robot[robotID] <= 1)

        # Constrain to use only one robot per target
        wwsum_target = dict()
        for rsid in wwtr:
            tlist = [wwtr[rsid][r] for r in wwtr[rsid]]
            wwsum_target[rsid] = cp_model.LinearExpr.Sum(tlist)
            allowed = self.assignments['allowed'][self.rsid2indx[rsid], iexp]
            if(allowed == False):
                model.Add(wwsum_target[rsid] == 0)
            elif(rsid in force):
                model.Add(wwsum_target[rsid] == 1)
            elif(rsid in deny):
                model.Add(wwsum_target[rsid] == 0)
            else:
                model.Add(wwsum_target[rsid] <= 1)

        if(calibrations):
            calibsum = dict()
            for c in self.calibration_order:
                minimum = self.required_calibrations[c][0]
                if(minimum > 0):
                    clist = []
                    icalib = np.where(self.targets['category'] == c)[0]
                    for rsid in self.targets['rsid'][icalib]:
                        if(rsid in wwtr):
                            for robotID in wwtr[rsid]:
                                clist.append(wwtr[rsid][robotID])
                    calibsum[c] = cp_model.LinearExpr.Sum(clist)
                    model.Add(calibsum[c] >= int(minimum))

        # Do not allow collisions
        if(check_collisions):

            # Find potention collisions
            collisions = []
            for robotID1 in rg.robotDict:
                r1 = rg.robotDict[robotID1]
                for rsid1 in r1.validTargetIDs:
                    rg.assignRobot2Target(robotID1, rsid1)
                    for robotID2 in r1.robotNeighbors:
                        r2 = rg.robotDict[robotID2]
                        for rsid2 in r2.validTargetIDs:
                            if(rsid1 != rsid2):
                                rg.assignRobot2Target(robotID2, rsid2)
                                if(rg.isCollidedWithAssigned(robotID1)[0]):
                                    collisions.append((robotID1,
                                                       rsid1,
                                                       robotID2,
                                                       rsid2))
                                rg.homeRobot(robotID2)
                    rg.homeRobot(robotID1)

            # Now add constraint that collisions can't occur
            for robotID1, rsid1, robotID2, rsid2 in collisions:
                ww1 = wwrt[robotID1][rsid1]
                ww2 = wwrt[robotID2][rsid2]
                tmp_collision = cp_model.LinearExpr.Sum([ww1, ww2])
                model.Add(tmp_collision <= 1)

        # Maximize the total sum of science targets
        wwsum_all = cp_model.LinearExpr.Sum(ww_science)
        model.Maximize(wwsum_all)

        # But need to decide about all targets
        model.AddDecisionStrategy(ww_list,
                                  cp_model.CHOOSE_FIRST,
                                  cp_model.SELECT_MAX_VALUE)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 16
        status = solver.Solve(model)

        assignedRobotIDs = np.zeros(len(rsids), dtype=np.int32) - 1
        if status == cp_model.OPTIMAL:
            for robotID in wwrt:
                for rsid in wwrt[robotID]:
                    assigned = solver.Value(wwrt[robotID][rsid])
                    if(assigned):
                        irsid = np.where(rsids == rsid)[0]
                        assignedRobotIDs[irsid] = robotID

        return(assignedRobotIDs)

    def assign_full_cp_model(self, rsids=None):
        """Assigns rsids exactly matching field cadence using the CP-SAT module

        Parameters
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        Returns
        -------

        success : ndarray of bool
            True if successful, False otherwise

        Notes
        -----

        Assigns only the ones matching the field cadence
"""
        all_rsids = self.targets['rsid']

        # Weeds out science targets not in field cadence or list
        bad = np.zeros(len(all_rsids), dtype=np.int32)
        for i, rsid in enumerate(all_rsids):
            if(self.targets['category'][i] == 'science'):
                if(self.targets['cadence'][i] != self.field_cadence.name):
                    bad[i] = 1
                if(rsid not in rsids):
                    bad[i] = 1
        deny = all_rsids[(bad > 0) & (self.assignments['assigned'] == 0)]

        # Force any assigned
        iforce = np.where(self.assignments['assigned'] > 0)[0]
        force = all_rsids[iforce]

        # This step reassigns everything
        robotIDs = self._assign_cp_model(force=force, deny=deny)

        # So we have to unassign everything
        self.unassign(rsids=self.targets['rsid'])
        
        # Then reassign
        for rsid, robotID in zip(all_rsids, robotIDs):
            if(robotID >= 0):
                for epoch in range(self.field_cadence.nepochs):
                    nexp = self.field_cadence.nexp[epoch]
                    self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch, nexp=nexp)

        success = (robotIDs >= 0)
        return(success)

    def decollide_unassigned(self):
        """Decollide all unassigned robots"""
        if(not self.allgrids):
            return

        for iexp, rg in enumerate(self.robotgrids):
            for robotID in rg.robotDict:
                if(rg.robotDict[robotID].isAssigned() == False):
                    rg.decollideRobot(robotID)
        return

    def _select_calibs(self, mask):
        return(self.irancalib[np.where((mask[self.irancalib] == True) &
                                       (self._is_good_calibration[self.irancalib] == True))[0]])

    def assign_calibrations(self, stage='srd'):
        """Assign all calibration targets

        Parameters
        ----------

        stage : str
            stage of targets to use (default 'srd')

        Notes
        -----

        This assigns all targets with 'category' set to one of 
        the required calibrations for this Field and with stage
        as specified. 

        It calls assign_cadences(), which will assign the targets
        in order of their priority value. The order of assignment is
        randomized within each priority value. The random seed is 
        set according to the fieldid.

        This is usually not used, since calibrations are assigned
        in the assign_science_and_calibrations() method.
"""
        if(self.nocalib):
            return

        self.set_stage(stage=stage)

        if(self.verbose):
            print("fieldid {fid}: Assigning calibrations".format(fid=self.fieldid), flush=True)
        
        icalib = self._select_calibs(self._is_calibration)
        self.assign_cadences(rsids=self.targets['rsid'][icalib])

        self._set_satisfied(rsids=self.targets['rsid'][icalib])
        self._set_count(reset_equiv=False)
        self._set_has_spare_calib()

        if(self.verbose):
            print("fieldid {fid}:   (done assigning calibrations)".format(fid=self.fieldid), flush=True)

        self.set_stage(stage=None)
        return

    def set_standard_apogee_goodness(self, nperzone_min=3):
        """Set standard_apogee goodness to ensure reasonable flexibility

        Parameters
        ----------

        nperzone : int
            minimum achievable number of standard apogee targets per zone (default 3)

        Return:
        ------

        badzones : ndarray of bool
            for each zones, set True if ok, set False if not enough targets to guarantee nperzone_min

        Notes
        -----

        There are 8 zones, and it had better be that the number of zones times the
        minimum number of per zone is greater than the required number. 

        It uses the first exposure, unless it is marked as locked, in which case it 
        uses the earliest not locked.
"""
        zones = np.arange(robostrategy.standards.nzone, dtype=int)

        if(self.nocalib):
            return

        if(self.verbose):
            print("fieldid {fid}: Determining goodness thresholds for standard_apogee".format(fid=self.fieldid), flush=True)

        # Find the APOGEE standards, set all to good
        icalib = np.where(self.targets['category'] == 'standard_apogee')[0]

        if(self.verbose):
            print("fieldid {fid}: {a} standard_apogee targets total".format(fid=self.fieldid, a=len(icalib)))

        # First just check if we shouldn't have any threholds, because
        # all zones are sparse. If so, give up and return
        self._is_good_calibration[icalib] = True
        iexpunlocked = np.where(self.exposure_locked == False)[0][0]
        achievable, achievable_zones = self.determine_achievable(iexp=iexpunlocked, limit=False)
        badzones = np.zeros(len(zones), dtype=bool)
        for zone in zones:
            if(achievable_zones['standard_apogee'][zone] < nperzone_min):
                if(self.verbose):
                    print("fieldid {fid}: zone {z} has only {a} standard_apogee achievable".format(fid=self.fieldid, z=zone, a=achievable_zones['standard_apogee'][zone]))
                badzones[zone] = True
        if(badzones.min() == True):
            return(badzones)

        # Now set default goodness threshold per zone; we will lower thresholds 
        # in each zone until we can reach target. At the end of this process,
        # _is_good_calibration will be set according to a zone-dependent criterion.
        goodness_threshold = np.zeros(robostrategy.standards.nzone, dtype=np.float32)
        goodness = robostrategy.standards.apogee_standard_goodness(self.targets['magnitude'][icalib, :])
        finished = False
        while(finished is False):

            self._is_good_calibration[icalib] = goodness > goodness_threshold[self.targets['zone'][icalib]]
            achievable, achievable_zones = self.determine_achievable(iexp=iexpunlocked, limit=False)

            # Did this work? If so we are done
            finished = True
            for zone in zones:
                if(achievable_zones['standard_apogee'][zone] < nperzone_min):
                    finished = False
            if(finished):
                break

            # Or else let us lower thresholds of goodness in under performing zones
            # If we have run out of targets in all zones ... we are finished.
            finished = True
            print("fieldid {fid}: Lowering goodness threshold!".format(fid=self.fieldid), flush=True)
            for zone in zones:
                izone = np.where(self.targets['zone'][icalib] == zone)[0]
                if(len(izone) == 0):
                    continue
                izone = izone[np.flip(np.argsort(goodness[izone]))]
                cgoodness = goodness[izone]
                ilow = np.where(cgoodness > goodness_threshold[zone])[0]
                if(achievable_zones['standard_apogee'][zone] < nperzone_min):
                    dlow = nperzone_min - achievable_zones['standard_apogee'][zone]
                    ilow = np.where(cgoodness > goodness_threshold[zone])[0]
                    if(len(ilow) == 0):
                        ilow = 0
                    else:
                        ilow = ilow.max()
                    iupdate = ilow + np.int32(np.floor(dlow * 1.5))
                    if(iupdate > len(izone) - 1):
                        iupdate = len(izone) - 1
                    else:
                        finished = False
                    goodness_threshold[zone] = cgoodness[iupdate] - 1e-6

        return(badzones)

    def force_standard_apogee(self, badzones=None):
        """Force standard apogee targets in bad zones

        Parameters
        ----------

        badzones : ndarray of bool
            zones to force

        stage : str
            assignment stage (default 'srd')
"""

        if(self.nocalib):
            return

        ibadzones = np.where(badzones)[0]

        if(len(ibadzones) == 0):
            if(self.verbose):
                print("fieldid {fid}: No bad zones to force standard_apogee in".format(fid=self.fieldid), flush=True)
            return
            
        if(self.verbose):
            print("fieldid {fid}: Forcing standard_apogee in {n} bad zones".format(fid=self.fieldid, n=len(ibadzones)), flush=True)

        ic = list(self.calibration_order).index('standard_apogee')

        for zone in ibadzones:

            # Find the APOGEE standards, sort by decreasing goodness, and 
            # assign as FIXED until there are enough
            icalib = np.where((self.targets['category'] == 'standard_apogee') &
                              (self.targets['zone'] == zone))[0]
            self._is_good_calibration[icalib] = True
            goodness = robostrategy.standards.apogee_standard_goodness(self.targets['magnitude'][icalib, :])
            isort = icalib[np.flip(np.argsort(goodness))]
            rsids = self.targets['rsid'][isort]
            for rsid in rsids:
                iexps = np.where(self.calibrations_per_zone[ic + 1, :, zone] <
                                 self.required_calibrations_per_zone[ic + 1])[0]
                if(len(iexps) > 0):
                    self.assign_exposures(rsid=rsid, iexps=iexps, check_spare=False,
                                          reset_satisfied=False, reset_has_spare=False,
                                          set_fixed=True)

            self._set_satisfied(rsids=rsids)
            self._set_count(reset_equiv=False)
            self._set_has_spare_calib()

        if(self.verbose):
            print("fieldid {fid}:   (done forcing standard_apogee)".format(fid=self.fieldid), flush=True)

        return

    def assign_science(self, stage='srd'):
        """Assign all science targets
        
        Parameters
        ----------

        stage : str
            stage of assignment to use

        Notes
        -----

        This assigns all targets with 'category' set to 'science'
        and with 'stage' set to selected value

        It calls assign_cadences(), which will assign the targets
        in order of their priority value. The order of assignment is
        randomized within each priority value. The random seed is 
        set according to the fieldid.

        This is usually used for stages after SRD, since it does not
        assign calibrations.
"""
        self.set_stage(stage=stage)

        stage_select = stage
        if(stage == 'reassign'):
            stage_select = 'srd'

        if(self.verbose):
            print("fieldid {fid}: Assigning science".format(fid=self.fieldid), flush=True)

        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['within']) &
                            (self.assignments['incadence']) &
                            (self.target_duplicated == 0) &
                            (self.targets['stage'] == stage_select))[0]
        np.random.seed(self.fieldid)
        random.seed(self.fieldid)
        np.random.shuffle(iscience)
        self.assign_cadences(rsids=self.targets['rsid'][iscience])

        self.decollide_unassigned()
        nproblems = self.validate()
        if(nproblems == 0):
            print("fieldid {f}: No problems".format(f=self.fieldid), flush=True)
        else:
            print("fieldid {f}: {n} problems!!!".format(f=self.fieldid,
                                                        n=nproblems), flush=True)

        self._set_satisfied(rsids=self.targets['rsid'][iscience])
        self._set_count(reset_equiv=False)

        if(self.verbose):
            print("fieldid {fid}:   (done assigning science)".format(fid=self.fieldid), flush=True)

        self.set_stage(stage=None)

        self._set_satisfied()
        self._set_satisfied(science=True)
        self._set_count(reset_equiv=False)
        return

    def assign_science_cp(self, stage='srd'):
        """Assign all science targets with CP
        
        Parameters
        ----------

        stage : str
            stage of assignment to use

        Notes
        -----

        This assigns all targets with 'category' set to 'science'
        and with 'stage' set to selected value

        It assumes that there is just one cadence.
"""
        self.set_stage(stage=stage)

        stage_select = stage
        if(stage == 'reassign'):
            stage_select = 'srd'

        if(self.verbose):
            print("fieldid {fid}: Assigning science with CP".format(fid=self.fieldid), flush=True)

        isscience = ((self.targets['category'] == 'science') &
                     (self.targets['within']) &
                     (self.assignments['incadence']) &
                     (self.target_duplicated == 0) &
                     (self.targets['stage'] == stage_select))
        iscience = np.where(isscience)[0]

        priorities = np.unique(self.targets['priority'][iscience])
        
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)
            ipriority = np.where(isscience & (self.assignments['satisfied'] == 0) & (self.targets['priority'] == priority))[0]
            if(len(ipriority) == 0):
                print("fieldid {fid}:  - apparently all science targets at this priority level already satisfied".format(fid=self.fieldid), flush=True)
                continue
            if(self.verbose):
                print("fieldid {fid}:  - {n} assigning in CP".format(n=len(ipriority), fid=self.fieldid), flush=True)
            self.assign_full_cp_model(rsids=self.targets['rsid'][ipriority])

            if(self.verbose):
                print(self.assess())

            if(priority != priorities[-1]):
                icalib = np.where(self.targets['category'] != 'science')[0]
                self.unassign(rsids=self.targets['rsid'][icalib])

            self.decollide_unassigned()
            self._set_satisfied(rsids=self.targets['rsid'][ipriority])
            self._set_count(reset_equiv=False)

            igot = np.where(self.assignments['satisfied'][ipriority] != 0)[0]
            if(self.verbose):
                print("fieldid {fid}:    (assigned {n})".format(n=len(igot), fid=self.fieldid), flush=True)

        nproblems = self.validate()
        if(nproblems == 0):
            print("fieldid {f}: No problems".format(f=self.fieldid), flush=True)
        else:
            print("fieldid {f}: {n} problems!!!".format(f=self.fieldid,
                                                        n=nproblems), flush=True)
        if(self.verbose):
            print("fieldid {fid}:   (done assigning science with CP)".format(fid=self.fieldid), flush=True)

        self.set_stage(stage=None)
        return

    def determine_achievable(self, iexp=None, ok=None, limit=True):
        """Determine achievable number of calibrations for an exposure

        Parameters
        ----------

        iexp : int or np.int32
            exposure number

        ok : ndarray of bool
            for each target, is it ok to include?

        limit : bool
            if set, limit to required number (default True)

        Returns
        -------

        achievable : dict with ints
            for each calibration category, number achievable

        achievable_zones : dict with ndarrys of ints
            for each calibration category, number achievable in each zone
        
        Notes
        -----
        
        Assumes nothing else is assigned
"""
        if(ok is None):
            ok = True

        self._assign_temporary_calibs(ok=ok, iexps=np.array([iexp], dtype=int))

        achievable = dict()
        achievable_zones = dict()
        for ic, c in enumerate(self.calibration_order):
            if((self.calibrations[c][iexp] <
                self.required_calibrations[c][iexp]) |
               (limit == False)):
                achievable[c] = self.calibrations[c][iexp]
            else:
                achievable[c] = self.required_calibrations[c][iexp]
            achievable_zones[c] = np.zeros(robostrategy.standards.nzone, dtype=int)
            for zone in np.arange(robostrategy.standards.nzone, dtype=int):
                if((self.calibrations_per_zone[ic + 1, iexp, zone] < 
                    self.required_calibrations_per_zone[ic + 1]) |
                   (limit == False)):
                    achievable_zones[c][zone] = self.calibrations_per_zone[ic + 1, iexp, zone]
                else:
                    achievable_zones[c][zone] = self.required_calibrations_per_zone[ic + 1]

        self._unassign_temporary_calibs()

        return(achievable, achievable_zones)

    def assign_science_and_calibs(self, stage='srd',
                                  coordinated_targets=None):
        """Assign all science targets and calibrations

        Parameters
        ----------

        stage : str
            stage of assignment (default 'srd')

        coordinated_targets : dict
            dictionary of coordinated targets (keys are rsids, values are bool)
            [ DEPRECATED ]

        Notes
        -----

        Does not try to assign any targets for which
        coordinated_targets[rsid] is True.
"""
        self.set_stage(stage=stage)

        stage_select = stage
        if(stage == 'reassign'):
            stage_select = 'srd'

        if(self.verbose):
            print("fieldid {fid}: Assigning science".format(fid=self.fieldid), flush=True)
        np.random.seed(self.fieldid)
        random.seed(self.fieldid)

        # Deal with any targets duplicated
        self.target_duplicated[:] = 0
        if(coordinated_targets is not None):
            for id_idx, rsid in enumerate(self.targets['rsid']):
                if rsid in coordinated_targets.keys():
                    if coordinated_targets[rsid]:
                        self.target_duplicated[id_idx] = 1
                        self.set_flag(rsid=rsid,
                                      flagname='ALREADY_ASSIGNED')

        # Set goodness threhold of standard_apogee targets
        ic_standard_apogee = list(self.calibration_order).index('standard_apogee')
        if(self.required_calibrations_per_zone[ic_standard_apogee + 1] > 0):
            nper = self.required_calibrations_per_zone[ic_standard_apogee + 1]
            badzones = self.set_standard_apogee_goodness(nperzone_min=nper * 2)

        # Assign calibration to one exposure to determine achievable
        # requirements and then unassign
        if(self.verbose):
            print("fieldid {fieldid}: Assigning calibrations to determine achievable".format(fieldid=self.fieldid), flush=True)

        # Uniquify design modes here; only need to check one exposure
        # per design mode
        udesign_mode = np.unique(self.design_mode)
        epochs = self.field_cadence.epochs
        self.achievable_calibrations_per_zone = self.calibrations_per_zone * 0
        for design_mode in udesign_mode:
            iexpall = np.where(self.design_mode[epochs] == design_mode)[0]
            iunlocked = np.where(self.exposure_locked[iexpall] == False)[0]
            if(len(iunlocked) == 0):
                iexp = iexpall[0]  # doesn't matter if they are locked, just pick first
            else:
                iexp = iexpall[iunlocked[0]]  # use an unlocked exposure
            achievable, achievable_zones = self.determine_achievable(iexp=iexp)
            for c in self.calibration_order:
                self.achievable_calibrations[c][iexpall] = achievable[c]
            for ic, c in enumerate(self.calibration_order):
                for iexp in iexpall:
                    self.achievable_calibrations_per_zone[ic + 1, iexp, :] = achievable_zones[c]

        # Fix standard_apogee targets where necessary
        if(self.required_calibrations_per_zone[ic_standard_apogee + 1] > 0):
            self.force_standard_apogee(badzones)

        if(self.verbose):
            print("fieldid {fid}: Unassigning calibrations (except fixed ones)".format(fid=self.fieldid),
                  flush=True)
        iassigned = np.where(self.assignments['assigned'])[0]
        self.unassign(rsids=self.targets['rsid'][iassigned],
                      respect_fixed=True)

        inotscience = np.where(self.targets['category'] != 'science')[0]
        self.set_flag(rsid=self.targets['rsid'][inotscience],
                      flagname='NOT_SCIENCE')

        inotincadence = np.where(self.assignments['incadence'] == 0)[0]
        self.set_flag(rsid=self.targets['rsid'][inotincadence],
                      flagname='NOT_INCADENCE')

        inotstage = np.where(self.targets['stage'] == 'none')[0]
        self.set_flag(rsid=self.targets['rsid'][inotstage],
                      flagname='STAGE_IS_NONE')

        inotcovered = np.where(self.targets['within'] == 0)[0]
        self.set_flag(rsid=self.targets['rsid'][inotcovered],
                      flagname='NOT_COVERED')
        
        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['within']) &
                            (self.assignments['incadence']) &
                            (self.target_duplicated == 0) &
                            (self.targets['stage'] == stage_select))[0]
        np.random.shuffle(iscience)

        permanent_exposure_calib = collections.OrderedDict()
        for c in self.calibration_order:
            permanent_exposure_calib[c] = np.zeros(self.field_cadence.nexp_total,
                                                   dtype=bool)

        priorities = np.unique(self.targets['priority'][iscience])
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority level {p}".format(p=priority, fid=self.fieldid), flush=True)

            # Assign science
            ipriority = np.where(self.targets['priority'][iscience] == priority)[0]
            ipriority = iscience[ipriority]

            self.assign_cadences(rsids=self.targets['rsid'][ipriority])

            # See how the calibrations are doing
            self._assign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

            # If there is a calibration shortfall, try to fix it
            anyshortfall, shortfalls = self._find_shortfall()
            cycle = 0
            ncycle = 4
            while((anyshortfall > 0) & (cycle < ncycle)):
                
                if(self.verbose):
                    print("fieldid {fid}: Found a shortfall (cycle {c})".format(fid=self.fieldid, c=cycle), flush=True)
                    print("fieldid {fid}: Unassigning science".format(fid=self.fieldid), flush=True)

                # Back out the last round of science targets
                self.unassign(rsids=self.targets['rsid'][ipriority], respect_fixed=True)

                # Make sure any temporary calibrations are unassigned
                self._unassign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

                # Then assign the shortfall cases permanently (this also assigns
                # temporary calibs along the way to make sure everything stays as
                # before).
                self._assign_permanent_calibs(shortfalls=shortfalls,
                                              permanent_exposure_calib=permanent_exposure_calib) 

                # Make sure any temporary calibrations are unassigned
                self._unassign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

                # Reassign the science
                self.assign_cadences(rsids=self.targets['rsid'][ipriority])

                # Check if this has solved the problem
                self._assign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)
                anyshortfall, shortfalls = self._find_shortfall()

                cycle = cycle + 1

            if(anyshortfall):
                print("fieldid {fid}: Still have shortfall after {nc} cycles.".format(fid=self.fieldid, nc=ncycle), flush=True)
                
                # Back out the last round of science targets
                self.unassign(rsids=self.targets['rsid'][ipriority], respect_fixed=True)

                # Make sure any temporary calibrations are unassigned
                self._unassign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

                # Then try again with all calibrations made permanent
                for c in self.calibration_order:
                    if(len(shortfalls[c]) > 0):
                        shortfalls[c] = set(list(np.where(permanent_exposure_calib[c] == False)[0]))
                self._assign_permanent_calibs(shortfalls=shortfalls,
                                              permanent_exposure_calib=permanent_exposure_calib)
                        
                # Reassign the science
                self.assign_cadences(rsids=self.targets['rsid'][ipriority])

                # Another go at calibs
                self._assign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

                anyshortfall, shortfalls = self._find_shortfall()
                if(anyshortfall):
                    print("fieldid {fid}: Still have shortfall. Did our best.".format(fid=self.fieldid))

            # For all exposures that did not get assigned
            # calibrations, remove the calibration targets.
            # UNLESS this is the last priority.
            if(priority != priorities[-1]):
                self._unassign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib)

        if(self.verbose):
            print("fieldid {fid}: Decolliding unassigned".format(fid=self.fieldid), flush=True)
        self.decollide_unassigned()

        self._set_satisfied()
        self._set_satisfied(science=True)
        self._set_count(reset_equiv=False)
        if(self.nocalib is False):
            self._set_has_spare_calib()

        self.set_stage(stage=None)

        if(self.verbose):
            print("fieldid {fid}:   (done assigning science and calib)".format(fid=self.fieldid), flush=True)
        return

    def _find_shortfall(self):
        """Search for shortfalls in calibration"""
        if(self.verbose):
            print("fieldid {fid}: Checking for shortfalls".format(fid=self.fieldid), flush=True)
        shortfalls = collections.OrderedDict()
        anyshortfall = False
        for ic, c in enumerate(self.calibration_order):
            shortfalls[c] = set()
            for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
                if(self.exposure_locked[iexp]):  # skip if this exposure is locked anyway
                    continue
                if(self.calibrations[c][iexp] < self.achievable_calibrations[c][iexp]):
                    shortfalls[c].add(iexp)
                    if(self.verbose):
                        print("fieldid {fid}:  - shortfall in {c}, exposure {iexp}, {gc}/{ac}".format(fid=self.fieldid, c=c, iexp=iexp, gc=self.calibrations[c][iexp], ac=self.achievable_calibrations[c][iexp]), flush=True)
                    anyshortfall = True
                if(self.required_calibrations_per_zone[ic + 1] > 0):
                    ninzone = self.calibrations_per_zone[ic + 1, iexp, :]
                    achinzone = self.achievable_calibrations_per_zone[ic + 1, iexp, :]
                    bad = np.any(ninzone < achinzone)
                    if(bad):
                        if(self.verbose):
                            print("fieldid {fid}:  - shortfall in {c}, exposure {iexp}, zones {z}".format(fid=self.fieldid, c=c, iexp=iexp, z=np.where(ninzone < achinzone)[0]), flush=True)
                        shortfalls[c].add(iexp)
                        anyshortfall = True
        return(anyshortfall, shortfalls)

    def _assign_permanent_calibs(self, shortfalls=None,
                                 permanent_exposure_calib=None,
                                 report=True):
        """Assign permanent calibrations for category and exposure, converting science to calib"""
        if(self.verbose):
            print("fieldid {fid}: Assigning calibs permanently in shortfall exposures, converting science targets".format(fid=self.fieldid), flush=True)

        # Go through assignment again; notice that we have 
        # left the original assignments in place.
        for c in self.calibration_order:
            print("fieldid {fid}:  ... {c}".format(fid=self.fieldid, c=c), flush=True)

            # If it is a shortfall skip the temporary assignment
            # (note that this DOES do the temporary assignment
            # for permanent cases)
            iexps = np.array(list(shortfalls[c]), dtype=int)
            exps_temp = np.ones(self.field_cadence.nexp_total, dtype=bool)
            exps_temp[iexps] = False
            iexps_temp = np.where(exps_temp)[0]

            if(len(iexps_temp) > 0):
                # If this is not a category with a short fall, assign temporary
                # to stake claims; otherwise move on to permanent assignment in
                # affected exposures
                self._assign_temporary_calibs(permanent_exposure_calib=permanent_exposure_calib,
                                              category=c, iexps=iexps_temp)

            if(len(iexps) == 0):
                continue

            if(self.verbose):
                print("fieldid {fid}:  ... assigning {c} permanently in exposures {iexps}".format(c=c, iexps=iexps, fid=self.fieldid))
            icalib = self._select_calibs(self.targets['category'] == c)
            isort = np.argsort(self.targets['priority'][icalib],
                               kind='stable')
            icalib = icalib[isort]

            # First check if we should convert any science targets to
            # the equivalent calib; but we have to make sure they are 
            # 'fixed' so that they will never be removed. Make sure
            # to only do this for cases that are "allowed", so check
            # AssignmentStatus
            self._set_equiv(rsids=self.targets['rsid'][icalib], iexps=iexps,
                            science=True)
            for iexp in iexps:
                isciencegot = np.where((self.assignments['scienceRobotID'][icalib, iexp] >= 0) &
                                       (self.assignments['robotID'][icalib, iexp] < 0))[0]
                scienceRobotIDs = self.assignments['scienceRobotID'][icalib[isciencegot], iexp]
                for i, scienceRobotID in zip(icalib[isciencegot], scienceRobotIDs):
                    status = AssignmentStatus(rsid=self.targets['rsid'][i],
                                              robotID=scienceRobotID,
                                              iexps=np.int32([iexp]))
                    self.set_assignment_status(status=status)
                    if(status.assignable[0]):
                        self.assign_robot_exposure(rsid=self.targets['rsid'][i],
                                                   robotID=scienceRobotID,
                                                   iexp=iexp, reset_satisfied=False,
                                                   reset_has_spare=False,
                                                   reset_count=False,
                                                   set_fixed=True)

            self._set_satisfied(rsids=self.targets['rsid'][icalib])
            self._set_has_spare_calib()

            # Then set any others
            for i in icalib:
                notgot = self.assignments['robotID'][i, iexps] < 0
                self.assign_exposures(rsid=self.targets['rsid'][i],
                                      iexps=iexps[notgot],
                                      reset_satisfied=False)
            permanent_exposure_calib[c][iexps] = True
            self._set_satisfied(rsids=self.targets['rsid'][icalib])
            self._set_has_spare_calib()

            if(report):
                for iexp in iexps:
                    if(self.calibrations[c][iexp] < self.achievable_calibrations[c][iexp]):
                        print("fieldid {fid}:   still short in {c}, exposure {iexp}; {nc}/{nac}".format(fid=self.fieldid, c=c, iexp=iexp, nc=self.calibrations[c][iexp], nac=self.achievable_calibrations[c][iexp]), flush=True)

        return
                       
    def _unassign_temporary_calibs(self, permanent_exposure_calib=None):
        """Unassign temporary calibrations unless their assignment is fixed for a category and exposure"""
        if(self.verbose):
            print("fieldid {fid}: Unassigning temporary calibs".format(fid=self.fieldid), flush=True)

        if(permanent_exposure_calib == None):
            permanent_exposure_calib = collections.OrderedDict()
            for c in self.calibration_order:
                permanent_exposure_calib[c] = np.zeros(self.field_cadence.nexp_total,
                                                       dtype=bool)

        for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
            for c in self.calibration_order:
                if(permanent_exposure_calib[c][iexp] == False):
                    icalib = self._select_calibs(self.targets['category'] == c)
                    for i in icalib:
                        if(self.check_expflag(rsid=self.targets['rsid'][i],
                                              iexp=iexp, flagname='FIXED') == False):
                            self.unassign_exposure(rsid=self.targets['rsid'][i],
                                                   iexp=iexp,
                                                   reset_satisfied=False,
                                                   reset_count=False,
                                                   reset_has_spare=False)

        self._set_has_spare_calib()
        icalib = self._select_calibs(self.targets['category'] != 'science')
        self._set_satisfied(rsids=self.targets['rsid'][icalib])
        return

    def _assign_temporary_calibs(self, ok=None, iexps=None, permanent_exposure_calib=None,
                                 category=None):
        """Assign temporary calibrations unless their assignment is fixed for a category and exposure"""
        if(self.verbose):
            print("fieldid {fid}: Assigning calibrations for each exposure".format(fid=self.fieldid), flush=True)

        if(ok is None):
            ok = True

        if(permanent_exposure_calib == None):
            permanent_exposure_calib = collections.OrderedDict()
            for c in self.calibration_order:
                permanent_exposure_calib[c] = np.zeros(self.field_cadence.nexp_total,
                                                       dtype=bool)

        assigned_rsids = []
        for ic, c in enumerate(self.calibration_order):
            # If we have specified a category, assign only that one
            if((category is not None) & (c != category)):
                continue
            if(self.verbose):
                print("fieldid {fid}:  ... assigning temporarily {c}".format(fid=self.fieldid, c=c), flush=True)

            if(iexps is None):
                iexps_assign = np.where(permanent_exposure_calib[c] == False)[0]
                if(len(iexps_assign) == 0):
                    continue
            else:
                iexps_assign = iexps
            icalib = self._select_calibs(self.targets['category'] == c)

            isort = np.argsort(self.targets['priority'][icalib],
                               kind='stable')
            icalib = icalib[isort]
            for i in icalib:
                inotdone = np.where(self.assignments['robotID'][i, iexps_assign] == -1)[0]
                if(len(inotdone) > 0):
                    self.assign_exposures(rsid=self.targets['rsid'][i], iexps=iexps_assign[inotdone],
                                          reset_satisfied=False, set_expflag=True)

        icalib = self._select_calibs(self.targets['category'] != 'science')
        self._set_satisfied(rsids=self.targets['rsid'][icalib])
        return(assigned_rsids)

    def assign(self, coordinated_targets=None):
        """Assign all targets

        Parameters
        ----------

        coordinated_targets : dict
            dictionary of coordinated targets (keys are rsids, values are bool)


        Notes
        -----

        Does not try to assign any targets for which
        coordinated_targets[rsid] is True.

        Normally not called.
"""

        # Deal with any targets duplicated
        self.target_duplicated[:] = 0
        if(coordinated_targets is not None):
            for id_idx, rsid in enumerate(self.targets['rsid']):
                if rsid in coordinated_targets.keys():
                    if coordinated_targets[rsid]:
                        self.target_duplicated[id_idx] = 1

        self.assign_calibrations()
        self.assign_science()
        self.decollide_unassigned()
        for itarget in np.arange(len(self.assignments), dtype=np.int32):
            self._set_assigned(itarget=itarget)
        self._set_satisfied()
        self._set_satisfied(science=True)
        self._set_count(reset_equiv=False)
        return

    def get_preferred_robotid(self, epoch=None, rsid=None):
        """Get the preferred robotID for an epoch for a target

        Parameters 
        ----------

        epoch : int or np.int32
            epoch to get preferred robotID for
        
        rsid : np.int64
            identifier for targets to check

        Returns
        -------

        preferredRobotID : np.int32 or None
            preferred robotID or None
"""
        if(self.preferred_robotids is None):
            return(None)
        iexps = (self.field_cadence.epoch_indx[epoch] +
                 np.arange(self.field_cadence.nexp[epoch], dtype=int))
        itarget = self.rsid2indx[rsid]
        iwasassigned = np.where(self.preferred_robotids[itarget, iexps] >= 0)[0]
        if(len(iwasassigned) > 0):
            preferredRobotID = self.preferred_robotids[itarget,
                                                       iexps[iwasassigned[0]]]
        else:
            preferredRobotID = None
        return(preferredRobotID)

    def are_preferred_robotids(self, robotIDs=None, rsid=None):
        """Return whether robotIDs are preferred for this rsid

        Parameters 
        ----------

        rsid : np.int64
            identifier for target to check

        robotIDs : ndarray of np.int32
            robotIDs to check

        Returns
        -------

        preferred : ndarray of bool
            whether each input robotID is a preferred one
"""
        if(self.preferred_robotids is None):
            return(np.zeros(len(robotIDs), dtype=bool))
        itarget = self.rsid2indx[rsid]
        preferred = np.array([rid in self.preferred_robotids[itarget, :]
                              for rid in robotIDs], dtype=bool)
        return(preferred)

    def set_preferred_robotids(self, iexp=None, rsids=None, holeIDs=None):
        """Set a set of preferred robotIDs for an exposure

        Parameters 
        ----------

        iexp : int or np.int32
            exposure to apply to
        
        rsids : ndarray of np.int64
            identifiers for targets to set for

        holeIDs : ndarray of str
            holes to set preferences for


        Notes
        -----

        Creates or updates attribute "preferred_robotids", which 
        is an [ntarget, nexp] ndarray of np.int32 with the robotID
        to be preferred for that exposure of that target.

        The use case for this is within apply_observed_status().
        For exposures which have been planned before, but haven't
        been observed yet, we set the original robotID as the preferred
        one, which will lead to a more consistent set of assignments
        with the original, and thus more complete.

        Although the preferred robotID is set as a function of 
        exposure, it gets used as a function of epoch.
"""
        if(len(rsids) == 0):
            return

        if(self.preferred_robotids is None):
            self.preferred_robotids = np.zeros((len(self.targets),
                                                self.field_cadence.nexp_total),
                                               dtype=np.int32) - 1

        holeID2robotID = dict()
        for robotID in self.mastergrid.robotDict:
            holeID = self.mastergrid.robotDict[robotID].holeID 
            holeID2robotID[holeID] = robotID
        robotIDs = np.array([holeID2robotID[hid] for hid in holeIDs],
                            dtype=int)
        
        iok = np.where(rsids >= 0)[0]
        itargets = np.array([self.rsid2indx[r] for r in rsids[iok]], dtype=int)
        self.preferred_robotids[itargets, iexp] = robotIDs[iok]
        return

    def apply_observed_status(self, observed_status=None):
        """Apply the information from the observational status

        Parameters
        ----------

        status : ndarray
            array of status information, with 'carton_to_target_pk', 'catalogid'
              'fiberType', 'lambda_eff', 'delta_ra', 'delta_dec',
              'field_exposure', 'status', 'mjd', and 'holeid'

        Notes
        -----

        Uses assign_done_exposure() method to set each completed exposure.
        Only assigns on field exposures which have at least one done target 
        (and this will mean nothing is changed for those exposures at all).

        For other exposures, it will set a preferred robot for each 
        previously observed target.
"""
        anydone = (observed_status['status'] > 0).any()
        if(anydone is False):
            print("fieldid {fid}: Nothing marked as done in field".format(fid=self.fieldid),
                  flush=True)
            return

        c2t_to_rsid = dict()
        for t in self.targets:
            c2t_to_rsid[t['carton_to_target_pk']] = t['rsid']

        rsid_obs_to_istatus = dict()

        # Find observed cases in each exposure. Track down rsid for each observation.
        # If there is an rsid appropriate, and the observed status is done, then
        # assign it as done. We gather the information for all observations here because
        # in the next section of code we may COUNT those as done if it is the only
        # option we have to "complete" the cadence.
        iexps = np.unique(observed_status['field_exposure'])
        for iexp in iexps:
            iobs = np.where((observed_status['field_exposure'] == iexp) &
                            (observed_status['mjd'] != 0))[0]
            if(len(iobs) > 0):
                if(self.verbose):
                    print("fieldid {fid}: Accounting for exposure={iexp} completion".format(fid=self.fieldid, iexp=iexp), flush=True)
                rsids = np.zeros(len(iobs), dtype=np.int64) - 1
                for indx, cstatus in enumerate(observed_status[iobs]):
                    tmp_rsids = self.equiv_target(cstatus)
                    if(cstatus['carton_to_target_pk'] in c2t_to_rsid):
                        # If this carton_to_target is explicitly in status table,
                        # assign that rsid
                        rsids[indx] = c2t_to_rsid[cstatus['carton_to_target_pk']]
                    else:
                        # Else take an equivalent based on catalogid/fiberType
                        if(len(tmp_rsids) > 0):
                            rsids[indx] = tmp_rsids[0]
                    # Associate the status with ANY equivalent target for step below
                    # "completing" cadences with bad observations; this relies on
                    # the observed_status only existing for one equivalent rsid
                    # per exposure, which really ought to be true
                    for tmp_rsid in tmp_rsids:
                        rsid_obs_to_istatus[iexp, tmp_rsid] = iobs[indx]

                infield_and_done = ((rsids >= 0) & (observed_status['status'][iobs] > 0))

                if(infield_and_done.max() > 0):
                    if(self.design_status['status'][iexp] != 'done'):
                        raise ValueError("""fieldid {fid}: Assignments marked done for exposure {e}, which is not done!
   {f}
""".format(fid=self.fieldid, f=self.design_status, e=iexp))

                self.assign_done_exposure(iexp=iexp, rsids=rsids[infield_and_done],
                                          holeIDs=observed_status['holeid'][iobs[infield_and_done]],
                                          force=True, lock=True)
            else:
                iassigned = np.where(observed_status['field_exposure'] == iexp)[0]
                rsids = np.zeros(len(iassigned), dtype=np.int64) - 1
                for indx, cstatus in enumerate(observed_status[iassigned]):
                    if(cstatus['carton_to_target_pk'] in c2t_to_rsid):
                        rsids[indx] = c2t_to_rsid[cstatus['carton_to_target_pk']]
                    else:
                        tmp_rsids = self.equiv_target(cstatus)
                        if(len(tmp_rsids) > 0):
                            rsids[indx] = tmp_rsids[0]
                self.set_preferred_robotids(iexp=iexp, rsids=rsids,
                                            holeIDs=observed_status['holeid'][iassigned])

        # For SRD targets that are NOT satisfied, that DID get one good observation,
        # check if they are satisfiable AT ALL any more; if not, assign the missing MJDs
        # to do the best we can; but only if they are in the cadence
        gotone = ((self.assignments['equivRobotID'] >= 0).sum(axis=1) > 0)
        isci = np.where((self.assignments['satisfied'] == 0) & (gotone) &
                        (self.targets['stage'] == 'srd') &
                        (self.assignments['incadence'] > 0))[0]
        for indx, target in enumerate(self.targets[isci]):
            success = self.assign_cadences(rsids=np.array([target['rsid']]),
                                           test_only=True)
            if(success == False) :
                for iexp in iexps:
                    if((iexp, target['rsid']) in rsid_obs_to_istatus):
                        # Bail if this assignment was equivalently done already
                        # earlier in this loop
                        if(self.assignments['equivRobotID'][isci[indx], iexp] >= 0):
                            continue
                        istatus = rsid_obs_to_istatus[iexp, target['rsid']]
                        if(observed_status['status'][istatus] == 0):
                            self.assign_done_exposure(iexp=iexp, rsids=np.array([target['rsid']]),
                                                      holeIDs=np.array([observed_status['holeid'][istatus]]),
                                                      force=True, lock=True, override_lock=True)

        return
    
    def assign_done_exposure(self, iexp=None, rsids=None, holeIDs=None, force=False,
                             lock=False, override_lock=False):
        """Record robot assignments for a design as done

        Parameters
        ----------

        iexp : int, np.int32
            design index to record as done
        
        rsids : ndarray of np.int64
            unique IDs of completed exposures

        holeIDs : ndarray of np.int32
            hole IDs of completed exposures

        force : bool
            if True, force cases to be assigned even if robot does not reach (default False) 

        lock : bool
            if True, lock down any unassigned robots in this exposure so they can't be assigned

        override_lock: bool
            if True, ignore any locks on holeIDs

        Notes
        -----

        rsids and holeIDs should be the same length.

        The caller has to find the right rsids depending on their input targets. 
        This method finds the right robotIDs given the holeIDs.

        If "lock" is set robots in the exposure that are not used are marked as unusable,
        and the exposure itself is marked as locked (uses _robot_locked and exposure_locked).

        If "force" is set, cases where the rsid cannot be reached by the robot in 
        the holeID are fudged to work using the "force" option in the 
        assign_robot_exposure method.

        If "override_lock" is set, then the user is expecting that some of these 
        robots might have been locked already. So we unlock them first. But
        ONLY if they are not actually already assigned to something.  Since
        this override is necessary to reinstate an exposure to a target which
        was observed but not declared done, it should never happen that the 
        robot is already assigned, It spews a warning if this state is reached.
"""
        holeID2robotID = dict()
        for robotID in self.mastergrid.robotDict:
            holeID = self.mastergrid.robotDict[robotID].holeID 
            holeID2robotID[holeID] = robotID
        robotIDs = np.array([holeID2robotID[hid] for hid in holeIDs], dtype=int)

        if(override_lock):
            for robotID in robotIDs:
                robotindx = self.robotID2indx[robotID]
                if(self._robot2indx[robotindx, iexp] == -1):
                    self._robot_locked[robotindx, iexp] = False
                else:
                    print("fieldid {fid}: WARNING, tried to unlock robot with assignment rsid={rsid} iexp={iexp} robotID={robotID}".format(rsid=self.targets['rsid'][self._robot2indx[robotindx, iexp]], iexp=iexp, robotID=robotID, fid=self.fieldid, flush=True))
        
        for robotID, rsid in zip(robotIDs, rsids):
            self.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp,
                                       set_fixed=True, force=force)

        if(lock):
            # Make sure we check against _robot2indx since if force is set we cannot
            # rely on RobotGrid; should not really matter though.
            self.exposure_locked[iexp] = True
            for robotID in self.mastergrid.robotDict:
                robotindx = self.robotID2indx[robotID]
                if(self._robot2indx[robotindx, iexp] == -1):
                    self._robot_locked[robotindx, iexp] = True

        return

    def complete_epochs_assigned(self):
        """Complete the epochs of any assigned science targets"""

        if(self.verbose):
            print("fieldid {fid}: Complete any assigned targets within their epochs".format(fid=self.fieldid), flush=True)

        # Go through each epoch. For all science targets
        # observed in that epoch, go through them in priority order
        # and see if you can assign them
        ntargets = 0
        ntargets_added = 0
        nexposures_added = 0
        for epoch in np.arange(self.field_cadence.nepochs, dtype=int):
            iexps_epoch = (self.field_cadence.epoch_indx[epoch] +
                           np.arange(self.field_cadence.nexp[epoch], dtype=int))
            observed_in_epoch = (self.assignments['equivRobotID'][:, iexps_epoch] >= 0).sum(axis=1) > 0
            iscience = np.where((self.targets['category'] == 'science') &
                                (observed_in_epoch))[0]
            if(len(iscience) > 0):
                isort = np.argsort(self.targets['priority'][iscience])
                iscience = iscience[isort]
                ntargets = ntargets + len(iscience)
                for itarget in iscience:
                    rsid = self.targets['rsid'][itarget]
                    not_observed = (self.assignments['equivRobotID'][itarget, iexps_epoch] < 0)
                    done = self.assign_exposures(rsid=rsid, iexps=iexps_epoch[not_observed])
                    ndone = done.sum()
                    if(ndone > 0):
                        ntargets_added = ntargets_added + 1
                        nexposures_added = nexposures_added + ndone

        if(self.verbose):
            print("fieldid {fid}:   {n} assigned targets checked".format(fid=self.fieldid, n=ntargets), flush=True)
            print("fieldid {fid}:   {n} targets added".format(fid=self.fieldid, n=ntargets_added), flush=True)
            print("fieldid {fid}:   {n} exposures added".format(fid=self.fieldid, n=nexposures_added), flush=True)

        self.decollide_unassigned()
        return

    def complete_assigned(self):
        """Complete any available exposures of any assigned science targets"""

        if(self.verbose):
            print("fieldid {fid}: Complete any assigned targets across all exposures".format(fid=self.fieldid), flush=True)

        # For all science targets observed in that epoch, go through 
        # them in priority order and see if you can assign them
        ntargets = 0
        ntargets_added = 0
        nexposures_added = 0
        iexps = np.arange(self.field_cadence.nexp_total, dtype=int)
        iscience = np.where((self.targets['category'] == 'science') &
                            ((self.assignments['equivRobotID'] >= 0).sum(axis=1) > 0))[0]
        skybrightness = self.field_cadence.skybrightness[self.field_cadence.epochs]
        if(len(iscience) > 0):
            isort = np.argsort(self.targets['priority'][iscience])
            iscience = iscience[isort]
            ntargets = ntargets + len(iscience)
            for itarget in iscience:
                rsid = self.targets['rsid'][itarget]
                minskybrightness = clist.cadences[self.targets['cadence'][itarget]].skybrightness.min()
                not_observed = (self.assignments['equivRobotID'][itarget, iexps] < 0)
                sky_ok = (skybrightness[iexps] <= minskybrightness)
                done = self.assign_exposures(rsid=rsid, iexps=iexps[not_observed & sky_ok])
                ndone = done.sum()
                if(ndone > 0):
                    ntargets_added = ntargets_added + 1
                    nexposures_added = nexposures_added + ndone

        if(self.verbose):
            print("fieldid {fid}:   {n} assigned targets checked".format(fid=self.fieldid, n=ntargets), flush=True)
            print("fieldid {fid}:   {n} targets added".format(fid=self.fieldid, n=ntargets_added), flush=True)
            print("fieldid {fid}:   {n} exposures added".format(fid=self.fieldid, n=nexposures_added), flush=True)
            
        self.decollide_unassigned()
        return

    def complete_unassigned(self):
        """Complete any available exposures of any unassigned science targets"""

        if(self.verbose):
            print("fieldid {fid}: Complete any unassigned targets".format(fid=self.fieldid), flush=True)

        # Try any unassigned targets, and assign them whatever can be 
        # assigned
        iscience = np.where((self.targets['category'] == 'science') &
                            ((self.assignments['equivRobotID'] >= 0).sum(axis=1) == 0))[0]
        print("fieldid {fid}:    {n} unassigned science targets to check".format(fid=self.fieldid, n=len(iscience)), flush=True)
        if(len(iscience) == 0):
            return

        iexps = np.arange(self.field_cadence.nexp_total, dtype=int)
        isort = np.argsort(self.targets['priority'][iscience])
        iscience = iscience[isort]
        ntargets_added = 0
        nexposures_added = 0
        skybrightness = self.field_cadence.skybrightness[self.field_cadence.epochs]
        for itarget in iscience:
            rsid = self.targets['rsid'][itarget]
            minskybrightness = clist.cadences[self.targets['cadence'][itarget]].skybrightness.min()
            sky_ok = (skybrightness[iexps] <= minskybrightness)
            done = self.assign_exposures(rsid=rsid, iexps=iexps[sky_ok])
            ndone = done.sum()
            if(ndone > 0):
                ntargets_added = ntargets_added + 1
                nexposures_added = nexposures_added + ndone

        if(self.verbose):
            print("fieldid {fid}:   {n} targets added".format(fid=self.fieldid, n=ntargets_added), flush=True)
            print("fieldid {fid}:   {n} exposures added".format(fid=self.fieldid, n=nexposures_added), flush=True)
        self.decollide_unassigned()
        return

    def complete_calibrations(self, category=''):
        """Add any available exposures for any calibration targets"""

        if(self.verbose):
            print("fieldid {fid}: Add any calibration targets".format(fid=self.fieldid), flush=True)

        # Assign any calibrations that can be additionally
        # assigned WITHOUT allowing spare calibrations to be
        # bumped.
        icalib = self._select_calibs(self.targets['category'] == category)
        if(self.verbose):
            print("fieldid {fid}:    {n} {cc} targets to check".format(fid=self.fieldid, n=len(icalib), cc=category), flush=True)
        if(len(icalib) == 0):
            return

        iexps = np.arange(self.field_cadence.nexp_total, dtype=int)
        isort = np.argsort(self.targets['priority'][icalib])
        icalib = icalib[isort]
        ntargets_added = 0
        nexposures_added = 0
        for itarget in icalib:
            rsid = self.targets['rsid'][itarget]
            not_observed = (self.assignments['equivRobotID'][itarget, iexps] < 0)
            done = self.assign_exposures(rsid=rsid, iexps=iexps[not_observed],
                                         check_spare=False)
            ndone = done.sum()
            if(ndone > 0):
                ntargets_added = ntargets_added + 1
                nexposures_added = nexposures_added + ndone

        if(self.verbose):
            print("fieldid {fid}:   {n} targets added".format(fid=self.fieldid, n=ntargets_added), flush=True)
            print("fieldid {fid}:   {n} exposures added".format(fid=self.fieldid, n=nexposures_added), flush=True)

        self.decollide_unassigned()
        return

    def complete(self):
        """Fill out any extra exposures possible

        Notes
        -----

        Runs::

           complete_epochs_assigned()
           complete_assigned()
           complete_unassigned()
           complete_calibrations()
           decollide_unassigned()

"""
        self.set_stage(stage="complete")
        self.complete_epochs_assigned()
        self.complete_assigned()
        self.complete_unassigned()
        for category in ['standard_apogee', 'standard_boss', 'sky_apogee', 'sky_boss']:
            self.complete_calibrations(category=category)
        self.decollide_unassigned()
        self.set_stage(stage=None)
        self._set_satisfied()
        self._set_satisfied(science=True)
        self._set_count(reset_equiv=False)
        return

    def assess_data(self):
        """Return dictionary with assessment of current results

        Returns
        -------

        results_data : dict
            dictionary of results
"""
        results = dict()
        results['field_cadence'] = self.field_cadence.name
        results['nepochs'] = self.field_cadence.nepochs
        results['nexp_total'] = self.field_cadence.nexp_total
        
        results['nocalib'] = self.nocalib
        if(self.nocalib is False):
            results['calibration_order'] = self.calibration_order
            results['calibrations'] = dict()
            results['required_calibrations'] = dict()
            results['achievable_calibrations'] = dict()
            # TODO REPORT N ZONES
            for c in self.calibration_order:
                results['calibrations'][c] = list(self.calibrations[c])
                results['required_calibrations'][c] = list(self.required_calibrations[c])
                results['achievable_calibrations'][c] = list(self.achievable_calibrations[c])

        iboss = np.where((self.targets['fiberType'] == 'BOSS') &
                         (self.assignments['assigned']) &
                         (self.targets['category'] == 'science'))[0]
        results['nboss_science'] = len(iboss)
        iapogee = np.where((self.targets['fiberType'] == 'APOGEE') &
                           (self.assignments['assigned']) &
                           (self.targets['category'] == 'science'))[0]
        results['napogee_science'] = len(iapogee)

        nperexposure = np.zeros(self.field_cadence.nexp_total, dtype=int)
        nhasapogee = np.zeros(self.field_cadence.nexp_total, dtype=int)
        nnoapogee = np.zeros(self.field_cadence.nexp_total, dtype=int)
        for iexp in range(self.field_cadence.nexp_total):
            iin = np.where((self.assignments['robotID'][:, iexp] >= 1) &
                           (self.targets['category'] == 'science'))[0]
            nperexposure[iexp] = len(iin)

            iapogee = np.where((self._robot2indx[:, iexp] >= 0) &
                               (self.robotHasApogee == True))[0]
            nhasapogee[iexp] = len(iapogee)

            iboss = np.where((self._robot2indx[:, iexp] >= 0) &
                             (self.robotHasApogee == False))[0]
            nnoapogee[iexp] = len(iboss)

        results['nperexposure_science'] = list(nperexposure)
        results['nperexposure_hasapogee'] = list(nhasapogee)
        results['nperexposure_noapogee'] = list(nnoapogee)

        nperepoch = np.zeros(self.field_cadence.nepochs, dtype=int)
        for epoch in range(self.field_cadence.nepochs):
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            iin = np.where(((self.assignments['robotID'][:, iexpst:iexpnd] >= 1).sum(axis=1) > 0) &
                           (self.targets['category'] == 'science'))[0]
            nperepoch[epoch] = len(iin)
        results['nperepoch_science'] = list(nperepoch)

        nboss_spare, napogee_spare, nboss_unused, napogee_unused = self.count_spares(return_unused=True)

        results['nboss_spare'] = list(nboss_spare)
        results['napogee_spare'] = list(napogee_spare)
        results['nboss_unused'] = list(nboss_unused)
        results['napogee_unused'] = list(napogee_unused)

        cartons = np.unique(self.targets['carton'])
        results['cartons'] = dict()
        for carton in cartons:
            isscience = (self.targets['category'] == 'science')
            incarton = (self.targets['carton'] == carton)
            within = (self.targets['within'] != 0)
            issatisfied = (self.assignments['satisfied'] > 0)
            icarton = np.where(incarton & isscience & within)[0]
            igot = np.where(incarton & issatisfied & isscience)[0]
            nexposures = (self.assignments['equivRobotID'][icarton, :] >= 0).sum()
            if(len(icarton) > 0):
                results['cartons'][carton] = dict()
                results['cartons'][carton]['nwithin'] = len(icarton)
                results['cartons'][carton]['nsatisfied'] = len(igot)
                results['cartons'][carton]['nexposures'] = nexposures

        return(results)

    def assess(self):
        """Assess the current results of assignment in field

        Returns
        -------

        results : str
            String describing results
"""
        tstr = """
Field cadence: {{field_cadence}}

{% if nocalib %} No calibrations included.
{% else %} Calibration targets:
{% for c in calibration_order %} {{c}}:{% for cn in calibrations[c] %} {{cn}}/{{required_calibrations[c][loop.index0]}}{% endfor %}
{% endfor %}{% endif %}
Science targets:
 BOSS targets assigned: {{nboss_science}}
 APOGEE targets assigned: {{napogee_science}}
 Per epoch:{% for n in nperepoch_science %} {{n}}{% endfor %}

Robots used per exposure:
 BOSS-only:{% for n in nperexposure_noapogee %} {{n}}{% endfor %}
 APOGEE-BOSS:{% for n in nperexposure_hasapogee %} {{n}}{% endfor %}

Spare fibers per exposure:
 BOSS:{% for n in nboss_spare %} {{n}}{% endfor %}
 APOGEE:{% for n in napogee_spare %} {{n}}{% endfor %}

Unassigned fibers per exposure:
 BOSS:{% for n in nboss_unused %} {{n}}{% endfor %}
 APOGEE:{% for n in napogee_unused %} {{n}}{% endfor %}

Carton completion:
{% for c in cartons %} {{c}}: {{cartons[c].nsatisfied}} / {{cartons[c].nwithin}} ({{cartons[c].nexposures}} exp)
{% endfor %}
"""
        assessment = self.assess_data()

        env = jinja2.Environment()
        template = env.from_string(tstr)
        out = template.render(assessment)

        return(out)

    def warnings(self):
        """Check for unusual circumstances

        Returns
        -------

        nwarnings : int
            number of unusual circumstances

        Notes
        -----

        Checks if different exposures in the same epoch use different 
        robotIDs for the same objects

        Checks if exposure is wrong skybrightness for target
"""
        nwarnings = 0

        for epoch in np.arange(self.field_cadence.nepochs, dtype=int):
            iexps = np.where(self.field_cadence.epochs == epoch)[0]
            gotinepoch = (((self.assignments['equivRobotID'][:, iexps] >= 0).sum(axis=1) > 0) &
                          (self.targets['category'] == 'science'))
            igotinepoch = np.where(gotinepoch)[0]
            for itarget in igotinepoch:
                igot = np.where(self.assignments['equivRobotID'][itarget, iexps] >= 0)[0]
                robotIDs = np.unique(self.assignments['equivRobotID'][itarget, iexps[igot]])
                if(len(robotIDs) > 1):
                    print("Multiple robotIDs in epoch {epoch} for rsid {rsid}".format(epoch=epoch, rsid=self.targets['rsid'][itarget]), flush=True)
                    nwarnings = nwarnings + 1


        target_skybrightness = np.array([clist.cadences[c].skybrightness[0]
                                         for c in self.targets['cadence']],
                                         dtype=np.float32)
        for iexp in np.arange(self.field_cadence.nexp_total, dtype=int):
            epoch = self.field_cadence.epochs[iexp]
            field_skybrightness = self.field_cadence.skybrightness[epoch]
            itargets = np.where(self.assignments['robotID'][:, iexp] >= 0)[0]
            ibad = np.where((target_skybrightness[itargets] < field_skybrightness) &
                            (self.targets['category'][itargets] == 'science'))[0]
            if(len(ibad) > 0):
                print("{n} targets in exposure {iexp} with skybrightness limits less than {fsb}".format(n=len(ibad), iexp=iexp, fsb=field_skybrightness))
                print(self.targets['rsid'][itargets[ibad]])
        return(nwarnings)

    def validate(self):
        """Validate a field solution

        Returns
        -------

        nproblems : int
            Number of problems discovered

        Notes
        -----

        Prints nature of problems identified to stdout

        Checks self-consistency between the robotGrid assignments and
        the assignments array in the object.

        Checks self-consistency between the calibrations dictionary
        and the number of actually assigned calibration targets.

        Checks that there are no collisions.
"""
        nproblems = 0

        if(not self.allgrids):
            print("allgrids is False, so no collisions are accounted for")
            nproblems = nproblems + 1

        if(self.nocalib is False):
            test_calibrations = dict()
            for c in self.calibration_order:
                test_calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                                dtype=np.int32)

            for target, assignment in zip(self.targets, self.assignments):
                if(target['category'] in self.required_calibrations):
                    for iexp, robotID in enumerate(assignment['robotID']):
                        if(robotID >= 1):
                            test_calibrations[target['category']][iexp] += 1

        for indx, target in enumerate(self.targets):
            assignment = self.assignments[indx]
            isassigned = assignment['robotID'].max() >= 0
            if((isassigned) != (assignment['assigned'])):
                print("rsid={rsid} : assigned misclassification (assigned is set to {assigned}, category is {cat})".format(rsid=target['rsid'], assigned=assignment['assigned'], cat=target['category']))
                nproblems += 1

        # Check that no locked robots are used
        for indx, target in enumerate(self.targets):
            assignment = self.assignments[indx]
            for iexp in np.arange(self.field_cadence.nexp_total, dtype=int):
                robotID = assignment['robotID'][iexp]
                if(robotID >= 0):
                    robotindx = self.robotID2indx[robotID]
                    if(self._robot_locked[robotindx, iexp]):
                        print("robotID={robotID} iexp={iexp}: locked robot used for assignment".format(robotID=robotID, iexp=iexp))
                        nproblems += 1

        # Check that the bright neighbors are respected
        if(self.bright_neighbors):
            dms = self.design_mode[self.field_cadence.epochs]
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=int):
                design_mode = dms[iexp]
                if(self.allgrids is False):
                    rg = self.mastergrid
                else:
                    rg = self.robotgrids[iexp]
                for irobot, indx in enumerate(self._robot2indx[:, iexp]):
                    if(indx >= 0):
                        rsid = self.targets['rsid'][indx]
                        robotID = self.robotIDs[irobot]
                        if(self.check_expflag(rsid=rsid, iexp=iexp, flagname='FORCED') == False):
                            allowed = self._bright_allowed_robot(rsid=rsid, robotID=robotID,
                                                                 design_mode=design_mode)
                        else:
                            allowed = True
                        if(allowed is False):
                            print("bright neighbor to rsid={rsid} on robotID={robotID} in iexp={iexp}".format(rsid=rsid, robotID=robotID, iexp=iexp))
                            nproblems = nproblems + 1

        # Check that the number of calibrators has been tracked right
        if(self.nocalib is False):
            for c in self.calibration_order:
                for iexp in range(self.field_cadence.nexp_total):
                    if(test_calibrations[c][iexp] != self.calibrations[c][iexp]):
                        print("number of {c} calibrators tracked incorrectly ({nc} found instead of {nct})".format(c=c, nc=test_calibrations[c][iexp], nct=self.calibrations[c][iexp]))

        # Check that assignments and _robot2indx agree with each other
        for itarget, assignment in enumerate(self.assignments):
            for iexp, robotID in enumerate(assignment['robotID']):
                if(robotID >= 1):
                    robotindx = self.robotID2indx[robotID]
                    if(itarget != self._robot2indx[robotindx, iexp]):
                        rsid = self.targets['rsid'][itarget]
                        print("assignments['robotID'] for rsid={rsid} and iexp={iexp} is robotID={robotID}, but _robot2indx[robotID, iexp] is {i}, meaning rsid={rsidtwo}".format(rsid=rsid, iexp=iexp, robotID=robotID, i=self._robot2indx[robotindx, iexp], rsidtwo=self.targets['rsid'][self._robot2indx[robotindx, iexp]]))
                        nproblems = nproblems + 1

        # Check that _robot2indx and _robotnexp agree with each other
        for robotID in self.mastergrid.robotDict:
            robotindx = self.robotID2indx[robotID]
            nn = self.field_cadence.nexp.copy()
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=np.int32):
                if(self._robot2indx[robotindx, iexp] >= 0):
                    epoch = self.field_cadence.epochs[iexp]
                    nn[epoch] = nn[epoch] - 1
            for epoch in np.arange(self.field_cadence.nepochs, dtype=np.int32):
                if(nn[epoch] != self._robotnexp[robotindx, epoch]):
                    print("_robotnexp for robotID={robotID} and epoch={epoch} is {rnexp}, but should be {nn}".format(robotID=robotID, epoch=epoch, rnexp=self._robotnexp[robotindx, epoch], nn=nn[epoch]))
                    nproblems = nproblems + 1

        for robotID in self.mastergrid.robotDict:
            robotindx = self.robotID2indx[robotID]
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=np.int32):
                itarget = self._robot2indx[robotindx, iexp]
                if(itarget >= 0):
                    if(robotID != self.assignments['robotID'][itarget, iexp]):
                        print("_robot2indx is {i} for robotID=robotID and iexp={iexp} but assignments['robotID'] for itarget={i} is robotID={robotID}".format(iexp=iexp, robotID=robotID, i=itarget))
                        nproblems = nproblems + 1

        if(self.allgrids):
            # Check for collisions
            for iexp, rg in enumerate(self.robotgrids):
                if(self.exposure_locked[iexp] == False):
                    for robotID in rg.robotDict:
                        c = rg.isCollided(robotID)
                        if(c):
                            if(rg.robotDict[robotID].isAssigned()):
                                print("robotID={robotID} iexp={iexp} : collision of assigned robot".format(robotID=robotID, iexp=iexp))
                            else:
                                print("robotID={robotID} iexp={iexp} : collision of unassigned robot".format(robotID=robotID, iexp=iexp))
                            nproblems = nproblems + 1

            # Check _robot2indx, assignments is tracking things correctly
            for iexp, rg in enumerate(self.robotgrids):
                if(self.exposure_locked[iexp] == False):
                    for robotID in rg.robotDict:
                        robotindx = self.robotID2indx[robotID]
                        if(rg.robotDict[robotID].isAssigned()):
                            tid = rg.robotDict[robotID].assignedTargetID
                            itarget = self.rsid2indx[tid]
                        else:
                            itarget = -1
                        if(self._robot2indx[robotindx, iexp] != itarget):
                            print("robotID={robotID} iexp={iexp} : expected {i1} in _robot2indx got {i2}".format(robotID=robotID, iexp=iexp, i1=itarget, i2=self._robot2indx[robotindx, iexp]))
                            nproblems = nproblems + 1

                        if(itarget != -1):
                            if(self.assignments['robotID'][itarget, iexp] !=
                               robotID):
                                print("rsid={rsid} iexp={iexp} : expected {robotID} in assignments['robotID'], got {actual}".format(rsid=tid, iexp=iexp, robotID=robotID, actual=self.assignments['robotID'][itarget, iexp]))
                                nproblems = nproblems + 1

            # Check assignments is tracking things correctly
            for iexp, rg in enumerate(self.robotgrids):
                if(self.exposure_locked[iexp] == False):
                    for itarget, assignment in enumerate(self.assignments):
                        if(assignment['robotID'][iexp] >= 0):
                            if(rg.robotDict[assignment['robotID'][iexp]].assignedTargetID != 
                               self.targets['rsid'][itarget]):
                                print("robotID={robotID} iexp={iexp} : expected {rsid} in assignedTargetID, got {actual}".format(rsid=self.targets['rsid'][itarget], iexp=iexp, robotID=robotID, actual=rg.robotDict[assignment['robotID'][iexp]].assignedTargetID))
                                nproblems = nproblems + 1

        epochs = self.field_cadence.epochs
        inotallowed = np.where((self.assignments['allowed'][:, epochs] == 0) &
                               (self.assignments['robotID'] >= 0))[0]
        if(len(inotallowed) > 0):
            nproblems = nproblems + len(inotallowed)
            uinotallowed = np.unique(inotallowed)
            print("Unallowed exposures observed {n} times".format(n=len(uinotallowed)))

        return(nproblems)

    def validate_cadences(self):
        """Validate the cadences

        Returns
        -------

        nproblems : int
            Number of problems discovered

        Notes
        -----

        Prints nature of problems identified to stdout

        Checks that assigned targets got the right number and type of epochs.

        Not tested recently.
"""
        nproblems = 0
        for indx, target in enumerate(self.targets):
            target_cadence = clist.cadences[target['cadence']]
            nepochs = 0
            for epoch in range(self.field_cadence.nepochs):
                iexpst = self.field_cadence.epoch_indx[epoch]
                iexpnd = self.field_cadence.epoch_indx[epoch + 1]
                nexp = (self.assignments['robotID'][iexpst:iexpnd] >= 0).sum()
                if(nexp > 0):
                    if(self.nocalib is False):
                        if(target['category'] in self.required_calibrations):
                            for iexp in range(iexpst, iexpnd):
                                test_calibrations[target['category']][iexp] += 1

                    # Check that the number of exposures assigned is right for this epoch
                    if(target_cadence.nexp[nepochs] != nexp):
                        print("rsid={rsid} epoch={epoch} : nexp mismatch".format(rsid=target['rsid'],
                                                                                 epoch=epoch))
                        nproblems += 1

                    # Check that the skybrightness is right for this epoch
                    if(target_cadence.skybrightness[nepochs] < self.field_cadence.skybrightness[epoch]):
                        print("rsid={rsid} epoch={epoch} : skybrightness mismatch".format(rsid=target['rsid'],
                                                                                          epoch=epoch))
                        nproblems += 1

                    # Check that the right number of exposures have this robotID assignment
                    nexpr = 0
                    for iexp in range(iexpst, iexpnd):
                        rg = self.robotgrids[iexp]
                        robotID = assignment['robotID'][iexp]
                        if(robotID >= 1):
                            if(rg.robotDict[robotID].assignedTargetID == target['rsid']):
                                nexpr = nexpr + 1
                    if(nexpr != target_cadence.nexp[nepochs]):
                        print("rsid={rsid} epoch={epoch} : robots not assigned ({ne} out of {nt})".format(rsid=target['rsid'],
                                                                                                          epoch=epoch, ne=nexp, nt=target_cadence.nexp[nepochs]))
                        nproblems += 1
                    nepochs = nepochs + 1

            # Check that if the target is assigned, it has the right number of epochs
            if((nepochs > 0) & (nepochs != target_cadence.nepochs) &
               (self._is_calibration[indx] == False)):
                print("rsid={rsid} : target assigned with wrong nepochs".format(rsid=target['rsid']))
                nproblems += 1

    def count_spares(self, return_unused=False):
        """Count spare fibers (accounting for spare calibrations)

        Parameters
        ----------

        return_unused : bool
            return nboss_unused and napogee_unused

        Returns
        -------

        nboss_spare : ndarray of np.int32
            Number of spare BOSS fibers (all, including APOGEE+BOSS robots) in each exposure

        napogee_spare : ndarray of np.int32
            Number of spare APOGEE fibers in each exposure

        nboss_unused : ndarray of np.int32
            If return_unused is True, number of unused BOSS fibers (all, including APOGEE+BOSS robots) in each exposure

        napogee_unused : ndarray of np.int32
            If return_unused is True, number of unused APOGEE fibers in each exposure
"""
        hasApogee = np.array([self.mastergrid.robotDict[self.robotIDs[x]].hasApogee
                              for x in range(500)], dtype=bool)

        iapogee = np.where(hasApogee)[0]
        napogee_spare = len(iapogee)
        nboss_spare = len(hasApogee)

        if(self.assignments is not None):
            nun_apogee = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)
            nun_boss = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)

            # Count the ones with literally no assignment
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=np.int32):
                ina_apogee = np.where(self._robot2indx[iapogee, iexp] < 0)[0]
                ina_boss = np.where(self._robot2indx[:, iexp] < 0)[0]
                nun_apogee[iexp] = len(ina_apogee)
                nun_boss[iexp] = len(ina_boss)
        
            nsp = collections.OrderedDict()
            for calibration in self.calibration_order:
                nsp[calibration] = (self.calibrations[calibration] -
                                    self.required_calibrations[calibration])
                iz = np.where(nsp[calibration] < 0)[0]
                nsp[calibration][iz] = 0

            for bosscalib in ['standard_boss', 'sky_boss']:
                nsp[bosscalib + '_wapogee'] = np.zeros(self.field_cadence.nexp_total,
                                                       dtype=np.int32)
                for iexp in np.arange(self.field_cadence.nexp_total,
                                      dtype=np.int32):
                    ia = np.where((self._robot2indx[:, iexp] >= 0) &
                                  hasApogee)[0]
                    if(len(ia) > 0):
                        ia = self._robot2indx[ia, iexp]
                        ic = np.where(self.targets['category'][ia] == bosscalib)[0]
                        nc = len(ic)
                    else:
                        nc = 0
                    if(nc < nsp[bosscalib][iexp]):
                        nsp[bosscalib + '_wapogee'][iexp] = nc
                    else:
                        nsp[bosscalib + '_wapogee'][iexp] = nsp[bosscalib][iexp]

            # The below is correct because every APOGEE robot is
            # also potentially a BOSS robot, but not vice-versa.
            nboss_spare = (nun_boss +
                           nsp['standard_boss'] +
                           nsp['sky_boss'] +
                           nsp['standard_apogee'] +
                           nsp['sky_apogee'])
            napogee_spare = (nun_apogee +
                             nsp['standard_apogee'] +
                             nsp['sky_apogee'] + 
                             nsp['standard_boss_wapogee'] +
                             nsp['sky_boss_wapogee'])

        return(nboss_spare, napogee_spare, nun_boss, nun_apogee)
        
    def _plot_robot(self, robot, color=None, ax=None):
        """Plot a single robot

        Parameters
        ----------

        robot : Robot object
            instance of robot to plot

        color : str
            color to make beta arm

        ax : Axes object
            matplotlib Axes object to plot on
"""
        alphaPoint = robot.betaCollisionSegment[0]
        betaPoint = robot.betaCollisionSegment[1]
        alphaX = alphaPoint[0]
        alphaY = alphaPoint[1]
        betaX = betaPoint[0]
        betaY = betaPoint[1]
        xa = [robot.xPos, alphaX]
        ya = [robot.yPos, alphaY]
        xb = [alphaX, betaX]
        yb = [alphaY, betaY]
        ax.plot(np.array(xa), np.array(ya), color=color, alpha=0.5)
        ax.plot(np.array(xb), np.array(yb), color=color, linewidth=2)

    def plot(self, iexp=None, robotID=False, catalogid=False):
        """Plot assignments of robots to targets for field

        Parameters
        ----------

        iexp : int or np.int32
            index of exposure to plot

        robotID : bool
            if True, plot the robotID for each robot (default False)

        catalogid : bool
            if True, plot to catalogid for each target (default False)
"""
        if(not self.allgrids):
            print("Cannot plot if allgrids is False")
            return

        fig = plt.figure(figsize=(10 * 0.7, 7 * 0.7))
        axfig = fig.add_axes([0., 0., 0.7, 1.])
        axleg = fig.add_axes([0.71, 0., 0.26, 1.])

        itarget = np.where(self.targets['category'] == 'science')[0]
        axfig.scatter(self.assignments['x'][itarget],
                      self.assignments['y'][itarget], s=1, color='black',
                      label='Science targets', alpha=0.2)
        axleg.plot(self.assignments['x'][itarget],
                   self.assignments['y'][itarget], linewidth=4, alpha=0.2, color='black',
                   label='Science targets')

        itarget = np.where(self.targets['category'] != 'science')[0]
        axfig.scatter(self.assignments['x'][itarget],
                      self.assignments['y'][itarget], s=1, color='blue',
                      label='Calib targets', alpha=0.1)
        axleg.plot(self.assignments['x'][itarget],
                   self.assignments['y'][itarget], linewidth=4, alpha=0.2, color='blue',
                   label='Calib targets')

        if(self.assignments is not None):
            target_got = np.zeros(len(self.targets), dtype=np.int32)
            target_robotid = np.zeros(len(self.targets), dtype=np.int32)
            itarget = np.where(self.assignments['robotID'][:, iexp] >= 1)[0]
            target_got[itarget] = 1
            target_robotid[itarget] = self.assignments['robotID'][itarget, iexp]
            itarget = np.where(target_got > 0)[0]
            
            axfig.scatter(self.assignments['x'][itarget],
                          self.assignments['y'][itarget], s=3, color='black')

            for i in itarget:
                if(self.targets['category'][i] == 'science'):
                    color='blue'
                else:
                    color='black'
                robot = self.robotgrids[iexp].robotDict[target_robotid[i]]
                self._plot_robot(robot, color=color, ax=axfig)

        used = (self._robot2indx[:, iexp] >= 0)
        inot = np.where(used == False)[0]
        for i in self.robotIDs[inot]:
            self._plot_robot(self.robotgrids[iexp].robotDict[int(i)],
                             color='grey', ax=axfig)

        plt.xlim([-370., 370.])
        plt.ylim([-370., 370.])

        h, ell = axleg.get_legend_handles_labels()
        axleg.clear()
        axleg.legend(h, ell, loc='upper left')
        axleg.axis('off')


class FieldSpeedy(Field):
    """FieldSpeedy class

    Notes
    -----

    Subclass of Field, with all the same attributes and methods.

    Relative to Field, this class behaves as follows: 
     * bright_neighbors is set to False
     * nocalib is set True, so calibrations are skipped, which allows a
       a substantial simplification.
     * nocollide is set True, so collisions are not considered
     * allgrids is set False, so that the robotgrid overhead
       is avoided
"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=2.,
                 fieldid=1, verbose=False):
        super().__init__(filename=filename, racen=racen, pa=pa,
                         observatory=observatory, field_cadence=field_cadence,
                         collisionBuffer=collisionBuffer, fieldid=fieldid,
                         verbose=verbose, bright_neighbors=False,
                         nocalib=True, nocollide=True, allgrids=False)
        self.bright_neighbors = False
        self.nocalib = True
        self.nocollide = True
        self.allgrids = False
        return


