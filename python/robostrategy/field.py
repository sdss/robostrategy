#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import re
import random
import datetime
import numpy as np
import fitsio
import collections
import matplotlib.pyplot as plt
import ortools.sat.python.cp_model as cp_model
import roboscheduler
import roboscheduler.cadence
import kaiju
import kaiju.robotGrid
import robostrategy
import robostrategy.targets
import robostrategy.header
import robostrategy.obstime as obstime
import coordio.time
import coordio.utils
import mugatu.designmode
import sdss_access.path

sdss_path = sdss_access.path.Path(release='sdss5', preserve_envvars=True)

# Default collision buffer
defaultCollisionBuffer = 2.


# intersection of lists
def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))


# Type for targets array
targets_dtype = robostrategy.targets.target_dtype
targets_dtype = targets_dtype + [('x', np.float64),
                                 ('y', np.float64),
                                 ('z', np.float64),
                                 ('within', np.int32),
                                 ('incadence', np.int32)]

# Dictionary defining meaning of flags
_flagdict = {'NOT_TO_ASSIGN':1,
             'NOT_SCIENCE':2,
             'NOT_INCADENCE': 4,
             'NOT_COVERED': 8,
             'NONE_ALLOWED': 16,
             'NO_AVAILABILITY': 32,
             'ALREADY_ASSIGNED': 64}

__all__ = ['Field']

"""Field module class.

Dependencies:

 numpy
 fitsio
 matplotlib
 roboscheduler
 kaiju
"""

# Establish access to the CadenceList singleton
clist = roboscheduler.cadence.CadenceList(skybrightness_only=True)


def read_field(plan=None, observatory=None, fieldid=None,
               version='', targets=False):
    """Convenience function to read a field object

    Parameters:
    ----------

    plan : str
        plan name

    observatory : str
        observatory name ('apo' or 'lco')

    version : str
        version of assignments ('', 'Open', 'Filler', 'Reassign')

    targets : bool
        if True, read rsFieldTargets file, do not set cadence (default False)

    fieldid : int
        field id

    Returns:
    -------

    field : Field object
        field object read in
"""
    cadences_file = sdss_path.full('rsCadences', plan=plan,
                                   observatory=observatory)
    clist.fromfits(filename=cadences_file, unpickle=False)

    base = 'rsFieldAssignments'
    if(targets):
        base = 'rsFieldTargets'

    field_file = sdss_path.full(base,
                                plan=plan, observatory=observatory,
                                fieldid=fieldid)
    if(version == 'Reassign'):
        field_file = field_file.replace('rsFieldAssignments',
                                        'rsFieldReassignments')
    if(version == 'Open'):
        field_file = field_file.replace(base, base + 'Open')
    if(version == 'Filler'):
        field_file = field_file.replace(base, base + 'Filler')
    if(version == 'Final'):
        field_file = field_file.replace('targets/' + base,
                                        'final/' + base + 'Final')

    f = Field(filename=field_file, fieldid=fieldid)
    return(f)


class AssignmentStatus(object):
    """Status of a prospective assignment for a set of exposures

    Parameters:
    ----------

    rsid : np.int64
        prospective target

    robotID : np.int32
        prospective robotID

    iexps : ndarray of np.int32
        prospective exposure numbers

    Attributes:
    ----------

    rsid : np.int64
        prospective target

    robotID : np.int32
        prospective robotID

    iexps : ndarray of np.int32
        prospective exposure numbers

    expindx : ndarray of np.int32
        mapping of iexp to index of iexps array

    assignable : ndarray of bool
        is the fiber free to assign and uncollided in exposure? 
        (initialized to True)

    collided : ndarray of bool
        is the fiber collided in exposure? (initialized to False)

    spare : ndarray of bool
        is fiber already assigned a spare calibration target in exposure?
        (initialized to False)

    spare_colliders : list of ndarrays of np.int32
        array of spare calibration targets that assignment collides with
        (initialized to list of empty arrays)

    Methods:
    -------

    assignable_exposures()

    Notes:
    -----

    These objects are used to track information about prospective assignments.
    They only make sense in the context of the Field class, which has
    several methods to manipulate these objects.
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
        self.collided = np.zeros(len(self.iexps), dtype=bool)
        self.spare = np.zeros(len(self.iexps), dtype=bool)
        self.spare_colliders = [np.zeros(0, dtype=np.int64)] * len(self.iexps)
        return

    def assignable_exposures(self):
        """List of assignable exposures
        
        Returns:
        --------
        
        iexps : ndarray of np.int32
            list of assignable exposures
"""
        return(self.iexps[np.where(self.assignable)[0]])


class Field(object):
    """Field class

    Parameters:
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
    
    collisionBuffer : float or np.float32
        collision buffer to send to kaiju in mm (default 2)
        (if set, will override setting in rsFieldTargets)
        IGNORED

    field_cadence : str
        field cadence (default 'none')

    nocalib : bool
        if True, do not account for calibrations (default False)

    allgrids : bool
        if True, keep track of all robotgrids (default True); if False
        automatically sets nocollide to True

    nocollide : bool
        if True,  do not check collisions (default False)

    verbose : bool
        if True, issue a lot of output statements

    veryverbose : bool
        if True, really issue a lot of output statements

    Attributes:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    pa : np.float32
        position angle of field (deg E of N)

    observatory : str
        observatory field observed from ('apo' or 'lco')

    field_cadence : Cadence object
        cadence associated with field

    design_mode : np.array of str
        keys to DesignModeDict for each exposure

    collisionBuffer : float
        collision buffer for kaiju (in mm)
        IGNORED

    radius : np.float32
        distance from racen, deccen to search for for targets (deg);
        set to 1.5 for observatory 'apo' and 0.95 for observatory 'lco'

    flagdict : Dict
        dictionary of assignment flag values

    rsid2indx : Dict
        dictionary linking rsid (key) to index of targets and assignments arrays.
        (values). E.g. targets['rsid'][f.rsid2indx[rsid]] == rsid

    mastergrid : RobotGrid object
        robotGrid used to inquire about robots & targets (not for assignment)

    robotgrids : list of RobotGrid objects
        robotGrids associated with each exposure

    targets : ndarray
        array of targets, including 'ra', 'dec', 'x', 'y', 'within',
        'priority', 'category', 'cadence', 'catalogid', 'rsid', 'fiberType'

    target_duplicated : ndarray of np.int32
        [len(targets)] initially 0s; set in assign() if there are
        coordinated targets which have already been assigned

    assignments : ndarray or None
        [len(targets)] array with 'assigned', 'satisfied', 
          'robotID', 'rsflags', 'fiberType'
        for each target; set to None prior to definition of field_cadence

    designModeDict : dict of DesignMode objects
        possible design modes

    required_calibrations : OrderedDict
        dictionary with numbers of required calibration sources specified
        for 'sky_boss', 'standard_boss', 'sky_apogee', 'standard_apogee'

    achievable_calibrations : OrderedDict
        dictionary of lists with number of achievable calibration
        sources specified for 'sky_boss', 'standard_boss',
        'sky_apogee', 'standard_apogee' (i.e. equal to
        required_calibrations if they all can be achieved even without
        science targets, or the maximum possible if less than that).

    calibrations : OrderedDict
        dictionary of lists with numbers of calibration sources assigned
        for each epoch for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    obstime : coordio Time object
        nominal time of observation to use for calculating x/y

    nocalib : bool
        if True, do not account for calibrations (default False)

    allgrids : bool
        if True, keep track of all robotgrids (default True); if False
        automatically sets nocollide to True

    nocollide : bool
        if True,  do not check collisions (default False)

    verbose : bool
        if True, issue a lot of output statements

    veryverbose : bool
        if True, really issue a lot of output statements

    _robot2indx : ndarray of int32 or None
        [nrobots, nexp_total] array of indices into targets from robots

    _robotnexp : ndarray of int32 or None
        [nrobots, nepochs] array of number of exposures available per epoch

    _is_calibration : ndarray of bool
        [len(targets)] list of whether the target is a calibration target

    _has_spare_calib : 2D ndarray of bool
        [len(targets) + 1, nexp_total] indicates whether a particular target
        is a spare calibration target in this exposure. The first axis should
        be referenced with rsid2indx + 1; the 0th element is there to deal
        with unassigned cases "-1".

    _calibration_index : ndarray of np.int32
        [len(targets)] indicates which type of calibration target this object
        is; 0 for a science target, and 1..4 for each of the required_calibration
        categories in order.

    _competing_targets : ndarray of np.int32
        [nrobots] count of how many targets are competing for a given
        robot; used only in certain methods of assignment.

    _ot : ObsTime object
        observing time object for convenience

    _unique_catalogids : ndarray of np.int64
        list of unique catalogids for convenience

    Notes:
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
                 verbose=False, veryverbose=False):
        self.verbose = verbose
        self.veryverbose = veryverbose
        self.fieldid = fieldid
        self.nocalib = nocalib
        self.nocollide = nocollide
        self.allgrids = allgrids
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
        self._calibration_index = np.zeros(1, dtype=bool)
        self._unique_catalogids = None
        if(filename is not None):
            if(self.verbose):
                print("fieldid {fid}: Reading from {f}".format(f=filename, fid=self.fieldid), flush=True)
            self.fromfits(filename=filename)
        else:
            self.racen = racen
            self.deccen = deccen
            self.pa = pa
            self.observatory = observatory
            self._ot = obstime.ObsTime(observatory=self.observatory)
            self.obstime = coordio.time.Time(self._ot.nominal(lst=self.racen))
            if(self.collisionBuffer is None):
                self.collisionBuffer = defaultCollisionBuffer
            self.mastergrid = self._robotGrid()
            self.robotIDs = np.array([x for x in self.mastergrid.robotDict.keys()],
                                     dtype=int)
            self.robotID2indx = dict()
            for indx, robotID in enumerate(self.robotIDs):
                self.robotID2indx[robotID] = indx
            self.designModeDict = mugatu.designmode.allDesignModes() 
            if(self.designModeDict is None):
                default_dm_file= os.path.join(os.getenv('ROBOSTRATEGY_DIR'),
                                              'data',
                                              'default_designmodes.fits')
                mugatu.designmode.allDesignModes(filename=default_dm_file)
            if(self.nocalib is False):
                self.required_calibrations = collections.OrderedDict()
                self.required_calibrations['sky_boss'] = np.zeros(0, dtype=np.int32)
                self.required_calibrations['standard_boss'] = np.zeros(0, dtype=np.int32)
                self.required_calibrations['sky_apogee'] = np.zeros(0, dtype=np.int32)
                self.required_calibrations['standard_apogee'] = np.zeros(0, dtype=np.int32)
                self.calibrations = collections.OrderedDict()
                for n in self.required_calibrations:
                    self.calibrations[n] = np.zeros(0, dtype=np.int32)
                self.achievable_calibrations = collections.OrderedDict()
                for n in self.required_calibrations:
                    self.achievable_calibrations[n] = self.required_calibrations[n].copy()
            self.set_field_cadence(field_cadence)
        self._set_radius()
        self.flagdict = _flagdict
        self._competing_targets = None
        self.methods = dict()
        self.methods['assign_epochs'] = 'first'
        self._add_dummy_cadences()
        return

    def _add_dummy_cadences(self): 
        """Adds some dummy cadences necessary to check singlebright and multibright"""
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

    def fromfits(self, filename=None):
        """Read field from FITS file

        Parameters:
        ----------

        filename : str
            name of file to read in

        Comments:
        --------

        Expects HDU0 header to contain keywords:

           RACEN (J2000 deg)
           DECCEN (J2000 deg)
           PA (position each deg E of N)
           OBS ('apo' or 'lco')
           CBUFFER ("collision buffer")
           FCADENCE  ("field cadence", can be 'none')
           RCNAME# (name of a required calibration category)
           RCNUM# (required calibration number)

        If NOCALIB is in header, the Field will be initialized
        with nocalib= True.

        IF ACNAME# and ACNUM# are set in header (and HDU2 is present)
        these are interpreted as the "achievable calibrations".

        Expects HDU1 data to be a table containing targets. See
        targets_fromarray() method for expectations about its structure.

        If HDU2 is present, it is expected to be the assignments
        table.  This table should have a row for each target that is
        parallel with the targets table, and at least one column
        called 'robotID', which should be an array of length
        field_cadence.nexp_total with an np.int32 for each exposure
        containing the number for the assigned robot (or -1 if the
        target is not assigned in that exposure). This method does
        not copy the assignments table directly, it adds the assignments
        using assign_robot_exposure(), so all columns in HDU2 other
        than 'robotID' are ignored.

        In the context of a robostrategy run, this method can read in
        an rsFieldTargets file (i.e. the input files to assignment)
        or an rsFieldAssignments file (i.e. the output files from
        assignment).
"""
        duf, hdr = fitsio.read(filename, ext=0, header=True)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        if(self.collisionBuffer is None):
            self.collisionBuffer = hdr['CBUFFER']
        if(('NOCALIB' in hdr) & (self.nocalib == False)):
            self.nocalib = np.bool(hdr['NOCALIB'])
        self.mastergrid = self._robotGrid()
        self.robotIDs = np.array([x for x in self.mastergrid.robotDict.keys()],
                                 dtype=int)
        self.robotID2indx = dict()
        for indx, robotID in enumerate(self.robotIDs):
            self.robotID2indx[robotID] = indx
        self._ot = obstime.ObsTime(observatory=self.observatory)
        self.obstime = coordio.time.Time(self._ot.nominal(lst=self.racen))
        field_cadence = hdr['FCADENCE']
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
            self.calibrations = collections.OrderedDict()
            for n in self.required_calibrations:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
            self.achievable_calibrations = collections.OrderedDict()
            for n in self.required_calibrations:
                self.achievable_calibrations[n] = self.required_calibrations[n].copy()
        self.designModeDict = mugatu.designmode.allDesignModes(filename,
                                                               ext='DESMODE')
        try:
            self.designModeDict = mugatu.designmode.allDesignModes(filename,
                                                                   ext='DESMODE')
            named_ext = True
        except:
            default_dm_file= os.path.join(os.getenv('ROBOSTRATEGY_DIR'),
                                          'data',
                                          'default_designmodes.fits')
            self.designModeDict = mugatu.designmode.allDesignModes(default_dm_file)
            named_ext = False
        self.set_field_cadence(field_cadence)
        if(named_ext):
            targets = fitsio.read(filename, ext='TARGET')
        else:
            targets = fitsio.read(filename, ext=1)
        self.targets_fromarray(target_array=targets)
        if(named_ext):
            try:
                assignments = fitsio.read(filename, ext='ASSIGN')
            except:
                assignments = None
        else:
            try:
                assignments = fitsio.read(filename, ext=2)
            except:
                assignments = None
        if(assignments is not None):
            self.achievable_calibrations = collections.OrderedDict()
            for n in self.required_calibrations:
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
                                               reset_count=False)
            else:
                iassigned = np.where(assignments['robotID'] >= 1)
                for itarget, iexp in zip(iassigned[0], iassigned[1]):
                    self.assign_robot_exposure(robotID=assignments['robotID'][itarget, iexp],
                                               rsid=targets['rsid'][itarget],
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False,
                                               reset_count=False)

            for assignment, target in zip(assignments, targets):
                indx = self.rsid2indx[target['rsid']]
                self.assignments['rsflags'][indx] = assignment['rsflags']
            self._set_has_spare_calib()
            self._set_satisfied()
            self._set_count(reset_equiv=False)
            self.decollide_unassigned()
        return

    def clear_assignments(self):
        """Clear the assignments for this field

        Comments:
        --------

        Uses unassign() to unassign every target.
"""
        if(self.assignments is not None):
            iassigned = np.where(self.assignments['assigned'])[0]
            self.unassign(self.targets['rsid'][iassigned])
            self.assignments['assigned'] = 0
            self.assignments['satisfied'] = 0
        return

    def clear_field_cadence(self):
        """Resets the field cadence to 'none' and clears all the ancillary data

        Comments:
        --------

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
        self._robot2indx = None
        self._robotnexp = None
        self._robotnexp_max = None
        self.field_cadence = None
        self.assignments_dtype = None
        self.assignments = None
        self.design_mode = None
        if(self.nocalib is False):
            for n in self.required_calibrations:
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
        """Return a RobotGridAPO or RobotGridLCO instance, with all robots at home"""
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

        Parameters:
        ----------

        field_cadence : str
            Name of field cadence

        Notes:
        ------

        Sets the field cadence. This can only be done once, and note that
        if the object is instantiated with parameters including field_cadence,
        this routine is called in the initialization. If the object is
        instantiated with a file name, if the file header has the FCADENCE
        keyword set to anything but 'none', this routine will be called.

        The cadence must be one in the CadenceList singleton. Upon
        setting the field cadence with this routine, the robotgrids,
        assignments_dtype, assignments, calibrations, and
        field_cadence attributes.  be configured.
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
                                               ('nexps', np.int32),
                                               ('nepochs', np.int32),
                                               ('allowed', np.int32,
                                                (self.field_cadence.nepochs,)),
                                               ('robotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('holeID', np.dtype("|U15"), self.field_cadence.nexp_total),
                                               ('equivRobotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('target_skybrightness', np.float32,
                                                (self.field_cadence.nexp_total,)),
                                               ('field_skybrightness', np.float32,
                                                (self.field_cadence.nexp_total,)),
                                               ('fiberType', np.unicode_, 10),
                                               ('rsflags', np.int32)])
            self.assignments = np.zeros(0, dtype=self.assignments_dtype)

            try:
                obsmode_pk = self.field_cadence.obsmode_pk
            except AttributeError:
                obsmode_pk = np.array([''] * self.field_cadence.nexp_total)

            if(obsmode_pk[0] != ''):
                if(self.verbose):
                    print("obsmode_pk has been set", flush=True)
                if((type(obsmode_pk) == list) |
                   (type(obsmode_pk) == np.ndarray)):
                    self.design_mode = np.array(obsmode_pk)
                else:
                    self.design_mode = np.array([obsmode_pk])
            else:
                if(self.verbose):
                    print("Using heuristics for obsmode_pk", flush=True)
                self.design_mode = np.array([''] *
                                            self.field_cadence.nexp_total)
                for iexp in np.arange(self.field_cadence.nexp_total):
                    epoch = self.field_cadence.epochs[iexp]
                    if(self.field_cadence.skybrightness[epoch] >= 0.5):
                        self.design_mode[iexp] = 'bright_time'
                    else:
                        if(('dark_100x8' in self.field_cadence.name) |
                           ('dark_174x8' in self.field_cadence.name)):
                            self.design_mode[iexp] = 'dark_rm'
                        elif(('dark_10x4' in self.field_cadence.name) |
                             ('dark_2x4' in self.field_cadence.name) |
                             ('dark_3x4' in self.field_cadence.name)):
                            self.design_mode[iexp] = 'dark_monit'
                        elif(('dark_1x1' in self.field_cadence.name) |
                             ('dark_1x2' in self.field_cadence.name) |
                             ('dark_2x1' in self.field_cadence.name) |
                             ('mixed2' in self.field_cadence.name)):
                            self.design_mode[iexp] = 'dark_plane'
                        else:
                            self.design_mode[iexp] = 'dark_faint'
                    
            if(self.nocalib is False):
                dms = self.design_mode[self.field_cadence.epochs]
                for c in self.required_calibrations:
                    if(c == 'standard_boss'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_stds_min['BOSS'] for d in dms], dtype=np.int32)
                    elif(c == 'standard_apogee'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_stds_min['APOGEE'] for d in dms], dtype=np.int32)
                    elif(c == 'sky_boss'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_skies_min['BOSS'] for d in dms], dtype=np.int32)
                    elif(c == 'sky_apogee'):
                        self.required_calibrations[c] = np.array([self.designModeDict[d].n_skies_min['APOGEE'] for d in dms], dtype=np.int32)
                for c in self.calibrations:
                    self.calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                                    dtype=np.int32)
                for c in self.calibrations:
                    self.achievable_calibrations[c] = self.required_calibrations[c].copy()
            self.targets = self._setup_targets_for_cadence(self.targets)
            self.assignments = self._setup_assignments_for_cadence(self.targets)
            if(self.nocalib is False):
                self._set_has_spare_calib()
            if(self.verbose):
                print("fieldid {fid}:   (done setting field cadence)".format(fid=self.fieldid), flush=True)
        else:
            self.field_cadence = None
            if(self.allgrids):
                self.robotgrids = []
            else:
                self.robotgrids = None
            self.assignments_dtype = None
            self._has_spare_calib = None

        return

    def set_flag(self, rsid=None, flagname=None):
        """Set a bitmask flag for a target

        Parameters:
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

        Parameters:
        ----------

        rsid : np.int64 or ndarray
            IDs of the target-cadence

        flagname : str
            name of flag to set

        Returns:
        -------

        setornot : ndarray of bool
            True if flag is set, flag otherwise
"""
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)], dtype=int)
        setornot = ((self.assignments['rsflags'][indxs] & self.flagdict[flagname]) != 0)
        return(setornot)

    def get_flag_names(self, flagval=None):
        """Return names associated with flag

        Parameters:
        ----------

        flagval : np.int32
            flag

        Returns:
        -------

        flagnames : list
            strings corresponding to each set bit
"""
        flagnames = []
        for fn in self.flagdict:
            if(flagval & self.flagdict[fn]):
                flagnames.append(fn)
        return(flagnames)

    def _offset_radec(self, ra=None, dec=None, delta_ra=0., delta_dec=0.):
        """Offsets ra and dec according to specified amount
        
        Parameters:
        ----------

        ra : np.float64 or ndarray of np.float64
        right ascension, deg

        dec : np.float64 or ndarray of np.float64
            declination, deg

        delta_ra : np.float64 or ndarray of np.float64
            right ascension direction offset, arcsec

        delta_dec : np.float64 or ndarray of np.float64
            declination direction offset, arcsec

        Returns:
        -------

        offset_ra : np.float64 or ndarray of np.float64
            offset right ascension, deg

        offset_dec : np.float64 or ndarray of np.float64
            offset declination, deg

        Notes:
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

    def radec2xyz(self, ra=None, dec=None, epoch=None, pmra=None,
                  pmdec=None, delta_ra=0., delta_dec=0., fiberType=None):
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
        pmra = np.zeros(len(raoff), dtype=np.float64) # BECAUSE PM FAILS!!
        pmdec = np.zeros(len(raoff), dtype=np.float64)
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
        """X and Y back to RA, Dec, without proper motions or deltas"""
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
"""
        t = fitsio.read(filename, ext=1)
        self.targets_fromarray(t)
        return

    def _mags_allowed(self, targets=None, designMode=None):
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

    def _targets_to_robotgrid(self, targets=None, robotgrid=None):
        for indx, target in enumerate(targets):
            if(target['fiberType'] == 'APOGEE'):
                fiberType = kaiju.cKaiju.ApogeeFiber
            else:
                fiberType = kaiju.cKaiju.BossFiber
            robotgrid.addTarget(targetID=target['rsid'],
                                xyzWok=[target['x'],
                                        target['y'],
                                        target['z']],
                                priority=np.float64(target['priority']),
                                fiberType=fiberType)
        return

    def _setup_targets_for_cadence(self, targets=None):
        if(targets is None):
            return(None)

        # Determine if it is within the field cadence
        for itarget, target_cadence in enumerate(targets['cadence']):
            if(target_cadence in clist.cadences):
                ok, solns = clist.cadence_consistency(target_cadence,
                                                      self.field_cadence.name)
                targets['incadence'][itarget] = ok

        if(self.allgrids):
            for rg in self.robotgrids:
                self._targets_to_robotgrid(targets=targets,
                                           robotgrid=rg)

        return(targets)

    def _setup_assignments_for_cadence(self, targets=None,
                                       assignment_array=None):
        if(targets is None):
            return(None)

        # Set up outputs
        assignments = np.zeros(len(targets),
                               dtype=self.assignments_dtype)

        field_skybrightness = self.field_cadence.skybrightness[self.field_cadence.epochs]
        assignments['field_skybrightness'] = np.outer(np.ones(len(targets)),
                                                      field_skybrightness)

        for epoch, mode in enumerate(self.design_mode):
            dm = self.designModeDict[mode]
            assignments['allowed'][:, epoch] = self._mags_allowed(designMode=dm,
                                                                  targets=targets)

        if(assignment_array is None):
            assignments['fiberType'] = targets['fiberType']
            assignments['robotID'] = -1
            assignments['equivRobotID'] = -1
            assignments['target_skybrightness'] = -1.
        else:
            for n in self.assignments_dtype.names:
                listns = ['robotID', 'equivRobotID', 'target_skybrightness',
                          'field_skybrightness']
                if((n in listns) & (self.field_cadence.nexp_total == 1)):
                    assignments[n][:, 0] = assignment_array[n]
                else:
                    assignments[n] = assignment_array[n]
        return(assignments)

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
                targets[n] = target_array[n]

        # Default value of 1 for priority, value, and rsassign
        if('value' not in target_array.dtype.names):
            targets['value'] = 1.
        if('priority' not in target_array.dtype.names):
            targets['priority'] = 1.
        if('rsassign' not in target_array.dtype.names):
            targets['rsassign'] = 1

        # Convert ra/dec to x/y
        if(self.verbose):
            print("Convert targets coords to x/y", flush=True)
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

        # Add targets to robotGrids
        if(self.verbose):
            print("Assign targets to robot grid", flush=True)
        self._targets_to_robotgrid(targets=targets,
                                   robotgrid=self.mastergrid)

        # Determine if within
        if(self.verbose):
            print("Check whether targets are within grid", flush=True)
        self.masterTargetDict = self.mastergrid.targetDict
        for itarget, rsid in enumerate(targets['rsid']):
            t = self.masterTargetDict[rsid]
            targets['within'][itarget] = len(t.validRobotIDs) > 0

        # Create internal look-up of whether it is a calibration target
        _is_calibration = np.zeros(len(targets), dtype=bool)
        _calibration_index = np.zeros(len(targets), dtype=np.int32)
        if(self.nocalib is False):
            for icategory, category in enumerate(self.required_calibrations):
                icat = np.where(targets['category'] == category)[0]
                _is_calibration[icat] = True
                _calibration_index[icat] = icategory + 1
        else:
            inotsci = np.where(targets['category'] != 'science')[0]
            _is_calibration[inotsci] = True
            _calibration_index[inotsci] = 1

        # Connect rsid with index of list
        for itarget, t in enumerate(targets):
            if(t['rsid'] in self.rsid2indx.keys()):
                print("Cannot replace identical rsid={rsid}. Will not add array.".format(rsid=t['rsid']))
                return
            else:
                self.rsid2indx[t['rsid']] = len(self.targets) + itarget

        # If field_cadence is set, set up potential outputs
        if(self.field_cadence is not None):
            targets = self._setup_targets_for_cadence(targets)
            assignments = self._setup_assignments_for_cadence(targets,
                                                              assignment_array)
        else:
            assignments = None

        target_duplicated = np.zeros(len(targets), dtype=np.int32)

        self.targets = np.append(self.targets, targets)
        self.target_duplicated = np.append(self.target_duplicated,
                                           target_duplicated)
        self._is_calibration = np.append(self._is_calibration,
                                         _is_calibration)
        self._calibration_index = np.append(self._calibration_index,
                                            _calibration_index)

        self._unique_catalogids = np.unique(self.targets['catalogid'])

        # Set up lists of equivalent observation conditions, meaning
        # that for each target we can look up all of the other targets
        # whose catalog, fiberType, lambda_eff, delta_ra, delta_dec 
        # are the same
        self._equivindx = collections.OrderedDict()
        self._equivkey = collections.OrderedDict()
        for itarget, target in enumerate(self.targets):
            ekey = (target['catalogid'], target['fiberType'],
                    target['lambda_eff'], target['delta_ra'],
                    target['delta_dec'])
            if(ekey not in self._equivindx):
                self._equivindx[ekey] = np.zeros(0, dtype=np.int32)
            self._equivindx[ekey] = np.append(self._equivindx[ekey],
                                              np.array([itarget], dtype=int))
            self._equivkey[itarget] = ekey

        if(assignments is not None):
            self.assignments = np.append(self.assignments, assignments, axis=0)
            self._set_satisfied()
            self._set_count(reset_equiv=False)

        return

    def _set_holeid(self):
        if(self.field_cadence.nexp_total == 1):
            self.assignments['holeID'][:] = ' '
        else:
            self.assignments['holeID'][:, :] = ' '
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

        Parameters:
        ----------

        filename : str
            file name to write to

        Notes:
        -----

        HDU0 header has keywords:

            RACEN
            DECCEN
            PA
            OBS
            FCADENCE (field cadence)
            RCNAME0 .. RNAMEN (required calibrations names)
            RCNUM0 .. RNUMN (required calibrations numbers)

        HDU1 has targets array
        HDU1 has assignments array
"""
        hdr = robostrategy.header.rsheader()
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
            for indx, rc in enumerate(self.required_calibrations):
                name = 'RCNAME{indx}'.format(indx=indx)
                num = 'RCNUM{indx}'.format(indx=indx)
                hdr.append({'name':name,
                            'value':rc,
                            'comment':'calibration category'})
                ns = ' '.join([str(int(n)) for n in self.required_calibrations[rc]])
                hdr.append({'name':num,
                            'value':ns,
                            'comment':'number required per exposure'})
            for indx, ac in enumerate(self.achievable_calibrations):
                name = 'ACNAME{indx}'.format(indx=indx)
                num = 'ACNUM{indx}'.format(indx=indx)
                hdr.append({'name':name,
                            'value':ac,
                            'comment':'calibration category'})
                ns = ' '.join([str(int(n)) for n in self.achievable_calibrations[ac]])
                hdr.append({'name':num,
                            'value':ns,
                            'comment':'number achievable per exposure'})

        fitsio.write(filename, None, header=hdr, clobber=True)
        fitsio.write(filename, self.targets, extname='TARGET')
        if(self.assignments is not None):
            self._set_holeid()
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
            robotIDs = np.sort(np.array([r for r in self.mastergrid.robotDict],
                                        dtype=np.int32))
            robots = np.zeros(len(robotIDs), dtype=robots_dtype) 
            for indx, robotID in enumerate(robotIDs):
                robots['robotID'][indx] = robotID
                robots['holeID'][indx] = self.mastergrid.robotDict[robotID].holeID
                robots['hasBoss'][indx] = self.mastergrid.robotDict[robotID].hasBoss
                robots['hasApogee'][indx] = self.mastergrid.robotDict[robotID].hasApogee
                if(self.field_cadence.nexp_total == 1):
                    robots['rsid'][indx] = self.robotgrids[0].robotDict[robotID].assignedTargetID
                    if(robots['rsid'][indx] == -1):
                        robots['itarget'][indx] = -1
                        robots['catalogid'][indx] = -1
                        robots['fiberType'][indx] = ''
                    else:
                        robots['itarget'][indx] = self.rsid2indx[robots['rsid'][indx]]
                        robots['catalogid'][indx] = self.targets['catalogid'][robots['itarget'][indx]]
                        robots['fiberType'][indx] = self.targets['fiberType'][robots['itarget'][indx]]
                else:
                    for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
                        robots['rsid'][indx, iexp] = self.robotgrids[iexp].robotDict[robotID].assignedTargetID
                        if(robots['rsid'][indx, iexp] == -1):
                            robots['itarget'][indx, iexp] = -1
                            robots['catalogid'][indx, iexp] = -1
                            robots['fiberType'][indx, iexp] = ''
                        else:
                            robots['itarget'][indx, iexp] = self.rsid2indx[robots['rsid'][indx, iexp]]
                            robots['catalogid'][indx, iexp] = self.targets['catalogid'][robots['itarget'][indx, iexp]]
                            robots['fiberType'][indx, iexp] = self.targets['fiberType'][robots['itarget'][indx, iexp]]

            fitsio.write(filename, robots, extname='ROBOTS')
                    
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

    def set_assignment_status(self, status=None, isspare=None):
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
            free = (status.currindx < 0)
            hasspare = self._has_spare_calib[self._calibration_index[status.currindx + 1], status.iexps]
            status.spare = (hasspare > 0) & (isspare == False) & (free == False)
            status.assignable = (free | status.spare) & (allowed > 0)
        else:
            # Consider exposures for this epoch
            status.currindx = self._robot2indx[robotindx, status.iexps]
            if(status.rsid is not None):
                indx = self.rsid2indx[status.rsid]
                epochs = self.field_cadence.epochs
                allowed = self.assignments['allowed'][indx, epochs[status.iexps]]
            else:
                allowed = True
            status.assignable = (status.currindx < 0) & (allowed > 0)

        if(status.rsid is not None):
            for iexp in status.assignable_exposures():
                self.set_collided_status(status=status, iexp=iexp)

        return

    def set_collided_status(self, status=None, iexp=None):
        # leave alone if collisions are being ignored
        if((not self.allgrids) | (self.nocollide)):
            return
        if(status.rsid is None):
            return
        i = status.expindx[iexp]
        if(status.assignable[i] == False):
            return
        rg = self.robotgrids[iexp]
        collided, fcollided, gcollided, colliders = rg.wouldCollideWithAssigned(status.robotID, status.rsid)
        colliders = np.array(colliders, dtype=np.int32)
        status.collided[i] = collided | fcollided | gcollided
        if(fcollided or gcollided):
            status.assignable[i] = False
        if((len(colliders) > 0) and
           (fcollided is False) and
           (gcollided is False)):
            colliderindxs = np.array([self.robotID2indx[x]
                                      for x in colliders], dtype=int)
            robotindx = self._robot2indx[colliderindxs, iexp]
            hasspare = self._has_spare_calib[self._calibration_index[robotindx + 1], iexp] > 0
            # If the collision is created ONLY by spare calibration targets
            # we will look at them in more detail
            if(hasspare.min() > 0):
                toremove = dict()
                for c in self.required_calibrations:
                    toremove[c] = 0
                if(status.spare[i]):
                    toremove[self.targets['category'][status.currindx[i]]] += 1
                for ri in robotindx:
                    toremove[self.targets['category'][ri]] += 1
                enough = True
                for c in self.required_calibrations:
                    excess = self.calibrations[c][iexp] - self.achievable_calibrations[c][iexp]
                    if(toremove[c] > excess):
                        enough = False
                status.assignable[i] = enough
                if(enough):
                    status.spare_colliders[i] = self.targets['rsid'][robotindx]
            else:
                status.assignable[i] = False

        return

    def unassign_assignable(self, status=None, iexp=None,
                            reset_satisfied=True, reset_has_spare=True,
                            reset_count=True):
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

    def collide_robot_exposure(self):
        """TODO GET THIS BACK
        collide : bool
            True if it causes a collision, False if not

        Notes:
        -----

        If there is no RobotGrid to check collisions and/or nocollide
        is set for this object, it doesn't actually check collisions.
        However, it does report a collision if any OTHER equivalent 
        target was assigned.
"""
        if((not self.allgrids) |
           (self.nocollide)):
            indx = self.rsid2indx[rsid]
            allindxs = set(self._equivindx[self._equivkey[indx]])
            if(len(allindxs) > 1):
                allindxs.discard(indx)
                allindxs = np.array(list(allindxs), dtype=np.int32)
                if(self.assignments['robotID'][allindxs, iexp].max() >= 0):
                    return(True)
                else:
                    return(False)
            else:
                return(False)

        rg = self.robotgrids[iexp]
        return rg.wouldCollideWithAssigned(robotID, rsid)[0]

    def available_robot_epoch(self, rsid=None,
                              robotID=None, epoch=None, nexp=None,
                              isspare=None):
        """Check if a robot-epoch has enough exposures

        Parameters:
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

        Returns:
        -------

        available : bool
            is it available or not?

        status : list of AssignmentStatus
            which exposures in the epoch are free?

        Comments:
        --------

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
        available = nfree >= nexp

        return available, status

    def available_robot_exposures(self, rsid=None, robotID=None, isspare=False):
        """Return available robot exposures for an rsid

        Parameters:
        ----------

        rsid : np.int64
            rsid

        robotID : np.int64
            robotID to check

        isspare : bool
            True if this is a spare calibration target (default False)

        Returns:
        -------

        status : AssignmentStatus for object
            for each exposure, is it available or not?

        Comments:
        --------

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
        the following:

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
        if(iexps is None):
            iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        return(self._has_spare_calib[self._calibration_index[self.rsid2indx[rsid] + 1], iexps] > 0)

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None, nexp=None,
                           reset_satisfied=True, reset_has_spare=True,
                           status=None, reset_count=True):
        """Assign an rsid to a particular robot-epoch

        Parameters:
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

        Returns:
        --------

        success : bool
            True if successful, False otherwise

        Comments:
        --------
"""
        # Only try to assign if you can.
        if(rsid not in self.mastergrid.robotDict[robotID].validTargetIDs):
            return False

        # Get list of available exposures in the epoch
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
        if(status is None):
            isspare = self._is_spare(rsid=rsid, iexps=iexps)
            status = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexps)
            self.set_assignment_status(status=status, isspare=isspare)

        assignable = status.assignable_exposures()

        # Bomb if there aren't enough available
        if(len(assignable) < nexp):
            return False

        # Now actually assign (to first available exposures)
        for iexp in assignable[0:nexp]:
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

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsid values to count for each robot

        Notes:
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
                              reset_count=True):
        """Assign an rsid to a particular robot-exposure

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        robotID : np.int64
            robotID to assign to

        iexp : int or np.int32
            exposure to assign to

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix
            (default True)

        reset_count : bool
            if True, reset the 'nexp' and 'nepochs' columns
            (default True)

        Returns:
        --------

        success : bool
            True if successful, False otherwise
"""
        itarget = self.rsid2indx[rsid]

        if(self.assignments['robotID'][itarget, iexp] >= 0):
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=True,
                                   reset_satisfied=True, reset_has_spare=True)

        robotindx = self.robotID2indx[robotID]
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

        # If this is a calibration target, update calibration target tracker
        if(self.nocalib is False):
            if(self._is_calibration[itarget]):
                category = self.targets['category'][itarget]
                self.calibrations[category][iexp] = self.calibrations[category][iexp] + 1

        if(self.allgrids):
            rg = self.robotgrids[iexp]
            rg.assignRobot2Target(robotID, rsid)

        if(reset_satisfied | reset_count):
            self._set_equiv(rsids=[rsid], iexps=[iexp])

        if(reset_satisfied):
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_count):
            self._set_count(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return

    def assign_exposures(self, rsid=None, iexps=None, reset_satisfied=True,
                         reset_has_spare=True):
        """Assign an rsid to particular exposures

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        iexps : ndarray of np.int32
            exposures to assign to

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix
            (default True)

        Returns:
        --------

        success : ndarray of bool
            for each exposure, True if successful, False otherwise
"""
        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs, dtype=np.int32)
        validRobotIndxs = np.array([self.robotID2indx[x]
                                    for x in validRobotIDs], dtype=int)
        hasApogee = self.robotHasApogee[validRobotIndxs]
        validRobotIDs = validRobotIDs[np.argsort(hasApogee)]
        done = np.zeros(len(iexps), dtype=bool)
        # will not work if iexps is not all exposures!
        for robotID in validRobotIDs:
            cexps = iexps[np.where(done == False)[0]]
            if(len(cexps) == 0):
                break
            status = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=cexps)
            self.set_assignment_status(status=status)
            for iexp in status.assignable_exposures():
                self.unassign_assignable(status=status, iexp=iexp,
                                         reset_count=False,
                                         reset_satisfied=False,
                                         reset_has_spare=True)
                self.assign_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp,
                                           reset_count=False,
                                           reset_satisfied=False,
                                           reset_has_spare=True)
                iorig = np.where(iexps == iexp)[0]
                done[iorig] = True

        if(reset_satisfied):
            self._set_equiv(rsids=[rsid], iexps=iexps)
            self._set_satisfied(rsids=[rsid], reset_equiv=False)

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return done

    def _set_assigned(self, itarget=None):
        if(itarget is None):
            print("Must specify a target.")
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        return

    def unassign_exposure(self, rsid=None, iexp=None, reset_assigned=True,
                          reset_satisfied=True, reset_has_spare=True,
                          reset_count=True):
        """Unassign an rsid from a particular exposure

        Parameters:
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
"""
        itarget = self.rsid2indx[rsid]
        category = self.targets['category'][itarget]
        robotID = self.assignments['robotID'][itarget, iexp]
        if(robotID >= 1):
            robotindx = self.robotID2indx[robotID]
            if(self.allgrids):
                rg = self.robotgrids[iexp]
                rg.unassignTarget(rsid)
            self.assignments['robotID'][itarget, iexp] = -1
            self._robot2indx[robotindx, iexp] = -1
            epoch = self.field_cadence.epochs[iexp]
            self._robotnexp[robotindx, epoch] = self._robotnexp[robotindx, epoch] + 1
            if(self.targets['category'][itarget] == 'science'):
                self._robotnexp_max[robotindx, epoch] = self._robotnexp_max[robotindx, epoch] + 1
            if(self.nocalib is False):
                if(self._is_calibration[itarget]):
                    self.calibrations[category][iexp] = self.calibrations[category][iexp] - 1
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
                       reset_count=True):
        """Unassign an rsid from a particular epoch

        Parameters:
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

        Returns:
        -------

        status : int
            0 if the target had been assigned and was successfully removed
"""
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        iexps = np.arange(iexpst, iexpnd)
        for iexp in iexps:
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=False,
                                   reset_satisfied=False, reset_has_spare=False)

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
                 reset_has_spare=True, reset_count=True):
        """Unassign a set of rsids entirely

        Parameters:
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
"""
        if(len(rsids) == 0):
            return

        for rsid in rsids:
            for epoch in range(self.field_cadence.nepochs):
                self.unassign_epoch(rsid=rsid, epoch=epoch, reset_assigned=False,
                                    reset_satisfied=False, reset_has_spare=False)

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

        Parameters:
        ----------

        epochs : ndarray of np.int32
            epochs to assign to (default all)

        nexps : ndarray of np.int32
            number of exposures needed

        Returns:
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

        Parameters:
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

        Returns:
        --------

        available : dictionary, with key value pairs below
            'available' : bool
                is the overall ask available

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
        if(bad.max() > 0):
            available = dict()
            available['available'] = False
            available['nAvailableRobotIDs'] = nAvailableRobotIDs
            available['availableRobotIDs'] = availableRobotIDs
            available['statuses'] = statuses
            return(available)

        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs, dtype=np.int32)
        np.random.shuffle(validRobotIDs)
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
        validRobotIDs = validRobotIDs[hasApogee.argsort()]

        if(self.nocalib is False):
            isspare = self._is_spare(rsid=rsid)
        else:
            isspare = np.zeros(self.field_cadence.nexp_total, dtype=bool)

        for iepoch, epoch in enumerate(epochs):
            nexp = nexps[iepoch]
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
            arlist = []
            slist = []
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

    def assign_epochs(self, rsid=None, epochs=None, nexps=None):
        """Assign target to robots in a set of epochs

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        epochs : ndarray of np.int32
            epochs to assign to

        nexps : ndarray of np.int32
            number of exposures needed

        method : str
            method to use to pick which robot ('first')

        Returns:
        --------

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

        # Assign to each epoch
        for iepoch, epoch in enumerate(epochs):
            currRobotIDs = np.array(availableRobotIDs[iepoch], dtype=np.int32)
            currRobotIndxs = np.array([self.robotID2indx[x]
                                       for x in currRobotIDs], dtype=int)
            if(self.methods['assign_epochs'] == 'first'):
                irobot = 0
            if(self.methods['assign_epochs'] == 'fewestcompeting'):
                irobot = np.argmin(self._competing_targets[currRobotIndxs])
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

    def assign_cadence(self, rsid=None):
        """Assign target to robots according to its cadence

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        Returns:
        --------

        success : bool
            True if successful, False otherwise

        Coments
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
                    if(self.assign_epochs(rsid=rsid, epochs=epochs, nexps=nexps)):
                        return True

        if(any_allowed is False):
            self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
        else:
            self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        if(self.veryverbose):
            print("rsid={r}: no epochs worked".format(r=rsid))
                
        return False

    def _set_equiv(self, rsids=None, iexps=None):
        """Set equivRobotID to reflect any compatible observations with this rsid

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids to update (default all currently assigned)

        iexps : ndarray of np.int32
            exposures to update (default all field exposures)

        Notes:
        -----

        This finds ALL entries with the same:

            catalogid
            fiberType
            lambda_eff
            delta_ra
            delta_dec

        and sets the robotIDs for all of them.
"""
        if(rsids is None):
            iassigned = np.where(self.assignments['assigned'])[0]
            rsids = self.targets['rsid'][iassigned]

        if(iexps is None):
            iexps = np.arange(self.field_cadence.nexp_total, dtype=int)

        for rsid in rsids:
            indx = self.rsid2indx[rsid]
            allindxs = self._equivindx[self._equivkey[indx]]

            if(len(allindxs) == 1):
                self.assignments['equivRobotID'][allindxs, :] = self.assignments['robotID'][allindxs, :]
                continue

            for iexp in iexps:
                robotIDs = self.assignments['robotID'][allindxs, iexp]
                robotIDs = robotIDs[robotIDs >= 0]
                if(len(robotIDs) > 0):
                    if(len(robotIDs) > 1):
                        print("Inconsistency: multiple equivalent rsids with robots assigned")
                        return
                    self.assignments['equivRobotID'][allindxs, iexp] = robotIDs[0]
                else:
                    self.assignments['equivRobotID'][allindxs, iexp] = -1

        return
            
    def _set_satisfied(self, rsids=None, reset_equiv=True):
        """Set satisfied flag based on assignments

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids to set (defaults to apply to all targets)

        reset_equiv : bool
            whether to reset equivRobotID before assessing (default True)

        Notes:
        -----

        'satisfied' means that the exposures obtained satisfy
        the cadence for an rsid and the right instrument.

        Uses equivRobotID to assess whether the conditions are
        satisfied.

        Only set reset_equiv=False if you have already just run
        _set_equiv() for these rsids (or all of them). Doing so 
        will save doing that twice.
"""
        if(reset_equiv):
            self._set_equiv(rsids=rsids)

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
            iexp = np.where(self.assignments['equivRobotID'][indx, :] >= 0)[0]
            if(self.targets['cadence'][indx] != ''):
                sat = clist.exposure_consistency(self.targets['cadence'][indx],
                                                 self.field_cadence.name, iexp)
                self.assignments['satisfied'][indx] = sat
            else:
                self.assignments['satisfied'][indx] = 0

        return

    def _set_count(self, rsids=None, reset_equiv=True):
        """Set exposure and epochs based on assignments

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids to set (defaults to apply to all targets)

        reset_equiv : bool
            whether to reset equivRobotID before assessing (default True)

        Notes:
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

    def _assign_one_by_one(self, rsids=None, check_satisfied=True):
        """Assign a set of targets to robots

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        check_satisfied : bool
            if True, do not try to reassign targets that are already satisfied

        Returns:
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes:
        -----

        Performs assigment in order given
"""
        success = np.zeros(len(rsids), dtype=bool)
        for i, rsid in enumerate(rsids):
            # Perform the assignment
            if((check_satisfied == False) |
               (self.assignments['satisfied'][self.rsid2indx[rsid]] == 0)):
                success[i] = self.assign_cadence(rsid=rsid)
        return(success)

    def assign_cadences(self, rsids=None, check_satisfied=True):
        """Assign a set of targets to robots

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        check_satisfied : bool
            if True, do not try to reassign targets that are already satisfied

        Returns:
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes:
        -----

        Sorts cadences by priority for assignment.
"""
        success = np.zeros(len(rsids), dtype=bool)
        indxs = np.array([self.rsid2indx[r] for r in rsids], dtype=np.int32)

        # Find single bright cases
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

        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            iassign = np.where((singlebright[indxs] == False) &
                               (multibright[indxs] == False) &
                               (multidark[indxs] == False) &
                               (self.assignments['satisfied'][indxs] == 0) &
                               (self.targets['priority'][indxs] == priority))[0]

            if(self.verbose):
                iall = np.where((self.assignments['satisfied'][indxs] == 0) &
                               (self.targets['priority'][indxs] == priority))[0]

                outstr = "fieldid {fid}: Includes cadences ".format(fid=self.fieldid)
                pcads = np.unique(self.targets['cadence'][indxs[iall]])
                for pcad in pcads:
                    outstr = outstr + pcad + " "
                print(outstr, flush=True)

                outstr = "fieldid {fid}: Includes cartons ".format(fid=self.fieldid)
                pcarts = np.unique(self.targets['carton'][indxs[iall]])
                for pcart in pcarts:
                    outstr = outstr + pcart + " "
                print(outstr, flush=True)
            
            if(len(iassign) > 0):
                if(self.verbose):
                    print("fieldid {fid}:  - {n} assigning one-by-one".format(n=len(iassign), fid=self.fieldid), flush=True)
                    
                success[iassign] = self._assign_one_by_one(rsids=rsids[iassign],
                                                           check_satisfied=check_satisfied)  
                    
                if(self.verbose):
                    print("fieldid {fid}:    (assigned {n})".format(n=success[iassign].sum(), fid=self.fieldid), flush=True)

            # It is always affordable to run through the single bright
            # cases twice. Why does it matter? Because when they displace
            # calibration targets on the first cycle, that can change the
            # collision situation on the second round. This is a 1% effect.
            # A second cycle might be worth doing for one-by-one cases, but
            # it is more expensive in that case in terms of run-time.
            for icycle in range(2):
                isinglebright = np.where(singlebright[indxs] &
                                         (self.assignments['satisfied'][indxs] == 0) &
                                         (self.targets['priority'][indxs] == priority))[0]
                if(len(isinglebright) > 0):
                    if(self.verbose):
                        print("fieldid {fid}:  - {n} assigning as single bright (cycle {i})".format(n=len(isinglebright), i=icycle, fid=self.fieldid), flush=True)
                    self._assign_singlebright(indxs=indxs[isinglebright])
                    success[isinglebright] = self.assignments['satisfied'][indxs[isinglebright]]

                    if(self.verbose):
                        print("fieldid {fid}:    (assigned {n})".format(n=success[isinglebright].sum(), fid=self.fieldid), flush=True)

            for icycle in range(1):
                imultibright = np.where(multibright[indxs] &
                                        (self.assignments['satisfied'][indxs] == 0) &
                                        (self.targets['priority'][indxs] == priority))[0]
                if(len(imultibright) > 0):
                    if(self.verbose):
                        print("fieldid {fid}:  - {n} assigning as multi bright (cycle {i})".format(n=len(imultibright), i=icycle, fid=self.fieldid), flush=True)
                    self._assign_multibright(indxs=indxs[imultibright])
                    success[imultibright] = self.assignments['satisfied'][indxs[imultibright]]

                    if(self.verbose):
                        print("fieldid {fid}:    (assigned {n})".format(n=success[imultibright].sum(), fid=self.fieldid), flush=True)

            for icycle in range(1):
                imultidark = np.where(multidark[indxs] &
                                      (self.assignments['satisfied'][indxs] == 0) &
                                      (self.targets['priority'][indxs] == priority))[0]
                if(len(imultidark) > 0):
                    if(self.verbose):
                        print("fieldid {fid}:  - {n} assigning as multi dark (cycle {i})".format(n=len(imultidark), i=icycle, fid=self.fieldid), flush=True)
                    self._assign_multidark(indxs=indxs[imultidark])
                    success[imultidark] = self.assignments['satisfied'][indxs[imultidark]]

                    if(self.verbose):
                        print("fieldid {fid}:    (assigned {n})".format(n=success[imultidark].sum(), fid=self.fieldid), flush=True)

            self._competing_targets = None

        return(success)

    def _assign_singlebright(self, indxs=None):
        """Assigns 1x1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign
"""
        rsids = self.targets['rsid'][indxs]
        iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)

        tdict = self.mastergrid.targetDict

        inotsat = np.where(self.assignments['satisfied'][indxs] == 0)[0]
        for rsid in rsids[inotsat]:
            indx = self.rsid2indx[rsid]
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            robotindx = np.array([self.robotID2indx[x] for x in robotIDs],
                                 dtype=int)
            hasApogee = self.robotHasApogee[robotindx]
            robotIDs = robotIDs[np.argsort(hasApogee)]

            succeed = False
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexps)
                self.set_assignment_status(status=s)
                cexps = s.assignable_exposures()
                if(len(cexps) > 0):
                    self.unassign_assignable(status=s, iexp=cexps[0])
                    self.assign_robot_exposure(robotID=robotID,
                                               rsid=rsid,
                                               iexp=cexps[0],
                                               reset_count=False,
                                               reset_satisfied=False,
                                               reset_has_spare=True)
                    succeed = True
                    break

            if(succeed is False):
                if(self.assignments['allowed'][indx].sum() == 0):
                    self.set_flag(rsid=rsid, flagname='NONE_ALLOWED')
                else:
                    self.set_flag(rsid=rsid, flagname='NO_AVAILABILITY')

        self._set_satisfied(rsids=rsids[inotsat])
        return

    def _assign_multibright(self, indxs=None):
        """Assigns nx1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign
"""
        rsids = self.targets['rsid'][indxs]
        iexpsall = np.arange(self.field_cadence.nexp_total, dtype=np.int32)

        tdict = self.mastergrid.targetDict

        inotsat = np.where(self.assignments['satisfied'][indxs] == 0)[0]
        for rsid in rsids[inotsat]:
            indx = self.rsid2indx[rsid]
            nexp_cadence = clist.cadences[self.targets['cadence'][indx]].nexp_total
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            robotindx = np.array([self.robotID2indx[x]
                                  for x in robotIDs], dtype=int)
            hasApogee = self.robotHasApogee[robotindx]
            robotIDs = robotIDs[np.argsort(hasApogee)]
            robotindx = robotindx[np.argsort(hasApogee)]

            statusDict = dict()
            expRobotIDs = [[] for _ in range(self.field_cadence.nexp_total)]
            nExpRobotIDs = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID, iexps=iexpsall)
                self.set_assignment_status(status=s)
                statusDict[robotID] = s
                for iexp in s.assignable_exposures():
                    expRobotIDs[iexp].append(robotID)
                    nExpRobotIDs[iexp] = nExpRobotIDs[iexp] + 1

            # if number of exposures with at least one free robot is high
            # enough, go ahead
            iexps = np.where(nExpRobotIDs > 0)[0]
            if(len(iexps) >= nexp_cadence):

                for iexp in iexps[0:nexp_cadence]:
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

        return

    def _assign_multidark(self, indxs=None):
        """Assigns nx1 dark targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign
"""
        rsids = self.targets['rsid'][indxs]
        iexpsall = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        ok, epochs_list = clist.cadence_consistency('_field_dark_single_1x1', self.field_cadence.name)
        iexpsall = np.array([self.field_cadence.epoch_indx[x[0]] +
                             np.arange(self.field_cadence.nexp[x[0]],
                                       dtype=int) for x in epochs_list],
                            dtype=int).flatten()
        tdict = self.mastergrid.targetDict

        inotsat = np.where(self.assignments['satisfied'][indxs] == 0)[0]
        for rsid in rsids[inotsat]:
            indx = self.rsid2indx[rsid]
            nexp_cadence = clist.cadences[self.targets['cadence'][indx]].nexp_total
            robotIDs = np.array(tdict[rsid].validRobotIDs, dtype=int)
            np.random.shuffle(robotIDs)
            hasApogee = self.robotHasApogee[robotIDs - 1]
            robotIDs = robotIDs[np.argsort(hasApogee)]

            statusDict = dict()
            expRobotIDs = [[] for _ in range(self.field_cadence.nexp_total)]
            nExpRobotIDs = np.zeros(self.field_cadence.nexp_total, dtype=np.int32)
            for robotID in robotIDs:
                s = AssignmentStatus(rsid=rsid, robotID=robotID,
                                     iexps=iexpsall)
                self.set_assignment_status(status=s)
                statusDict[robotID] = s
                for iexp in s.assignable_exposures():
                    expRobotIDs[iexp].append(robotID)
                    nExpRobotIDs[iexp] = nExpRobotIDs[iexp] + 1

                iexps = np.where(nExpRobotIDs > 0)[0]
                if(len(iexps) >= nexp_cadence):
                    break

            # if number of exposures with at least one free robot is high
            # enough, go ahead
            iexps = np.where(nExpRobotIDs > 0)[0]
            if(len(iexps) >= nexp_cadence):

                for iexp in iexps[0:nexp_cadence]:
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

        return

    def _assign_cp_model(self, rsids=None, robotIDs=None, check_collisions=True):
        """Assigns using CP-SAT to optimize number of targets

        Parameters
        ----------

        rsids : ndarray of np.int64
            [N] rsids of targets to assign

        robotIDs : ndarray of np.int32
            robots which are available to assign

        check_collisions : bool
            if set, check for collisions (default True)

        Returns:
        -------

        assignedRobotIDs : ndarray of np.int32
            [N] robots to assign to

        Notes:
        -----

        Doesn't yet limit to robotIDs input

        Plan to also allow certain rsids to be guaranteed
"""
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
        ww_list = [wwrt[y][x] for y in wwrt for x in wwrt[y]]

        # Constrain to use only one target per robot
        wwsum_robot = dict()
        for robotID in wwrt:
            rlist = [wwrt[robotID][c] for c in wwrt[robotID]]
            wwsum_robot[robotID] = cp_model.LinearExpr.Sum(rlist)
            model.Add(wwsum_robot[robotID] <= 1)

        # Constrain to use only one robot per target
        wwsum_target = dict()
        for rsid in wwtr:
            tlist = [wwtr[rsid][r] for r in wwtr[rsid]]
            wwsum_target[rsid] = cp_model.LinearExpr.Sum(tlist)
            model.Add(wwsum_target[rsid] <= 1)

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

        # Maximize the total sum
        wwsum_all = cp_model.LinearExpr.Sum(ww_list)
        model.Maximize(wwsum_all)

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

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        Returns:
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes:
        -----

        Assigns only the ones matching the field cadence
"""
        # Weeds out ones not in field cadence
        keep = np.ones(len(rsids), dtype=np.int32)
        for i, rsid in enumerate(rsids):
            if(self.targets['cadence'][self.rsid2indx[rsid]] != self.field_cadence.name):
                keep[i] = 0
        ikeep = np.where(keep)[0]
        rsids = rsids[ikeep]

        robotIDs = self._assign_cp_model(rsids=rsids)

        for rsid, robotID in zip(rsids, robotIDs):
            if(robotID >= 1):
                for epoch in range(self.field_cadence.nepochs):
                    nexp = self.field_cadence.nexp[epoch]
                    self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch, nexp=nexp)

        success = (robotIDs >= 1)
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

    def assign_calibrations(self):
        """Assign all calibration targets

        Notes
        -----

        This assigns all targets with 'category' set to one of 
        the required calibrations for this Field and with 'rsassign' 
        set to 1.

        It calls assign_cadences(), which will assign the targets
        in order of their priority value. The order of assignment is
        randomized within each priority value. The random seed is 
        set according to the fieldid.
"""
        if(self.nocalib):
            return

        if(self.verbose):
            print("fieldid {fid}: Assigning calibrations".format(fid=self.fieldid), flush=True)
        
        icalib = np.where(self._is_calibration &
                          (self.targets['rsassign'] != 0))[0]
        np.random.shuffle(icalib)
        self.assign_cadences(rsids=self.targets['rsid'][icalib])

        self._set_satisfied(rsids=self.targets['rsid'][icalib])
        self._set_count(reset_equiv=False)
        self._set_has_spare_calib()

        if(self.verbose):
            print("fieldid {fid}:   (done assigning calibrations)".format(fid=self.fieldid), flush=True)
        return

    def assign_science(self, rsassign=1):
        """Assign all science targets
        
        Parameters:
        ----------

        rsassign : int, np.int32
            value of rsassign for selecting targets (default 1)

        Notes
        -----

        This assigns all targets with 'category' set to 'science'
        and with 'rsassign' set to selected value

        It calls assign_cadences(), which will assign the targets
        in order of their priority value. The order of assignment is
        randomized within each priority value. The random seed is 
        set according to the fieldid.
"""
        if(self.verbose):
            print("fieldid {fid}: Assigning science".format(fid=self.fieldid), flush=True)

        print(self.validate())

        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['incadence']) &
                            (self.target_duplicated == 0) &
                            (self.targets['rsassign'] == rsassign))[0]
        np.random.seed(self.fieldid)
        random.seed(self.fieldid)
        np.random.shuffle(iscience)
        self.assign_cadences(rsids=self.targets['rsid'][iscience])

        self.decollide_unassigned()
        print(self.validate())

        self._set_satisfied(rsids=self.targets['rsid'][iscience])
        self._set_count(reset_equiv=False)

        if(self.verbose):
            print("fieldid {fid}:   (done assigning science)".format(fid=self.fieldid), flush=True)
        return

    def assign_science_and_calibs(self, coordinated_targets=None):
        """Assign all science targets and calibrations

        Parameters:
        ----------

        coordinated_targets : dict
            dictionary of coordinated targets (keys are rsids, values are bool)


        Notes:
        -----

        Does not try to assign any targets for which
        coordinated_targets[rsid] is True.
"""
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

        # Assign calibration to one exposure to determine achievable
        # requirements and then unassign
        if(self.verbose):
            print("fieldid {fieldid}: Assigning calibrations to determine achievable".format(fieldid=self.fieldid), flush=True)
        # Uniquify design modes here
        udesign_mode = np.unique(self.design_mode)
        for design_mode in udesign_mode:
            iexpall = np.where(self.design_mode == design_mode)[0]
            iexp = iexpall[0]
            for c in self.required_calibrations:
                icalib = np.where(self.targets['category'] != 'science')[0]
                np.random.shuffle(icalib)
                isort = np.argsort(self.targets['priority'][icalib])
                icalib = icalib[isort]
                for i in icalib:
                    self.assign_exposures(rsid=self.targets['rsid'][i],
                                          iexps=np.array([iexp],
                                                         dtype=np.int32))
                for c in self.required_calibrations:
                    if(self.calibrations[c][iexp] <
                       self.required_calibrations[c][iexp]):
                        self.achievable_calibrations[c][iexpall] = self.calibrations[c][iexp]
                    else:
                        self.achievable_calibrations[c][iexpall] = self.required_calibrations[c][iexp]

        iassigned = np.where(self.assignments['assigned'])[0]
        self.unassign(rsids=self.targets['rsid'][iassigned])

        inotscience = np.where(self.targets['category'] != 'science')[0]
        self.set_flag(rsid=self.targets['rsid'][inotscience],
                      flagname='NOT_SCIENCE')

        inotincadence = np.where(self.targets['incadence'] == 0)[0]
        self.set_flag(rsid=self.targets['rsid'][inotincadence],
                      flagname='NOT_INCADENCE')

        inotrsassign = np.where(self.targets['rsassign'] == 0)[0]
        self.set_flag(rsid=self.targets['rsid'][inotrsassign],
                      flagname='NOT_TO_ASSIGN')

        inotcovered = np.where(self.targets['within'] == 0)[0]
        self.set_flag(rsid=self.targets['rsid'][inotcovered],
                      flagname='NOT_COVERED')
        
        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['within']) &
                            (self.targets['incadence']) &
                            (self.target_duplicated == 0) &
                            (self.targets['rsassign'] != 0))[0]
        np.random.shuffle(iscience)

        assigned_exposure_calib = collections.OrderedDict()
        for c in self.required_calibrations:
            assigned_exposure_calib[c] = np.zeros(self.field_cadence.nexp_total,
                                                  dtype=bool)

        priorities = np.unique(self.targets['priority'][iscience])
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)

            # Assign science
            ipriority = np.where(self.targets['priority'][iscience] == priority)[0]
            ipriority = iscience[ipriority]
            self.assign_cadences(rsids=self.targets['rsid'][ipriority])

            # For exposures without assigned calibrations in
            # some category, assign the calibs just in those exposures
            if(self.verbose):
                print("Checking calibrations for each exposure")
            for c in self.required_calibrations:
                if(self.verbose):
                    print("   ... {c}".format(c=c))
                iexps = np.where(assigned_exposure_calib[c] == False)[0]
                icalib = np.where((self.targets['category'] == c) &
                                  (self.targets['rsassign'] != 0))[0]
                np.random.shuffle(icalib)
                for i in icalib:
                    self.assign_exposures(rsid=self.targets['rsid'][i], iexps=iexps)

            # If any exposure didn't get the achievable calibrations
            # in any category, remove all science targets, assign the
            # calibrations for those exposures and categories, and then
            # reassign the science. Mark exposure and category as having
            # assigned calibrations.
            shortfalls = collections.OrderedDict()
            anyshortfall = False
            for c in self.required_calibrations:
                shortfalls[c] = []
                for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
                    if(self.calibrations[c][iexp] < self.achievable_calibrations[c][iexp]):
                        shortfalls[c].append(iexp)
                        anyshortfall = True
            if(anyshortfall):
                if(self.verbose):
                    print("fieldid {fid}: Found a shortfall".format(fid=self.fieldid), flush=True)
                    print("fieldid {fid}: Unassigning science".format(fid=self.fieldid), flush=True)
                self.unassign(rsids=self.targets['rsid'][ipriority])
                if(self.verbose):
                    print("fieldid {fid}: Assigning calibs in shortfall exposures".format(fid=self.fieldid), flush=True)
                for c in self.required_calibrations:
                    if(self.verbose):
                        print("   ... {c}".format(c=c))
                    icalib = np.where((self.targets['category'] == c) &
                                      (self.targets['rsassign'] != 0))[0]
                    np.random.shuffle(icalib)
                    for i in icalib:
                        self.assign_exposures(rsid=self.targets['rsid'][i], iexps=iexps)
                    assigned_exposure_calib[c][iexps] = True
                self.assign_cadences(rsids=self.targets['rsid'][ipriority])

            # For all exposures that did not get assigned
            # calibrations, remove the calibration targets.
            # UNLESS this is the last priority
            if(priority != priorities[-1]):
                if(self.verbose):
                    print("fieldid {fid}: Unassigning temporary calibs".format(fid=self.fieldid), flush=True)
                for iexp in np.arange(self.field_cadence.nexp_total, dtype=np.int32):
                    for c in self.required_calibrations:
                        if(assigned_exposure_calib[c][iexp] == False):
                            icalib = np.where(self.targets['category'] == c)[0]
                            for i in icalib:
                                self.unassign_exposure(rsid=self.targets['rsid'][i],
                                                       iexp=iexp,
                                                       reset_satisfied=False,
                                                       reset_count=False,
                                                       reset_has_spare=False)
                            self.assignments['satisfied'][icalib] = 0
                self._set_count()
                if(self.nocalib is False):
                    self._set_has_spare_calib()
                            
        if(self.verbose):
            print("fieldid {fid}: Decolliding unassigned".format(fid=self.fieldid), flush=True)
        self.decollide_unassigned()

        self._set_satisfied()
        self._set_count(reset_equiv=False)
        if(self.nocalib is False):
            self._set_has_spare_calib()

        if(self.verbose):
            print("fieldid {fid}:   (done assigning science and calib)".format(fid=self.fieldid), flush=True)
        return

    def assign(self, coordinated_targets=None):
        """Assign all targets

        Parameters:
        ----------

        coordinated_targets : dict
            dictionary of coordinated targets (keys are rsids, values are bool)


        Notes:
        -----

        Does not try to assign any targets for which
        coordinated_targets[rsid] is True.
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
        self._set_count(reset_equiv=False)
        return

    def assess(self):
        """Assess the current results of assignment in field

        Parameters
        ----------

        Returns
        -------

        results : str
            String describing results
"""
        out = ""

        out = out + "Field cadence: {fc}\n".format(fc=self.field_cadence.name)

        if(self.nocalib is False):
            out = out + "\n"
            out = out + "Calibration targets:\n"
            for c in self.required_calibrations:
                tmp = " {c}:"
                out = out + tmp.format(c=c)
                for cn, rcn in zip(self.calibrations[c], self.required_calibrations[c]):
                    out = out + " {cn}/{rcn}".format(cn=cn, rcn=int(rcn))
                out = out + "\n"
        else:
            out = out + "No calibrations\n"

        out = out + "\n"
        out = out + "Science targets:\n"
        iboss = np.where((self.targets['fiberType'] == 'BOSS') &
                         (self.assignments['assigned']) &
                         (self._is_calibration == False))[0]
        out = out + " BOSS targets assigned: {n}\n".format(n=len(iboss))
        iapogee = np.where((self.targets['fiberType'] == 'APOGEE') &
                           (self.assignments['assigned']) &
                           (self._is_calibration == False))[0]
        out = out + " APOGEE targets assigned: {n}\n".format(n=len(iapogee))

        perepoch = np.zeros(self.field_cadence.nepochs, dtype=np.int32)
        out = out + " Targets per epoch:"
        for epoch in range(self.field_cadence.nepochs):
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            rids = np.where(((self.assignments['robotID'][:, iexpst:iexpnd] >= 1).sum(axis=1) > 0) &
                            (self._is_calibration == False))[0]
            perepoch[epoch] = len(rids)
            out = out + " {p}".format(p=perepoch[epoch])
        out = out + "\n"
        out = out + "\n"

        out = out + "Robots used:\n"
        hasApogee = np.array([self.mastergrid.robotDict[self.robotIDs[x]].hasApogee
                              for x in range(500)], dtype=bool)
        out = out + " BOSS-only:"
        for iexp in range(self.field_cadence.nexp_total):
            iused_boss = np.where((self._robot2indx[:, iexp] >= 0) &
                                  (hasApogee == False))[0]
            out = out + " {p}".format(p=len(iused_boss))
        out = out + "\n"
        out = out + " APOGEE-BOSS:"
        for iexp in range(self.field_cadence.nexp_total):
            iused_apogee = np.where((self._robot2indx[:, iexp] >= 0) &
                                    (hasApogee == True))[0]
            out = out + " {p}".format(p=len(iused_apogee))
        out = out + "\n"

        out = out + "\nSpare fibers per exposure (including spare calibs):\n"
        nboss_spare, napogee_spare = self.count_spares()
        out = out + " BOSS: "
        for iexp in range(self.field_cadence.nexp_total):
            out = out + " {p}".format(p=nboss_spare[iexp])
        out = out + "\n APOGEE: "
        for iexp in range(self.field_cadence.nexp_total):
            out = out + " {p}".format(p=napogee_spare[iexp])
        out = out + "\n"

        out = out + "\nCarton completion:\n"
        cartons = np.unique(self.targets['carton'])
        for carton in cartons:
            isscience = (self.targets['category'] == 'science')
            incarton = (self.targets['carton'] == carton)
            issatisfied = (self.assignments['satisfied'] > 0)
            icarton = np.where(incarton & isscience)[0]
            igot = np.where(incarton & issatisfied & isscience)[0]
            if(len(icarton) > 0):
                tmp = " {carton}: {ngot} of {ncarton}\n".format(carton=carton,
                                                                ngot=len(igot),
                                                                ncarton=len(icarton))
                out = out + tmp 
        out = out + "\n"

        return(out)

    def validate(self):
        """Validate a field solution

        Parameters:
        -------

        Returns:
        -------

        nproblems : int
            Number of problems discovered

        Comments:
        --------

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
            for c in self.required_calibrations:
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

        # Check that the number of calibrators has been tracked right
        if(self.nocalib is False):
            for c in self.required_calibrations:
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
            print("Unallowed exposures observed {n} times".format(n=len(inotallowed)))

        return(nproblems)

    def validate_cadences(self):
        """Validate the cadences

        Parameters:
        -------

        Returns:
        -------

        nproblems : int
            Number of problems discovered

        Comments:
        --------

        Prints nature of problems identified to stdout

        Checks that assigned targets got the right number and type of epochs.
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

    def count_spares(self):
        """Count spare fibers (accounting for spare calibrations)

        Returns:
        -------

        nboss_spare : np.int32
            Number of spare BOSS fibers (all, including APOGEE+BOSS robots)

        napogee_spare : np.int32
            Number of spare APOGEE fibers
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
            for calibration in self.calibrations:
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

        return(nboss_spare, napogee_spare)
        
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

        target_cadences = np.sort(np.unique(self.targets['cadence']))

        colors = ['black', 'green', 'blue', 'cyan', 'purple', 'red',
                  'magenta', 'grey']

        fig = plt.figure(figsize=(10 * 0.7, 7 * 0.7))
        axfig = fig.add_axes([0., 0., 0.7, 1.])
        axleg = fig.add_axes([0.71, 0., 0.26, 1.])

        itarget = np.where(self.targets['category'] == 'science')[0]
        axfig.scatter(self.targets['x'][itarget],
                      self.targets['y'][itarget], s=1, color='black',
                      label='Science targets', alpha=0.2)
        axleg.plot(self.targets['x'][itarget],
                   self.targets['y'][itarget], linewidth=4, alpha=0.2, color='black',
                   label='Science targets')

        itarget = np.where(self.targets['category'] != 'science')[0]
        axfig.scatter(self.targets['x'][itarget],
                      self.targets['y'][itarget], s=1, color='blue',
                      label='Calib targets', alpha=0.1)
        axleg.plot(self.targets['x'][itarget],
                   self.targets['y'][itarget], linewidth=4, alpha=0.2, color='blue',
                   label='Calib targets')

        if(self.assignments is not None):
            target_got = np.zeros(len(self.targets), dtype=np.int32)
            target_robotid = np.zeros(len(self.targets), dtype=np.int32)
            itarget = np.where(self.assignments['robotID'][:, iexp] >= 1)[0]
            target_got[itarget] = 1
            target_robotid[itarget] = self.assignments['robotID'][itarget, iexp]
            itarget = np.where(target_got > 0)[0]
            
            axfig.scatter(self.targets['x'][itarget],
                          self.targets['y'][itarget], s=3, color='black')

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

    Parameters:
    ----------

    filename : str
        if set, reads from file (ignores other inputs)

    fieldid : np.int32
        field ID number

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    pa : np.float32
        position angle of field (deg E of N)

    observatory : str
        observatory field observed from, 'apo' or 'lco' (default 'apo')

    field_cadence : str
        field cadence (default 'none')

    nocalib : bool
        if True, do not account for calibrations (default False)

    speedy : bool
        if True, perform approximations for speed up (default False)

    Attributes:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    pa : np.float32
        position angle of field (deg E of N)

    observatory : str
        observatory field observed from ('apo' or 'lco')

    field_cadence : Cadence object
        cadence associated with field

    collisionBuffer : float
        collision buffer for kaiju (in mm)

    radius : np.float32
        distance from racen, deccen to search for for targets (deg);
        set to 1.5 for observatory 'apo' and 0.95 for observatory 'lco'

    flagdict : Dict
        dictionary of assignment flag values

    rsid2indx : Dict
        dictionary linking rsid (key) to index of targets and assignments arrays.
        (values). E.g. targets['rsid'][f.rsid2indx[rsid]] == rsid

    targets : ndarray
        array of targets, including 'ra', 'dec', 'x', 'y', 'within',
        'priority', 'category', 'cadence', 'catalogid', 'rsid', 'fiberType'

    assignments : ndarray or None
        [len(targets)] array with 'assigned', 'satisfied', 
          'robotID', 'rsflags', 'fiberType'
        for each target; set to None prior to definition of field_cadence

    required_calibrations : OrderedDict
        dictionary with numbers of required calibration sources specified
        for 'sky_boss', 'standard_boss', 'sky_apogee', 'standard_apogee'

    calibrations : OrderedDict
        dictionary of lists with numbers of calibration sources assigned
        for each epoch for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    _robot2indx : ndarray of int32 or None
        [nrobots, nexp_total] array of indices into targets

    _robotnexp : ndarray of int32 or None
        [nrobots, nepochs] array of number of exposures available per epoch

    _is_calibration : ndarray of bool
        [len(targets)] list of whether the target is a calibration target

    Notes:
    -----

    This class internally assumes that robotIDs are sequential integers starting at 1.

    Relative to Field, this class behaves as follows: 
     * nocalib is set True, so calibrations are skipped, which allows a
       a substantial simplification.
     * Any 1x1 bright cadences are performed en masse
     * Any cadences consistent with 12x1 bright cadences are performed en masse
"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=2.,
                 fieldid=1, verbose=False):
        super().__init__(filename=filename, racen=racen, pa=pa,
                         observatory=observatory, field_cadence=field_cadence,
                         collisionBuffer=collisionBuffer, fieldid=fieldid,
                         verbose=verbose,
                         nocalib=True, nocollide=True, allgrids=False)
        return

    def assign_cadences(self, rsids=None, check_satisfied=True):
        """Assign a set of targets to robots

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        check_satisfied : bool
            if True, do not try to reassign targets that are already satisfied

        Returns:
        --------

        success : ndarray of bool
            True if successful, False otherwise

        Notes:
        -----

        Sorts cadences by priority for assignment.
"""
        success = np.zeros(len(rsids), dtype=bool)
        indxs = np.array([self.rsid2indx[r] for r in rsids], dtype=np.int32)

        # Find single bright cases
        if(self.verbose):
            print("fieldid {fid}: Finding single bright cases".format(fid=self.fieldid), flush=True)
        cadences = np.unique(self.targets['cadence'][indxs])
        singlebright = np.zeros(len(self.targets), dtype=bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_1x1',
                                         return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                singlebright[indxs[icad]] = True

        # Find multiple single exposure bright cases
        if(self.verbose):
            print("fieldid {fid}: Finding multi bright cases".format(fid=self.fieldid), flush=True)
        multibright = np.zeros(len(self.targets), dtype=bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_12x1',
                                         return_solutions=False)):
                icad = np.where((self.targets['cadence'][indxs] == cadence) &
                                (singlebright[indxs] == False))[0]
                multibright[indxs[icad]] = True

        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)

            if(self.verbose):
                print("fieldid {fid}: Set competing targets".format(fid=self.fieldid), flush=True)
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            if(self.verbose):
                iall = np.where((self.assignments['satisfied'][indxs] == 0) &
                                (self.targets['priority'][indxs] == priority))[0]
                outstr = "fieldid {fid}: Includes cadences ".format(fid=self.fieldid)
                pcads = np.unique(self.targets['cadence'][indxs[iall]])
                for pcad in pcads:
                    outstr = outstr + pcad + " "
                print(outstr, flush=True)
                    
                outstr = "fieldid {fid}: Includes cartons ".format(fid=self.fieldid)
                pcarts = np.unique(self.targets['carton'][indxs[iall]])
                for pcart in pcarts:
                    outstr = outstr + pcart + " "
                print(outstr, flush=True)

            # Since we are in speedy mode, skip the single-bright and
            # multibright cases
            iassign = np.where((singlebright[indxs] == False) &
                               (multibright[indxs] == False) &
                               (self.assignments['satisfied'][indxs] == 0) &
                               (self.targets['priority'][indxs] == priority))[0]

            if(len(iassign) > 0):

                if(self.verbose):
                    print("fieldid {fid}: Assign one by ones".format(fid=self.fieldid), flush=True)
                success[iassign] = self._assign_one_by_one(rsids=rsids[iassign],
                                                           check_satisfied=check_satisfied)

            imultibright = np.where(multibright[indxs] &
                                    (self.assignments['satisfied'][indxs] == 0) &
                                    (self.targets['priority'][indxs] == priority))[0]
            if(len(imultibright) > 0):
                if(self.verbose):
                    print("fieldid {fid}: Assign multibrights".format(fid=self.fieldid), flush=True)
                self._assign_multibright(indxs=indxs[imultibright])

            isinglebright = np.where(singlebright[indxs] &
                                     (self.assignments['satisfied'][indxs] == 0) &
                                     (self.targets['priority'][indxs] == priority))[0]
            if(len(isinglebright) > 0):
                if(self.verbose):
                    print("fieldid {fid}: Assign singlebrights".format(fid=self.fieldid), flush=True)
                self._assign_singlebright(indxs=indxs[isinglebright])

            self._competing_targets = None
        return(success)
