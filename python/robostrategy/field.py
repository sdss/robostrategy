#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import re
import random
import numpy as np
import fitsio
import collections
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import ortools.sat.python.cp_model as cp_model
import roboscheduler.cadence
import kaiju
import kaiju.robotGrid
import robostrategy.obstime as obstime
import coordio.time
import coordio.utils
import mugatu.designmode
    

# alpha and beta lengths for plotting
_alphaLen = 7.4
_betaLen = 15

# Make these to save some time later
onetrue = np.ones(1, dtype=np.bool)
onefalse = np.zeros(1, dtype=np.bool)


# intersection of lists
def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))


# Type for targets array
targets_dtype = np.dtype([('ra', np.float64),
                          ('dec', np.float64),
                          ('epoch', np.float32),
                          ('pmra', np.float32),
                          ('pmdec', np.float32),
                          ('parallax', np.float32),
                          ('lambda_eff', np.float32),
                          ('delta_ra', np.float64),
                          ('delta_dec', np.float64),
                          ('magnitude', np.float32, 7),
                          ('x', np.float64),
                          ('y', np.float64),
                          ('within', np.int32),
                          ('incadence', np.int32),
                          ('priority', np.int32),
                          ('value', np.float32),
                          ('program', np.unicode_, 30),
                          ('carton', np.unicode_, 50),
                          ('category', np.unicode_, 30),
                          ('cadence', np.unicode_, 30),
                          ('fiberType', np.unicode_, 10),
                          ('catalogid', np.int64),
                          ('carton_to_target_pk', np.int64),
                          ('rsid', np.int64),
                          ('target_pk', np.int64),
                          ('rsassign', np.int32)])

# Dictionary defining meaning of flags
_flagdict = {'CADENCE_INCONSISTENT': 1,
             'NOT_COVERED_BY_APOGEE': 2,
             'NOT_COVERED_BY_BOSS': 4,
             'ASSIGNED_IN_PREVIOUS_FIELD': 8,
             'COLLISION': 16}

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
clist = roboscheduler.cadence.CadenceList()

class Field(object):
    """Field class

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
    
    collisionBuffer : float or np.float32
        collision buffer to send to kaiju in mm (default 2)

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

    design_mode : list of str
        keys to DesignModeDict for each exposure

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

    robotgrids : list of RobotGrid objects
        robotGrids associated with each exposure

    targets : ndarray
        array of targets, including 'ra', 'dec', 'x', 'y', 'within',
        'priority', 'category', 'cadence', 'catalogid', 'rsid', 'fiberType'

    assignments : ndarray or None
        [len(targets)] array with 'assigned', 'satisfied', 
          'robotID', 'rsflags', 'fiberType'
        for each target; set to None prior to definition of field_cadence

    designModeDict : dict of DesignMode objects
        possible design modes

    required_calibrations : OrderedDict
        dictionary with numbers of required calibration sources specified
        for 'sky_boss', 'standard_boss', 'sky_apogee', 'standard_apogee'

    calibrations : OrderedDict
        dictionary of lists with numbers of calibration sources assigned
        for each epoch for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    obstime : coordio Time object
        nominal time of observation to use for calculating x/y

    _robot2indx : ndarray of int32 or None
        [nrobots, nexp_total] array of indices into targets

    _robotnexp : ndarray of int32 or None
        [nrobots, nepochs] array of number of exposures available per epoch

    _is_calibration : ndarray of np.bool
        [len(targets)] list of whether the target is a calibration target

    nocalib : bool
        if True, do not account for calibrations (default False)

    allgrids : bool
        if True, keep track of all robotgrids (default True); if False
        automatically sets nocollide to True

    nocollide : bool
        if True,  do not check collisions (default False)

    verbose : bool
        if True, issue a lot of output statements

    Notes:
    -----

    This class internally assumes that robotIDs are sequential integers starting at 0.
"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=2.,
                 fieldid=1, allgrids=True, nocalib=False, nocollide=False,
                 verbose=False):
        self.verbose = verbose
        self.fieldid = fieldid
        self.nocalib = nocalib
        self.nocollide = nocollide
        self.allgrids = allgrids
        if(self.allgrids is False):
            self.nocollide = True
        if(self.nocollide):
            self.allgrids = False
        if(self.allgrids):
            self.robotgrids = []
        else:
            self.robotgrids = None
        self.assignments = None
        self.rsid2indx = dict()
        self.targets = np.zeros(0, dtype=targets_dtype)
        self.target_duplicated = np.zeros(0, dtype=np.int32)
        self._is_calibration = np.zeros(0, dtype=np.bool)
        self._calibration_index = np.zeros(1, dtype=np.bool)
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
            self.collisionBuffer = collisionBuffer
            self.mastergrid = self._robotGrid()
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
            self.set_field_cadence(field_cadence)       
        self._set_radius()
        self.flagdict = _flagdict
        self._competing_targets = None
        self.methods = dict()
        self.methods['assign_epochs'] = 'first'
        self._add_dummy_cadences()
        return

    def _add_dummy_cadences(self): 
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
        return

    def fromfits(self, filename=None):
        duf, hdr = fitsio.read(filename, ext=0, header=True)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        self.collisionBuffer = hdr['CBUFFER']
        if(('NOCALIB' in hdr) & (self.nocalib == False)):
            self.nocalib = np.bool(hdr['NOCALIB'])
        self.mastergrid = self._robotGrid()
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
                            self.required_calibrations[hdr[name]] = np.array([np.int32(x) for x in hdr[num].split()])
                        else:
                            self.required_calibrations[hdr[name]] = np.zeros(0, dtype=np.int32)
            self.calibrations = collections.OrderedDict()
            for n in self.required_calibrations:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
        self.set_field_cadence(field_cadence)
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
            if(self.field_cadence.nexp_total == 1):
                iassigned = np.where(assignments['robotID'])
                for itarget in iassigned[0]:
                    self.assign_robot_exposure(robotID=assignments['robotID'][itarget],
                                               rsid=targets['rsid'][itarget],
                                               iexp=0, reset_satisfied=False,
                                               reset_has_spare=False)
            else:
                iassigned = np.where(assignments['robotID'] >= 0)
                for itarget, iexp in zip(iassigned[0], iassigned[1]):
                    self.assign_robot_exposure(robotID=assignments['robotID'][itarget, iexp],
                                               rsid=targets['rsid'][itarget],
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False)
            self._set_has_spare_calib()
            self._set_satisfied()
            self.decollide_unassigned()
        return

    def clear_assignments(self):
        if(self.assignments is not None):
            iassigned = np.where(self.assignments['assigned'])[0]
            for i in iassigned:
                self.unassign(self.targets['rsid'][i], reset_assigned=False,
                              reset_satisfied=False)
            self.assignments['assigned'] = 0
            self.assignments['satisfied'] = 0
        return

    def clear_field_cadence(self):
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
        """Return a RobotGridFilledHex instance, with all robots at home"""
        rg = kaiju.robotGrid.RobotGridFilledHex(collisionBuffer=self.collisionBuffer)
        for k in rg.robotDict.keys():
            rg.homeRobot(k)
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
                                               ('allowed', np.int32,
                                                (self.field_cadence.nepochs,)),
                                               ('robotID', np.int32,
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
                if((type(obsmode_pk) == list) |
                   (type(obsmode_pk) == np.ndarray)):
                    self.design_mode = np.array(obsmode_pk)
                else:
                    self.design_mode = np.array([obsmode_pk])
            else:
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
                        self.required_calibrations[c] = [self.designModeDict[d].n_stds_min['BOSS'] for d in dms]
                    elif(c == 'standard_apogee'):
                        self.required_calibrations[c] = [self.designModeDict[d].n_stds_min['APOGEE'] for d in dms] 
                    elif(c == 'sky_boss'):
                        self.required_calibrations[c] = [self.designModeDict[d].n_skies_min['BOSS'] for d in dms]
                    elif(c == 'sky_apogee'):
                        self.required_calibrations[c] = [self.designModeDict[d].n_skies_min['APOGEE'] for d in dms]
                for c in self.calibrations:
                    self.calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                                    dtype=np.int32)
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
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)])
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
        indxs = np.array([self.rsid2indx[r] for r in self._arrayify(rsid)])
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

    def radec2xy(self, ra=None, dec=None, epoch=None, pmra=None,
                 pmdec=None, fiberType=None):
        if(isinstance(fiberType, str)):
            wavename = fiberType.capitalize()
        else:
            wavename = np.array([x.capitalize() for x in fiberType])
        epoch = self.obstime.jd
        x, y, warn, ha, pa = coordio.utils.radec2wokxy(ra, dec, epoch,
                                                       wavename,
                                                       self.racen, self.deccen,
                                                       self.pa,
                                                       self.observatory.upper(),
                                                       self.obstime.jd,
                                                       pmra=pmra,
                                                       pmdec=pmdec)
        return(x, y)

    def xy2radec(self, x=None, y=None, fiberType=None):
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
        target_allowed = np.ones(len(targets), dtype=np.bool)
        for fiberType in fiberTypes:
            for category in categories:
                icurr = np.where((targets['fiberType'] == fiberType) &
                                 (target_category == category))[0]
                mags = targets['magnitude'][icurr, :]
                if(category == 'science'):
                    limits = designMode.bright_limit_targets[fiberType]
                if(category == 'standard'):
                    limits = designMode.stds_mags[fiberType]
                ok = np.ones(len(icurr), dtype=np.bool)
                for i in np.arange(limits.shape[0], dtype=np.int32):
                    #if((limits[i, 0] != - 999.) |
                    #   (limits[i, 1] != - 999.)):
                        #ok = ok & ((np.isnan(mags[:, i]) == False) &
                        #           (mags[:, i] != 0.) &
                        #           (mags[:, i] != 99.9) &
                        #           (mags[:, i] != 999.) &
                        #           (mags[:, i] != - 999.) &
                        #           (mags[:, i] != - 9999.))
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
        for target in targets:
            if(target['fiberType'] == 'APOGEE'):
                fiberType = kaiju.ApogeeFiber
            else:
                fiberType = kaiju.BossFiber
            robotgrid.addTarget(targetID=target['rsid'], x=target['x'],
                                y=target['y'],
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
            assignments['target_skybrightness'] = -1.
        else:
            for n in self.assignments_dtype.names:
                listns = ['robotID', 'target_skybrightness', 'field_skybrightness']
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
        targets['x'], targets['y'] = self.radec2xy(ra=targets['ra'],
                                                   dec=targets['dec'],
                                                   fiberType=targets['fiberType'])

        # Add targets to robotGrids
        self._targets_to_robotgrid(targets=targets,
                                   robotgrid=self.mastergrid)

        # Determine if within
        self.masterTargetDict = self.mastergrid.targetDict
        for itarget, rsid in enumerate(targets['rsid']):
            t = self.masterTargetDict[rsid]
            targets['within'][itarget] = len(t.validRobotIDs) > 0

        # Create internal look-up of whether it is a calibration target
        _is_calibration = np.zeros(len(targets), dtype=np.bool)
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

        if(assignments is not None):
            self.assignments = np.append(self.assignments, assignments, axis=0)
            self._set_satisfied()

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
        hdr = dict()
        hdr['RACEN'] = self.racen
        hdr['DECCEN'] = self.deccen
        hdr['OBS'] = self.observatory
        hdr['PA'] = self.pa
        if(self.field_cadence is not None):
            hdr['FCADENCE'] = self.field_cadence.name
            hdr['NEXP'] = self.field_cadence.nexp_total
            hdr['DESMODE'] = ' '.join(list(self.design_mode[self.field_cadence.epochs]))
        else:
            hdr['FCADENCE'] = 'none'
        hdr['CBUFFER'] = self.collisionBuffer
        hdr['NOCALIB'] = self.nocalib
        if(self.nocalib is False):
            for indx, rc in enumerate(self.required_calibrations):
                name = 'RCNAME{indx}'.format(indx=indx)
                num = 'RCNUM{indx}'.format(indx=indx)
                hdr[name] = rc
                hdr[num] = ' '.join([str(n) for n in self.required_calibrations[rc]])
        fitsio.write(filename, None, header=hdr, clobber=True)
        fitsio.write(filename, self.targets, extname='TARGET')
        if(self.assignments is not None):
            fitsio.write(filename, self.assignments, extname='ASSIGN')
        dmarr = None
        for i, d in enumerate(self.designModeDict):
            arr = self.designModeDict[d].toarray()
            if(dmarr is None):
                dmarr = np.zeros(len(self.designModeDict), dtype=arr.dtype)
            dmarr[i] = arr
        fitsio.write(filename, dmarr, extname='DESMODE')
        return

    def collide_robot_exposure(self, rsid=None, robotID=None, iexp=None):
        """Check if assigning an rsid to a robot would cause collision

        Parameters:
        ----------

        rsid : np.int64
            rsid (for checking collisions)

        robotID : np.int64
            robotID to check

        iexp : int or np.int32
            exposure to check

        Returns:
        -------

        collide : bool
            True if it causes a collision, False if not
"""
        if((not self.allgrids) |
           (self.nocollide)):
            return False
        rg = self.robotgrids[iexp]
        return rg.wouldCollideWithAssigned(robotID, rsid)[0]

    def _set_has_spare_calib(self):
        """Set _has_spare for each exposure"""
        self._has_spare_calib = np.zeros((len(self.required_calibrations) + 1,
                                          self.field_cadence.nexp_total),
                                         dtype=np.bool)
        for icategory, category in enumerate(self.required_calibrations):
            self._has_spare_calib[icategory + 1, :] = (self.calibrations[category] >
                                                       self.required_calibrations[category])
        return

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

        free : ndarray of bool
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

        if(self.nocalib is True):
            # Checks obvious case that this epoch doesn't have enough exposures
            available = False
            cnexp = self.field_cadence.nexp[epoch]
            if(cnexp < nexp):
                return available, np.zeros(cnexp, dtype=np.bool)

            # Now optimize case where nexp=1
            if(cnexp == 1):
                iexp = self.field_cadence.epoch_indx[epoch]
                robot2indx = self._robot2indx[robotID, iexp]
                free = robot2indx < 0

                if(free is False):
                    return False, onefalse
            
                if(self.nocalib is False):
                    if((free == False) & (isspare[iexp] == False)):
                        free = self._has_spare_calib[self._calibration_index[robot2indx + 1], iexp]

                if(free & (rsid is not None)):
                    free = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                       iexp=iexp) == False

                if(free):
                    return True, onetrue
                else:
                    return False, onefalse

        if(self.nocalib is False):

            if(isspare is None):
                isspare = np.zeros(self.field_cadence.nexp_total, dtype=np.bool)

            # Consider exposures for this epoch
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]

            # Get indices of assigned targets to this robot
            # and make Boolean arrays of which are assigned and not
            iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
            robot2indx = self._robot2indx[robotID, iexps]
            free = robot2indx < 0
            hasspare = self._has_spare_calib[self._calibration_index[robot2indx + 1], iexps]

            free = free | (hasspare & (isspare[iexpst:iexpnd] == False))

        else:
            # Consider exposures for this epoch
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            iexps = np.arange(iexpst, iexpnd, dtype=np.int32)
            robot2indx = self._robot2indx[robotID, iexps]
            free = robot2indx < 0

        if(rsid is not None):
            for ifree in np.where(free)[0]:
                free[ifree] = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                          iexp=iexpst + ifree) == False

        # Count this exposure as available if there are enough free exposures.
        # Package list of which calibrations are considered spare.
        nfree = free.sum()
        available = nfree >= nexp

        return available, free

    def available_robot_exposures(self, rsid=None, robotID=None, iscalib=False):
        """Return available robot exposures for an rsid

        Parameters:
        ----------

        rsid : np.int64
            rsid (optional; will check for collisions)

        robotID : np.int64
            robotID to check

        iscalib : bool
            True if this is a calibration target

        Returns:
        -------

        available : ndarray of bool
            for each exposure, is it available or not?

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
        # Check if this is an "extra" calibration target in any exposures;
        # i.e. not necessary so should not bump any other calibration targets
        isspare = False
        if(self.nocalib is False):
            if(iscalib):
                cat = self.targets['category'][self.rsid2indx[rsid]]
                isspare = self.calibrations[cat] > self.required_calibrations[cat]

        # Get indices of assigned targets to this robot
        # and make Boolean arrays of which are assigned and not
        robot2indx = self._robot2indx[robotID, :]
        free = robot2indx < 0

        # Check if the assigned robots are to "spare" calibration targets.
        # These may be bumped if necessary (but won't be if the target under
        # consideration is, itself, a "spare" calibration targets. This logic
        # is not so straightforward but avoids expensive for loops.
        if(self.nocalib is False):
            iassigned = np.where(free == False)[0]
            icalib = iassigned[np.where(self._is_calibration[robot2indx[iassigned]])[0]]
            if(len(icalib) > 0):
                spare = np.zeros(self.field_cadence.nexp_total, dtype=np.bool)
                category = self.targets['category'][robot2indx[icalib]]
                calibspare = np.array([self.calibrations[category[i]][icalib[i]] >
                                       self.required_calibrations[category[i]][icalib[i]]
                                       for i in range(len(category))], dtype=np.bool)
                spare[icalib] = calibspare
                spare = spare & (isspare == False)

                # Now classify exposures as "free" or not (free if unassigned OR assigned to
                # a calibration target that may be bumped).
                free = free | spare

        # Check allowability
        free = free & (self.assignments['allowed'][self.field_cadence.epochs])

        # Now (if there is an actual target under consideration) check for collisions.
        for ifree in np.where(free)[0]:
            free[ifree] = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                      iexp=ifree) == False

        return(free)

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None, nexp=None,
                           reset_satisfied=True, reset_has_spare=True,
                           free=None):
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

        free : ndarray of bool
            True for each exposure in epoch which should be treated as free

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
            (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix based on this assignment
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
        indxs = self._robot2indx[robotID, iexpst:iexpnd]
        assigned = indxs >= 0
        if(free is None):
            available = np.zeros(0, dtype=np.int32)
            for iexp in np.arange(iexpst, iexpnd):
                ok = True
                if(self.nocalib is False):
                    isspare = self._has_spare_calib[self._calibration_index[self.rsid2indx[rsid] + 1], iexp]
                    ok = False
                    currindx = indxs[iexp - iexpst]
                    if((currindx >= 0) & (isspare == False)):
                        if(self._has_spare_calib[self._calibration_index[currindx + 1],
                                                 iexp]):
                            ok = True
                    if(currindx < 0):
                        ok = True
                if(ok & (self.collide_robot_exposure(rsid=rsid,
                                                     robotID=robotID,
                                                     iexp=iexp) != True)):
                    available = np.append(available, iexp)
        else:
            available = iexpst + np.where(free)[0]

        # Bomb if there aren't enough available
        if(len(available) < nexp):
            return False

        # Now actually assign (to first available exposures)
        for iexp in available[0:nexp]:
            if(assigned[iexp - iexpst]):
                currindx = self._robot2indx[robotID, iexp]
                currrsid = self.targets['rsid'][currindx]
                self.unassign_exposure(rsid=currrsid,
                                       iexp=iexp,
                                       reset_satisfied=False,
                                       reset_has_spare=False)
            self.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp,
                                       reset_satisfied=False,
                                       reset_has_spare=False)

        if(reset_satisfied):
            indx = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][indx]
            self._set_satisfied(catalogids=[catalogid])

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
            robotIDs = np.array(self.masterTargetDict[rsid].validRobotIDs, dtype=np.int32)
            self._competing_targets[robotIDs] += 1
        return

    def assign_robot_exposure(self, robotID=None, rsid=None, iexp=None,
                              reset_satisfied=True, reset_has_spare=True):
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

        Returns:
        --------

        success : bool
            True if successful, False otherwise
"""
        itarget = self.rsid2indx[rsid]

        if(self.assignments['robotID'][itarget, iexp] >= 0):
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=True,
                                   reset_satisfied=True, reset_has_spare=True)

        if(self._robot2indx[robotID, iexp] >= 0):
            rsid_unassign = self.targets['rsid'][self._robot2indx[robotID,
                                                                  iexp]]
            self.unassign_exposure(rsid=rsid_unassign, iexp=iexp,
                                   reset_assigned=True, reset_satisfied=True,
                                   reset_has_spare=True)

        self.assignments['robotID'][itarget, iexp] = robotID
        self._robot2indx[robotID, iexp] = itarget
        epoch = self.field_cadence.epochs[iexp]
        self._robotnexp[robotID, epoch] = self._robotnexp[robotID, epoch] - 1
        if(self.targets['category'][itarget] == 'science'):
            self._robotnexp_max[robotID, epoch] = self._robotnexp_max[robotID, epoch] - 1
        self.assignments['assigned'][itarget] = 1

        # If this is a calibration target, update calibration target tracker
        if(self.nocalib is False):
            if(self._is_calibration[itarget]):
                category = self.targets['category'][itarget]
                self.calibrations[category][iexp] = self.calibrations[category][iexp] + 1

        if(self.allgrids):
            rg = self.robotgrids[iexp]
            rg.assignRobot2Target(robotID, rsid)

        if(reset_satisfied):
            indx = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][indx]
            self._set_satisfied(catalogids=[catalogid])

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return

    def _set_assigned(self, itarget=None):
        if(itarget is None):
            print("Must specify a target.")
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        return

    def unassign_exposure(self, rsid=None, iexp=None, reset_assigned=True,
                          reset_satisfied=True, reset_has_spare=True):
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

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix after unassignment
            (default True)
"""
        itarget = self.rsid2indx[rsid]
        category = self.targets['category'][itarget]
        robotID = self.assignments['robotID'][itarget, iexp]
        if(robotID >= 0):
            if(self.allgrids):
                rg = self.robotgrids[iexp]
                rg.unassignTarget(rsid)
            self.assignments['robotID'][itarget, iexp] = -1
            self._robot2indx[robotID, iexp] = -1
            epoch = self.field_cadence.epochs[iexp]
            self._robotnexp[robotID, epoch] = self._robotnexp[robotID, epoch] + 1
            if(self.targets['category'][itarget] == 'science'):
                self._robotnexp_max[robotID, epoch] = self._robotnexp_max[robotID, epoch] + 1
            if(self.nocalib is False):
                if(self._is_calibration[itarget]):
                    self.calibrations[category][iexp] = self.calibrations[category][iexp] - 1

        if(reset_assigned == True):
            self._set_assigned(itarget=itarget)

        if(reset_satisfied):
            catalogid = self.targets['catalogid'][itarget]
            self._set_satisfied(catalogids=[catalogid])

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return

    def unassign_epoch(self, rsid=None, epoch=None, reset_assigned=True,
                       reset_satisfied=True, reset_has_spare=True):
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
        for iexp in np.arange(iexpst, iexpnd):
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=False,
                                   reset_satisfied=False, reset_has_spare=False)

        if(reset_assigned):
            self._set_assigned(itarget=self.rsid2indx[rsid])

        if(reset_satisfied):
            itarget = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][itarget]
            self._set_satisfied(catalogids=[catalogid])

        if(reset_has_spare & (self.nocalib is False)):
            self._set_has_spare_calib()

        return 0

    def unassign(self, rsid=None, reset_assigned=True, reset_satisfied=True,
                 reset_has_spare=True):
        """Unassign an rsid entirely

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        reset_assigned : bool
            if True, resets assigned flag for this rsid (default True)

        reset_satisfied : bool
            if True, resets satified flag for this rsid (default True)

        reset_has_spare : bool
            if True, reset the '_has_spare' matrix after unassignment
            (default True)
"""
        for epoch in range(self.field_cadence.nepochs):
            self.unassign_epoch(rsid=rsid, epoch=epoch, reset_assigned=False,
                                reset_satisfied=False, reset_has_spare=False)

        if(reset_assigned):
            self._set_assigned(itarget=self.rsid2indx[rsid])

        if(reset_satisfied):
            itarget = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][itarget]
            self._set_satisfied(catalogids=[catalogid])

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
                         iscalib=False, first=False, strict=False):
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

            'freeExposures' : list of lists of ndarrays
                for each epoch, and each available robotID, ndarray of bool
                regarding whether each exposure is "free"
"""
        if(epochs is None):
            epochs = np.arange(self.field_cadence.nepochs, dtype=np.int32)
        if(nexps is None):
            nexps = np.ones(len(epochs))

        nAvailableRobotIDs = np.zeros(len(epochs), dtype=np.int32)
        availableRobotIDs = [[]] * len(epochs)
        frees = [[]] * len(epochs)

        bad = (self.assignments['allowed'][self.rsid2indx[rsid], epochs] == 0)
        if(bad.max() > 0):
            available = dict()
            available['available'] = False
            available['nAvailableRobotIDs'] = nAvailableRobotIDs
            available['availableRobotIDs'] = availableRobotIDs
            available['freeExposures'] = frees
            return(available)

        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs, dtype=np.int32)

        if(len(validRobotIDs) == 0):
            available = dict()
            available['available'] = False
            available['nAvailableRobotIDs'] = nAvailableRobotIDs
            available['availableRobotIDs'] = availableRobotIDs
            available['freeExposures'] = frees
            return(available)

        # If we are going to require ALL epochs can be fulfilled, we
        # can punt early
        if(strict):
            if(len(epochs) == 1):
                if(self._robotnexp_max[validRobotIDs, epochs[0]].max() < nexps[0]):
                    available = dict()
                    available['available'] = False
                    available['nAvailableRobotIDs'] = nAvailableRobotIDs
                    available['availableRobotIDs'] = availableRobotIDs
                    available['freeExposures'] = frees
                    return(available)
            else:
                for iepoch, epoch in enumerate(epochs):
                    if(self._robotnexp_max[validRobotIDs, epoch].max() < nexps[iepoch]):
                        available = dict()
                        available['available'] = False
                        available['nAvailableRobotIDs'] = nAvailableRobotIDs
                        available['availableRobotIDs'] = availableRobotIDs
                        available['freeExposures'] = frees
                        return(available)

        validRobotIDs.sort()
        if(self.nocalib is False):
            isspare = self._has_spare_calib[self._calibration_index[self.rsid2indx[rsid] + 1], :]
        else:
            isspare = False

        for iepoch, epoch in enumerate(epochs):
            nexp = nexps[iepoch]
            arlist = []
            flist = []
            ican = np.where(self._robotnexp_max[validRobotIDs, epoch] >= nexp)[0]
            for robotID in validRobotIDs[ican]:
                ok, free = self.available_robot_epoch(rsid=rsid,
                                                      robotID=robotID,
                                                      epoch=epoch,
                                                      nexp=nexp,
                                                      isspare=isspare)

                if(ok):
                    arlist.append(robotID)
                    flist.append(free)
                    # If this robot was good, then let's just return it
                    if(first):
                        break
            availableRobotIDs[iepoch] = arlist
            nAvailableRobotIDs[iepoch] = len(arlist)
            frees[iepoch] = flist

        available = dict()
        available['available'] = nAvailableRobotIDs.min() > 0
        available['nAvailableRobotIDs'] = nAvailableRobotIDs
        available['availableRobotIDs'] = availableRobotIDs
        available['freeExposures'] = frees
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
        iscalib = False
        if(self.nocalib is False):
            if(self.targets['category'][self.rsid2indx[rsid]] in self.required_calibrations):
                iscalib = True

        if(self.methods['assign_epochs'] == 'first'):
            first = True
        else:
            first = False

        available = self.available_epochs(rsid=rsid, epochs=epochs,
                                          nexps=nexps, iscalib=iscalib,
                                          strict=True, first=first)
        availableRobotIDs = available['availableRobotIDs']
        freeExposures = available['freeExposures']

        # Check if there are robots available
        nRobotIDs = np.array([len(x) for x in availableRobotIDs])
        if(nRobotIDs.min() < 1):
            return False

        # Assign to each epoch
        for iepoch, epoch in enumerate(epochs):
            currRobotIDs = np.array(availableRobotIDs[iepoch], dtype=np.int32)
            if(self.methods['assign_epochs'] == 'first'):
                irobot = 0
            if(self.methods['assign_epochs'] == 'fewestcompeting'):
                irobot = np.argmin(self._competing_targets[currRobotIDs])
            robotID = currRobotIDs[irobot]
            free = freeExposures[iepoch][irobot]
            nexp = nexps[iepoch]

            self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch,
                                    nexp=nexp, free=free,
                                    reset_satisfied=False,
                                    reset_has_spare=False)

        indx = self.rsid2indx[rsid]
        catalogid = self.targets['catalogid'][indx]
        self._set_satisfied(catalogids=[catalogid])
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
            return False

        # Check for all potential epochs whether they can accomodate at
        # least the minimum number of exposures; if not we can eliminate
        # them.
        if(len(epochs_list) > 100):
            epochs = np.arange(self.field_cadence.nepochs, dtype=np.int32)
            nexps = (np.zeros(self.field_cadence.nepochs, dtype=np.int32) + 
                     clist.cadences[target_cadence].nexp.min())
        else:
            epochs = np.unique(np.array([e for es in epochs_list for e in es]))
            nexps = np.zeros(len(epochs), dtype=np.int32) + np.array([ne for nes in nexps_list for ne in nes], dtype=np.int32).min()
            
        available = self.available_epochs(rsid, epochs=epochs, nexps=nexps,
                                          iscalib=False, strict=False, first=True)
        ibad = np.where(available['nAvailableRobotIDs'] == 0)[0]
        epoch_bad = np.zeros(self.field_cadence.nepochs, dtype=np.bool)
        epoch_bad[epochs[ibad]] = True
        
        for indx, epochs in enumerate(epochs_list):
            if(epoch_bad[epochs].max() == False):
                nexps = nexps_list[indx]
                if(self.assign_epochs(rsid=rsid, epochs=epochs, nexps=nexps)):
                    return True
                
        return False

    def _set_satisfied(self, catalogids=None):
        """Set satisfied flag based on assignments

        Parameters:
        ----------

        catalogids : ndarray of np.int64
            catalogids to set (defaults to apply to all targets)

        Notes:
        -----

        'satisfied' means that the exposures obtained for this catalog ID satisfy
        the cadence for an rsid.
"""
        if(catalogids is None):
            catalogids = self._unique_catalogids

        for catalogid in catalogids:
            # Check for other instances of this catalogid, and whether
            # assignments have satisfied their cadence
            icats = np.where((self.targets['catalogid'] == catalogid) &
                             (self.targets['incadence']))[0]
            if(len(icats) > 0):
                gotexp = (self.assignments['robotID'][icats, :] >= 0).sum(axis=0)
                iexp = np.where(gotexp > 0)[0]
                self.assignments['satisfied'][icats] = 0
                for icat in icats:
                    other_cadence_name = self.targets['cadence'][icat]

                    fits = clist.exposure_consistency(other_cadence_name,
                                                      self.field_cadence.name,
                                                      iexp)
                    if(fits):
                        self.assignments['satisfied'][icat] = 1

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
        success = np.zeros(len(rsids), dtype=np.bool)
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
        success = np.zeros(len(rsids), dtype=np.bool)
        indxs = np.array([self.rsid2indx[r] for r in rsids], dtype=np.int32)

        # Find single bright cases
        cadences = np.unique(self.targets['cadence'][indxs])
        singlebright = np.zeros(len(self.targets), dtype=np.bool)
        multibright = np.zeros(len(self.targets), dtype=np.bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_1x1',
                                         return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                singlebright[indxs[icad]] = True
            elif(clist.cadence_consistency(cadence, '_field_single_12x1',
                                           return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                multibright[indxs[icad]] = True

        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            if(self.verbose):
                print("fieldid {fid}: Assigning priority {p}".format(p=priority, fid=self.fieldid), flush=True)
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            iassign = np.where((singlebright[indxs] == False) &
                               (multibright[indxs] == False) &
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
        epochs = self.field_cadence.epochs

        tdict = self.mastergrid.targetDict

        for rsid in rsids:
            indx = self.rsid2indx[rsid]
            robotIDs = np.array(tdict[rsid].validRobotIDs)

            if((len(robotIDs) > 0) &
               (self.assignments['satisfied'][indx] == 0)):

                gotem = False
                for epoch, iexp in zip(epochs, iexps):
                    if(self.assignments['allowed'][indx, epoch]):
                        # Only check possibly free robots
                        robot2indx = self._robot2indx[robotIDs, iexp]
                        if(self.nocalib is False):
                            spare = self._has_spare_calib[self._calibration_index[robot2indx + 1], iexp]
                            ifree = np.where((robot2indx < 0) | (spare == True))[0]
                        else:
                            ifree = np.where(robot2indx < 0)[0]
                        for robotID in robotIDs[ifree]:
                            collided = self.collide_robot_exposure(rsid=rsid,
                                                                   robotID=robotID,
                                                                   iexp=iexp)
                            if(collided == False):
                                self.assign_robot_exposure(robotID=robotID,
                                                           rsid=rsid,
                                                           iexp=iexp,
                                                           reset_satisfied=True,
                                                           reset_has_spare=True)
                                gotem = True
                                break

                    if(gotem):
                        break

        return

    def _assign_multibright(self, indxs=None):
        """Assigns nx1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign
"""
        rsids = self.targets['rsid'][indxs]
        iexps = np.arange(self.field_cadence.nexp_total, dtype=np.int32)
        epochs = self.field_cadence.epochs

        tdict = self.mastergrid.targetDict

        for rsid in rsids:
            indx = self.rsid2indx[rsid]
            nexp = clist.cadences[self.targets['cadence'][indx]].nexp_total
            robotIDs = np.array(tdict[rsid].validRobotIDs)

            if((len(robotIDs) > 0) &
               (self.assignments['satisfied'][indx] == 0)):

                iexpgot = []
                for epoch, iexp in zip(epochs, iexps):
                    # Only check possibly free robots
                    if(self.assignments['allowed'][indx, epoch]):
                        robot2indx = self._robot2indx[robotIDs, iexp]
                        if(self.nocalib is False):
                            spare = self._has_spare_calib[self._calibration_index[robot2indx + 1], iexp]
                            ifree = np.where((robot2indx < 0) | (spare == True))[0]
                        else:
                            ifree = np.where(robot2indx < 0)[0]
                        for robotID in robotIDs[ifree]:
                            collided = self.collide_robot_exposure(rsid=rsid,
                                                                   robotID=robotID,
                                                                   iexp=iexp)
                            if(collided == False):
                                iexpgot.append(iexp)
                                break

                    if(len(iexpgot) >= nexp):
                        break
                    
                if(len(iexpgot) >= nexp):
                    for iexp in iexpgot[0:nexp]:
                        self.assign_robot_exposure(robotID=robotID,
                                                   rsid=rsid,
                                                   iexp=iexp,
                                                   reset_satisfied=True,
                                                   reset_has_spare=True)

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
            if(robotID >= 0):
                for epoch in range(self.field_cadence.nepochs):
                    nexp = self.field_cadence.nexp[epoch]
                    self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch, nexp=nexp)

        success = robotIDs >= 0
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
        self.assign_cadences(rsids=self.targets['rsid'][icalib])
        self._set_has_spare_calib()
        if(self.verbose):
            print("fieldid {fid}:   (done assigning calibrations)".format(fid=self.fieldid), flush=True)
        return

    def assign_science(self):
        """Assign all science targets

        Notes
        -----

        This assigns all targets with 'category' set to 'science'
        and with 'rsassign' set to 1.

        It calls assign_cadences(), which will assign the targets
        in order of their priority value. The order of assignment is
        randomized within each priority value. The random seed is 
        set according to the fieldid.
"""
        if(self.verbose):
            print("fieldid {fid}: Assigning science".format(fid=self.fieldid), flush=True)

        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['incadence']) &
                            (self.target_duplicated == 0) &
                            (self.targets['rsassign'] != 0))[0]
        np.random.seed(self.fieldid)
        random.seed(self.fieldid)
        np.random.shuffle(iscience)
        self.assign_cadences(rsids=self.targets['rsid'][iscience])

        if(self.verbose):
            print("fieldid {fid}:   (done assigning science)".format(fid=self.fieldid), flush=True)
        return

    def assign(self, coordinated_targets=None):
        """Assign all targets

        Parameters:
        ----------

        coordinated_targets : dict
            dictionary of coordinated targets (keys are rsids, values are bool)


        Notes:
        -----

        Does not true to assign any targets for which
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
            rids = np.where(((self.assignments['robotID'][:, iexpst:iexpnd] >= 0).sum(axis=1) > 0) &
                            (self._is_calibration == False))[0]
            perepoch[epoch] = len(rids)
            out = out + " {p}".format(p=perepoch[epoch])
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
                        if(robotID >= 0):
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
                if(robotID >= 0):
                    if(itarget != self._robot2indx[robotID, iexp]):
                        rsid = self.targets['rsid'][itarget]
                        print("assignments['robotID'] for rsid={rsid} and iexp={iexp} is robotID={robotID}, but _robot2indx[robotID, iexp] is {i}, meaning rsid={rsidtwo}".format(rsid=rsid, iexp=iexp, robotID=robotID, i=self._robot2indx[robotID, iexp], rsidtwo=self.targets['rsid'][self._robot2indx[robotID, iexp]]))
                        nproblems = nproblems + 1

        # Check that _robot2indx and _robotnexp agree with each other
        for robotID in self.mastergrid.robotDict:
            nn = self.field_cadence.nexp.copy()
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=np.int32):
                if(self._robot2indx[robotID, iexp] >= 0):
                    epoch = self.field_cadence.epochs[iexp]
                    nn[epoch] = nn[epoch] - 1
            for epoch in np.arange(self.field_cadence.nepochs, dtype=np.int32):
                if(nn[epoch] != self._robotnexp[robotID, epoch]):
                    print("_robotnexp for robotID={robotID} and epoch={epoch} is {rnexp}, but should be {nn}".format(robotID=robotID, epoch=epoch, rnexp=self._robotnexp[robotID, epoch], nn=nn[epoch]))
                    nproblems = nproblems + 1

        for robotID in self.mastergrid.robotDict:
            for iexp in np.arange(self.field_cadence.nexp_total,
                                  dtype=np.int32):
                itarget = self._robot2indx[robotID, iexp]
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
                    if(rg.robotDict[robotID].isAssigned()):
                        tid = rg.robotDict[robotID].assignedTargetID
                        itarget = self.rsid2indx[tid]
                    else:
                        itarget = -1
                    if(self._robot2indx[robotID, iexp] != itarget):
                        print("robotID={robotID} iexp={iexp} : expected {i1} in _robot2indx got {i2}".format(robotID=robotID, iexp=iexp, i1=itarget, i2=self._robot2indx[robotID, iexp]))
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
                nexp = (assignment['robotID'][iexpst:iexpnd] >= 0).sum()
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
                        if(robotID >= 0):
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
        xr = robot.xPos
        yr = robot.yPos
        xa = xr + _alphaLen * np.cos(robot.alpha / 180. * np.pi)
        ya = yr + _alphaLen * np.sin(robot.alpha / 180. * np.pi)
        xb = xa + _betaLen * np.cos((robot.alpha + robot.beta) / 180. * np.pi)
        yb = ya + _betaLen * np.sin((robot.alpha + robot.beta) / 180. * np.pi)
        ax.plot(np.array([xr, xa]), np.array([yr, ya]), color=color, alpha=0.5)
        ax.plot(np.array([xa, xb]), np.array([ya, yb]), color=color, linewidth=3)

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

        if(self.assignments is not None):
            target_got = np.zeros(len(self.targets), dtype=np.int32)
            target_robotid = np.zeros(len(self.targets), dtype=np.int32)
            itarget = np.where(self.assignments['robotID'][:, iexp] >= 0)[0]
            target_got[itarget] = 1
            target_robotid[itarget] = self.assignments['robotID'][itarget, iexp]
            for indx in np.arange(len(target_cadences)):
                itarget = np.where((target_got > 0) & (self.targets['cadence'] ==
                                                       target_cadences[indx]))[0]

                axfig.scatter(self.targets['x'][itarget],
                              self.targets['y'][itarget], s=4)

                icolor = indx % len(colors)
                for i in itarget:
                    robot = self.robotgrids[iexp].robotDict[target_robotid[i]]
                    self._plot_robot(robot, color=colors[icolor], ax=axfig)

        for indx in np.arange(len(target_cadences)):
            itarget = np.where(self.targets['cadence'] == target_cadences[indx])[0]
            icolor = indx % len(colors)
            axfig.scatter(self.targets['x'][itarget],
                          self.targets['y'][itarget], s=2, color=colors[icolor],
                          label=target_cadences[indx])
            axleg.plot(self.targets['x'][itarget],
                       self.targets['y'][itarget], linewidth=4, color=colors[icolor],
                       label=target_cadences[indx])

        xcen = np.array([self.robotgrids[iexp].robotDict[r].xPos
                         for r in self.robotgrids[iexp].robotDict],
                        dtype=np.float32)
        ycen = np.array([self.robotgrids[iexp].robotDict[r].yPos
                         for r in self.robotgrids[iexp].robotDict],
                        dtype=np.float32)
        robotid = np.array([str(r)
                            for r in self.robotgrids[iexp].robotDict])
        axfig.scatter(xcen, ycen, s=6, color='grey', label='Used robot')
        axleg.plot(xcen, ycen, linewidth=4, color='grey', label='Used robot')

        if(robotID):
            for cx, cy, cr in zip(xcen, ycen, robotid):
                plt.text(cx, cy, cr, color='grey', fontsize=8,
                         clip_on=True)

        if(catalogid):
            for cx, cy, ct in zip(self.target_x, self.target_y,
                                  self.target_catalogid):
                plt.text(cx, cy, ct, fontsize=8, clip_on=True)

        used = (self._robot2indx[iexp, :] >= 0)

        inot = np.where(used == False)[0]
        axfig.scatter(xcen[inot], ycen[inot], s=20, color='grey',
                      label='Unused robot')
        axleg.plot(xcen[inot], ycen[inot], color='grey',
                   linewidth=4, label='Unused robot')
        for i in robotid[inot]:
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

    _is_calibration : ndarray of np.bool
        [len(targets)] list of whether the target is a calibration target

    Notes:
    -----

    This class internally assumes that robotIDs are sequential integers starting at 0.

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
        success = np.zeros(len(rsids), dtype=np.bool)
        indxs = np.array([self.rsid2indx[r] for r in rsids], dtype=np.int32)

        # Find single bright cases
        cadences = np.unique(self.targets['cadence'][indxs])
        singlebright = np.zeros(len(self.targets), dtype=np.bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_1x1',
                                         return_solutions=False)):
                icad = np.where(self.targets['cadence'][indxs] == cadence)[0]
                singlebright[indxs[icad]] = True

        # Find multiple single exposure bright cases
        multibright = np.zeros(len(self.targets), dtype=np.bool)
        for cadence in cadences:
            if(clist.cadence_consistency(cadence, '_field_single_12x1',
                                         return_solutions=False)):
                icad = np.where((self.targets['cadence'][indxs] == cadence) &
                                (singlebright[indxs] == False))[0]
                multibright[indxs[icad]] = True

        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            # Since we are in speedy mode, skip the single-bright and
            # multibright cases
            iassign = np.where((singlebright[indxs] == False) &
                               (multibright[indxs] == False) &
                               (self.targets['priority'][indxs] == priority))[0]

            success[iassign] = self._assign_one_by_one(rsids=rsids[iassign],
                                                       check_satisfied=check_satisfied)

            imultibright = np.where(multibright[indxs] &
                                    (self.assignments['satisfied'][indxs] == 0) &
                                    (self.targets['priority'][indxs] == priority))[0]
            if(len(imultibright) > 0):
                self._assign_multibright(indxs=indxs[imultibright])

            isinglebright = np.where(singlebright[indxs] &
                                     (self.assignments['satisfied'][indxs] == 0) &
                                     (self.targets['priority'][indxs] == priority))[0]
            if(len(isinglebright) > 0):
                self._assign_singlebright(indxs=indxs[isinglebright])

            self._competing_targets = None

        return(success)

    def _assign_singlebright(self, indxs=None):
        """Assigns 1x1 bright targets en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign

        Notes
        -----

        First, uniquifies on catalogid so as not to repeat itself.

        Second, loops through robots, and assigns its unused exposures to
        singlebrights.

        Ignores collisions!
"""
        inindx = np.zeros(len(self.targets), dtype=np.bool)
        inindx[indxs] = 1

        # Find unique set of catalogid and create index into
        catalogids, iunique = np.unique(self.targets['catalogid'][indxs],
                                        return_index=True)
        indxs = indxs[iunique]
        rsids = set(self.targets['rsid'][indxs])

        for robotID in self.mastergrid.robotDict:
            r = self.mastergrid.robotDict[robotID]
            robot_rsids = set(r.validTargetIDs)
            curr_rsids = np.array(list(robot_rsids.intersection(rsids)),
                                  dtype=np.int64)
            if(len(curr_rsids) > 0):
                np.random.shuffle(curr_rsids)
                ifree = np.where(self._robot2indx[robotID, :] < 0)[0]
                if(len(ifree) >= len(curr_rsids)):
                    ifree = ifree[0:len(curr_rsids)]
                for icurr, iexp in enumerate(ifree):
                    self.assign_robot_exposure(robotID=robotID,
                                               rsid=curr_rsids[icurr],
                                               iexp=iexp,
                                               reset_satisfied=False,
                                               reset_has_spare=False)
                    indx = self.rsid2indx[curr_rsids[icurr]]
                    icat = np.where((self.targets['catalogid'] == 
                                     self.targets['catalogid'][indx])
                                    & (inindx))[0]
                    self.assignments['satisfied'][icat] = 1
                    rsids.remove(curr_rsids[icurr])

        self.decollide_unassigned()

    def _assign_multibright(self, indxs=None):
        """Assigns nx1 bright targets (no constraints) en masse

        Parameters
        ----------

        indxs : ndarray of np.int32
            indices into self.targets of targets to assign

        Notes
        -----

        Loops through robots, and assigns its unused exposures to
        multibrights (does not take advantage of using different 
        robots at different epochs).

        Ignores collisions! Doesn't account for spare calib fibers.
"""
        rsids = set(self.targets['rsid'][indxs])

        for robotID in self.mastergrid.robotDict:
            r = self.mastergrid.robotDict[robotID]
            robot_rsids = set(r.validTargetIDs)
            curr_rsids = np.array(list(robot_rsids.intersection(rsids)),
                                  dtype=np.int64)
            if(len(curr_rsids) > 0):
                np.random.shuffle(curr_rsids)
                ifree = np.where(self._robot2indx[robotID, :] < 0)[0]
                icurr = 0
                irsid = 0
                while((icurr < len(ifree)) & (irsid < len(curr_rsids))):
                    curr_rsid = curr_rsids[irsid]
                    indx = self.rsid2indx[curr_rsid]
                    nexp = clist.cadences[self.targets['cadence'][indx]].nepochs
                    if(icurr + nexp < len(ifree)):
                        for i in np.arange(nexp, dtype=np.int32):
                            self.assign_robot_exposure(robotID=robotID,
                                                       rsid=curr_rsid,
                                                       iexp=ifree[icurr],
                                                       reset_satisfied=False,
                                                       reset_has_spare=False)
                            icurr = icurr + 1
                        self._set_satisfied(catalogids=[self.targets['catalogid'][indx]])
                        rsids.remove(curr_rsid)
                    irsid = irsid + 1

        self.decollide_unassigned()
