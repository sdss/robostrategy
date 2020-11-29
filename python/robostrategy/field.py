#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import re
import numpy as np
import fitsio
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import mplcursors

import roboscheduler.cadence as cadence
import kaiju
import kaiju.robotGrid

# alpha and beta lengths for plotting
_alphaLen = 7.4
_betaLen = 15

# Type for targets array
targets_dtype = np.dtype([('ra', np.float64),
                          ('dec', np.float64),
                          ('x', np.float64),
                          ('y', np.float64),
                          ('within', np.int32),
                          ('priority', np.int32),
                          ('category', np.unicode_, 30),
                          ('cadence', np.unicode_, 30),
                          ('fiberType', np.unicode_, 10),
                          ('catalogid', np.int64),
                          ('rsid', np.int64),
                          ('target_pk', np.int64)])

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
clist = cadence.CadenceList()


class Field(object):
    """Field class

    Parameters:
    ----------

    filename : str
        if set, reads from file (ignores other inputs)

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    pa : np.float32
        position angle of field (deg E of N)

    observatory : str
        observatory field observed from, 'apo' or 'lco' (default 'apo')

    field_cadence : str
        field cadence (default 'none'; if not set explicitly, you need
        to call Field.set_field_cadence() subsequently)

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

    robotgrids : list of RobotGrid objects
        robotGrids associated with each exposure

    targets : ndarray
        array of targets, including 'ra', 'dec', 'x', 'y', 'within',
        'priority', 'category', 'cadence', 'catalogid', 'rsid', 'fiberType'

    assignments : ndarray or None
        [len(targets)] array of 'assigned', 'robotID', 'rsflags', 'fiberType'
        for each target; set to None prior to definition of field_cadence

    required_calibrations : Dict
        dictionary with numbers of required calibration sources specified
        for 'sky_boss', 'standard_boss', 'sky_apogee', 'standard_apogee'

    calibrations : Dict
        dictionary of lists with numbers of calibration sources assigned
        for each epoch for 'sky_boss', 'standard_boss', 'sky_apogee',
        'standard_apogee'

    _robot2indx : ndarray of int32 or None
        [nrobots, nexp_total] array of indices into targets

    _is_calibration : ndarray of np.bool
        [len(targets)] list of whether the target is a calibration target
"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=2.):
        self.robotgrids = []
        self.assignments = None
        self.rsid2indx = dict()
        self.targets = np.zeros(0, dtype=targets_dtype)
        if(filename is not None):
            self.fromfits(filename=filename)
        else:  
            self.racen = racen
            self.deccen = deccen
            self.pa = pa
            self.observatory = observatory
            self.collisionBuffer = collisionBuffer
            self.required_calibrations = dict()
            self.required_calibrations['sky_boss'] = 80
            self.required_calibrations['standard_boss'] = 80
            self.required_calibrations['sky_apogee'] = 30
            self.required_calibrations['standard_apogee'] = 20
            self.calibrations = dict()
            for n in self.required_calibrations:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
            self._set_field_cadence(field_cadence)
        self._set_radius()
        self.flagdict = _flagdict
        return

    def fromfits(self, filename=None):
        duf, hdr = fitsio.read(filename, ext=0, header=True)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        self.collisionBuffer = hdr['CBUFFER']
        field_cadence = hdr['FCADENCE']
        self.required_calibrations = dict()
        for name in hdr:
            m = re.match('^RCNAME([0-9]*)$', name)
            if(m is not None):
                num = 'RCNUM{d}'.format(d=m.group(1))
                if(num in hdr):
                    self.required_calibrations[hdr[name]] = np.int32(hdr[num])
        self.calibrations = dict()
        for n in self.required_calibrations:
            self.calibrations[n] = np.zeros(0, dtype=np.int32)
        self._set_field_cadence(field_cadence)
        targets = fitsio.read(filename, ext=1)
        self.targets_fromarray(targets)
        self.assignments = fitsio.read(filename, ext=2)
        for assignment, target in zip(self.assignments, self.targets):
            for iexp in range(self.field_cadence.nexp_total):
                if(assignment['robotID'][iexp] >= 0):
                    self.assign_robot_exposure(robotID=assignment['robotID'][iexp],
                                               rsid=target['rsid'], iexp=iexp)
        self.decollide_unassigned()
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
            rg.robotDict[k].setAlphaBeta(0., 180.)
        return(rg)

    def _set_radius(self):
        """Set radius limit in deg depending on observatory"""
        if(self.observatory == 'apo'):
            self.radius = 1.5
        if(self.observatory == 'lco'):
            self.radius = 0.95
        return

    def _set_field_cadence(self, field_cadence='none'):
        """Set the field cadence, and set up robotgrids and assignments output"""
        if(len(self.robotgrids) > 0):
            print("Cannot reset field_cadence")
            return
        if(field_cadence != 'none'):
            self.field_cadence = clist.cadences[field_cadence]
            for i in range(self.field_cadence.nexp_total):
                self.robotgrids.append(self._robotGrid())
            self._robot2indx = np.zeros((len(self.robotgrids[0].robotDict),
                                         self.field_cadence.nexp_total),
                                        dtype=np.int32) - 1
        else:
            self.field_cadence = None
            self.robotgrids = []

        self.assignments_dtype = np.dtype([('assigned', np.int32),
                                           ('robotID', np.int32,
                                            (self.field_cadence.nexp_total,)),
                                           ('fiberType', np.unicode_, 10),
                                           ('rsflags', np.int32)])
        self.assignments = np.zeros(0, dtype=self.assignments_dtype)
        self._is_calibration = np.zeros(0, dtype=np.bool)
        for c in self.calibrations:
            self.calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                            dtype=np.int32)
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

    # Temporary method to deal with x&y to ra&dec conversion
    def radec2xy(self, ra=None, dec=None):
        # Yikes!
        if(self.observatory == 'apo'):
            scale = 218.
        if(self.observatory == 'lco'):
            scale = 329.

        # From Meeus Ch. 17
        deccen_rad = self.deccen * np.pi / 180.
        racen_rad = self.racen * np.pi / 180.
        dec_rad = dec * np.pi / 180.
        ra_rad = ra * np.pi / 180.
        x = (np.cos(deccen_rad) * np.sin(dec_rad) -
             np.sin(deccen_rad) * np.cos(dec_rad) *
             np.cos(ra_rad - racen_rad))
        y = np.cos(dec_rad) * np.sin(ra_rad - racen_rad)
        z = (np.sin(deccen_rad) * np.sin(dec_rad) +
             np.cos(deccen_rad) * np.cos(dec_rad) *
             np.cos(ra_rad - racen_rad))
        d_rad = np.arctan2(np.sqrt(x**2 + y**2), z)

        pay = np.sin(ra_rad - racen_rad)
        pax = (np.cos(deccen_rad) * np.tan(dec_rad) -
               np.sin(deccen_rad) * np.cos(ra_rad - racen_rad))
        pa_rad = np.arctan2(pay, pax)  # I think E of N?

        pa_rad = pa_rad - self.pa * np.pi / 180.

        x = d_rad * 180. / np.pi * scale * np.sin(pa_rad)
        y = d_rad * 180. / np.pi * scale * np.cos(pa_rad)

        return(x, y)

    # Temporary method to deal with x&y to ra&dec conversion
    def _min_xy_diff(self, radec, xt, yt):
        x, y = self.radec2xy(ra=radec[0], dec=radec[1])
        resid2 = (x - xt)**2 + (y - yt)**2
        return(resid2)

    # Temporary method to deal with x&y to ra&dec conversion
    def xy2radec(self, x=None, y=None):
        # This doesn't handle poles well
        # Yikes!
        if(self.observatory == 'apo'):
            scale = 218.
        if(self.observatory == 'lco'):
            scale = 329.
        xa = self._arrayify(x, dtype=np.float64)
        ya = self._arrayify(y, dtype=np.float64)
        rast = self.racen - xa / scale / np.cos(self.deccen * np.pi / 180.)
        decst = self.deccen + ya / scale
        ra = np.zeros(len(xa), dtype=np.float64)
        dec = np.zeros(len(xa), dtype=np.float64)
        for i in np.arange(len(xa)):
            res = optimize.minimize(self._min_xy_diff, [rast[i], decst[i]],
                                    (xa[i], ya[i]))
            ra[i] = res.x[0]
            dec[i] = res.x[1]
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

    def targets_fromarray(self, target_array=None):
        """Read in targets from ndarray

        Parameters
        ----------

        target_array : ndarray
            array with target information
"""
        # Read in
        targets = np.zeros(len(target_array), dtype=targets_dtype)
        for n in targets.dtype.names:
            if(n in target_array.dtype.names):
                targets[n] = target_array[n]

        # Connect rsid with index of list
        for itarget, t in enumerate(targets):
            if(t['rsid'] in self.rsid2indx.keys()):
                print("Cannot replace identical rsid={rsid}. Will not add array.".format(rsid=t['rsid']))
                return
            else:
                self.rsid2indx[t['rsid']] = itarget

        # Set fiber type
        targets['fiberType'] = np.array(['APOGEE' if clist.cadences[c].instrument == cadence.Instrument.ApogeeInstrument
                                         else 'BOSS'
                                         for c in targets['cadence']])

        # Convert ra/dec to x/y
        targets['x'], targets['y'] = self.radec2xy(ra=targets['ra'],
                                                   dec=targets['dec'])

        # Add targets to robotGrids
        for rg in self.robotgrids:
            for target in targets:
                if(target['fiberType'] == 'APOGEE'):
                    fiberType = kaiju.ApogeeFiber
                else:
                    fiberType = kaiju.BossFiber
                rg.addTarget(targetID=target['rsid'], x=target['x'],
                             y=target['y'],
                             priority=np.float64(target['priority']),
                             fiberType=fiberType)

        # Determine if within
        self.masterTargetDict = self.robotgrids[0].targetDict
        for itarget, rsid in enumerate(targets['rsid']):
            t = self.masterTargetDict[rsid]
            targets['within'][itarget] = len(t.validRobotIDs) > 0

        # Set up outputs
        assignments = np.zeros(len(targets),
                               dtype=self.assignments_dtype)
        assignments['fiberType'] = targets['fiberType']
        assignments['robotID'] = -1

        _is_calibration = np.zeros(len(targets), dtype=np.bool)
        for i, t in enumerate(targets):
            _is_calibration[i] = t['category'] in self.required_calibrations

        self.targets = np.append(self.targets, targets)
        self.assignments = np.append(self.assignments, assignments, axis=0)
        self._is_calibration = np.append(self._is_calibration,
                                         _is_calibration)

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
        hdr['FCADENCE'] = self.field_cadence.name
        hdr['CBUFFER'] = self.collisionBuffer
        for indx, rc in enumerate(self.required_calibrations):
            name = 'RCNAME{indx}'.format(indx=indx)
            num = 'RCNUM{indx}'.format(indx=indx)
            hdr[name] = rc
            hdr[num] = self.required_calibrations[rc]
        fitsio.write(filename, None, header=hdr, clobber=True)
        fitsio.write(filename, self.targets)
        fitsio.write(filename, self.assignments)
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
        rg = self.robotgrids[iexp]
        return rg.wouldCollideWithAssigned(robotID, rsid)

    def available_robot_epoch(self, rsid=None,
                              robotID=None, epoch=None, nexp=None, iscalib=False):
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

        iscalib : bool
            True if this is a calibration target

        Returns:
        -------

        available : bool
            is it available or not?

        competing_targets : ndarray of np.int64
            competing target IDs

        spare_calibration : np.int64
            spare calibration targetID if that is what is assigned

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

        # Checks obvious case that this epoch doesn't have enough exposures
        available = False
        competing_targets = np.zeros(0, dtype=np.int64)
        spare_calibrations = np.zeros(0, dtype=np.int64)
        if(self.field_cadence.nexp[epoch] < nexp):
            return available, competing_targets, spare_calibrations

        # Consider exposures for this epoch
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]

        # Check if this is an "extra" calibration target; i.e. not necessary
        # so should not bump any other calibration targets
        isspare = False
        if(iscalib & (rsid is not None)):
            cat = self.targets['category'][self.rsid2indx[rsid]]
            if(self.calibrations[cat][iexpst:iexpnd].min() >= self.required_calibrations[cat]):
                isspare = True

        # Get indices of assigned targets to this robot
        # and make Boolean arrays of which are assigned and not
        robot2indx = self._robot2indx[robotID, iexpst:iexpnd]
        unassigned = robot2indx < 0
        assigned = robot2indx >= 0

        # Check if the assigned robots are to "spare" calibration target.
        # These may be bumped if necessary (but won't be if the target under
        # consideration is, itself, a "spare" calibration targets. This logic
        # is not so straightforward but avoids expensive for loops.
        iassigned = np.flatnonzero(assigned)
        spare = np.zeros(iexpnd - iexpst, dtype=np.bool)
        icalib = iassigned[np.flatnonzero(self._is_calibration[robot2indx[iassigned]])]
        category = self.targets['category'][robot2indx[icalib]]
        calibspare = np.array([self.calibrations[category[i]][iexpst + icalib[i]] >
                               self.required_calibrations[category[i]]
                               for i in range(len(category))], dtype=np.bool)
        spare[icalib] = calibspare
        spare = spare & (isspare is False)

        # Now classify exposures as "free" or not (free if unassigned OR assigned to
        # a calibration target that may be bumped).
        free = unassigned | spare

        # And identify exposures which have competing targets
        competing = assigned & (spare == False)

        # Now (if there is an actual target under consideration) check for collisions.
        if(rsid is not None):
            for ifree in np.flatnonzero(free):
                free[ifree] = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                          iexp=iexpst + ifree) is False

        # Count this exposure as available if there are enough free exposures.
        # Package list of competing targets, and which calibrations are considered
        # spare. 
        available = free.sum() >= nexp
        competing_targets = np.unique(self.targets['rsid'][robot2indx[competing]])
        spare_calibrations = np.unique(self.targets['rsid'][robot2indx[spare]])

        return available, competing_targets, spare_calibrations

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None, nexp=None):
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

        Returns:
        --------

        success : bool
            True if successful, False otherwise

        Comments:
        --------
"""
        # Only try to assign if you can.
        if(rsid not in self.robotgrids[0].robotDict[robotID].validTargetIDs):
            return False

        # Get list of available exposures in the epoch
        available = []
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        for iexp in np.arange(iexpst, iexpnd):
            rg = self.robotgrids[iexp]
            if((rg.robotDict[robotID].isAssigned() != True) &
               (self.collide_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp) != True)):
                available.append(iexp)

        # Bomb if there aren't enough available
        if(len(available) < nexp):
            return False

        # Now actually assign (to first available exposures)
        for iexp in available[0:nexp]:
            self.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp)
        return True

    def assign_robot_exposure(self, robotID=None, rsid=None, iexp=None):
        """Assign an rsid to a particular robot-exposure

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        robotID : np.int64
            robotID to assign to

        iexp : int or np.int32
            exposure to assign to

        Returns:
        --------

        success : bool
            True if successful, False otherwise
"""
        rg = self.robotgrids[iexp]
        itarget = self.rsid2indx[rsid]
        rg.assignRobot2Target(robotID, rsid)
        self.assignments['robotID'][itarget, iexp] = robotID
        self._robot2indx[robotID, iexp] = itarget
        self.assignments['assigned'][itarget] = 1

        # If this is a calibration target, update calibration target tracker
        if(self._is_calibration[itarget]):
            category = self.targets['category'][itarget]
            self.calibrations[category][iexp] = self.calibrations[category][iexp] + 1
        return

    def unassign_epoch(self, rsid=None, epoch=None):
        """Unassign an rsid from a particular epoch

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        epoch : int or np.int32
            epoch to check

        Returns:
        -------

        status : int
            0 if the target had been assigned and was successfully removed
"""
        itarget = self.rsid2indx[rsid]
        category = self.targets['category'][itarget]
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        nexp = 0
        for iexp in np.arange(iexpst, iexpnd):
            rg = self.robotgrids[iexp]
            robotID = self.assignments['robotID'][itarget, iexp]
            if(robotID >= 0):
                if(rg.robotDict[robotID].assignedTargetID == rsid):
                    rg.unassignTarget(rsid)
                    self.assignments['robotID'][itarget, iexp] = -1
                    self._robot2indx[robotID, iexp] = -1
                    nexp = nexp + 1
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        if(nexp > 0):
            if(category in self.calibrations):
                self.calibrations[category][iexpst:iexpnd] = self.calibrations[category][iexpst:iexpnd] - 1
        return 0

    def unassign(self, rsid=None):
        """Unassign an rsid entirely

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign
"""
        for epoch in range(self.field_cadence.nepochs):
            self.unassign_epoch(rsid=rsid, epoch=epoch)
        return

    def available_robot_epochs(self, rsid=None, epochs=None, nexps=None, iscalib=False):
        """Find robots available for each epoch

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        epochs : ndarray of np.int32
            epochs to assign to

        nexps : ndarray of np.int32
            number of exposures needed

        calib : bool
            True if this is a calibration target; will not bump other calibs

        Returns:
        --------

        availableRobotIDs : list of lists
            for each epoch, list of available robotIDs sorted by robotID

        spareCalibrations : list of list of lists
            for each epoch and each robotID, list of spare calibrations
            availability relies upon removing
"""
        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs)
        validRobotIDs.sort()
        availableRobotIDs = [[]] * len(epochs)
        spareCalibrations = [[[]]] * len(epochs)
        for iepoch, epoch in enumerate(epochs):
            nexp = nexps[iepoch]
            arlist = []
            sclist = []
            for robotID in validRobotIDs:
                ok, ct, sc = self.available_robot_epoch(rsid=rsid,
                                                        robotID=robotID,
                                                        epoch=epoch,
                                                        nexp=nexp,
                                                        iscalib=iscalib)
                if(ok):
                    arlist.append(robotID)
                    sclist.append(sc)
            availableRobotIDs[iepoch] = arlist
            spareCalibrations[iepoch] = sclist
        return availableRobotIDs, spareCalibrations

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

        Returns:
        --------

        success : bool
            True if successful, False otherwise
"""
        if(self.targets['category'][self.rsid2indx[rsid]] in self.required_calibrations):
            iscalib = True
        else:
            iscalib = False
        availableRobotIDs, spareCalibrations = self.available_robot_epochs(rsid=rsid,
                                                                           epochs=epochs,
                                                                           nexps=nexps,
                                                                           iscalib=iscalib)

        # Check if there are robots available
        nRobotIDs = np.array([len(x) for x in availableRobotIDs])
        if(nRobotIDs.min() < 1):
            return False

        # Assign to each epoch
        for iepoch, epoch in enumerate(epochs):
            irobot = 0
            robotID = availableRobotIDs[iepoch][irobot]
            nexp = nexps[iepoch]

            # If there are spare calibrations associated with this
            # epoch, they need to be removed.
            for sc in spareCalibrations[iepoch][irobot]:
                self.unassign_epoch(rsid=sc, epoch=epoch)

            self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch,
                                    nexp=nexp)

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
"""
        indx = self.rsid2indx[rsid]
        target_cadence = self.targets['cadence'][indx]

        ok, epochs_list = clist.cadence_consistency(target_cadence,
                                                    self.field_cadence.name,
                                                    return_solutions=True,
                                                    epoch_level=True)

        nexps = clist.cadences[target_cadence].nexp
        for epochs in epochs_list:
            if(self.assign_epochs(rsid=rsid, epochs=epochs, nexps=nexps)):
                return True

        return False

    def assign_cadences(self, rsids=None):
        """Assign a set of targets to robots

        Parameters:
        ----------

        rsids : ndarray of np.int64
            rsids of targets to assign

        Returns:
        --------

        success : ndarray of bool
            True if successful, False otherwise
"""
        success = np.zeros(len(rsids), dtype=np.bool)

        for indx, rsid in enumerate(rsids):
            success[indx] = self.assign_cadence(rsid=rsid)

        return(success)

    def decollide_unassigned(self):
        """Decollide all unassigned robots"""
        for iexp, rg in enumerate(self.robotgrids):
            for robotID in rg.robotDict:
                if(rg.robotDict[robotID].isAssigned() == False):
                    rg.decollideRobot(robotID)
        return

    def assign_calibrations(self):
        """Assign all calibration targets"""
        icalib = np.where(self._is_calibration)[0]
        icalib = icalib[np.argsort(self.targets['priority'][icalib])]
        self.assign_cadences(rsids=self.targets['rsid'][icalib])
        return

    def assign_science(self):
        """Assign all science targets"""
        iscience = np.where(self.targets['category'] == 'science')[0]
        iscience = iscience[np.argsort(self.targets['priority'][iscience])]
        self.assign_cadences(rsids=self.targets['rsid'][iscience])
        return

    def assign(self):
        """Assign all targets"""
        self.assign_calibrations()
        self.assign_science()
        self.decollide_unassigned()

    def assess(self):
        """Assess the current results of assignment in field"""
        out = ""

        out = out + "Field cadence: {fc}\n".format(fc=self.field_cadence.name)

        out = out + "\n"
        out = out + "Calibration targets:"
        for c in self.required_calibrations:
            tmp = " {c} (want {rc}):"
            out = out + tmp.format(c=c, rc=self.required_calibrations[c])
            for rcn in self.calibrations[c]:
                out = out + " {rcn}".format(rcn=rcn)
            out = out + "\n"

        out = out + "\n"
        out = out + "Science targets:\n"
        iboss = np.where((self.targets['fiberType'] == 'BOSS') &
                         (self.assignments['assigned']))[0]
        out = out + " BOSS targets assigned: {n}\n".format(n=len(iboss))
        iapogee = np.where((self.targets['fiberType'] == 'APOGEE') &
                           (self.assignments['assigned']))[0]
        out = out + " APOGEE targets assigned: {n}\n".format(n=len(iapogee))

        perepoch = np.zeros(self.field_cadence.nepochs, dtype=np.int32)
        out = out + " Targets per epoch:"
        for epoch in range(self.field_cadence.nepochs):
            iexpst = self.field_cadence.epoch_indx[epoch]
            iexpnd = self.field_cadence.epoch_indx[epoch + 1]
            perepoch[epoch] = len(np.where(self.assignments['robotID'][:, iexpst:iexpnd].sum(axis=1) >= 0)[0])
            out = out + " {p}".format(p=perepoch[epoch])
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

        Checks that assigned targets got the right number of epochs.

        Checks that there are no collisions.
"""
        nproblems = 0
        test_calibrations = dict()
        for c in self.required_calibrations:
            test_calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                            dtype=np.int32)
        for indx, target in enumerate(self.targets):
            assignment = self.assignments[indx]
            isassigned = assignment['robotID'].max() >= 0
            if((isassigned) != (assignment['assigned'])):
                print("rsid={rsid} : assigned misclassification".format(rsid=target['rsid']))
                nproblems += 1
            target_cadence = clist.cadences[target['cadence']]
            nepochs = 0
            for epoch in range(self.field_cadence.nepochs):
                iexpst = self.field_cadence.epoch_indx[epoch]
                iexpnd = self.field_cadence.epoch_indx[epoch + 1]
                nexp = (assignment['robotID'][iexpst:iexpnd] >= 0).sum()
                if(nexp > 0):
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

        # Check that the number of calibrators has been tracked right
        for c in self.required_calibrations:
            for iexp in range(self.field_cadence.nexp_total):
                if(test_calibrations[c][iexp] != self.calibrations[c][iexp]):
                    print("number of {c} calibrators tracked incorrectly ({nc} found instead of {nct})".format(c=c, nc=test_calibrations[c][iexp], nct=self.calibrations[c][iexp]))

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

        # Check _robot2indx is tracking things correctly           
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

        return(nproblems)

    def plot_robot(self, robot, color=None, ax=None):
        xr = robot.xPos
        yr = robot.yPos
        xa = xr + _alphaLen * np.cos(robot.alpha / 180. * np.pi)
        ya = yr + _alphaLen * np.sin(robot.alpha / 180. * np.pi)
        xb = xa + _betaLen * np.cos((robot.alpha + robot.beta) / 180. * np.pi)
        yb = ya + _betaLen * np.sin((robot.alpha + robot.beta) / 180. * np.pi)
        ax.plot(np.array([xr, xa]), np.array([yr, ya]), color=color, alpha=0.5)
        ax.plot(np.array([xa, xb]), np.array([ya, yb]), color=color, linewidth=3)

    def plot(self, iexp=None, robotID=False, catalogid=False):
        """Plot assignments of robots to targets for field """
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
                    self.plot_robot(robot, color=colors[icolor], ax=axfig)

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
            self.plot_robot(self.robotgrids[iexp].robotDict[int(i)],
                            color='grey', ax=axfig)

        plt.xlim([-370., 370.])
        plt.ylim([-370., 370.])

        h, ell = axleg.get_legend_handles_labels()
        axleg.clear()
        axleg.legend(h, ell, loc='upper left')
        axleg.axis('off')
