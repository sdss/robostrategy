#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import fitsio
import scipy.optimize as optimize

import roboscheduler.cadence as cadence
import kaiju
import kaiju.robotGrid


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
"""
    def __init__(self, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none'):
        self.collisionBuffer = 2.0  # for kaiju
        self.required_calibrations = dict()
        self.required_calibrations['sky_boss'] = 80
        self.required_calibrations['standard_boss'] = 80
        self.required_calibrations['sky_apogee'] = 30
        self.required_calibrations['standard_apogee'] = 20
        self.calibrations = dict()
        for n in self.required_calibrations:
            self.calibrations[n] = []
        self.racen = racen
        self.deccen = deccen
        self.pa = pa
        self.observatory = observatory
        self._set_radius()
        self.flagdict = _flagdict
        self.robotgrids = []
        self.assignments = None
        self.rsid2indx = dict()
        self.targets = np.zeros(0, dtype=targets_dtype)
        self._set_field_cadence(field_cadence)
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
        else:
            self.field_cadence = None
            self.robotgrids = []

        self.assignments_dtype = np.dtype([('assigned', np.int32),
                                           ('robotID', np.int32,
                                            (self.field_cadence.nexp_total,)),
                                           ('fiberType', np.unicode_, 10),
                                           ('rsflags', np.int32)])
        self.assignments = np.zeros(0, dtype=self.assignments_dtype)
        for c in self.calibrations:
            self.calibrations[c] = [0] * self.field_cadence.nepochs
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

    def _min_xy_diff(self, radec, xt, yt):
        x, y = self.radec2xy(ra=radec[0], dec=radec[1])
        resid2 = (x - xt)**2 + (y - yt)**2
        return(resid2)

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

        self.targets = np.append(self.targets, targets)
        self.assignments = np.append(self.assignments, assignments, axis=0)

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
        ct = -1
        if(rg.robotDict[robotID].isAssigned()):
            ct = rg.robotDict[robotID].assignedTargetID
        rg.assignRobot2Target(robotID, rsid)
        collide = rg.isCollidedWithAssigned(robotID)
        if(ct >= 0):
            rg.assignRobot2Target(robotID, ct)
        else:
            rg.unassignRobot(robotID)

        return collide

    def available_robot_epoch(self, rsid=None,
                              robotID=None, epoch=None, nexp=None, calib=False):
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

        calib : bool
            True if this is a calibration target; will not bump other calibs

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
"""
        available = False
        competing_targets = np.zeros(0, dtype=np.int64)
        spare_calibrations = np.zeros(0, dtype=np.int64)
        if(self.field_cadence.nexp[epoch] < nexp):
            return available, competing_targets, spare_calibrations

        nfree = 0
        competing_targets = []
        spare_calibrations = []
        for iexp in np.arange(self.field_cadence.epoch_indx[epoch],
                              self.field_cadence.epoch_indx[epoch + 1]):
            rg = self.robotgrids[iexp]
            free = False
            if(rg.robotDict[robotID].isAssigned() != True):
                # Available if the fiber is not assigned, and if assigning
                # it would not cause a collision
                free = True
            else:
                cid = rg.robotDict[robotID].assignedTargetID
                category = self.targets['category'][self.rsid2indx[cid]]
                if((category in self.required_calibrations) | (calib)):
                    if(self.calibrations[category][epoch] >
                       self.required_calibrations[category]):
                        # Or if the assignment is to a calibration target that can be spared
                        # (and a collision would not be caused)
                        free = True
                        spare_calibrations.append(cid)
                else:
                    free = False
                    competing_targets.append(cid)
            if((free) & (rsid is not None)):
                if(self.collide_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp)):
                    free = False
            if(free):
                nfree = nfree + 1

        competing_targets = np.unique(np.array(competing_targets, dtype=np.int64))
        spare_calibrations = np.unique(np.array(spare_calibrations, dtype=np.int64))
        if(nfree >= nexp):
            available = True
        return available, competing_targets, spare_calibrations

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None,
                           nexp=None, force=True):
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

        Does not check collisions.
"""
        itarget = self.rsid2indx[rsid]
        if(rsid not in self.robotgrids[0].robotDict[robotID].validTargetIDs):
            return False
        available = []
        for iexp in np.arange(self.field_cadence.epoch_indx[epoch],
                              self.field_cadence.epoch_indx[epoch + 1]):
            rg = self.robotgrids[iexp]
            if((rg.robotDict[robotID].isAssigned() != True) &
               (self.collide_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp) != True)):
                available.append(iexp)
        if(len(available) < nexp):
            return False
        self.assignments['assigned'][itarget] = 1
        for iexp in available[0:nexp]:
            rg = self.robotgrids[iexp]
            rg.assignRobot2Target(robotID, rsid)
            self.assignments['robotID'][itarget, iexp] = robotID
        category = self.targets['category'][itarget]
        if(category in self.calibrations):
            self.calibrations[category][epoch] = self.calibrations[category][epoch] + 1
        return True

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
                    nexp = nexp + 1
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        if(nexp > 0):
            if(category in self.calibrations):
                self.calibrations[category][epoch] = self.calibrations[category][epoch] - 1
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

    def collisions(self, rsid=None):
        """Check collisions for an rsid

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        Returns:
        -------

        status : int
            1 if the target is involved in a collision with an assigned robot
"""
        #itarget = self.rsid2indx[rsid]
        #for indx, rg in enumerate(self.robotgrids):
            #epoch = self.field_cadence.epochs[indx]
            #robotID = self.assignments['robotID'][itarget, epoch]
            #if(robotID != -1):
                #if(rg.isCollidedWithAssigned(robotID)):
                    #return 1
        return 0

    def available_robot_epochs(self, rsid=None, epochs=None, nexps=None, calib=False):
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
                                                        calib=calib)
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
            calib = True
        else:
            calib = False
        availableRobotIDs, spareCalibrations = self.available_robot_epochs(rsid=rsid,
                                                                           epochs=epochs,
                                                                           nexps=nexps,
                                                                           calib=calib)

        # Check if there are robots available
        nRobotIDs = np.array([len(x) for x in availableRobotIDs])
        if(nRobotIDs.min() < 1):
            return False

        for iepoch, epoch in enumerate(epochs):
            irobot = 0
            robotID = availableRobotIDs[iepoch][irobot]
            nexp = nexps[iepoch]
            for sc in spareCalibrations[iepoch][irobot]:
                self.unassign(rsid=sc)
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
        iscalib = np.zeros(len(self.targets), dtype=np.int32)
        for indx, target in enumerate(self.targets):
            iscalib[indx] = target['category'] in self.required_calibrations
        icalib = np.where(iscalib)[0]
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
        """Assess the results"""
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
            test_calibrations[c] = [0] * self.field_cadence.nepochs
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
                        test_calibrations[target['category']][epoch] += 1

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
                    expst = self.field_cadence.epoch_indx[epoch]
                    expnd = self.field_cadence.epoch_indx[epoch + 1]
                    nexp = 0
                    for iexp in range(expst, expnd):
                        rg = self.robotgrids[iexp]
                        robotID = assignment['robotID'][iexp]
                        if(robotID >= 0):
                            if(rg.robotDict[robotID].assignedTargetID == target['rsid']):
                                nexp = nexp + 1
                    if(nexp != target_cadence.nexp[nepochs]):
                        print("rsid={rsid} epoch={epoch} : robots not assigned ({ne} out of {nt})".format(rsid=target['rsid'],
                                                                                                          epoch=epoch, ne=nexp, nt=target_cadence.nexp[nepochs]))
                        nproblems += 1
                    nepochs = nepochs + 1

            # Check that if the target is assigned, it has the right number of epochs
            if((nepochs > 0) & (nepochs != target_cadence.nepochs)):
                print("rsid={rsid} : target assigned with wrong nepochs".format(rsid=target['rsid']))
                nproblems += 1

        # Check that the number of calibrators has been tracked right
        for c in self.required_calibrations:
            for epoch in range(self.field_cadence.nepochs):
                if(test_calibrations[c][epoch] != self.calibrations[c][epoch]):
                    print("number of {c} calibrators tracked incorrectly ({nc} found instead of {nct})".format(c=c, nc=test_calibrations[c][epoch], nct=self.calibrations[c][epoch]))

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

        return(nproblems)
