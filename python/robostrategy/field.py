#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import re
import numpy as np
import fitsio
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import ortools.sat.python.cp_model as cp_model
import roboscheduler.cadence as cadence
import kaiju
import kaiju.robotGrid

# alpha and beta lengths for plotting
_alphaLen = 7.4
_betaLen = 15


# intersection of lists
def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))


# Type for targets array
targets_dtype = np.dtype([('ra', np.float64),
                          ('dec', np.float64),
                          ('x', np.float64),
                          ('y', np.float64),
                          ('within', np.int32),
                          ('incadence', np.int32),
                          ('priority', np.int32),
                          ('value', np.float32),
                          ('program', np.unicode_, 30),
                          ('carton', np.unicode_, 30),
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
        [len(targets)] array with 'assigned', 'satisfied', 
          'robotID', 'rsflags', 'fiberType'
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

    Notes:
    -----

    This class internally assumes that robotIDs are sequential integers starting at 0.
"""
    def __init__(self, filename=None, racen=None, deccen=None, pa=0.,
                 observatory='apo', field_cadence='none', collisionBuffer=2.,
                 fieldid=1, allgrids=True):
        self.fieldid = fieldid
        self.allgrids = allgrids
        if(self.allgrids):
            self.robotgrids = []
        else:
            self.robotgrids = None
        self.assignments = None
        self.rsid2indx = dict()
        self.targets = np.zeros(0, dtype=targets_dtype)
        self.target_duplicated = np.zeros(0, dtype=np.int32)
        self._is_calibration = np.zeros(0, dtype=np.bool)
        if(filename is not None):
            self.fromfits(filename=filename)
        else:
            self.racen = racen
            self.deccen = deccen
            self.pa = pa
            self.observatory = observatory
            self.collisionBuffer = collisionBuffer
            self.mastergrid = self._robotGrid()
            self.required_calibrations = dict()
            self.required_calibrations['sky_boss'] = 80
            self.required_calibrations['standard_boss'] = 80
            self.required_calibrations['sky_apogee'] = 30
            self.required_calibrations['standard_apogee'] = 20
            self.calibrations = dict()
            for n in self.required_calibrations:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
            self.set_field_cadence(field_cadence)
        self._set_radius()
        self.flagdict = _flagdict
        self._competing_targets = None
        self.methods = dict()
        self.methods['assign_epochs'] = 'first'
        self.methods['assign_cadence'] = 'first'
        return

    def fromfits(self, filename=None):
        duf, hdr = fitsio.read(filename, ext=0, header=True)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        self.collisionBuffer = hdr['CBUFFER']
        self.mastergrid = self._robotGrid()
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
        self.set_field_cadence(field_cadence)
        targets = fitsio.read(filename, ext=1)
        try:
            assignments = fitsio.read(filename, ext=2)
        except OSError:
            assignments = None
        self.targets_fromarray(target_array=targets, assignment_array=assignments)
        if(self.assignments is not None):
            for assignment, target in zip(self.assignments, self.targets):
                for iexp in range(self.field_cadence.nexp_total):
                    if(assignment['robotID'][iexp] >= 0):
                        self.assign_robot_exposure(robotID=assignment['robotID'][iexp],
                                                   rsid=target['rsid'], iexp=iexp)
            self.decollide_unassigned()
        return

    def clear_assignments(self):
        if(self.assignments is not None):
            iassigned = np.where(self.assignments['assigned'])[0]
            for i in iassigned:
                self.unassign(self.targets['rsid'][i])
        return

    def clear_field_cadence(self):
        if(self.assignments is not None):
            self.clear_assignments()

        if(self.allgrids):
            for i in range(self.field_cadence.nexp_total):
                self.robotgrids[i] = None
            self.robotgrids = []
        self._robot2indx = None
        self.field_cadence = None
        self.assignments_dtype = None
        self.assignments = None
        for c in self.calibrations:
            for n in self.required_calibrations:
                self.calibrations[n] = np.zeros(0, dtype=np.int32)
            
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
            self.field_cadence = clist.cadences[field_cadence]
            if(self.allgrids):
                for i in range(self.field_cadence.nexp_total):
                    self.robotgrids.append(self._robotGrid())
            self._robot2indx = np.zeros((len(self.mastergrid.robotDict),
                                         self.field_cadence.nexp_total),
                                        dtype=np.int32) - 1
            self.assignments_dtype = np.dtype([('assigned', np.int32),
                                               ('satisfied', np.int32),
                                               ('robotID', np.int32,
                                                (self.field_cadence.nexp_total,)),
                                               ('fiberType', np.unicode_, 10),
                                               ('rsflags', np.int32)])
            self.assignments = np.zeros(0, dtype=self.assignments_dtype)
            for c in self.calibrations:
                self.calibrations[c] = np.zeros(self.field_cadence.nexp_total,
                                                dtype=np.int32)
            self.targets, self.assignments = self._setup_for_cadence(self.targets)
        else:
            self.field_cadence = None
            if(self.allgrids):
                self.robotgrids = []
            else:
                self.robotgrids = None
            self.assignments_dtype = None

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

    def _setup_for_cadence(self, targets=None, assignment_array=None):
        if(targets is None):
            return(None, None)

        # Determine if it is within the field cadence
        for itarget, target_cadence in enumerate(targets['cadence']):
            if(target_cadence in clist.cadences):
                ok = clist.cadence_consistency(target_cadence,
                                               self.field_cadence.name,
                                               return_solutions=False)
                targets['incadence'][itarget] = ok

        if(self.allgrids):
            for rg in self.robotgrids:
                self._targets_to_robotgrid(targets=targets,
                                           robotgrid=rg)

        # Set up outputs
        assignments = np.zeros(len(targets),
                               dtype=self.assignments_dtype)
        if(assignment_array is None):
            assignments['fiberType'] = targets['fiberType']
            assignments['robotID'] = -1
        else:
            for n in self.assignments_dtype.names:
                if((n == 'robotID') & (self.field_cadence.nexp_total == 1)):
                    assignments[n][:, 0] = assignment_array[n]
                else:
                    assignments[n] = assignment_array[n]
        return(targets, assignments)

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

        # Default value of 1 for priority and value
        if('value' not in target_array.dtype.names):
            targets['value'] = 1.
        if('priority' not in target_array.dtype.names):
            targets['priority'] = 1.

        # Set fiber type
        if('fiberType' not in target_array.dtype.names):
            targets['fiberType'] = np.array(['APOGEE' if clist.cadences[c].instrument == cadence.Instrument.ApogeeInstrument
                                             else 'BOSS'
                                             for c in targets['cadence']])

        # Convert ra/dec to x/y
        targets['x'], targets['y'] = self.radec2xy(ra=targets['ra'],
                                                   dec=targets['dec'])

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
        for i, t in enumerate(targets):
            _is_calibration[i] = t['category'] in self.required_calibrations

        # Connect rsid with index of list
        for itarget, t in enumerate(targets):
            if(t['rsid'] in self.rsid2indx.keys()):
                print("Cannot replace identical rsid={rsid}. Will not add array.".format(rsid=t['rsid']))
                return
            else:
                self.rsid2indx[t['rsid']] = len(self.targets) + itarget

        # If field_cadence is set, set up potential outputs
        if(self.field_cadence is not None):
            targets, assignments = self._setup_for_cadence(targets,
                                                           assignment_array)
        else:
            assignments = None

        target_duplicated = np.zeros(len(targets), dtype=np.int32)

        self.targets = np.append(self.targets, targets)
        self.target_duplicated = np.append(self.target_duplicated,
                                           target_duplicated)
        self._is_calibration = np.append(self._is_calibration,
                                         _is_calibration)
        if(assignments is not None):
            self.assignments = np.append(self.assignments, assignments, axis=0)

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
        else:
            hdr['FCADENCE'] = 'none'
        hdr['CBUFFER'] = self.collisionBuffer
        for indx, rc in enumerate(self.required_calibrations):
            name = 'RCNAME{indx}'.format(indx=indx)
            num = 'RCNUM{indx}'.format(indx=indx)
            hdr[name] = rc
            hdr[num] = self.required_calibrations[rc]
        fitsio.write(filename, None, header=hdr, clobber=True)
        fitsio.write(filename, self.targets)
        if(self.assignments is not None):
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
        if(not self.allgrids):
            return False
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
        spare_calibrations = np.zeros(0, dtype=np.int64)
        if(self.field_cadence.nexp[epoch] < nexp):
            return available, spare_calibrations

        # Now optimize case where nexp=1
        if(self.field_cadence.nexp[epoch] == 1):
            iexp = self.field_cadence.epoch_indx[epoch]

            robot2indx = self._robot2indx[robotID, iexp]
            unassigned = robot2indx < 0
            spare_calibrations = np.zeros(0, dtype=np.int64)
            free = False
            if(unassigned):
                free = True
            else:
                # Check if this is an "extra" calibration target; i.e. not necessary
                # so should not bump any other calibration targets
                isspare = False
                if(iscalib & (rsid is not None)):
                    cat = self.targets['category'][self.rsid2indx[rsid]]
                    if(self.calibrations[cat][iexp] > self.required_calibrations[cat]):
                        isspare = True

                spare = False
                if((isspare == False) & (self._is_calibration[robot2indx])):
                    category = self.targets['category'][robot2indx]
                    spare = self.calibrations[category][iexp] > self.required_calibrations[category]
                    if(spare):
                        spare_calibrations = np.array([self.targets['rsid'][robot2indx]])
                        free = True

            if(free):
                free = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                   iexp=iexp) == False

            if(free):
                return True, 1, spare_calibrations
            else:
                return False, 0, spare_calibrations

        # Consider exposures for this epoch
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]

        # Check if this is an "extra" calibration target; i.e. not necessary
        # so should not bump any other calibration targets
        isspare = False
        if(iscalib & (rsid is not None)):
            cat = self.targets['category'][self.rsid2indx[rsid]]
            if(self.calibrations[cat][iexpst:iexpnd].min() > self.required_calibrations[cat]):
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
        spare = np.zeros(iexpnd - iexpst, dtype=np.bool)
        iassigned = np.where(assigned)[0]
        icalib = iassigned[np.where(self._is_calibration[robot2indx[iassigned]])[0]]
        if(len(icalib) > 0):
            category = self.targets['category'][robot2indx[icalib]]
            calibspare = np.array([self.calibrations[category[i]][iexpst + icalib[i]] >
                                   self.required_calibrations[category[i]]
                                   for i in range(len(category))], dtype=np.bool)
            spare[icalib] = calibspare
            spare = spare & (isspare == False)

        # Now classify exposures as "free" or not (free if unassigned OR assigned to
        # a calibration target that may be bumped).
        free = unassigned | spare

        # Now (if there is an actual target under consideration) check for collisions.
        if(rsid is not None):
            for ifree in np.where(free)[0]:
                free[ifree] = self.collide_robot_exposure(rsid=rsid, robotID=robotID,
                                                          iexp=iexpst + ifree) == False

        # Count this exposure as available if there are enough free exposures.
        # Package list of which calibrations are considered spare.
        available = free.sum() >= nexp
        if(spare.sum() > 0):
            spare_calibrations = np.unique(self.targets['rsid'][robot2indx[spare]])
        else:
            spare_calibrations = np.zeros(0, dtype=np.int64)

        return available, free.sum(), spare_calibrations

    def assign_robot_epoch(self, rsid=None, robotID=None, epoch=None, nexp=None,
                           reset_satisfied=True):
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

        reset_satisfied : bool
            if True, reset the 'satisfied' column based on this assignment
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
        available = []
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        for iexp in np.arange(iexpst, iexpnd):
            if((self._robot2indx[robotID, iexp] < 0) &
               (self.collide_robot_exposure(rsid=rsid, robotID=robotID, iexp=iexp) != True)):
                available.append(iexp)

        # Bomb if there aren't enough available
        if(len(available) < nexp):
            return False

        # Now actually assign (to first available exposures)
        for iexp in available[0:nexp]:
            self.assign_robot_exposure(robotID=robotID, rsid=rsid, iexp=iexp,
                                       reset_satisfied=False)

        if(reset_satisfied):
            indx = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][indx]
            self._set_satisfied(catalogids=[catalogid])

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
                              reset_satisfied=True):
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

        Returns:
        --------

        success : bool
            True if successful, False otherwise
"""
        itarget = self.rsid2indx[rsid]
        self.assignments['robotID'][itarget, iexp] = robotID
        self._robot2indx[robotID, iexp] = itarget
        self.assignments['assigned'][itarget] = 1

        # If this is a calibration target, update calibration target tracker
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

        return

    def _set_assigned(self, itarget=None):
        self.assignments['assigned'][itarget] = (self.assignments['robotID'][itarget, :] >= 0).sum() > 0
        return

    def unassign_exposure(self, rsid=None, iexp=None, reset_assigned=True,
                          reset_satisfied=True):
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
            if(self._is_calibration[itarget]):
                self.calibrations[category][iexp] = self.calibrations[category][iexp] - 1

        if(reset_assigned == True):
            self._set_assigned(itarget=itarget)

        if(reset_satisfied):
            catalogid = self.targets['catalogid'][itarget]
            self._set_satisfied(catalogids=[catalogid])

        return

    def unassign_epoch(self, rsid=None, epoch=None, reset_assigned=True,
                       reset_satisfied=True):
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

        Returns:
        -------

        status : int
            0 if the target had been assigned and was successfully removed
"""
        iexpst = self.field_cadence.epoch_indx[epoch]
        iexpnd = self.field_cadence.epoch_indx[epoch + 1]
        for iexp in np.arange(iexpst, iexpnd):
            self.unassign_exposure(rsid=rsid, iexp=iexp, reset_assigned=False,
                                   reset_satisfied=False)

        if(reset_assigned):
            self._set_assigned(itarget=self.rsid2indx[rsid])

        if(reset_satisfied):
            itarget = self.rsid2indx[rsid]
            catalogid = self.targets['catalogid'][itarget]
            self._set_satisfied(catalogids=[catalogid])

        return 0

    def unassign(self, rsid=None):
        """Unassign an rsid entirely

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign
"""
        for epoch in range(self.field_cadence.nepochs):
            self.unassign_epoch(rsid=rsid, epoch=epoch, reset_assigned=False,
                                reset_satisfied=False)

        self._set_assigned(itarget=self.rsid2indx[rsid])

        itarget = self.rsid2indx[rsid]
        catalogid = self.targets['catalogid'][itarget]
        self._set_satisfied(catalogids=[catalogid])

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

    def available_epochs(self, rsid=None, epochs=None, nexps=None, iscalib=False):
        """Find robots available for each epoch

        Parameters:
        ----------

        rsid : np.int64
            rsid of target to assign

        epochs : ndarray of np.int32
            epochs to assign to (default all)

        nexps : ndarray of np.int32
            number of exposures needed (default 1 per epoch)

        calib : bool
            True if this is a calibration target; will not bump other calibs

        Returns:
        --------

        available : dictionary, with key value pairs below
            'availableRobotIDs' : list of lists
                for each epoch, list of available robotIDs sorted by robotID

            'nFrees' : list of lists
                for each epoch, list of number of free exposures for each available robot

            'spareCalibrations' : list of list of lists
                for each epoch and each robotID, list of spare calibrations
                availability relies upon removing
"""
        if(epochs is None):
            epochs = np.arange(self.field_cadence.nepochs, dtype=np.int32)
        if(nexps is None):
            nexps = np.ones(len(epochs))
        validRobotIDs = self.masterTargetDict[rsid].validRobotIDs
        validRobotIDs = np.array(validRobotIDs)
        validRobotIDs.sort()
        availableRobotIDs = [[]] * len(epochs)
        spareCalibrations = [[[]]] * len(epochs)
        nFrees = [[]] * len(epochs)
        for iepoch, epoch in enumerate(epochs):
            nexp = nexps[iepoch]
            arlist = []
            sclist = []
            nflist = []
            for robotID in validRobotIDs:
                ok, nfree, sc = self.available_robot_epoch(rsid=rsid,
                                                           robotID=robotID,
                                                           epoch=epoch,
                                                           nexp=nexp,
                                                           iscalib=iscalib)
                if(ok):
                    arlist.append(robotID)
                    nflist.append(nfree)
                    sclist.append(sc)
            availableRobotIDs[iepoch] = arlist
            spareCalibrations[iepoch] = sclist
            nFrees[iepoch] = nflist

        available = dict()
        available['availableRobotIDs'] = availableRobotIDs
        available['nFrees'] = nFrees
        available['spareCalibrations'] = spareCalibrations
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
        if(self.targets['category'][self.rsid2indx[rsid]] in self.required_calibrations):
            iscalib = True
        else:
            iscalib = False

        epochs_merged, nexps_merged = self._merge_epochs(epochs=epochs,
                                                         nexps=nexps)

        available = self.available_epochs(rsid=rsid, epochs=epochs_merged,
                                          nexps=nexps_merged, iscalib=iscalib)
        availableRobotIDs = available['availableRobotIDs']
        spareCalibrations = available['spareCalibrations']

        # Check if there are robots available
        nRobotIDs = np.array([len(x) for x in availableRobotIDs])
        if(nRobotIDs.min() < 1):
            return False

        # Assign to each epoch
        for iepoch, epoch in enumerate(epochs_merged):
            currRobotIDs = np.array(availableRobotIDs[iepoch], dtype=np.int32)
            if(self.methods['assign_epochs'] == 'first'):
                irobot = 0
            if(self.methods['assign_epochs'] == 'fewestcompeting'):
                irobot = np.argmin(self._competing_targets[currRobotIDs])
            robotID = currRobotIDs[irobot]
            nexp = nexps_merged[iepoch]

            # If there are spare calibrations associated with this
            # epoch, they need to be removed.
            for sc in spareCalibrations[iepoch][irobot]:
                self.unassign_epoch(rsid=sc, epoch=epoch)

            self.assign_robot_epoch(rsid=rsid, robotID=robotID, epoch=epoch,
                                    nexp=nexp, reset_satisfied=False)

        indx = self.rsid2indx[rsid]
        catalogid = self.targets['catalogid'][indx]
        self._set_satisfied(catalogids=[catalogid])

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

        ok, epochs_list = clist.cadence_consistency(target_cadence,
                                                    self.field_cadence.name,
                                                    return_solutions=True,
                                                    epoch_level=True)

        if(ok == False):
            return False

        nexps = clist.cadences[target_cadence].nexp

        # Find set of epochs which is "most available"
        if(self.methods['assign_cadence'] == 'mostavailable'):
            navailable = np.zeros(len(epochs_list), dtype=np.int32)
            for indx, epochs in enumerate(epochs_list):
                if(self.targets['category'][self.rsid2indx[rsid]] in self.required_calibrations):
                    iscalib = True
                else:
                    iscalib = False
                available = self.available_epochs(rsid=rsid, epochs=epochs, nexps=nexps,
                                                  iscalib=iscalib)
                availableRobotIDs = available['availableRobotIDs']

                navailable[indx] = 1
                for ari in availableRobotIDs:
                    navailable[indx] = navailable[indx] + len(ari)

            imost = np.argmax(navailable)
            epochs = epochs_list[imost]
            if(self.assign_epochs(rsid=rsid, epochs=epochs, nexps=nexps)):
                return True

        # Or just use first available
        if(self.methods['assign_cadence'] == 'first'):
            for indx, epochs in enumerate(epochs_list):
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
            catalogids = np.unique(self.targets['catalogid'])

        for catalogid in catalogids:
            # Check for other instances of this catalogid, and whether
            # assignments have satisfied their cadence
            icats = np.where(self.targets['catalogid'] == catalogid)[0]
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
        priorities = np.unique(self.targets['priority'][indxs])
        for priority in priorities:
            iormore = np.where((self.targets['priority'][indxs] >= priority) &
                               (self._is_calibration[indxs] == False))[0]
            self._set_competing_targets(rsids[iormore])

            iassign = np.where(self.targets['priority'][indxs] == priority)[0]

            for i, rsid in enumerate(rsids[iassign]):
                # Perform the assignment
                if((check_satisfied == False) |
                   (self.assignments['satisfied'][self.rsid2indx[rsid]] == 0)):
                    success[iassign[i]] = self.assign_cadence(rsid=rsid)
                    if(rsid == 0):
                        ii = np.where(self.targets['catalogid'] == 0)[0]

            self._competing_targets = None

        return(success)

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
                                if(rg.isCollidedWithAssigned(robotID1)):
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
        """Assign all calibration targets"""
        icalib = np.where(self._is_calibration)[0]
        self.assign_cadences(rsids=self.targets['rsid'][icalib])
        return

    def assign_science(self):
        """Assign all science targets"""
        iscience = np.where((self.targets['category'] == 'science') &
                            (self.targets['incadence']) &
                            (self.target_duplicated == 0))[0]
        np.random.seed(self.fieldid)
        np.random.shuffle(iscience)
        self.assign_cadences(rsids=self.targets['rsid'][iscience])
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
                print("rsid={rsid} : assigned misclassification".format(rsid=target['rsid']))
                nproblems += 1

        # Check that the number of calibrators has been tracked right
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
                        print("assignments['robotID'] for rsid={rsid} and iexp={iexp} is robotID={robotID}, but _robot2indx[robotID, iexp] is {i}".format(rsid=rsid, iexp=iexp, robotID=robotID, i=self._robot2indx[itarget, iexp]))
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
