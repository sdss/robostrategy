#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import fitsio
import matplotlib.pyplot as plt

import roboscheduler.cadence as cadence
import observesim.robot as robot

__all__ = ['Field']

"""Field module class.

Dependencies:

 numpy
 fitsio
 matplotlib
 roboscheduler
 observesim

"""


class Field(object):
    """Field class

    Parameters:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    observatory : str
        observatory field observed from, 'apo' or 'lco' (default 'apo')

    fps_layout : str
        name of FPS layout to assume (default 'filled_hex')

    db : boolean
        whether to use database when setting up Robot instance (default True)

    Attributes:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    observatory : str
        observatory field observed from ('apo' or 'lco')

    field_cadence : int, np.int32
        name of field cadence (as given in cadencelist)

    robot : Robot class
        instance of Robot (singleton)

    cadencelist : CadenceList class
        instance of CadenceList (singleton)

    ntarget : int or np.int32
        number of targets

    target_array : ndarray
        ndarray with target info, exact format varies

    target_ra : ndarray of np.float64
        RA of targets, J2000 deg

    target_dec : ndarray of np.float64
        Dec of targets, J2000 deg

    target_x : ndarray of np.float64
        x positions of targets, mm

    target_y : ndarray of np.float64
        y positions of targets, mm

    target_priority : ndarray of np.int32
        priorities of targets (lower is considered first)

    target_category : ndarray of strings
        category of targets ('SKY', 'STANDARD', 'SCIENCE')

    target_pk : ndarray of np.int64
        unique primary key for each target

    target_cadence : ndarray of np.int32
        cadences of targets

    target_type : ndarray of strings
        target types ('BOSS' or 'APOGEE')

    target_assigned : ndarray of np.int32
        (ntarget) array of 0 or 1, indicating whether target is assigned

    target_assignments : ndarray of np.int32
        (ntarget, nexposure) array of positionerid for each target

    assignment : ndarray of np.int32
        (npositioner, nexposure) array of target indices

    greedy_limit : int or np.int32
        number of exposures above which assign() uses greedy algorithm

    Methods:
    -------

    targets_fromarray() : read targets from an ndarray
    targets_fromfits() : read targets from a FITS file
    targets_toarray() : write targets to an ndarray
    tofits() : write targets (and assignments) to a FITS file
    assign() : assign targets to robots for cadence
    plot() : plot assignments of robots to targets

    Notes:
    -----

    assignments gives a direct index into the target_* arrays, or -1
    for unassigned positioner-exposures. It does not contain
    target_pk.
"""
    def __init__(self, racen=None, deccen=None,
                 db=True, fps_layout='filled_hex',
                 observatory='apo'):
        self.robot = robot.Robot(db=db, fps_layout=fps_layout)
        self.racen = racen
        self.deccen = deccen
        self.observatory = observatory
        self.cadencelist = cadence.CadenceList()
        self.field_cadence = None
        self.assignments = None
        self.target_assigned = None
        self.target_assignments = None
        self.greedy_limit = 100
        self.nsky_apogee = 20
        self.nstandard_apogee = 20
        self.nsky_boss = 50
        self.nstandard_boss = 20
        return

    def _arrayify(self, quantity=None, dtype=np.float64):
        """Cast quantity as ndarray of numpy.float64"""
        try:
            length = len(quantity)
        except TypeError:
            length = 1
        return np.zeros(length, dtype=dtype) + quantity

    def set_target_assignments(self):
        """Convert assignments array to per-target basis

        Notes:
        ------

        Sets attributes target_assignment and target_assigned based on
        the assignments attribute values.
"""
        if(self.assignments is None):
            return

        nexp = self.cadencelist.cadences[self.field_cadence].nexposures
        self.target_assignments = np.zeros((self.ntarget, nexp),
                                           dtype=np.int32) - 1
        self.target_assigned = np.zeros(self.ntarget, dtype=np.int32)
        for iexp in np.arange(nexp, dtype=np.int32):
            for irobot in np.arange(self.robot.npositioner, dtype=np.int32):
                curr_assignment = self.assignments[irobot, iexp]
                if(curr_assignment >= 0):
                    self.target_assigned[curr_assignment] = 1
                    self.target_assignments[curr_assignment, iexp] = self.robot.positionerid[irobot]

        return

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
        pa_rad = np.arctan2(pay, pax)

        x = d_rad * 180. / np.pi * scale * np.sin(pa_rad)
        y = d_rad * 180. / np.pi * scale * np.cos(pa_rad)

        return(x, y)

# def xy2radec_crude(self, x=None, y=None):
#    # Yikes!
#    if(self.observatory == 'apo'):
#        scale = 218.
#    if(self.observatory == 'lco'):
#        scale = 329.
#    ra = self.racen + (x / scale) / np.cos(self.deccen * np.pi / 180.)
#    dec = self.deccen + (y / scale)
#    return(ra, dec)
#
#    def xy2radec(self, x=None, y=None):
#        # Yikes!
#        if(self.observatory == 'apo'):
#            scale = 218.
#        if(self.observatory == 'lco'):
#            scale = 329.
#        ra = self.racen + (x / scale) / np.cos(self.deccen * np.pi / 180.)
#        dec = self.deccen + (y / scale)
#        return(ra, dec)

    def targets_fromarray(self, target_array=None):
        """Read targets from an ndarray

        Parameters:
        ----------

        target_array : ndarray
            ndarray with columns below

        Notes:
        ------

        Required columns of array:
         'ra', 'dec' should be np.float64
         'pk' should be np.int64
         'cadence', 'type' should be str or bytes

        Optional columns of array:
         'priority'
         'category'
"""
        self.target_array = target_array
        self.ntarget = len(self.target_array)
        self.target_ra = self.target_array['ra']
        self.target_dec = self.target_array['dec']
        self.target_pk = self.target_array['pk']
        self.target_x, self.target_y = self.radec2xy(self.target_ra,
                                                     self.target_dec)

        try:
            self.target_cadence = np.array(
                [c.decode().strip() for c in self.target_array['cadence']])
        except AttributeError:
            self.target_cadence = np.array(
                [c.strip() for c in self.target_array['cadence']])

        try:
            self.target_priority = self.target_array['priority']
        except ValueError:
            self.target_priority = np.ones(self.ntarget, dtype=np.int32)

        try:
            self.target_value = self.target_array['value']
        except ValueError:
            self.target_value = np.ones(self.ntarget, dtype=np.int32)

        try:
            self.target_category = np.array(
                [c.decode().strip() for c in self.target_array['category']])
        except AttributeError:
            self.target_category = np.array(
                [c.strip() for c in self.target_array['category']])
        except ValueError:
            self.target_category = np.array(['SCIENCE'] * self.ntarget)

        try:
            self.target_type = np.array(
                [t.decode().strip() for t in self.target_array['type']])
        except AttributeError:
            self.target_type = np.array(
                [t.strip() for t in self.target_array['type']])

        return

    def targets_fromfits(self, filename=None):
        """Read targets from a FITS file

        Parameters:
        ----------

        filename : str
            FITS file name, for file with columns listed below

        Notes:
        ------

        Required columns:
         'ra', 'dec' should be np.float64
         'pk' should be np.int64
         'cadence', 'type' should be str or bytes

        Optional columns:
         'priority'
         'category'
"""
        target_array = fitsio.read(filename)
        self.targets_fromarray(target_array)
        return

    def fromfits(self, filename=None, read_assignments=True):
        """Read field from a FITS file

        Parameters:
        ----------

        filename : str
            FITS file name, where HDU 2 has array of assignments
"""
        hdr = fitsio.read_header(filename, ext=1)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.field_cadence = hdr['FCADENCE'].strip()
        if((self.field_cadence != 'none') & (read_assignments)):
            self.assignments = fitsio.read(filename, ext=2)
        self.targets_fromfits(filename)
        if((self.field_cadence != 'none') & (read_assignments)):
            self.set_target_assignments()
        return

    def targets_toarray(self):
        """Write targets to an ndarray

        Returns:
        -------

        target_array : ndarray
            Array of targets, with columns:
              'ra', 'dec' (np.float64)
              'pk' (np.int64)
              'cadence', 'type' ('a30')
              'priority' (np.int32)
              'category' ('a30')
"""
        target_array_dtype = np.dtype([('ra', np.float64),
                                       ('dec', np.float64),
                                       ('pk', np.int64),
                                       ('cadence', cadence.fits_type),
                                       ('type', np.dtype('a30')),
                                       ('category', np.dtype('a30')),
                                       ('value', np.int32),
                                       ('priority', np.int32)])

        target_array = np.zeros(self.ntarget, dtype=target_array_dtype)
        target_array['ra'] = self.target_ra
        target_array['dec'] = self.target_dec
        target_array['pk'] = self.target_pk
        target_array['cadence'] = self.target_cadence
        target_array['type'] = self.target_type
        target_array['category'] = self.target_category
        target_array['value'] = self.target_value
        target_array['priority'] = self.target_priority
        return(target_array)

    def tofits(self, filename=None, clobber=True):
        """Write targets to a FITS file

        Parameters:
        ----------

        filename : str
            file name to write to

        clobber : boolean
            if True overwrite file, otherwise add an extension

        Notes:
        -----

        Writes header keywords:

            RACEN
            DECCEN
            FCADENCE (if determined)

        Tables has columns:

            'ra', 'dec' (np.float64)
            'pk' (np.int64)
            'cadence', 'type' ('a30')
            'priority' (np.int32)
            'category' ('a30')
"""
        hdr = dict()
        hdr['RACEN'] = self.racen
        hdr['DECCEN'] = self.deccen
        if(self.field_cadence is not None):
            hdr['FCADENCE'] = self.field_cadence
        tarray = self.targets_toarray()
        fitsio.write(filename, tarray, header=hdr, clobber=clobber)
        if(self.assignments is not None):
            fitsio.write(filename, self.assignments, clobber=False)
        return

    def plot(self, epochs=None):
        """Plot assignments of robots to targets for field

        Parameters:
        ----------

        epochs : list or ndarray, of int or np.int32
            list of epochs to plot (integers)
"""
        if(epochs is None):
            if(self.assignments is not None):
                epochs = np.arange(self.assignments.shape[1])
            else:
                epochs = np.arange(0)
        else:
            epochs = self._arrayify(epochs, dtype=np.int32)

        target_cadence = np.sort(np.unique(self.target_cadence))
        colors = ['black', 'green', 'blue', 'cyan', 'purple', 'red',
                  'magenta', 'grey']
        for indx in np.arange(len(target_cadence)):
            itarget = np.where(self.target_cadence ==
                               target_cadence[indx])[0]
            icolor = indx % len(colors)
            plt.scatter(self.target_x[itarget],
                        self.target_y[itarget], s=2, color=colors[icolor])

        if(self.assignments is not None):
            target_got = np.zeros(self.ntarget, dtype=np.int32)
            iassigned = np.where(self.assignments.flatten() >= 0)[0]
            itarget = self.assignments.flatten()[iassigned]
            target_got[itarget] = 1
            for indx in np.arange(len(target_cadence)):
                itarget = np.where((target_got > 0) &
                                   (self.target_cadence ==
                                    target_cadence[indx]))[0]
                icolor = indx % len(colors)
                plt.scatter(self.target_x[itarget],
                            self.target_y[itarget], s=20,
                            color=colors[icolor],
                            label=target_cadence[indx])

        realrobot = self.robot.apogee | self.robot.boss
        irobot = np.where(realrobot)[0]
        plt.scatter(self.robot.xcen[irobot], self.robot.ycen[irobot], s=6,
                    color='grey', label='Used robot')

        if(self.assignments is not None):
            used = (self.assignments >= 0).sum(axis=1) > 0
        else:
            used = np.zeros(len(self.robot.xcen), dtype=np.bool)

        inot = np.where((used == False) & realrobot)[0]
        plt.scatter(self.robot.xcen[inot], self.robot.ycen[inot], s=20,
                    color='grey', label='Unused robot')

        plt.xlim([-370., 370.])
        plt.ylim([-370., 370.])
        plt.legend()

    def assign_calibration(self, tcategory=None, ttype=None):
        """Assign calibration targets to robots within the field

        Notes:
        -----

        Assigns calibration targets. All it attempts at the moment
        is that a certain number will be assigned, according to the
        attribute:

           n{category}_{type}

        There is no guarantee regarding the spatial distribution.
        In addition, even the number is not guaranteed.

        The current method goes to each exposure, and does the following:

          * For each unassigned robot, tries to match it to
            one of the calibration targets. Assigns up to
            n{category}_{type} robots. It prefers robots used
            for calibration in previous exposures, but beyond
            that picks randomly.

          * If there are less than n{category}_{type} calibration
            targets assigned, for each robot assigned to a single
            exposure 'SCIENCE' target, tries to match it to one of the
            calibration targets. Assigns more calibration targets up
            to a total of n{category}_{type}, randomly selected. If
            there is more than one exposure in the field cadence,
            tries to assign the replaced targets back to their same
            fiber in an earlier (preferentially) or later exposure.

          * If there are still less than n{category}_{type}
            calibration targets assigned, for each robot assigned to a
            any other 'SCIENCE' target, tries to match it to one of
            the calibration targets. Assigns more calibration targets
            up to a total of n{category}_{type}. It prefers robots
            used for calibration in previous exposures, but beyond
            that picks randomly. The replaced targets are lost.

        This method is a hack. It will usually get the right number of
        calibration targets but isn't optimized.
"""
        icalib = np.where((self.target_category == tcategory) &
                          (self.target_type == ttype))[0]
        if(len(icalib) == 0):
            return

        # Match robots to targets (indexed into icalib)
        robot_targets = dict()
        for indx in np.arange(self.robot.npositioner):
            positionerid = self.robot.positionerid[indx]
            requires_boss = (ttype == 'BOSS')
            requires_apogee = (ttype == 'APOGEE')

            it = self.robot.targets(positionerid=positionerid,
                                    x=self.target_x[icalib],
                                    y=self.target_y[icalib],
                                    requires_apogee=requires_apogee,
                                    requires_boss=requires_boss)
            robot_targets[positionerid] = it

        ncalib = getattr(self, 'n{c}_{t}'.format(c=tcategory, t=ttype).lower())

        # Loop over exposures
        nexposures = self.cadencelist.cadences[self.field_cadence].nexposures

        robot_used = np.zeros(self.robot.npositioner, dtype=np.int32)
        for iexp in np.arange(nexposures, dtype=np.int32):
            calibration_assignments = (np.zeros(self.robot.npositioner,
                                                dtype=np.int32) - 1)

            # First, associate a calibration target with each robot
            # (preferentially but not necessarily one-to-one)
            robot_icalib = np.zeros(self.robot.npositioner, dtype=np.int32) - 1
            got_calib = np.zeros(len(icalib), dtype=np.int32)
            for indx in np.arange(self.robot.npositioner):
                # First try calibration targets not already taken
                it = robot_targets[positionerid]
                if(len(it) > 0):
                    ileft = np.where(got_calib[it] == 0)[0]
                    if(len(ileft) > 0):
                        robot_icalib[indx] = icalib[it[ileft[0]]]
                        got_calib[it[ileft[0]]] = 1

                # Then try any calibration targets
                if(robot_icalib[indx] == -1):
                    it = robot_targets[positionerid]
                    if(len(it) > 0):
                        robot_icalib[indx] = icalib[it[0]]
                        got_calib[it[0]] = 1

            # Now make ordered list of robots to use
            exposure_assignments = self.assignments[:, iexp]
            indx = np.where(exposure_assignments >= 0)[0]
            assignment_nexp = np.zeros(self.robot.npositioner, dtype=np.int32)
            iscience = np.where(self.target_category[exposure_assignments[indx]] == 'SCIENCE')[0]
            assignment_nexp[indx[iscience]] = np.array([
                self.cadencelist.cadences[x].nexposures
                for x in self.target_cadence[exposure_assignments[indx[iscience]]]])
            inot = np.where(self.target_category[exposure_assignments[indx]] != 'SCIENCE')[0]
            assignment_nexp[indx[inot]] = -1
            chances = np.random.random(size=self.robot.npositioner)
            sortby = (robot_used * (1 + chances) * 1 +
                      np.int32(assignment_nexp == 1) * (1 + chances) * 2 +
                      np.int32(assignment_nexp == 0) *
                      (1 + robot_used) * (1 + chances) * 4)
            indx_order = np.argsort(sortby)[::-1]

            # Set up calibration assignments in that priority
            nassigned = 0
            for indx in indx_order:
                if((robot_icalib[indx] >= 0) & (nassigned < ncalib) &
                   (assignment_nexp[indx] != -1)):
                    calibration_assignments[indx] = robot_icalib[indx]
                    robot_used[indx] = 1
                    nassigned = nassigned + 1

            # If there is a conflict with a single observation
            conflicts = ((calibration_assignments >= 0) &
                         (self.assignments[:, iexp] >= 0))
            single = (assignment_nexp == 1)
            isingle = np.where(conflicts & single)[0]
            for indx in isingle:
                ifree = np.sort(np.where(self.assignments[indx, :] == -1)[0])
                if(len(ifree) > 0):
                    itarget = self.assignments[indx, iexp]
                    self.assignments[indx, ifree] = itarget
                    self.assignments[indx, iexp] = -1

            # If there is a conflict with a multi-exposure observation
            multi = (assignment_nexp > 1)
            imulti = np.where(conflicts & multi)[0]
            for indx in imulti:
                iother = np.where(self.assignments[indx, :] ==
                                  self.assignments[indx, iexp])[0]
                if(len(iother) > 0):
                    self.assignments[indx, iother] = -1

            iassign = np.where(calibration_assignments >= 0)[0]
            self.assignments[iassign, iexp] = calibration_assignments[iassign]

    def assign(self, include_calibration=True):
        """Assign targets to robots within the field

        Parameters:
        ----------

        include_calibration : boolean
            Assign calibration targets if True, do not if False

        Notes:
        -----

        Field needs to have targets loaded into it. For each robot
        positioner, it searches for targets that are covered and that
        have not been assigned yet for a previous robot.

        It usually uses the pack_targets() method of CadenceList to
        pack the target cadences into the field cadence optimally.
        If the total number of exposures is greater than the value
        of the attribute "greedy_limit" (default 100) then it uses
        pack_targets_greedy().

        The results are stored in the attribute assignments, which is
        an (nposition, nexposures) array with the target index to
        observe for each positioner in each exposure of the field
        cadence.

        This method is optimal (in the usual case) for individual
        positioners, but not necessarily globally; i.e. trades of
        targets between positioners might be possible that would allow
        a better use of time.

        It first assigns targets in category 'SCIENCE', respecting
        cadence categories.

        Then for each exposure it assigns 'STANDARD' and then 'SKY'
        targets for 'APOGEE' and 'BOSS' fibers. It uses the 
        assign_calibration() method in each case.

        It does not use the target priorities yet.

"""

        # Initialize
        nexposures = self.cadencelist.cadences[self.field_cadence].nexposures
        self.assignments = (np.zeros((self.robot.npositioner, nexposures),
                                     dtype=np.int32) - 1)
        got_target = np.zeros(self.ntarget, dtype=np.int32)

        iscience = np.where(self.target_category == 'SCIENCE')[0]

        # Find which targets are viable at all
        ok_cadence = dict()
        for curr_cadence in np.unique(self.target_cadence[iscience]):
            ok = self.cadencelist.cadence_consistency(curr_cadence,
                                                      self.field_cadence,
                                                      return_solutions=False)
            ok_cadence[curr_cadence] = (
                ok | (self.cadencelist.cadences[curr_cadence].nepochs == 1))
        ok = [ok_cadence[tcadence]
              for tcadence in self.target_cadence[iscience]]
        iok = np.where(np.array(ok))[0]
        if(len(iok) == 0):
            return

        # Assign the robots
        target_requires_apogee = np.array(
            [self.cadencelist.cadences[c].requires_apogee
             for c in self.target_cadence[iscience]], dtype=np.int8)
        target_requires_boss = np.array(
            [self.cadencelist.cadences[c].requires_boss
             for c in self.target_cadence[iscience]], dtype=np.int8)
        for indx in np.arange(self.robot.npositioner):
            positionerid = self.robot.positionerid[indx]
            ileft = np.where(got_target[iok] == 0)[0]
            if(len(ileft) > 0):
                requires_apogee = target_requires_apogee[iok[ileft]]
                requires_boss = target_requires_boss[iok[ileft]]
                it = self.robot.targets(positionerid=positionerid,
                                        x=self.target_x[iscience[iok[ileft]]],
                                        y=self.target_y[iscience[iok[ileft]]],
                                        requires_apogee=requires_apogee,
                                        requires_boss=requires_boss)
                if(len(it) > 0):
                    ifull = iscience[iok[ileft[it]]]
                    if(nexposures < self.greedy_limit):
                        epoch_targets, itarget = (
                            self.cadencelist.pack_targets(
                                self.target_cadence[ifull],
                                self.field_cadence,
                                value=self.target_value[ifull]))
                    else:
                        epoch_targets, itarget = (
                            self.cadencelist.pack_targets_greedy(
                                self.target_cadence[ifull],
                                self.field_cadence,
                                value=self.target_value[ifull]))
                    iassigned = np.where(itarget >= 0)[0]
                    nassigned = len(iassigned)
                    if(nassigned > 0):
                        got_target[ifull[itarget[iassigned]]] = 1
                        self.assignments[indx, 0:nassigned] = (
                            ifull[itarget[iassigned]])

        if(include_calibration):
            self.assign_calibration(ttype='APOGEE', tcategory='SKY')
            self.assign_calibration(ttype='APOGEE', tcategory='STANDARD')
            self.assign_calibration(ttype='BOSS', tcategory='SKY')
            self.assign_calibration(ttype='BOSS', tcategory='STANDARD')

        self.set_target_assignments()

        return
