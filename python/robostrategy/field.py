#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import json
import numpy as np
import fitsio
import matplotlib.pyplot as plt

import roboscheduler.cadence as cadence
import kaiju.robotGrid

# import observesim.robot as robot

__all__ = ['Field']

"""Field module class.

Dependencies:

 numpy
 fitsio
 matplotlib
 roboscheduler
 kaiju

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

    robotgrids : list
        instances of RobotGrid, one per exposure

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

    target_within : ndarray of np.int32
        1 if target is within the robot hexagon, 0 otherwise

    target_priority : ndarray of np.int32
        priorities of targets (lower is considered first)

    target_program : ndarray of strings
        program of targets

    target_category : ndarray of strings
        category of targets ('SKY', 'STANDARD', 'SCIENCE')

    target_pk : ndarray of np.int64
        unique primary key for each target

    target_cadence : ndarray of np.int32
        cadences of targets

    target_incadence : ndarray of np.bool
        whether each target is allowed in the field cadence (set by assign())

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
                 db=True, observatory='apo'):
        self.stepSize = 1  # for kaiju
        self.collisionBuffer = 2.0  # for kaiju
        self.mastergrid = self._robotGrid()
        self.robotgrids = []
        self.stepSize = 1
        self.racen = racen
        self.deccen = deccen
        self.observatory = observatory
        self.cadencelist = cadence.CadenceList()
        self.set_field_cadence('none')
        self.assignments = None
        self.target_assigned = None
        self.target_assignments = None
        self.target_incadence = None
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

    def _robotGrid(self):
        """Return a RobotGrid instance"""
        rg = kaiju.robotGrid.RobotGridFilledHex(collisionBuffer=self.collisionBuffer)
        for r in rg.allRobots:
            r.setAlphaBeta(0., 180.)
        return(rg)

    def set_target_assignments(self):
        """Convert assignments array to per-target basis

        Notes:
        ------

        Sets attributes target_assignment and target_assigned based on
        the assignments attribute values.
"""
        if(self.assignments is None):
            return

        nexp = self.nexposures
        self.target_assignments = np.zeros((self.ntarget, nexp),
                                           dtype=np.int32) - 1
        self.target_assigned = np.zeros(self.ntarget, dtype=np.int32)
        for iexp in np.arange(nexp, dtype=np.int32):
            rg = self.mastergrid
            for irobot in np.arange(rg.nRobots, dtype=np.int32):
                curr_assignment = self.assignments[irobot, iexp]
                if(curr_assignment >= 0):
                    self.target_assigned[curr_assignment] = 1
                    self.target_assignments[curr_assignment, iexp] = rg.getRobot(irobot).id

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
         'program'
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
            self.target_program = np.array(
                [c.decode().strip() for c in self.target_array['program']])
        except AttributeError:
            self.target_program = np.array(
                [c.strip() for c in self.target_array['program']])
        except ValueError:
            self.target_program = np.array(['PROGRAM'] * self.ntarget)

        self.target_requires_apogee = np.zeros(self.ntarget, dtype=np.int8)
        iscience = np.where(self.target_category == 'SCIENCE')[0]
        self.target_requires_apogee[iscience] = [self.cadencelist.cadences[c].requires_apogee
                                                 for c in self.target_cadence[iscience]]
        self.target_requires_boss = np.zeros(self.ntarget, dtype=np.int8)
        self.target_requires_boss[iscience] = [self.cadencelist.cadences[c].requires_boss
                                               for c in self.target_cadence[iscience]]
        inotscience = np.where(self.target_category != 'SCIENCE')[0]
        ttype = [t.split('_')[-1] for t in self.target_category[inotscience]]
        self.target_requires_apogee[inotscience] = (ttype == 'APOGEE')
        self.target_requires_boss[inotscience] = (ttype == 'BOSS')

        # Add all targets to master grid.
        self.tlist = []
        for itarget in np.arange(self.ntarget, dtype=np.int32):
            if(self.target_requires_apogee[itarget]):
                fiberID = 1
            else:
                fiberID = 2
            self.tlist.append([itarget,
                               self.target_x[itarget],
                               self.target_y[itarget],
                               self.target_priority[itarget],
                               fiberID])

        self.mastergrid.setTargetList(self.tlist)
        self.target_id2indx = dict()
        self.target_indx2id = dict()
        for i, t in enumerate(self.mastergrid.targetList):
            self.target_id2indx[t.id] = i
            self.target_indx2id[i] = t.id

        self.target_within = np.array([(len(t.robotInds) > 0)
                                       for t in self.mastergrid.targetList],
                                      dtype=np.bool)

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
         'cadence' should be str or bytes

        Optional columns:
         'priority'
         'category'
         'program'
"""
        target_array = fitsio.read(filename)
        self.targets_fromarray(target_array)
        return

    def set_field_cadence(self, field_cadence='none'):
        self.field_cadence = field_cadence
        if(self.field_cadence != 'none'):
            self.nexposures = self.cadencelist.cadences[self.field_cadence].nexposures
        else:
            self.nexposures = 0
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
        self.set_field_cadence(hdr['FCADENCE'].strip())
        if((self.field_cadence != 'none') & (read_assignments)):
            self.assignments = fitsio.read(filename, ext=2)
        self.targets_fromfits(filename)
        self.make_robotgrids()
        if((self.field_cadence != 'none') & (read_assignments)):
            self.set_target_assignments()
        if(read_assignments):
            self._assignments_to_grids()
        return

    def targets_toarray(self):
        """Write targets to an ndarray

        Returns:
        -------

        target_array : ndarray
            Array of targets, with columns:
              'ra', 'dec' (np.float64)
              'pk' (np.int64)
              'cadence' ('a30')
              'priority' (np.int32)
              'category' ('a30')
              'program' ('a30')
"""
        target_array_dtype = np.dtype([('ra', np.float64),
                                       ('dec', np.float64),
                                       ('pk', np.int64),
                                       ('cadence', cadence.fits_type),
                                       ('category', np.dtype('a30')),
                                       ('program', np.dtype('a30')),
                                       ('value', np.int32),
                                       ('priority', np.int32)])

        target_array = np.zeros(self.ntarget, dtype=target_array_dtype)
        target_array['ra'] = self.target_ra
        target_array['dec'] = self.target_dec
        target_array['pk'] = self.target_pk
        target_array['cadence'] = self.target_cadence
        target_array['category'] = self.target_category
        target_array['program'] = self.target_program
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
            'program' ('a30')
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

        xcen = np.array([r.xPos for r in self.mastergrid.allRobots],
                        dtype=np.float32)
        ycen = np.array([r.yPos for r in self.mastergrid.allRobots],
                        dtype=np.float32)
        plt.scatter(xcen, ycen, s=6, color='grey', label='Used robot')

        if(self.assignments is not None):
            used = (self.assignments >= 0).sum(axis=1) > 0
        else:
            used = np.zeros(self.mastergrid.nRobots, dtype=np.bool)

        inot = np.where(used == False)[0]
        plt.scatter(xcen[inot], ycen[inot], s=20, color='grey',
                    label='Unused robot')

        plt.xlim([-370., 370.])
        plt.ylim([-370., 370.])
        plt.legend()

    def assign_calibration(self, category=None, kaiju=True):
        """Assign calibration targets to robots within the field

        Notes:
        -----

        Assigns calibration targets. All it attempts at the moment
        is that a certain number will be assigned, according to the
        attribute:

           n{category}

        There is no guarantee regarding the spatial distribution.
        In addition, even the number is not guaranteed.

        The current method goes to each exposure, and does the following:

          * For each unassigned robot, tries to match it to
            one of the calibration targets. Assigns up to
            n{category} robots. It prefers robots used
            for calibration in previous exposures, but beyond
            that picks randomly.

          * If there are less than n{category} calibration
            targets assigned, for each robot assigned to a single
            exposure 'SCIENCE' target, tries to match it to one of the
            calibration targets. Assigns more calibration targets up
            to a total of n{category}, randomly selected. If
            there is more than one exposure in the field cadence,
            tries to assign the replaced targets back to their same
            fiber in an earlier (preferentially) or later exposure.

          * If there are still less than n{category}
            calibration targets assigned, for each robot assigned to a
            any other 'SCIENCE' target, tries to match it to one of
            the calibration targets. Assigns more calibration targets
            up to a total of n{category}. It prefers robots
            used for calibration in previous exposures, but beyond
            that picks randomly. The replaced targets are lost.

        This method is a hack. It will usually get the right number of
        calibration targets but isn't optimized.
"""
        iscalib = (self.target_category == category)
        icalib = np.where(iscalib)[0]
        if(len(icalib) == 0):
            return

        ttype = category.split("_")[-1]

        # Match robots to targets (indexed into icalib)
        curr_robot_targets = dict()
        for rindx, robot in enumerate(self.mastergrid.allRobots):
            requires_boss = (ttype == 'BOSS')
            requires_apogee = (ttype == 'APOGEE')

            curr_robot_targets[rindx] = np.zeros(0, dtype=np.int32)
            if(len(robot.targetList) > 0):
                robot_targets = np.array([self.target_indx2id[x]
                                          for x in robot.targetList])
                curr_icalib = np.where((iscalib[robot_targets] > 0) &
                                       ((requires_boss == 0) |
                                        (robot.hasBoss > 0)) &
                                       ((requires_apogee == 0) |
                                        (robot.hasApogee > 0)))[0]
                if(len(curr_icalib) > 0):
                    curr_robot_targets[rindx] = robot_targets[curr_icalib]

        ncalib = getattr(self, 'n{c}'.format(c=category).lower())

        # Loop over exposures
        robot_used = np.zeros(self.mastergrid.nRobots, dtype=np.int32)
        for iexp, rg in enumerate(self.robotgrids):
            calibration_assignments = (np.zeros(self.mastergrid.nRobots,
                                                dtype=np.int32) - 1)

            # Initial consistency check
            if(kaiju):
                for indx in np.arange(rg.nRobots):
                    if(rg.allRobots[indx].isAssigned() is False):
                        if(self.assignments[indx, iexp] >= 0):
                            print("PRECHECK {i}: UH OH DID NOT ASSIGN ROBOT".format(i=iexp))
                    elif(self.target_indx2id[rg.allRobots[indx].assignedTarget] !=
                         self.assignments[indx, iexp]):
                        print("PRECHECK: UH OH ROBOT DOES NOT MATCH ASSIGNMENT")

            # Now make ordered list of robots to use
            exposure_assignments = self.assignments[:, iexp]
            indx = np.where(exposure_assignments >= 0)[0]
            assignment_nexp = np.zeros(self.mastergrid.nRobots,
                                       dtype=np.int32)
            iscience = np.where(self.target_category[exposure_assignments[indx]] == 'SCIENCE')[0]
            assignment_nexp[indx[iscience]] = np.array([
                self.cadencelist.cadences[x].nexposures
                for x in self.target_cadence[exposure_assignments[indx[iscience]]]])
            inot = np.where(self.target_category[exposure_assignments[indx]] != 'SCIENCE')[0]
            assignment_nexp[indx[inot]] = -1
            chances = np.random.random(size=self.mastergrid.nRobots)
            sortby = (robot_used * (1 + chances) * 1 +
                      np.int32(assignment_nexp == 1) * (1 + chances) * 2 +
                      np.int32(assignment_nexp == 0) *
                      (1 + robot_used) * (1 + chances) * 4)
            indx_order = np.argsort(sortby)[::-1]

            # Set up calibration assignments in that priority
            got_calib = np.zeros(self.ntarget, dtype=np.int32)
            nassigned = 0
            for indx in indx_order:
                icalib = curr_robot_targets[indx]
                for itry in icalib:
                    if(got_calib[itry] == 0):
                        got = True
                        if(kaiju):
                            if(rg.allRobots[indx].isAssigned()):
                                try:
                                    rg.unassignRobot(indx)
                                except:
                                    print("unassign failure 1")
                            rg.assignRobot2Target(indx, itry)
                            if(rg.isCollidedWithAssigned(rg.allRobots[indx]) == False):
                                got = True
                            else:
                                got = False
                        if(got):
                            calibration_assignments[indx] = self.target_indx2id[itry]
                            got_calib[itry] = 1
                            robot_used[indx] = 1
                            nassigned = nassigned + 1
                            break
                        if(kaiju):
                            try:
                                rg.unassignRobot(indx)
                            except RuntimeError:
                                print("unassign failure 2")
                            if(self.assignments[indx, iexp] >= 0):
                                rg.assignRobot2Target(indx,
                                                      self.assignments[indx, iexp])
                if(nassigned >= ncalib):
                    break

            # If there is a conflict with a single observation
            # swap with another exposure (need to check collision)
            conflicts = ((calibration_assignments >= 0) &
                         (self.assignments[:, iexp] >= 0))
            single = (assignment_nexp == 1)
            isingle = np.where(conflicts & single)[0]
            for indx in isingle:
                ifree = np.sort(np.where(self.assignments[indx, :] == -1)[0])
                for itry in ifree:
                    itarget = self.assignments[indx, iexp]
                    if(kaiju):
                        self.robotgrids[itry].assignRobot2Target(indx,
                                                                 itarget)
                        ica = self.robotgrids[itry].isCollidedWithAssigned(self.robotgrids[itry].allRobots[indx])
                    else:
                        ica = False
                    if(ica == False):
                        self.assignments[indx, itry] = itarget
                        self.assignments[indx, iexp] = -1
                        #if(kaiju):
                            # if((iexp == 0) & (indx == 361)):
                                # print("HERE!!")
                            #self.robotgrids[iexp].unassignRobot(indx)
                        break
                    if(kaiju):
                        try:
                            self.robotgrids[itry].unassignRobot(indx)
                        except:
                            print("unassign failure 3")

            # If there is a conflict with a multi-exposure observation
            # just completely unassign (need to actually unassign)
            multi = (assignment_nexp > 1)
            imulti = np.where(conflicts & multi)[0]
            for indx in imulti:
                iother = np.where(self.assignments[indx, :] ==
                                  self.assignments[indx, iexp])[0]
                for iun in iother:
                    self.assignments[indx, iun] = -1
                    try:
                        self.robotgrids[iun].unassignRobot(indx)
                    except:
                        print("unassign failure4")

            iassign = np.where(calibration_assignments >= 0)[0]
            self.assignments[iassign, iexp] = calibration_assignments[iassign]
            if(kaiju):
                for indx in np.arange(rg.nRobots):
                    if(rg.allRobots[indx].isAssigned() is False):
                        if(self.assignments[indx, iexp] >= 0):
                            print("UH OH DID NOT ASSIGN ROBOT")
                            print("indx={indx}".format(indx=indx))
                            print("iexp={iexp}".format(iexp=iexp))
                            print("itarget={itarget}".format(itarget=self.assignments[indx, iexp]))
                    elif(self.target_indx2id[rg.allRobots[indx].assignedTarget] !=
                         self.assignments[indx, iexp]):
                        print("UH OH ROBOT DOES NOT MATCH ASSIGNMENT")

        if(kaiju):
            for iexp, rg in enumerate(self.robotgrids):
                for indx in np.arange(rg.nRobots):
                    if(rg.allRobots[indx].isAssigned() is False):
                        if(self.assignments[indx, iexp] >= 0):
                            print("UH OH DID NOT ASSIGN ROBOT")
                        if(rg.isCollidedInd(indx)):
                            try:
                                rg.unassignRobot(indx)
                            except:
                                print("unassign failure 5")

    def make_robotgrids(self):
        self.robotgrids = []
        for i in np.arange(self.nexposures):
            self.robotgrids.append(self._robotGrid())
            ta = self.mastergrid.target_array()
            self.robotgrids[i].clearTargetList()
            self.robotgrids[i].target_fromarray(ta)
        return

    def assign(self, include_calibration=True, kaiju=True):
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

        It uses the pack_targets_greedy() method of CadenceList to
        pack the target cadences into the field cadence greedily.

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

        assign() also sets the target_incadence attribute, which
        tells you wich targets fit somehow into the field cadence.
"""

        # Initialize
        np.random.seed(int(self.racen))
        nexposures = self.nexposures
        self.assignments = (np.zeros((self.mastergrid.nRobots, nexposures),
                                     dtype=np.int32) - 1)
        got_target = np.zeros(self.ntarget, dtype=np.int32)

        iscience = np.where(self.target_category == 'SCIENCE')[0]

        # Find which targets are viable at all
        ok_cadence = dict()
        for curr_cadence in np.unique(self.target_cadence[iscience]):
            ok, s = self.cadencelist.cadence_consistency(curr_cadence,
                                                         self.field_cadence,
                                                         return_solutions=True)
            ok_cadence[curr_cadence] = (
                ok | (self.cadencelist.cadences[curr_cadence].nepochs == 1))
        ok = [ok_cadence[tcadence]
              for tcadence in self.target_cadence[iscience]]
        self.target_incadence = np.zeros(self.ntarget, dtype=np.int32)
        iok = np.where(np.array(ok))[0]
        self.target_incadence[iscience[iok]] = 1
        if(len(iok) == 0):
            return

        # Set up robotgrids
        if(kaiju):
            self.make_robotgrids()
        else:
            self.robotgrids = [None] * self.nexposures

        # Assign the robots
        nexp = self.nexposures
        allRobots = self.mastergrid.allRobots
        doneRobots = np.zeros(self.mastergrid.nRobots, dtype=np.bool)
        for indx in np.arange(self.mastergrid.nRobots, dtype=np.int32):
            irobot = self._next_robot(allRobots=allRobots,
                                      doneRobots=doneRobots,
                                      got_target=got_target,
                                      kaiju=kaiju)
            doneRobots[irobot] = True
            cRobot = allRobots[irobot]
            if(len(cRobot.targetList) > 0):
                itargets = np.array([self.target_indx2id[x]
                                     for x in cRobot.targetList])
                it = np.where((got_target[itargets] == 0) &
                              (self.target_incadence[itargets] > 0) &
                              ((self.target_requires_boss[itargets] == 0) |
                               (cRobot.hasBoss > 0)) &
                              ((self.target_requires_apogee[itargets] == 0) |
                               (cRobot.hasApogee > 0)))[0]
                if(len(it) > 0):
                    ifull = itargets[it]
                    # Create mask to pass to pack_targets_greedy() based
                    # on collisions
                    emask = np.zeros((len(ifull), nexp), dtype=np.bool)
                    if(kaiju):
                        for tindx, itarget in enumerate(ifull):
                            for iexp in np.arange(nexp, dtype=np.int32):
                                self.robotgrids[iexp].assignRobot2Target(irobot,
                                                                         itarget)
                                eRobot = self.robotgrids[iexp].allRobots[irobot]
                                if(self.robotgrids[iexp].isCollidedWithAssigned(eRobot)):
                                    emask[tindx, iexp] = True
                                # Reset robot -- perhaps there are more elegant ways
                                try:
                                    self.robotgrids[iexp].unassignRobot(irobot)
                                except:
                                    print("unassign failure 6")
                    p = cadence.Packing(self.field_cadence)
                    p.pack_targets_greedy(
                        target_ids=ifull,
                        target_cadences=self.target_cadence[ifull],
                        value=self.target_value[ifull],
                        exposure_mask=emask)
                    itarget = p.exposures  # make sure this returns targetid
                    iassigned = np.where(itarget >= 0)[0]
                    nassigned = len(iassigned)
                    if(nassigned > 0):
                        got_target[itarget[iassigned]] = 1
                    self.assignments[irobot, :] = itarget
                    if(kaiju):
                        for iexp, rg in enumerate(self.robotgrids):
                            ctarget = self.assignments[irobot, iexp]
                            if(ctarget >= 0):
                                try:
                                    rg.assignRobot2Target(irobot, ctarget)
                                except RuntimeError:
                                    print(irobot)
                                    print(iexp)
                                    print(ctarget)
                                    print(rg.allRobots[irobot].targetList)
                                    print([rg.targetList[x].id
                                           for x in rg.allRobots[irobot].targetList])
                                    sys.exit(1)
                            else:
                                try:
                                    rg.unassignRobot(irobot)
                                except:
                                    print("unassign failure 7")
                            if(rg.isCollidedWithAssigned(rg.allRobots[irobot])):
                                print("INCONSISTENCY")

        # Explicitly unassign all unassigned robots so they
        # are out of the way.
        if(kaiju):
            for iexp, rg in enumerate(self.robotgrids):
                iun = np.where(self.assignments[:, iexp] < 0)[0]
                for irobot in iun:
                    try:
                        rg.unassignRobot(irobot)
                    except:
                        print("unassign failure 8")

        # Make sure all assigned robots are assigned
        if(kaiju):
            for iexp, rg in enumerate(self.robotgrids):
                for indx in np.arange(rg.nRobots):
                    if(rg.allRobots[indx].isAssigned() is False):
                        if(self.assignments[indx, iexp] >= 0):
                            print("UH OH DID NOT ASSIGN ROBOT")
                    elif(self.target_indx2id[rg.allRobots[indx].assignedTarget] !=
                         self.assignments[indx, iexp]):
                        print("UH OH ROBOT DOES NOT MATCH ASSIGNMENT")

        if(include_calibration):
            self.assign_calibration(category='SKY_APOGEE', kaiju=kaiju)
            self.assign_calibration(category='STANDARD_APOGEE', kaiju=kaiju)
            self.assign_calibration(category='SKY_BOSS', kaiju=kaiju)
            self.assign_calibration(category='STANDARD_BOSS', kaiju=kaiju)

        if(kaiju):
            for iexp, rg in enumerate(self.robotgrids):
                for indx in np.arange(rg.nRobots):
                    if(rg.allRobots[indx].isAssigned() is False):
                        if(self.assignments[indx, iexp] >= 0):
                            print("UH OH DID NOT ASSIGN ROBOT")
                        if(rg.isCollidedInd(indx)):
                            try:
                                rg.unassignRobot(indx)
                            except:
                                print("unassign failure 9")

        self.set_target_assignments()

        return

    def tojs(self, filename):
        """Write Javascript for each exposure

        Parameters:
        ----------

        filename : str
            name of file to write to

        Comments:
        ---------

        Javascript assigns target information to target_obj dict
        and robot information to robot_obj, an array of dicts
"""
        ii = np.where(self.target_within)[0]
        nwithin = len(ii)
        js_str = "{"
        js_str = js_str + '"racen" : ' + str(self.racen) + ',\n'
        js_str = js_str + '"deccen" : ' + str(self.deccen) + ',\n'
        js_str = js_str + '"field_cadence" : "' + str(self.field_cadence) + '",\n'
        js_str = js_str + '"observatory" : "' + str(self.observatory) + '",\n'
        js_str = js_str + '"ntarget" : ' + str(self.ntarget) + ',\n'
        js_str = js_str + '"nwithin" : ' + str(nwithin) + ',\n'
        js_str = js_str + '"target_obj" : ' + json.dumps(self.mastergrid.target_dict()) + ',\n'
        js_str = js_str + '"robot_obj" : [\n'
        for iexp, rg in enumerate(self.robotgrids):
            js_str = js_str +\
                json.dumps(rg.robot_dict())
            if(iexp < len(self.robotgrids) - 1):
                js_str = js_str + ","
            js_str = js_str + "\n"
        js_str = js_str + ']}'
        fp = open(filename, "w")
        fp.write(js_str)
        fp.close()
        return

    def html(self, filename):
        """Write HTML format file for visualizing assignments

        Parameters:
        ----------

        filename : str
            name of file to write to
"""
        html_str = '<script type="text/javascript">\n'
        html_str = html_str + 'target_obj = ' + json.dumps(self.mastergrid.target_dict()) + ";\n"
        html_str = html_str + 'var robot_obj = new Array();\n'
        for iexp, rg in enumerate(self.robotgrids):
            html_str = html_str +\
                'robot_obj[{iexp}] = '.format(iexp=iexp) +\
                json.dumps(rg.robot_dict()) + ";\n"
        html_str = html_str + '</script>\n'
        fp = open(os.path.join(os.getenv('ROBOSTRATEGY_DIR'), 'data',
                               'field.html'), "r")
        for l in fp.readlines():
            html_str = html_str + l
        fp.close()
        fp = open(filename, "w")
        fp.write(html_str)
        fp.close()
        return

    def _next_robot(self, allRobots=None, doneRobots=None, got_target=None,
                    kaiju=True):
        """Get next robot in order of highest priority of remaining targets"""
        maxPriority = np.zeros(len(allRobots), dtype=np.int32) - 9999
        for indx, robot in enumerate(allRobots):
            if(doneRobots[indx] == np.bool(False)):
                if(len(robot.targetList) > 0):
                    itargets = np.array([self.target_indx2id[x]
                                         for x in robot.targetList])
                    inot = np.where(got_target[itargets] == 0)[0]
                    if(len(inot) > 0):
                        it = itargets[inot]
                        maxPriority[indx] = self.target_priority[it].max()
                else:
                    maxPriority[indx] = - 99999
        imax = np.argmax(maxPriority)
        return(imax)

    def add_observations(self):
        """For assigned targets, add observations if possible

        Notes:
        -----

        The assign() method needs to have been run so that the
        assignments attribute is set.

        The code then adds as many more observations as it can of
        the targets already observed by this fiber. Note that it is
        looking for full new observations.

        It does not use the target priorities yet.
"""

        # Assign the robots
        for indx in np.arange(self.mastergrid.nRobots):
            targetids = np.unique(self.assignments[indx, :])[0]
            igd = np.where(targetids != -1)[0]
            igd2 = np.where(self.target_category[igd] != 'CALIBRATION')[0]
            targetids = targetids[igd[igd2]]
            tcs = self.target_cadence[targetids]
            if(len(targetids) > 0):
                p = cadence.Packing(self.field_cadence)
                p.import_exposures(self.assignments[indx, :])
                ok = True
                while(ok):
                    ok = False
                    for tc in tcs:
                        if(p.add_target(tc)):
                            ok = True

        return

    def _assignments_to_grids(self):
        """Transfer assignments to RobotGrid objects"""
        nexp = self.nexposures
        for i in np.arange(nexp):
            for rindx, robot in enumerate(self.robotgrids[i].allRobots):
                itarget = self.assignments[rindx, i]
                if(itarget >= 0):
                    self.robotgrids[i].assignRobot2Target(rindx, itarget)
            for rindx, robot in enumerate(self.robotgrids[i].allRobots):
                itarget = self.assignments[rindx, i]
                if(itarget < 0):
                    try:
                        self.robotgrids[i].unassignRobot(rindx)
                    except:
                        print("unassign failure 10")
        return

    def apply_kaiju(self):
        """Apply kaiju conditions to field. Deprecated"""
        self._assignments_to_grids()
        nexposures = self.nexposures
        self.kaiju_assignments = (np.zeros((self.mastergrid.nRobots,
                                            nexposures), dtype=np.int32) - 1)
        for iexp, rg in enumerate(self.robotgrids):
            rg.pairwiseSwap()
            # for r in self.robotgrids[iexp].allRobots:
                # if(r.isCollided()):
                    # print(r.assignedTarget.id)
            try:
                rg.decollide()
                success = True
            except:
                print("Decollision failed")
                success = False

            if(success):
                for rindx, r in enumerate(rg.allRobots):
                    if(r.isAssigned()):
                        targetid = self.target_indx2id[r.assignedTarget]
                        if(targetid >= 0):
                            self.kaiju_assignments[rindx, iexp] = targetid
