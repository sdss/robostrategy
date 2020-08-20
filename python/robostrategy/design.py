#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: field.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import fitsio
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import ortools.sat.python.cp_model as cp_model

import kaiju
import kaiju.robotGrid

# import observesim.robot as robot

__all__ = ['Design']

"""Design module class.

Dependencies:

 numpy
 fitsio
 matplotlib
 roboscheduler
 kaiju

"""

_target_array_dtype = np.dtype([('ra', np.float64),
                                ('dec', np.float64),
                                ('catalogid', np.int64),
                                ('category', np.unicode_, 30),
                                ('program', np.unicode_, 30),
                                ('fiberType', np.unicode_, 30),
                                ('priority', np.int32),
                                ('within', np.int32)])

alphaLen = 7.4
betaLen = 15


class DesignBase(object):
    """Design base class

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

    robotgrid : RobotGrid object
        instance of RobotGrid

    xVert : ndarray of np.float32
        x positions of vertices of hexagon bounding the field (mm)

    yVert : ndarray of np.float32
        y positions of vertices of hexagon bounding the field (mm)

    raVert : ndarray of np.float32
        RA positions of vertices of hexagon bounding the field (deg J2000)

    decVert : ndarray of np.float32
        Dec positions of vertices of hexagon bounding the field (deg J2000)

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
        category of targets ('APOGEE_SKY', 'APOGEE_STANDARD',
        'BOSS_SKY', 'BOSS_STANDARD', 'SCIENCE')

    target_catalogid : ndarray of np.int64
        unique catalogid for each target

    target_assigned : ndarray of np.int32
        (ntarget) array of 0 or 1, indicating whether target is assigned

    target_assignments : ndarray of np.int32
        (ntarget) array of positionerid for each target

    assignment : ndarray of np.int32
        (npositioner) array of catalogid for each positioner
"""
    def __init__(self, racen=None, deccen=None, pa=0.,
                 observatory='apo'):
        self.stepSize = 1  # for kaiju
        self.collisionBuffer = 2.0  # for kaiju
        self.robotgrid = self._robotGrid()
        self.robotID2indx = dict()
        self.indx2RobotID = dict()
        for i, k in enumerate(self.robotgrid.robotDict):
            self.robotID2indx[k] = i
            self.indx2RobotID[i] = k
        self.racen = racen
        self.deccen = deccen
        self.pa = pa  # assume deg E of N
        self.observatory = observatory
        if((self.racen != None) &
           (self.deccen != None)):
            self.set_vertices()
        self.assignments = None
        self.target_assigned = None
        self.target_assignments = None
        self.target_incadence = None
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
        for k in rg.robotDict.keys():
            rg.robotDict[k].setAlphaBeta(0., 180.)
        return(rg)

    def set_vertices(self):
        """Set vertices bounding the field"""
        maxReach = self.robotgrid.robotDict[1].getMaxReach()
        xPos = np.array([self.robotgrid.robotDict[r].xPos
                         for r in self.robotgrid.robotDict])
        yPos = np.array([self.robotgrid.robotDict[r].yPos
                         for r in self.robotgrid.robotDict])
        rPos = np.sqrt(xPos**2 + yPos**2)
        ivert = np.argsort(rPos)[-6:]
        xVert = np.zeros(6, dtype=np.float64)
        yVert = np.zeros(6, dtype=np.float64)
        for i, cvert in enumerate(ivert):
            rOut = rPos[cvert] + maxReach
            xVert[i] = xPos[cvert] * rOut / rPos[cvert]
            yVert[i] = yPos[cvert] * rOut / rPos[cvert]
        thVert = np.arctan2(yVert, xVert)
        isort = np.argsort(thVert)
        self.xVert = np.zeros(7, dtype=np.float64)
        self.yVert = np.zeros(7, dtype=np.float64)
        self.xVert[0:6] = xVert[isort]
        self.yVert[0:6] = yVert[isort]
        self.xVert[6] = self.xVert[0]
        self.yVert[6] = self.yVert[0]
        self.raVert, self.decVert = self.xy2radec(self.xVert, self.yVert)
        return

    def set_assignments(self):
        """Convert robotgrid assignments to array

        Notes:
        ------

        Sets attributes assignments and target_assignments
"""
        self.assignments = np.zeros(len(self.robotgrid.robotDict),
                                    dtype=np.int32) - 1
        self.target_assignments = np.zeros(self.ntarget,
                                           dtype=np.int32) - 1
        for robotID in self.robotgrid.robotDict:
            irobot = self.robotID2indx[robotID]
            if(self.robotgrid.robotDict[robotID].isAssigned()):
                catalogid = self.robotgrid.robotDict[robotID].assignedTargetID
                self.assignments[irobot] = catalogid
                tindx = self.catalogid2indx[catalogid]
                self.target_assignments[tindx] = robotID

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

    def _targets_fromarray_robotgrid(self):
        # Add all targets to robot grid.
        for itarget in np.arange(self.ntarget, dtype=np.int32):
            if(self.target_apogee[itarget]):
                fiberType = kaiju.ApogeeFiber
            else:
                fiberType = kaiju.BossFiber
            self.robotgrid.addTarget(targetID=self.target_catalogid[itarget],
                                     x=self.target_x[itarget],
                                     y=self.target_y[itarget],
                                     priority=self.target_priority[itarget],
                                     fiberType=fiberType)
        return

    def _targets_fromarray_within(self):
        self.target_within = np.zeros(self.ntarget, dtype=np.bool)
        for tid, t in self.robotgrid.targetDict.items():
            itarget = self.catalogid2indx[tid]
            self.target_within[itarget] = len(t.validRobotIDs) > 0
        return

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
         'catalogid' should be np.int64
         'category' should be str or bytes
         'fiberType' should be 'APOGEE' or 'BOSS'

        Optional columns of array:
         'priority'
         'category'
         'program'
"""

        # Copy over data from array
        self.target_array = target_array
        self.ntarget = len(self.target_array)
        self.target_ra = self.target_array['ra']
        if(type(self.target_ra[0]) != np.float64):
            print("WARNING: TARGET_RA NOT 64-bit")
        self.target_dec = self.target_array['dec']
        if(type(self.target_dec[0]) != np.float64):
            print("WARNING: TARGET_DEC NOT 64-bit")
        self.target_catalogid = self.target_array['catalogid']

        # Optional data
        if('priority' in self.target_array.dtype.names):
            self.target_priority = self.target_array['priority']
        else:
            self.target_priority = np.ones(self.ntarget, dtype=np.int32)
        if('category' in self.target_array.dtype.names):
            self.target_category = np.array(
                [c.strip() for c in self.target_array['category']])
        else:
            self.target_category = np.array(['SCIENCE'] * self.ntarget)
        if('program' in self.target_array.dtype.names):
            self.target_program = np.array(
                [c.strip() for c in self.target_array['program']])
        else:
            self.target_program = np.array(['PROGRAM'] * self.ntarget)

        # Build dictionary for catalogid2indx
        self.catalogid2indx = dict()
        for itarget in np.arange(self.ntarget, dtype=np.int32):
            self.catalogid2indx[self.target_catalogid[itarget]] = itarget

        self.target_x, self.target_y = self.radec2xy(self.target_ra,
                                                     self.target_dec)

        self.target_apogee = np.array(self.target_array['fiberType'] ==
                                      'APOGEE')
        self.target_boss = np.array(self.target_array['fiberType'] ==
                                    'BOSS')

        self._targets_fromarray_robotgrid()
        self._targets_fromarray_within()

        return

    def targets_fromfits(self, filename=None):
        """Read targets from a FITS file

        Parameters:
        ----------

        filename : str
            FITS file name, for file with columns listed below

        Notes:
        ------

        Required columns of array:
         'ra', 'dec' should be np.float64
         'catalogid' should be np.int64
         'category' should be str or bytes
         'fiberType' should be 'APOGEE' or 'BOSS'

        Optional columns of array:
         'priority'
         'category'
         'program'
"""
        target_array = fitsio.read(filename)
        self.targets_fromarray(target_array)
        return

    def fromfits(self, filename=None):
        """Read design from a FITS file

        Parameters:
        ----------

        filename : str
            FITS file name, where HDU 2 has array of assignments
"""
        hdr = fitsio.read_header(filename, ext=1)
        self.racen = np.float64(hdr['RACEN'])
        self.deccen = np.float64(hdr['DECCEN'])
        self.pa = np.float32(hdr['PA'])
        self.observatory = hdr['OBS']
        self.targets_fromfits(filename)
        f = fitsio.FITS(filename)
        if(len(f) > 2):
            self.assignments = fitsio.read(filename, ext=2)
            self.set_target_assignments()
        return

    def targets_toarray(self):
        """Write targets to an ndarray

        Returns:
        -------

        target_array : ndarray
            Array of targets, with columns:
              'ra', 'dec' (np.float64)
              'catalogid' (np.int64)
              'fiberType' (np.int32)
              'within' (np.int32)
              'priority' (np.int32)
              'category' ('a30')
              'program' ('a30')
"""
        target_array = np.zeros(self.ntarget, dtype=_target_array_dtype)
        target_array['ra'] = self.target_ra
        target_array['dec'] = self.target_dec
        target_array['catalogid'] = self.target_catalogid
        target_array['category'] = self.target_category
        target_array['program'] = self.target_program
        target_array['fiberType'] = ['BOSS' if c else 'APOGEE'
                                     for c in self.target_boss]
        target_array['priority'] = self.target_priority
        target_array['within'] = self.target_within
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
            PA

        Tables has columns:

            'ra', 'dec' (np.float64)
            'pa' (np.float32)
            'cadence', 'type' (np.unicode_)
            'priority' (np.int32)
            'category' (np.unicode_)
            'program' (np.unicode_)
            'fiberType' (np.unicode_)
"""
        hdr = dict()
        hdr['RACEN'] = self.racen
        hdr['DECCEN'] = self.deccen
        hdr['PA'] = self.pa
        hdr['OBS'] = self.observatory
        tarray = self.targets_toarray()
        fitsio.write(filename, tarray, header=hdr, clobber=clobber)
        if(self.assignments is not None):
            fitsio.write(filename, self.assignments, clobber=False)
        return

    def plot_robot(self, robot, color=None):
        xr = robot.xPos
        yr = robot.yPos
        xa = xr + alphaLen * np.cos(robot.alpha /
                                    180. * np.pi)
        ya = yr + alphaLen * np.sin(robot.alpha /
                                    180. * np.pi)
        xb = xa + betaLen * np.cos((robot.alpha + robot.beta) /
                                   180. * np.pi)
        yb = ya + betaLen * np.sin((robot.alpha + robot.beta) /
                                   180. * np.pi)
        plt.plot(np.array([xr, xa]), np.array([yr, ya]),
                 color=color, alpha=0.5)
        plt.plot(np.array([xa, xb]), np.array([ya, yb]),
                 color=color, linewidth=3)

    def plot(self, robotID=False, catalogid=False):
        """Plot assignments of robots to targets for field """
        target_programs = np.sort(np.unique(self.target_program))

        colors = ['black', 'green', 'blue', 'cyan', 'purple', 'red',
                  'magenta', 'grey']

        if(self.assignments is not None):
            target_got = np.zeros(self.ntarget, dtype=np.int32)
            target_robotid = np.zeros(self.ntarget, dtype=np.int32)
            iassigned = np.where(self.assignments >= 0)[0]
            itarget = np.array([self.catalogid2indx[x] for x in
                                self.assignments[iassigned]])
            target_got[itarget] = 1
            target_robotid[itarget] = self.target_assignments[itarget]
            for indx in np.arange(len(target_programs)):
                itarget = np.where((target_got > 0) &
                                   (self.target_program ==
                                    target_programs[indx]))[0]

                plt.scatter(self.target_x[itarget],
                            self.target_y[itarget], s=4)

                icolor = indx % len(colors)
                for i in itarget:
                    robot = self.robotgrid.robotDict[target_robotid[i]]
                    self.plot_robot(robot, color=colors[icolor])

        for indx in np.arange(len(target_programs)):
            itarget = np.where(self.target_program ==
                               target_programs[indx])[0]
            icolor = indx % len(colors)
            plt.scatter(self.target_x[itarget],
                        self.target_y[itarget], s=2, color=colors[icolor])

        xcen = np.array([self.robotgrid.robotDict[r].xPos
                         for r in self.robotgrid.robotDict],
                        dtype=np.float32)
        ycen = np.array([self.robotgrid.robotDict[r].yPos
                         for r in self.robotgrid.robotDict],
                        dtype=np.float32)
        robotid = np.array([str(r)
                            for r in self.robotgrid.robotDict])
        plt.scatter(xcen, ycen, s=6, color='grey', label='Used robot')

        if(robotID):
            for cx, cy, cr in zip(xcen, ycen, robotid):
                plt.text(cx, cy, cr, color='grey', fontsize=8,
                         clip_on=True)

        if(catalogid):
            for cx, cy, ct in zip(self.target_x, self.target_y,
                                  self.target_catalogid):
                plt.text(cx, cy, ct, fontsize=8, clip_on=True)

        used = (self.assignments >= 0)

        inot = np.where(used == False)[0]
        plt.scatter(xcen[inot], ycen[inot], s=20, color='grey',
                    label='Unused robot')
        for i in robotid[inot]:
            self.plot_robot(self.robotgrid.robotDict[int(i)],
                            color='grey')

        plt.xlim([-370., 370.])
        plt.ylim([-370., 370.])
        plt.legend()


class DesignGreedy(DesignBase):
    def __init__(self, racen=None, deccen=None, pa=0.,
                 observatory='apo'):
        super().__init__(racen=racen, deccen=deccen, pa=pa,
                         observatory=observatory)
        return

    def assign(self):

        # Sort by priority, and randomly for ties
        iorder = np.arange(self.ntarget, dtype=np.int32)
        np.random.shuffle(iorder)
        isort = iorder[np.argsort(self.target_priority[iorder])]

        count = 0
        for i in isort:
            catalogid = self.target_catalogid[i]
            t = self.robotgrid.targetDict[catalogid]
            for robotID in t.validRobotIDs:
                if(self.robotgrid.robotDict[robotID].isAssigned() == False):
                    self.robotgrid.assignRobot2Target(robotID, catalogid)
                    if(self.robotgrid.isCollidedWithAssigned(robotID) == True):
                        self.robotgrid.decollideRobot(robotID)
                    else:
                        count = count + 1

        for robotID in self.robotgrid.robotDict:
            if(self.robotgrid.isCollidedWithAssigned(robotID) == True):
                if(self.robotgrid.robotDict[robotID].isAssigned()):
                    print("INCONSISTENCY.")
                self.robotgrid.decollideRobot(robotID)

        self.set_assignments()


class DesignOptimize(DesignBase):
    def __init__(self, racen=None, deccen=None, pa=0.,
                 observatory='apo'):
        super().__init__(racen=racen, deccen=deccen, pa=pa,
                         observatory=observatory)
        return

    def assign(self, check_collisions=True):
        """Assigns using CP-SAT to optimize number of targets

        Parameters
        ----------

        check_collisions : boolean
            whether to add collision constraints (default True)

        Notes
        -----

        Assigns the robots in the robotGrid object attribute "robotgrid"
        Sets ndarray attributes "assignments" and "target_assignments"
"""

        rg = self.robotgrid

        # Initialize Model
        model = cp_model.CpModel()

        # Add variables; one for each robot-target pair
        # Make a dictionary to organize them as wwrt[robotID][catalogid],
        # and one to organize them as wwtr[catalogid][robotID], and
        # also a flattened list
        wwrt = dict()
        wwtr = dict()
        for robotID in rg.robotDict:
            r = rg.robotDict[robotID]
            for catalogid in r.validTargetIDs:
                name = 'ww[{r}][{c}]'.format(r=robotID, c=catalogid)
                if(catalogid not in wwtr):
                    wwtr[catalogid] = dict()
                if(robotID not in wwrt):
                    wwrt[robotID] = dict()
                wwrt[robotID][catalogid] = model.NewBoolVar(name)
                wwtr[catalogid][robotID] = wwrt[robotID][catalogid]
        ww_list = [wwrt[y][x] for y in wwrt for x in wwrt[y]]

        # Constrain only one target per robot
        wwsum_robot = dict()
        for robotID in wwrt:
            rlist = [wwrt[robotID][c] for c in wwrt[robotID]]
            wwsum_robot[robotID] = cp_model.LinearExpr.Sum(rlist)
            model.Add(wwsum_robot[robotID] <= 1)

        # Constrain only one robot per target
        wwsum_target = dict()
        for catalogid in wwtr:
            tlist = [wwtr[catalogid][r] for r in wwtr[catalogid]]
            wwsum_target[catalogid] = cp_model.LinearExpr.Sum(tlist)
            model.Add(wwsum_target[catalogid] <= 1)

        # Do not allow collisions
        if(check_collisions):

            # Find potention collisions
            collisions = []
            for robotID1 in rg.robotDict:
                r1 = rg.robotDict[robotID1]
                for catalogid1 in r1.validTargetIDs:
                    rg.assignRobot2Target(robotID1, catalogid1)
                    for robotID2 in r1.robotNeighbors:
                        r2 = rg.robotDict[robotID2]
                        for catalogid2 in r2.validTargetIDs:
                            if(catalogid1 != catalogid2):
                                rg.assignRobot2Target(robotID2, catalogid2)
                                if(rg.isCollidedWithAssigned(robotID1)):
                                    collisions.append((robotID1,
                                                       catalogid1,
                                                       robotID2,
                                                       catalogid2))
                                rg.homeRobot(robotID2)
                    rg.homeRobot(robotID1)

            # Now add constraint that collisions can't occur
            for robotID1, catalogid1, robotID2, catalogid2 in collisions:
                ww1 = wwrt[robotID1][catalogid1]
                ww2 = wwrt[robotID2][catalogid2]
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

        if status == cp_model.OPTIMAL:
            print('Count: {ov}'.format(ov=solver.ObjectiveValue()))

            for robotID in wwrt:
                for catalogid in wwrt[robotID]:
                    assigned = solver.Value(wwrt[robotID][catalogid])
                    if(assigned):
                        rg.assignRobot2Target(robotID, catalogid)
                        if(rg.isCollidedWithAssigned(robotID)):
                            print("Unexpected collision occurred:")
                            print(" r1, c1: {r} {c}".format(r=robotID,
                                                            c=catalogid))
                            rids = rg.robotColliders(robotID)
                            for rid in rids:
                                cid = rg.robotDict[rid].assignedTargetID
                                print(" ro, co: {r} {c}".format(r=rid,
                                                                c=cid))
            for robotID in rg.robotDict:
                if(rg.robotDict[robotID].isAssigned() is False):
                    if(rg.isCollided(robotID)):
                        rg.decollideRobot(robotID)

        self.set_assignments()


class DesignOptimalFast(DesignBase):
    """Test class. Not actually faster."""

    def __init__(self, racen=None, deccen=None, pa=0.,
                 observatory='apo'):
        super().__init__(racen=racen, deccen=deccen, pa=pa,
                         observatory=observatory)
        return

    def assign(self):

        # Initialize Model
        model = cp_model.CpModel()

        # Add variables; one for each robot; store as a
        # dictionary but also as a list.
        ww = dict()
        assigned = dict()
        for robotID in self.robotgrid.robotDict:
            r = self.robotgrid.robotDict[robotID]
            tlist = [int(x) for x in r.validTargetIDs]  # allowed targets
            tlist = [int(- robotID - 1)] + tlist  # unassigned -> robotID (unique)
            dom = cp_model.Domain.FromValues(tlist)
            name = 'ww[{r}]'.format(r=robotID)
            ww[robotID] = model.NewIntVarFromDomain(dom, name)
            name = 'assigned[{r}]'.format(r=robotID)
            assigned[robotID] = model.NewBoolVar(name)
            model.Add(ww[robotID] >= 0).OnlyEnforceIf(assigned[robotID])
            model.Add(ww[robotID] < 0).OnlyEnforceIf(assigned[robotID].Not())

        ww_list = [ww[robotID] for robotID in ww]
        assigned_list = [assigned[robotID] for robotID in assigned]

        # Constrain only one robot per target (i.e. you can't set two
        # robots to the same target). Every robot has the option
        # of not being assigned (i.e. of "- robotID - 1", which is unique)
        model.AddAllDifferent(ww_list)

        # Maximize the total sum
        assigned_sum = cp_model.LinearExpr.Sum(assigned_list)
        model.Maximize(assigned_sum)

        # Search for values in decreasing order.
        model.AddDecisionStrategy(assigned_list,
                                  cp_model.CHOOSE_FIRST,
                                  cp_model.SELECT_MAX_VALUE)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        print(solver.StatusName(status=status))

        if status == cp_model.OPTIMAL:
            print('Count: {ov}'.format(ov=solver.ObjectiveValue()))
