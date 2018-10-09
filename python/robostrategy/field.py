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
 astropy
 matplotlib

"""


class Field(object):
    """Field class

    Parameters:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    Attributes:
    ----------

    racen : np.float64
        boresight RA, J2000 deg

    deccen : np.float64
        boresight Dec, J2000 deg

    field_cadence : int, np.int32
        index of field cadence in cadencelist

    robot : Robot class
        instance of Robot

    target_x : ndarray of np.float64
        x positions of targets

    target_y : ndarray of np.float64
        y positions of targets

    target_cadence : ndarray of np.int32
        cadences of targets

    target_type : ndarray of strings
        target types ('boss' or 'apogee')

    Methods:
    -------

    read_cadences() : read cadences from a file
    read_targets() : read targets from a file
    assign() : assign targets to robots for cadence

    Notes:
    -----

    This class is definitely going to need to be refactored (please
    email me if you are reading this comment in 2025 ...).

"""
    def __init__(self, racen=None, deccen=None,
                 db=True, fps_layout='filled_hex'):
        self.robot = robot.Robot(db=db, fps_layout=fps_layout)
        self.racen = racen
        self.deccen = deccen
        self.cadencelist = cadence.CadenceList()
        self.field_cadence = None
        self.assignments = None
        return

    def _arrayify(self, quantity=None, dtype=np.float64):
        """Cast quantity as ndarray of numpy.float64"""
        try:
            length = len(quantity)
        except TypeError:
            length = 1
        return np.zeros(length, dtype=dtype) + quantity

    def radec2xy(self, ra=None, dec=None):
        # Yikes!
        scale = 218.
        x = (ra - self.racen) * np.cos(self.deccen * np.pi / 180.) * scale
        y = (dec - self.deccen) * scale
        return(x, y)

    def xy2radec(self, x=None, y=None):
        # Yikes!
        scale = 218.
        ra = self.racen + (x / scale) / np.cos(self.deccen * np.pi / 180.)
        dec = self.deccen + (y / scale)
        return(ra, dec)

    def targets_fromarray(self, target_array):
        self.target_array = target_array
        self.ntarget = len(self.target_array)
        self.target_ra = self.target_array['ra']
        self.target_dec = self.target_array['dec']
        self.target_pk = self.target_array['pk']
        self.target_x, self.target_y = self.radec2xy(self.target_ra,
                                                     self.target_dec)
        self.target_cadence = np.array([c.decode().strip()
                                        for c in self.target_array['cadence']])
        self.target_type = np.array([t.decode().strip()
                                     for t in self.target_array['type']])
        return

    def targets_toarray(self):
        target_array_dtype = np.dtype([('ra', np.float64),
                                       ('dec', np.float64),
                                       ('pk', np.int64),
                                       ('cadence', np.dtype('a20')),
                                       ('type', np.dtype('a20'))])

        target_array = np.zeros(self.ntarget, dtype=target_array_dtype)
        target_array['ra'] = self.target_ra
        target_array['dec'] = self.target_dec
        target_array['pk'] = self.target_pk
        target_array['cadence'] = self.target_cadence
        target_array['type'] = self.target_type
        return(target_array)

    def tofits(self, filename=None, clobber=True):
        hdr = dict()
        hdr['RACEN'] = self.racen
        hdr['DECCEN'] = self.deccen
        if(self.field_cadence is not None):
            hdr['FCADENCE'] = self.field_cadence
        tarray = self.targets_toarray()
        fitsio.write(filename, tarray, header=hdr, clobber=clobber)
        if(self.assignments is not None):
            fitsio.write(filename, self.assignments, clobber=False)

    def targets_fromfits(self, filename=None):
        target_array = fitsio.read(filename)
        self.targets_fromarray(target_array)
        return

    def plot(self, epochs=None):
        if(epochs is None):
            epochs = np.arange(self.assignments.shape[1])
        else:
            epochs = self._arrayify(epochs, dtype=np.int32)
        colors = ['black', 'green', 'blue', 'cyan', 'purple']
        plt.scatter(self.robot.xcen, self.robot.ycen, s=3, color='black')
        plt.scatter(self.target_x, self.target_y, s=3, color='red')
        for irobot in np.arange(self.assignments.shape[0]):
            for iepoch in np.array(epochs):
                icolor = iepoch % len(colors)
                itarget = self.assignments[irobot, iepoch]
                if(itarget >= 0):
                    xst = self.robot.xcen[irobot]
                    yst = self.robot.ycen[irobot]
                    xnd = self.target_x[itarget]
                    ynd = self.target_y[itarget]
                    plt.plot([xst, xnd], [yst, ynd], color=colors[icolor])

    def assign(self):
        nexposures = self.cadencelist.cadences[self.field_cadence].nexposures
        self.assignments = (np.zeros((self.robot.npositioner, nexposures),
                                     dtype=np.int32) - 1)
        got_target = np.zeros(self.ntarget, dtype=np.int32)
        ok_cadence = dict()
        print(np.unique(self.target_cadence))
        for curr_cadence in np.unique(self.target_cadence):
            print(curr_cadence)
            ok = self.cadencelist.cadence_consistency(curr_cadence, self.field_cadence, return_solutions=False)
            ok_cadence[curr_cadence] = (ok |
                                        (self.cadencelist.cadences[curr_cadence].nepochs == 1))
        ok = [ok_cadence[tcadence] for tcadence in self.target_cadence]
        iok = np.where(np.array(ok))[0]
        if(len(iok) == 0):
            return
        target_requires_apogee = np.array([self.cadencelist.cadences[c].requires_apogee
                                           for c in self.target_cadence],
                                          dtype=np.int8)
        target_requires_boss = np.array([self.cadencelist.cadences[c].requires_boss
                                         for c in self.target_cadence],
                                        dtype=np.int8)
        for indx in np.arange(self.robot.npositioner):
            positionerid = self.robot.positionerid[indx]
            print(positionerid)
            ileft = np.where(got_target[iok] == 0)[0]
            if(len(ileft) > 0):
                requires_apogee = target_requires_apogee[iok[ileft]]
                requires_boss = target_requires_boss[iok[ileft]]
                it = self.robot.targets(positionerid=positionerid,
                                        x=self.target_x[iok[ileft]],
                                        y=self.target_y[iok[ileft]],
                                        requires_apogee=requires_apogee,
                                        requires_boss=requires_boss)
                if(len(it) > 0):
                    print(positionerid)
                    print(self.target_cadence[iok[ileft[it]]])
                    print(self.field_cadence)
                    epoch_targets, itarget = self.cadencelist.pack_targets(self.target_cadence[iok[ileft[it]]], self.field_cadence)
                    iassigned = np.where(itarget >= 0)[0]
                    nassigned = len(iassigned)
                    if(nassigned > 0):
                        got_target[iok[ileft[it[itarget[iassigned]]]]] = 1
                        self.assignments[indx, 0:nassigned] = iok[ileft[it[itarget[iassigned]]]]
        return
