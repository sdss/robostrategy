# @Author: Michael R. Blanton
# @Date: Aug 3, 2018
# @Filename: slots.py
# @License: BSD 3-Clause
# @Copyright: Michael R. Blanton

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import fitsio
import robostrategy
import roboscheduler
import roboscheduler.scheduler


class Slots(object):
    """Slots of time, in LST and sky brightness

    Parameters
    ----------

    nlst : int, np.int32
        number of LST bins (default 24)

    skybrightness : list of float or np.float32
        edges of sky brightness bins in increasing order (N+1 length
        for N bins; default [0., 0.35, 1.])

    observatory : str
        observatory to calculate slots for, 'apo' or 'lco' (default 'apo')

    exptimes : ndarray of np.float32
        duration of a single nominal exposure, in hours (2-element array in dark, bright order) (defaults 15. / 60.)

    exposure_overhead : float, np.float32
        overhead of a single exposure, in hours (default 3. / 60.)

    schedule : str
        name of schedule to use (default 'normal')

    fclear : float, np.float32
        fraction of clear time; note this does not change the numbers in the
        slots array (default 0.5 if observatory='apo', 0.7 if observatory='lco')

    Attributes
    ----------

    mjd_start : int
        MJD to start counting schedule from

    nlst : int, np.int32
        number of LST bins

    lst : ndarray of np.float32
        centers of LST bins (hours)

    skybrightness : list of float or np.float32
        edges of sky brightness bins (N+1 length for N bins)

    nskybrightness : int
        number of sky brightness bins N

    observatory : str
        observatory to calculate slots for, 'apo' or 'lco'

    exptimes : ndarray of np.float32
        durations of a single nominal exposure, in hours, in dark and bright

    exposure_overhead : float, np.float32
        overhead of a single exposure, in hours

    schedule : str
        name of schedule to use

    fclear : float, np.float32
        fraction of clear time; note this does not change the numbers in the
        slots array

    durations : ndarray np.float32
        duration of a single nominal exposure plus overhead, in hours, in dark and bright

    slots : ndarray of np.float32
        number of available hours in LST, sky brightness slots
        [nlst, nskybrightness], created only when fill() is called

    Notes
    -----

    fclear is not applied to the number of hours in slots. It is there
    just as a place to set the assumption of clear hours for the allocation
    code to consult.
"""
    def __init__(self, nlst=24, skybrightness=[0., 0.35, 1.],
                 observatory='apo',
                 exptimes=np.array([15., 15.], dtype=np.float32) / 60.,
                 exposure_overhead=3. / 60.,
                 schedule='normal', fclear=None):
        self.nlst = nlst
        self.lst = ((np.arange(nlst, dtype=np.float32) + 0.5) * 24. /
                    np.float32(self.nlst))
        self.skybrightness = np.array(skybrightness)
        self.nskybrightness = len(skybrightness) - 1
        self.observatory = observatory
        self.schedule = schedule
        if(fclear is None):
            if(self.observatory == 'apo'):
                self.fclear = 0.5
            if(self.observatory == 'lco'):
                self.fclear = 0.7
        else:
            self.fclear = fclear
        self.exposure_overhead = exposure_overhead
        self.exptimes = exptimes
        self.durations = self.exposure_overhead + self.exptimes
        self.mjd_start = None
        return

    def fill(self, mjd_start=None):
        """Fill slots attribute with available hours

        Parameters
        ----------
        
        mjd_start : int or None
            start planning at this MJD (i.e. only count time after this)

        Notes
        -----

        Sets the attribute slots to an ndarray of np.float32 with
        shape (nlst, nskybrightness).

        Uses roboscheduler to step through every night of the survey
        and count the number of hours per LST and skybrightness.

        Does NOT apply the fclear factor.
"""
        self.mjd_start = mjd_start
        self.slots = np.zeros((self.nlst, self.nskybrightness),
                              dtype=np.float32)
        scheduler = roboscheduler.scheduler.Scheduler(observatory=self.observatory,
                                                      schedule=self.schedule)
        for mjd in scheduler.mjds:
            # Do not count MJDs in the past
            if(mjd_start is not None):
                if(mjd < mjd_start):
                    continue

            mjd_evening_twilight = scheduler.evening_twilight(mjd)
            mjd_morning_twilight = scheduler.morning_twilight(mjd)
            curr_mjd = mjd_evening_twilight
            on, next_mjd = scheduler.on(curr_mjd)
            if(on == 'off'):
                continue
            curr_duration = 0.
            while(curr_mjd + curr_duration < mjd_morning_twilight and
                  curr_mjd + curr_duration < scheduler.end_mjd()):
                lst = scheduler.lst(curr_mjd)
                skybrightness = scheduler.skybrightness(curr_mjd)
                ilst = np.int32(np.floor((lst / 15. / 24.) * self.nlst))
                iskybrightness = np.where((skybrightness >=
                                           self.skybrightness[0:-1]) &
                                          (skybrightness <=
                                           self.skybrightness[1:]))[0][0]
                self.slots[ilst,
                           iskybrightness] = (self.slots[ilst,
                                                         iskybrightness] +
                                              self.durations[iskybrightness])
                curr_duration = self.durations[iskybrightness] / 24.
                curr_mjd = curr_mjd + curr_duration
        return

    def tofits(self, filename=None, clobber=True):
        """Write slots information to FITS file

        Parameters
        ----------

        filename : str
            file name to write to

        clobber : boolean
            whether to overwrite existing file (default True)

        Notes
        -----

        Will fail if the slots attribute has not yet been set.

        Writes header keywords (NLST, DURATION, FCLEAR, OBSERVAT, NSB,
        and SB0..SB[NSB+1]) with object attributes.

        Writes slots attribute as a binary image.
"""
        hdr = dict()
        hdr['STRATVER'] = robostrategy.__version__
        hdr['SCHEDVER'] = roboscheduler.__version__
        hdr['NLST'] = self.nlst
        hdr['DURATION_DARK'] = self.durations[0]
        hdr['DURATION_BRIGHT'] = self.durations[1]
        hdr['EXPOVER'] = self.exposure_overhead
        hdr['EXPTIME_DARK'] = self.exptimes[0]
        hdr['EXPTIME_BRIGHT'] = self.exptimes[1]
        hdr['FCLEAR'] = self.fclear
        hdr['OBSERVAT'] = self.observatory
        hdr['NSB'] = self.nskybrightness
        if(self.mjd_start is not None):
            hdr['MJD_START'] = self.mjd_start
        for indx in range(len(self.skybrightness)):
            hdr['SB{indx}'.format(indx=indx)] = self.skybrightness[indx]
        fitsio.write(filename, self.slots, header=hdr, clobber=clobber, extname='SLOTS')
        return

    def fromfits(self, filename=None, ext=0):
        """Read slots information from FITS file

        Parameters
        ----------

        filename : str
            file name to read from

        Notes
        ------

        Assumes the FITS file is of the form written by the tofits()
        method.
"""
        self.slots, hdr = fitsio.read(filename, ext=ext, header=True)
        self.nlst = np.int32(hdr['NLST'])
        if('MJD_START' in hdr):
            self.mjd_start = int(hdr['MJD_START'])
        if('DURATION' in hdr):
            self.durations = np.array([np.float32(hdr['DURATION']),
                                      np.float32(hdr['DURATION'])],
                                     dtype=np.float32)
        if(('DURATION_DARK' in hdr) &
           ('DURATION_BRIGHT' in hdr)):
            self.durations = np.array([np.float32(hdr['DURATION_DARK']),
                                       np.float32(hdr['DURATION_BRIGHT'])],
                                      dtype=np.float32)
        if('EXPTIME' in hdr):
            self.exptimes = np.array([np.float32(hdr['EXPTIME']),
                                      np.float32(hdr['EXPTIME'])],
                                     dtype=np.float32)
        if(('EXPTIME_DARK' in hdr) &
           ('EXPTIME_BRIGHT' in hdr)):
            self.exptimes = np.array([np.float32(hdr['EXPTIME_DARK']),
                                      np.float32(hdr['EXPTIME_BRIGHT'])],
                                     dtype=np.float32)
        if('EXPOVER' in hdr):
            self.exposure_overhead = np.float32(hdr['EXPOVER'])
        self.fclear = np.float32(hdr['FCLEAR'])
        self.observatory = hdr['OBSERVAT']
        self.nskybrightness = np.int32(hdr['NSB'])
        self.skybrightness = np.zeros(self.nskybrightness + 1,
                                      dtype=np.float32)
        for indx in range(len(self.skybrightness)):
            self.skybrightness[indx] = hdr['SB{indx}'.format(indx=indx)]
        return()
