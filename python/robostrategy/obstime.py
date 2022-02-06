#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: obstime.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import coordio
import datetime


class ObsTime(object):
    """Class for finding appropriate observing times

    Parameters
    ----------

    observatory : str
        'apo' or 'lco'

    year : int
        nominal year to consider (default 2021)

    Attributes
    ----------

    observatory : str
        'apo' or 'lco'

    year : int
        nominal year to consider

    utcoff : int
        offset of local time from UTC

    transit_lst : ndarray of np.float64
        [365] LST (deg) transiting at each local standard midnight of year

    midnights : list of datetime.datetime objects
        [365] datetime format for each local standard midnight of year

    Methods
    -------

    nominal(lst=) : returns nominal observing time for a given RA

    Notes
    -----

    This class provides a way to assign a nominal observation time for
    a given LST.

    nominal() returns the local midnight at which the the LST is
    closest to transiting. It differs slightly from this at the 0/360
    deg boundary of LSTs.

    It uses SDSS's coordio for the astronomy calculation.

    """
    def __init__(self, observatory='apo', year=2021):
        self.observatory = observatory
        self.year = year
        if(observatory == 'apo'):
            self.utcoff = - 7
        if(observatory == 'lco'):
            self.utcoff = - 4

        oneday = datetime.timedelta(days=1)
        onehour = datetime.timedelta(hours=1)

        site = coordio.site.Site(self.observatory.upper())

        self.transit_lst = np.zeros(365, dtype=np.float64)
        self.midnight = []

        day = datetime.datetime(year, 1, 1) - self.utcoff * onehour
        for n in range(365):
            midnight = day + oneday * n
            site.set_time(midnight)
            south = coordio.sky.Observed([[45., 180.]], site=site)
            self.transit_lst[n] = south.ra
            self.midnight.append(midnight)

        return

    def nominal(self, lst=None):
        """Return a nominal observation time for a given LST

        Parameters
        ----------

        lst : np.float64 or float
            LST desired for the observation (deg)

        Returns
        -------

        nominal_time : datetime object
            datetime object describing the midnight at which this LST
            is closest to transiting.

        Notes
        -----

        At 0/360 boundary picks the closest night to that boundary.
        This should be a very minor effect (few minutes).
"""
        imin = np.abs(self.transit_lst - lst).argmin()
        return(self.midnight[imin])
