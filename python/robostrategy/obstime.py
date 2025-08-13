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

    date : str
        date to start on ('YYYY-MM-DD' format)

    Attributes
    ----------

    observatory : str
        'apo' or 'lco'

    utcoff : int
        offset of local time from UTC

    transit_lst : ndarray of np.float64
        [365] LST (deg) transiting at each local standard midnight of year

    midnights : list of datetime.datetime objects
        [365] datetime format for each local standard midnight of year

    Notes
    -----

    This class provides a way to assign a nominal observation time for
    a given LST.

    nominal() returns the local midnight at which the the LST is
    closest to transiting. It differs slightly from this at the 0/360
    deg boundary of LSTs.

    It uses SDSS's coordio for the astronomy calculation.

    """
    def __init__(self, observatory='apo', date='2025-08-01'):
        if(date is None):
            date = '2025-08-01'
        year, month, mday = [int(x) for x in date.split('-')]
        self.observatory = observatory
        if(observatory == 'apo'):
            self.utcoff = - 7
        if(observatory == 'lco'):
            self.utcoff = - 4

        oneday = datetime.timedelta(days=1)
        onehour = datetime.timedelta(hours=1)

        site = coordio.site.Site(self.observatory.upper())

        self.transit_lst = np.zeros(365, dtype=np.float64)
        self.midnight = []

        day = datetime.datetime(year, month, mday) - self.utcoff * onehour
        for n in range(365):
            midnight = day + oneday * n
            site.set_time(midnight)
            try:
                south = coordio.sky.Observed([[45., 180.]], site=site)
            except coordio.exceptions.CoordIOError as e:
                outstr = "{e} (tried date {d})".format(e=e, d=date)
                raise coordio.exceptions.CoordIOError(outstr)
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
