import numpy as np

nzone = 8


def apogee_standard_goodness(magnitude=None):
    """Returns goodness of an APOGEE standard based on magnitudes

    Parameters
    ----------

    magnitude : ndarray of np.float32
        [10] or [N, 10] array of grizBGRJHK

    Returns
    -------

    goodness : np.float32 or ndarray of np.float32
        for each star, how good the standard is


    Notes
    -----

    Goodness is defined based on magnitude and color as:
        - 40 * ((J-K) - 0.1) - 3 * (H - 11)

    This definition is based on the mean absolute deviation
    plots of telluric feature models from pipeline compared
    to those derived directly from the stars.

    If any of J, H, K is 0, negative, or greater than
    20, then goodness is -999.
"""
    j = magnitude[..., 7]
    h = magnitude[..., 8]
    k = magnitude[..., 9]
    goodness = - 40 * ((j - k) - 0.1) - 3. * (h - 11.)
    bad = ((j <= 0) | (j > 20.) |
           (h <= 0) | (h > 20.) |
           (k <= 0) | (k > 20.))
    if(len(j.shape) > 0):
        goodness[bad] = - 999.
    else:
        goodness = - 999.
        
    return(goodness)


def standard_zone(x=None, y=None):
    """Returns zone(s) associated with a location

    Parameters
    ----------

    x : np.float64 (or ndarray of np.float64)
        X position in focal plane (mm)

    y : np.float64 (or ndarray of np.float64)
        Y position in focal plane (mm)

    Returns
    -------

    zone : np.int32 or ndarray of np.int32
        for each location, which zone is it in

    Notes
    -----

    There are 8 zones. Zones 0 and 1 are within a radius of 122mm of 
    center, and Zones 2-7 are outside that radius.
"""
    zone = np.zeros(len(x), dtype=int)
    radius = 122.
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y ,x)
    inside = r < radius
    zone[inside] = (x[inside] > 0)
    outside = r >= radius
    zone[outside] = 2 + np.floor(((th[outside] + 7. * np.pi / 6.) %
                                  (2. * np.pi))/ (np.pi / 3.))
    return(zone)
