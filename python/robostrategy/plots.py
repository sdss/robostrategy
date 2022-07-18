#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: plots.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shapely.geometry
import shapely.ops
import shapely.speedups
from descartes import PolygonPatch
import astropy.coordinates as coords
import kaiju.robotGrid
import robostrategy.obstime as obstime
import coordio.time
import coordio.utils

ot = dict()
ot['apo'] = obstime.ObsTime(observatory='apo')
ot['lco'] = obstime.ObsTime(observatory='lco')

if(shapely.speedups.available):
    shapely.speedups.enable()

try:
    import mpl_toolkits.basemap as basemap
except ImportError:
    basemap = None


def plot_field_points(fields=None):
    def on_pick(event):
        ind = event.ind
        ind = ind[0]
        print('fieldid : {fid}'.format(fid=fields['fieldid'][ind]))
        print(' ra/dec : {r:4f}, {d:4f}'.format(r=fields['racen'][ind],
                                                d=fields['deccen'][ind]))
        print(' l/b    : {l:4f}, {b:4f}'.format(l=l[ind], b=b[ind]))

    c = coords.SkyCoord(fields['racen'], fields['deccen'], unit='deg',
                        frame='icrs')
    b = c.galactic.b.value
    l = c.galactic.l.value

    fig, ax = plt.subplots()

    tolerance = 10 # points
    ax.scatter(fields['racen'], fields['deccen'], s=2, picker=tolerance, c=np.abs(b), cmap=cm.plasma)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()


def xyhex():
    hex_radius = 315.5
    n_vertex = 6
    angle0 = np.pi / 3.0

    # these are the pixel coords of the vertices on the tangent plane
    vertex_x = hex_radius * np.array([np.cos(- 2. * np.pi *
                                             (i / float(n_vertex)) +
                                             angle0)
                                      for i in np.arange(n_vertex)])

    vertex_y = hex_radius * np.array([np.sin(- 2. * np.pi *
                                             (i / float(n_vertex)) +
                                             angle0)
                                      for i in np.arange(n_vertex)])
    return(vertex_x, vertex_y)


def radechex(racen=180., deccen=0., pa=0., observatory='apo'):
    vertex_x, vertex_y = xyhex()
    wavename = np.array(['Boss'] * len(vertex_x))
    otime = coordio.time.Time(ot[observatory].nominal(lst=racen))
    ra, dec, warn = coordio.utils.wokxy2radec(vertex_x,
                                              vertex_y,
                                              wavename,
                                              racen,
                                              deccen,
                                              pa,
                                              observatory.upper(),
                                              otime.jd)
    return(ra, dec)

def ra_transform(ras):
    ras = 360. - ras  # flip
    ras = ras + 90.  # put original 270 at 180.
    ras = ras % 360.  # put range back in 0..360
    ras = ras - 180.  # shift range to -180..180
    return(ras)


def fieldshape(racen=180., deccen=0., pa=0., observatory='apo'):
    ras, decs = radechex(racen=racen, deccen=deccen, pa=pa, observatory=observatory)
    ras = ra_transform(ras)
    xy = np.array([ras, decs]).transpose()
    fsh = shapely.geometry.Polygon(xy)
    return(fsh)

def plot_field_shapes(racen=None, deccen=None, pa=None, observatory=None, types=None,
                      type2color=None):

    if(type2color is None):
        type2color = dict()
        type2color['AllSkySloane'] = 'black'
        type2color['BHMAqmesMedium'] = 'orange'
        type2color['BHMAqmesWide2'] = 'yellow'
        type2color['RM'] = 'red'
        type2color['RMlite'] = 'red'
        
    fieldshapes = []
    for i in np.arange(len(racen), dtype=int):
        print(i)
        fieldshapes.append(fieldshape(racen=racen[i], deccen=deccen[i],
                                      pa=pa[i], observatory=observatory[i]))
    allfieldshapes = shapely.geometry.MultiPolygon(fieldshapes)

    fig, ax = plt.subplots(figsize=(12, 6.))
    plt.subplots_adjust(0.005, 0.005, 0.995, 0.995, 0., 0.)
    
    m = basemap.Basemap(projection='moll', lon_0=0, resolution='c')
    
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 30.),
                    linewidth=0.5,
                    labels=[0, 0, 0, 0],
                    labelstyle='+/-')
    m.drawmeridians(np.arange(0., 420., 60.), linewidth=0.5)
    m.drawmapboundary()

    patches = []
    for i, fsh in enumerate(fieldshapes):
        if(types is None):
            color = 'black'
        else:
            color = type2color[types[i]]
        
        ras = np.array([x[0] for x in fsh.exterior.coords])
        decs = np.array([x[1] for x in fsh.exterior.coords])
        if(ras.max() - ras.min() < 180.):
            mpoly = shapely.ops.transform(m, fsh)
            patches.append(PolygonPatch(mpoly, fc=color, ec=None, alpha=0.5,
                                        linewidth=0))
        else:
            ras1 = ras
            ii = np.where(ras1 < 0.)[0]
            ras1[ii] = ras1[ii] + 360.
            p1 = shapely.geometry.box(0., -90., 180., 90.)
            xy1 = zip(ras1, decs)
            fsh1 = shapely.geometry.Polygon(xy1)
            fsh1 = fsh1.intersection(p1)
            if(fsh1.area > 0):
                mpoly = shapely.ops.transform(m, fsh1)
                patches.append(PolygonPatch(mpoly, fc=color, ec=None, alpha=0.5))
            ras2 = ras
            ii = np.where(ras2 > 0.)[0]
            ras2[ii] = ras2[ii] - 360.
            p2 = shapely.geometry.box(-180., -90., 0., 90.)
            xy2 = zip(ras2, decs)
            fsh2 = shapely.geometry.Polygon(xy2)
            fsh2 = fsh2.intersection(p2)
            if(fsh2.area > 0):
                mpoly = shapely.ops.transform(m, fsh2)
                patches.append(PolygonPatch(mpoly, fc=color, ec=None, alpha=0.5))
    ax.add_collection(matplotlib.collections.PatchCollection(patches, match_original=True))

    return


def plot_field_allocation(allocate=None, fieldid=None, ax=None, xlabel=True,
                          ylabel=True, legend='full'):

    if(ax is None):
        fig, ax = plt.subplots()

    ifield = np.where(allocate.field_array['fieldid'] == fieldid)[0][0]
    ioptions = np.where(allocate.field_options['fieldid'] == fieldid)[0]
    ioption = np.where((allocate.field_options['fieldid'] == fieldid) &
                       (allocate.field_options['cadence'] == allocate.field_array['cadence'][ifield]))[0][0]

    cadence = allocate.field_array['cadence'][ifield]
    exposures = allocate.field_array['slots_exposures'][ifield, :, :].sum(axis=1)
    dexposures = allocate.field_array['slots_exposures'][ifield, :, 0]

    lsts = np.arange(24) + 0.5
    if(legend == 'full'):
        label = 'Bright Observable'
    else:
        label = None
    ax.bar(lsts,
           allocate.field_slots['slots'][ioption, :, 1] * 4000.,
           width=1.,
           bottom=- 1000.,
           label=label,
           color='red',
           alpha=0.25)
    if(legend == 'full'):
        label = 'Dark Observable'
    else:
        label = None
    ax.bar(lsts,
           allocate.field_slots['slots'][ioption, :, 0] * 4000.,
           width=1.,
           bottom=- 1000.,
           label=label,
           color='black',
           alpha=0.25)
    if(legend == 'full'):
        label = 'Bright Allocated'
    else:
        label = None
    ax.bar(lsts,
           exposures + 0.01 * exposures.max(),
           width=1.,
           label=label,
           color='darkred',
           alpha=1.0)
    if(legend == 'full'):
        label = 'Dark Allocated'
    else:
        label = None
    ax.bar(lsts,
           dexposures + 0.01 * exposures.max(),
           width=1.,
           label=label,
           color='black',
           alpha=1.0)

    ax.plot(np.array([1., 1.]) * allocate.fields['racen'][ifield] / 15.,
            np.array([-1000., 4000.]), linestyle='dotted', color='black')

    ax.set_xlim([0., 24.])
    ax.set_ylim(np.array([-0.05, 1.1]) * exposures.max())

    if(xlabel):
        ax.set_xlabel('Local Sidereal Time (hours)')
    if(ylabel):
        ax.set_ylabel('Number of Designs')

    if(legend):
        ax.legend(title='{c}'.format(f=fieldid, c=cadence), fontsize=10)

    return


def plot_targets(targets=None, assignments=None, iexp=0,
                 robots=True, hexagon=True, categories=[],
                 observatory='apo'):
    """Plot targets in the field

    Parameters
    ----------

    targets : ndarray
        target information as stored in Field

    assignments : ndarray
        assignment information as stored in Field

    iexp : int
        exposure to plot (default 0)

    robots : bool
        if True, show robots and coverage (default True)

    hexagon : bool
        if True, show background hexagon (default True)

    categories : list of str
        which categories of targets to show (default [])

    observatory : str
        which observatory ('apo' or 'lco')
"""
    if(observatory == 'apo'):
        rg = kaiju.robotGrid.RobotGridAPO(stepSize=0.05)
    if(observatory == 'lco'):
        rg = kaiju.robotGrid.RobotGridLCO(stepSize=0.05)

    alphaLen = 7.4
    betaLen = 15.

    fig, ax = plt.subplots()

    ax.set_xlim([-370., 370.])
    ax.set_ylim([-370., 370.])

    if(robots):
        coverage = []
        colors = []
        for robotID in rg.robotDict:
            robot = rg.robotDict[robotID]
            if(robot.hasApogee):
                color = 'red'
            else:
                color = 'black'
            robotpatch = matplotlib.patches.Wedge((robot.xPos, robot.yPos),
                                                  betaLen + alphaLen, 
                                                  0., 360.,
                                                  width=2. * alphaLen)
            colors.append(color)
            coverage.append(robotpatch)

        coverage_collection = matplotlib.collections.PatchCollection(coverage)
        coverage_collection.set_color(colors)
        coverage_collection.set_alpha(np.zeros(len(coverage)) + 0.1)

        ax.add_collection(coverage_collection)

    if(hexagon):
        xv, yv = xyhex()
        ax.plot(xv, yv, linewidth=2, color='black')

    category_colors = dict()
    category_colors['science'] = 'black'
    category_colors['standard_apogee'] = 'red'
    category_colors['sky_apogee'] = 'darkred'
    category_colors['standard_boss'] = 'blue'
    category_colors['sky_boss'] = 'darkblue'

    for category in categories:
        color = category_colors[category]
        itarget = np.where(targets['category'] == category)[0]
        ax.scatter(targets['x'][itarget], targets['y'][itarget],
                   s=2, color='grey', alpha=0.45)
        if(len(assignments['robotID'].shape) == 2):
            igot = np.where((targets['category'] == category) &
                            (assignments['robotID'][:, iexp] >= 0))[0]
        else:
            igot = np.where((targets['category'] == category) &
                            (assignments['robotID'] >= 0))[0]
        ax.scatter(targets['x'][igot], targets['y'][igot],
                   s=18, color=color)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

    return(fig, ax)
