#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: plots.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import numpy as np
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import astropy.coordinates as coords
import kaiju.robotGrid


def plot_fields(fields=None):
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
                                      for i in np.arange(n_vertex + 1)])

    vertex_y = hex_radius * np.array([np.sin(- 2. * np.pi *
                                             (i / float(n_vertex)) +
                                             angle0)
                                      for i in np.arange(n_vertex + 1)])
    return(vertex_x, vertex_y)


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
