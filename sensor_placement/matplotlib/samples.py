# Drawing functions for raw samples
#
# Copyright (C) 2022 Simon Dobson
#
# This file is part of sensor-placement, an experiment in sensor placement
# and error.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software. If not, see <http://www.gnu.org/licenses/gpl.html>.

import numpy
from geopandas import GeoDataFrame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def drawRawSamples(tensor, samples,
                   ax=None, cmap=None, norm=None, fontsize=None, marker='o', markersize=None, shrink=1.0,
                   include_sample_labels=True, include_colorbar=True):
    '''Draw raw samples coloured by the given colourmap.'''

    # fill in default
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = cm.get_cmap('viridis')
    if norm is None:
        norm = Normalize(vmin=0, vmax=samples.max())

    # draw samples
    for i in range(len(tensor._samples)):
        p = tensor._samples.geometry.iloc[i]
        pt = list(p.coords)[0]
        ax.plot(pt[0], pt[1], marker=marker, markersize=markersize, color=cmap(norm(samples[i])))
        if include_sample_labels:
            ax.annotate(f'{i}', (pt[0], pt[1]),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=fontsize)

    # add colorbar
    if include_colorbar:
        cbar = plt.colorbar(ax=ax, cmap=cmap, norm=norm, format=format,
                            fraction=0.1, shrink=shrink)

        # ticks at the extrema, and at 0 if there's a change of sign
        if norm.vmin * norm.vmax < 0:
            cbar.set_ticks([norm.vmin, 0.0, norm.vmax])
        else:
            cbar.set_ticks([norm.vmin, norm.vmax])
    else:
        cbar = None

    # return the main axes and the colorbar
    return ax, cbar


def drawVoronoiCells(tensor,
                     ax=None, color='k', linewidth=None):
    '''Draw sample Voronoi cells.'''

    # fill in defaults
    if ax is None:
        ax = plt.gca()

    # draw cell boundaries
    for i in range(len(tensor._samples)):
        x, y = tensor._voronoi['geometry'].iloc[i].exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)

    # return the axes
    return ax


def drawSampleLabels(tensor, ss=None,
                     ax=None, marker=None, color='k', markersize=None, fontsize=None,
                     include_sample_labels=True, include_sample_indices=True):
    '''Draw points for the samples, optionally labelled.'''

    # fill in defaults
    if ax is None:
        ax = plt.gca()
    if ss is None:
        tss = tensor._samples
    else:
        tss = tensor._samples.loc[ss]

    # draw labels
    for i, r in tss.iterrows():
        # plot marker
        p = r.geometry
        pt = list(p.coords)[0]
        ax.plot(pt[0], pt[1], marker=marker, color=color, markersize=markersize)

        # add label if requested
        if include_sample_labels or include_sample_indices:
            l = None
            if include_sample_labels:
                l = f'{i}'
                if include_sample_indices:
                    n = tensor._samples.index.get_loc(i)
                    l += f'({n})'
            elif include_sample_indices:
                n = tensor._samples.index.get_loc(i)
                l = f'({n})'
            ax.annotate(l, (pt[0], pt[1]), xytext=(2, 2), textcoords='offset points', fontsize=fontsize)

    # return the axes
    return ax
