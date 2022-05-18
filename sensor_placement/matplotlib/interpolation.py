# Draweing functions for interpolated values
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


def drawGrid(g, xs, ys,
             ax=None, cmap=None, norm=None, format=None, shrink=1.0,
             include_colorbar=True, include_interpolation=True):
    '''Draw the interpolated values for the given grid.'''

    # fill in defaults
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = cm.get_cmap('viridis')
    if norm is None:
        norm = Normalize(vmin=0, vmax=g.max())
    if not include_interpolation:
        include_colorbar = False

    # create the colours on the mesh
    if include_interpolation:
        xx, yy = numpy.meshgrid(xs, ys)
        mp = ax.pcolormesh(xx, yy, g.T, cmap=cmap, norm=norm)
    ax.set_aspect(1.0)

    # add colorbar
    if include_colorbar:
        cbar = plt.colorbar(mappable=mp,
                            ax=ax, cmap=cmap, norm=norm, format=format,
                            fraction=0.1, shrink=shrink)

        # ticks at the extrema, and at 0 if there's a change of sign
        if norm.vmin * norm.vmax < 0:
            cbar.set_ticks([norm.vmin, 0.0, norm.vmax])
        else:
            cbar.set_ticks([norm.vmin, norm.vmax])
    else:
        cbar = None

    # return the main axes, colorbar, and the norm used
    return ax, cbar, norm


def drawInterpolation(tensor, samples,
                      ax=None, cmap=None, norm=None, format=None, shrink=1.0,
                      clipped=True, include_colorbar=True, include_interpolation=True):
    '''Draw an interpolated dataset using the given tensor and samples.'''
    g = tensor.apply(samples, clipped=clipped)
    return drawGrid(g, tensor._xs, tensor._ys,
                    ax=ax, cmap=cmap, norm=norm, format=format, shrink=shrink,
                    include_colorbar=include_colorbar, include_interpolation=include_interpolation)
