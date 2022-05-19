# Drawing functions for interpolation vectors
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
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# ---------- Support functions ----------

def makeVector(p, q, l):
    '''Construct the offset for a vector aimed of p to q of length l'''
    (x_p, y_p) = list(p.coords)[0]
    (x_q, y_q) = list(q.coords)[0]

    # compute counterclockwise angle, 0 = 3 o'clock
    h = numpy.sqrt((x_q - x_p) ** 2 + (y_q - y_p) ** 2)
    theta = numpy.arcsin(numpy.abs(y_q - y_p) / h)
    if x_q < x_p:
        theta = numpy.pi - theta
    if y_q < y_p:
        theta = 2 * numpy.pi - theta

    dx = l * numpy.cos(theta)
    dy = l * numpy.sin(theta)

    return dx, dy


def nearestPointTo(p, tensor):
    '''Return the interpolation point nearest to the given point.'''
    cloud = MultiPoint(list(tensor._grid.geometry))
    return nearest_points(cloud, p)[0]


def drawVectorOffset(p, dx, dy,
                     ax=None, color=None):
    '''Draw a vector with base at p and offsets dx and dy.'''

    # fill in defaults
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'r'

    # draw the vector
    xy = list(p.coords)[0]
    ax.arrow(xy[0], xy[1], dx, dy, color=color)  # width=0.002)


def drawVector(p, q, w,
               ax=None, radius=None, overwrite=True, color=None):
    '''Draw a vector from p towards q with lengfth that is a fraction w of
    the given radius.'''

    # cut-off the vector drawing if the arrow will overwrite the endpoint
    if (not overwrite) and (p.distance(q) < w * radius):
        return

    # construct vector offset
    (dx, dy) = makeVector(p, q, w * radius)

    # draw it
    drawVectorOffset(p, dx, dy, ax=ax, color=color)



# ---------- User functions ----------

def drawWeightVector(tensor, p, s,
                     ax=None, radius=None, overwrite=True, color=None):
    '''Draw a vector representing the weight give to sample s in
    the interpolation of point p.'''

    # fill in defaults
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'r'
    if radius is None:
        radius = 0.05

    # find tensor indices of point
    xy = list(p.coords)[0]
    i = tensor._xs.index(xy[0])
    j = tensor._ys.index(xy[1])

    # find point for sample
    q = tensor._samples.loc[s].geometry

    # find tensor index for sample
    si = tensor._samples.index.get_loc(s)

    # get weight
    w = tensor._tensor[i, j, si]

    drawVector(p, q, w, ax=ax, radius=radius, overwrite=overwrite, color=color)


def drawWeightVectors(tensor, p, cutoff=0.0,
                      ax=None, radius=None, overwrite=True, color=None):
    '''Draw all the weight vectors at point p, optionally with a cut-off for small weights.'''

    # find tensor indices of point
    xy = list(p.coords)[0]
    i = tensor._xs.index(xy[0])
    j = tensor._ys.index(xy[1])

    # find indices of non-zero weights
    ss = numpy.nonzero(tensor._tensor[i, j, :])[0]

    # draw vectors for all weights
    for si in ss:
        if tensor._tensor[i, j, si] >= cutoff:
            s = list(tensor._samples.index)[si]
            drawWeightVector(tensor, p, s, ax=ax, radius=radius, overwrite=overwrite, color=color)


def drawDominantWeightVector(tensor, p, cutoff=0.0,
                             ax=None, radius=None, overwrite=True, color=None):
    '''Draw the largest weight vector at point p, optionally with a cut-off for small weights.'''

    # find tensor indices of point
    xy = list(p.coords)[0]
    i = tensor._xs.index(xy[0])
    j = tensor._ys.index(xy[1])

    # find index of largest weight
    si = numpy.argmax(tensor._tensor[i, j, :])
    s = list(tensor._samples.index)[si]

    # draw vector
    if tensor._tensor[i, j, si] >= cutoff:
        drawWeightVector(tensor, p, s, ax=ax, radius=radius, overwrite=overwrite, color=color)


def drawResolvedVector(tensor, p,
                       ax=None, radius=None, overwrite=True, color=None):
    '''Resolve all vectors and draw the resolved summary vector.'''

    # find tensor indices of point
    xy = list(p.coords)[0]
    i = tensor._xs.index(xy[0])
    j = tensor._ys.index(xy[1])

    # find indices of non-zero weights
    ss = numpy.nonzero(tensor._tensor[i, j, :])[0]

    # resolve vectors by adding the offsets of all components
    dx, dy = 0.0, 0.0
    for si in ss:
        w = tensor._tensor[i, j, si]
        s = list(tensor._samples.index)[si]
        q = tensor._samples.loc[s].geometry
        ddx, ddy = makeVector(p, q, w * radius)
        dx += ddx
        dy += ddy

    drawVectorOffset(p, dx, dy, ax=ax, color=color)
