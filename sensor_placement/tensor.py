# Interpolation tensor
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

from itertools import product
import numpy
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union, voronoi_diagram
from geopandas import GeoDataFrame


class InterpolationTensor:

    def __init__(self, points, boundary, ys, xs):
        self._samples = points
        self._boundary = boundary
        self._voronoi = None
        self._ys = ys
        self._xs = xs
        self._grid = None
        self._tensor = None

        # construct the elements of the tensor
        self.voronoi()
        self.geometry()
        self.tensor()


    # ---------- Voronoi cell construction ----------

    def voronoi(self):
        '''Construct a table of natural nearest neighbour Voronoi cells
        and their adjacencies based on a set of samplepo points and a boundary.
        '''

        # check that all the sample points lie within the boundary
        if not self._samples.geometry.within(self._boundary).all():
            raise ValueError('At least one point lies on or outside the boundary')

        # create the Voronoi cells
        voronoi_cells = self.voronoiCells(list(self._samples.geometry), self._boundary)
        self._voronoi = GeoDataFrame({'centre': self._samples.geometry,
                                      'id': self._samples.index,
                                      'geometry': voronoi_cells})

        # use the sample identifier as the cell identifier
        self._voronoi.set_index('id', inplace=True)

        # add the neighbourhoods of each cell, their index and overall boundary
        neighbourhoods = []
        boundaries = []
        for i in self._voronoi.index:
            # indices of intersecting neighbour cells, which match both
            # the index of the Voronoi cells and the corresponding sample points
            neighbours = self.voronoiNeighboursOf(i) + [i]  # including the cell itself
            neighbourhoods.append(neighbours)

            # boundary of neighbourhood
            boundaries.append(self.voronoiBoundaryOf(neighbours))
        self._voronoi['neighbourhood'] = neighbourhoods
        self._voronoi['boundary'] = boundaries

        return self._voronoi

    def voronoiCells(self, points, boundary):
        '''Compute the Voronoi cells of the points in the given set, clipped
        by the boundary.'''

        # compute the diagram
        voronoi_cells = voronoi_diagram(MultiPoint(points))

        # annoyingly the Voronoi cells don't come out in the order of
        # their centres, so we need to order them
        geometries = []
        for p in points:
            # sd: could be faster
            g = [voronoi_cells.geoms[i] for i in range(len(voronoi_cells.geoms)) if p.within(voronoi_cells.geoms[i])][0]
            # clip to the boundary
            geometries.append(g.intersection(boundary))
        return geometries

    def voronoiNeighboursOf(self, real_cell):
        '''Return the cells neighbouring the one given.'''
        cell = self._voronoi.loc[real_cell]
        return list(self._voronoi[self._voronoi.geometry.touches(cell.geometry)].index)

    def voronoiBoundaryOf(self, cells):
        '''Return the boundary of the neighbours of the given cells.'''
        return unary_union(self._voronoi.loc[cells].geometry)


    # ---------- Grid construction----------

    def geometry(self):
        '''Construct the grid of interpolation points from a set of samples,
        a set of their Voronoi cells, and the sample point axes.
        '''

        # build the grid
        self._grid = GeoDataFrame({'x': [i for l in [[j] * len(self._ys) for j in range(len(self._xs))] for i in l],
                                   'y': list(range(len(list(self._ys)))) * len(self._xs),
                                   'geometry': [Point(x, y) for (x, y) in product(self._xs, self._ys)]})

        # add the cell containing each point
        # sd: this needs to be a lot faster
        cells = []
        for _, cell in self._grid.iterrows():
            cs = self._voronoi[self._voronoi.geometry.intersects(cell.geometry)]
            if len(cs) == 1:
                cells.append(cs.index[0])
            else:
                cells.append(-1)
        self._grid['cell'] = cells

        return self._grid


    # ---------- Tensor construction----------

    def tensor(self):
        raise NotImplementedError('tensor')


    # ---------- Tensor editing ----------


    # ---------- Access ----------

    def __getattr__(self, attr):
        if attr == 'shape':
            return self._tensor.shape

    def weights(self, y, x):
        '''Return a vector of tensor weights at the given point.'''
        return self._tensor[y, x, :]


    # ---------- Applying the tensor ----------

    def __call__(self, samples):
        '''Apply the tensor to the given sample vector to give
        a grid of interpolated points.

        :param samples: the sample vector
        :returns: a grid'''

        # check type and dimensions
        if not isinstance(samples, numpy.ndarray):
            # sd: should we try to build an array here?
            raise TypeError('Expected array, got {t}'.format(t=type(samples)))
        if len(samples.shape) != 1:
            raise TypeError('Expected column vector, got shape {s}'.format(s=samples.shape))
        if len(samples) != self._tensor.shape[2]:
            raise ValueError('Tensor needs {n} samples, got {m}'.format(n=self._tensor.shape[2],
                                                                        m=len(samples)))

        # create the result grid
        grid = numpy.zeros((self._tensor.shape[0], tensor.shape[1]))

        # apply the tensor, optimising for sparseness
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                # extract indices of the non-zero elements of each weighting row
                nz = numpy.nonzero(self._tensor[x, y, :])[0]

                # compute the weighted sum
                if len(nz) > 0:
                    # sparse dot product, including only the non-zero elements
                    grid[x, y] = numpy.dot(self._tensor[x, y, nz], samples[nz])

        return grid
