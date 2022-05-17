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

import logging
from itertools import product
from datetime import date, datetime
from multiprocessing import cpu_count
import numpy
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import unary_union, voronoi_diagram
from pandas import Series
from geopandas import GeoDataFrame
from netCDF4 import Dataset
from sensor_placement import Logger


logger = logging.getLogger(Logger)


class InterpolationTensor:
    '''The abstract class of interpolation tensors.

    The tensor is stored as a three-dimensional array with axes (lon, lat, sample).

    The operation can be run in parallel on a multicore be:
    machine. The degree of parallelism is given by cores, which may

    - 0 to use the maximum number of available cores
    - +n to use a specific number of cores
    - -n to leave n cores unused

    There is no benefit to using a degree of parallelism greater
    than the number of physical cores on the machine. In cases
    where there are very few cells to compute a smaller amount of
    parallelism will be used anyway.

    '''

    def __init__(self, points, boundary, xs, ys, cores = 1,
                 voronoi = None, grid = None, data = None):
        self._samples = points.copy()   # we'll change this if we edit the tensor
        self._boundary = boundary       # shape of overall boundary
        self._xs = xs                   # list of x axis of interpoolation points
        self._ys = ys                   # list of y axis of interpoolation points
        self._voronoi = voronoi         # DataFrame holding Voronboi cells
        self._grid = grid               # DataFrame mapping points to cells
        self._tensor = data             # array representing tensor

        # optimisations
        self._lastCell = None           # cache of the last real cell returned by
                                        # cellContaining(). If used in a multiprocessing
                                        # operation this will become the per-process
                                        # last cell, so the different execution paths
                                        # will have their own optimising behaviour

        # compute the nunber of cores to use when computing the tensor
        if cores == 0:
            # use all available
            self.limitNumberOfCores(cpu_count())
        elif cores < 0:
            # use fewer than available, down to a minimum of 1
            self.limitNumberOfCores(max(cpu_count() + cores, 1))   # cpu_count() + cores as cores is negative
        else:
            # use the number of cores requested, up to the maximum available
            self.limitNumberOfCores(min(cores, cpu_count()))

        # construct the elements of the tensor
        if self._voronoi is None:
            self.buildVoronoi()
        if self._grid is None:
            self.buildGeometry()
        if self._tensor is None:
            self.buildTensor()


    # ---------- Core usage ----------

    def limitNumberOfCores(self, c):
        '''Limit the maximum number of cores the tensor will use.'''
        self._cores = c
        logger.info('Tensor limited to {cores} core{s}'.format(cores=self._cores,
                                                               s='s' if self._cores > 1 else ''))

    def numberOfCores(self):
        '''Return the number of cores to use in an operation. This is
        bounded by the number given at creation, and also by the number
        of cells in the Voronoi diagram (since that's the appropriate granularity
        for parallelism).'''
        c = min(self._cores, len(self._voronoi))
        logger.info(f'Operation using {c} cores')
        return c


    # ---------- Voronoi cell construction ----------

    # id: identifier for sample and cell
    # centre: location of sample point
    # geometry: boundary of Voronoi cell around centre
    # neighbourhood: ids of neighbouring cells, including this one
    # boundary: boundary of the neighbourhood, union of neighbourhood's cell boundaries

    # order of samples wiuthin the DataFrame corresponds to iundex of sample within
    # the sample vector, and order of weights within the tensor

    def buildVoronoi(self):
        '''Construct a table of natural nearest neighbour Voronoi cells
        and their adjacencies based on a set of sample points and a boundary.
        '''

        # check that all the sample points lie within the boundary
        if not self._samples.geometry.within(self._boundary).all():
            raise ValueError(f'At least one point lies on or outside the boundary')

        # create the Voronoi cells
        then = datetime.now()
        logger.info('Computing Voronoi diagram')
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

        # report time
        dt = (datetime.now()- then).seconds
        logger.info(f'Created Voronoi diagram in {dt:.2f}s')

    def voronoiCells(self, points, boundary):
        '''Compute the Voronoi cells of the points in the given set, clipped
        by the boundary.'''

        # compute the diagram
        voronoi_cells = voronoi_diagram(MultiPoint(points), boundary)

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

        # There are some topologies in which points in non-neighbouring
        # cells (in the topological sense) can still be influenced, so
        # we build a circle to catch them with radius to the farthest
        # immediate neighbour. This will catch acute-angled intrusions
        c = self._voronoi.loc[real_cell]['centre']
        g = self._voronoi.loc[real_cell].geometry
        immediate = self._voronoi[self._voronoi.geometry.intersects(g)].index
        farthestImmediate = max([p.distance(c) for p in self._voronoi.loc[immediate]['centre']])
        neighbours = [s for s in self._voronoi.index if self._voronoi.loc[s].geometry.distance(c) <= farthestImmediate and s != real_cell]

        logger.debug(f'Neighbours of {real_cell} set to {neighbours}')
        return neighbours

    def voronoiBoundaryOf(self, cells):
        '''Return the boundary of the neighbours of the given cells.'''
        return unary_union(self._voronoi.loc[cells].geometry)


    # ---------- Grid construction----------

    # x: x co-ordinate of interpolation point
    # y: y- co-ordinate
    # geometry: Point at these co-ordinates
    # cell: id of Voronoi cell this point sits within
    # distance: distance to the cell centre

    def buildGeometry(self):
        '''Construct the grid of interpolation points from a set of samples,
        a set of their Voronoi cells, and the sample point axes.
        '''

        # The grid is held as a DataFrame rather than as an array to make it
        # easier to edit. This might not be a good enough reason, and holding
        # it as an array might make more sense (and be more compact).

        # build the grid
        then = datetime.now()
        logger.info('Constructing interpolation grid')
        self._grid = GeoDataFrame({'x': [i for l in [[j] * len(self._ys) for j in range(len(self._xs))] for i in l],
                                   'y': list(range(len(list(self._ys)))) * len(self._xs),
                                   'geometry': [Point(x, y) for (x, y) in product(self._xs, self._ys)]})

        # add the cell containing each point
        cells = []
        for _, pt in self._grid.iterrows():
            cells.append(self.cellContaining(pt.geometry))
        self._grid['cell'] = cells

        # compute the distances to the cell centres
        distances = []
        for _, pt in self._grid.iterrows():
            c = pt['cell']
            if c == -1:
                # uncovered point, set distance to -1
                distances.append(-1)
            else:
                distances.append(self.distanceToSample(pt.geometry, c))
        self._grid['distance'] = distances

        # report time
        dt = (datetime.now() - then).seconds
        logger.info(f'Created interpolation grid in {dt:.2f}s')

    def cellContaining(self, p):
        '''Return the cell containing the given point, or -1 if
        the point does not lie in a cell.'''
        if self._lastCell is not None:
            # check in the last cell, since we usually look at nearby points
            if self._voronoi.loc[self._lastCell].geometry.intersects(p):
                return self._lastCell

        # check all cells
        cs = self._voronoi[self._voronoi.geometry.intersects(p)]
        if len(cs) == 0:
            return -1
        else:
            # it's possible that there's multiple containment if a point
            # lies exactly on a cell boundary, in which case we pick one
            self._lastCell = cs.index[0]
            return cs.index[0]

    def distanceToSample(self, p, s):
        '''Return the distance from the given point to the given sample.'''
        return p.distance(self._voronoi.loc[s].centre)


    # ---------- Tensor construction----------

    def buildTensor(self):
        raise NotImplementedError('buildTensor')

    def iterateWeightsFor(self, real_cells = None):
        raise NotImplementedError('weightsFor')


    # ---------- Tensor editing ----------

    def removeSample(self, s):
        '''Remove sample from the tensor.'''
        logging.debug(f'Removing cell {s}')

        # retrieve the neighbourhood of the sample cell, not
        # including the cell itself, and the boundary
        pre_neighbours = self.voronoiNeighboursOf(s)
        pre_neighbourhood = list(self._samples.loc[pre_neighbours].geometry)
        pre_boundary = self.voronoiBoundaryOf(pre_neighbours + [s])

        # remove the sample from the data structures
        self._samples.drop([s], axis=0, inplace=True)
        i = self._voronoi.index.get_loc(s)
        self._tensor = numpy.delete(self._tensor, i, axis=2)
        self._voronoi.drop([s], axis=0, inplace=True)

        # re-compute the cells in the neighbourhood
        # (Need to use an explicit Series because the elements are themselves lists)
        cells = self.voronoiCells(pre_neighbourhood, pre_boundary)
        self._voronoi.loc[pre_neighbours, ['geometry']] = cells
        neighbourhoods = Series([self.voronoiNeighboursOf(i) + [i] for i in pre_neighbours],
                                index=pre_neighbours)
        self._voronoi.loc[pre_neighbours, 'neighbourhood'] = neighbourhoods
        boundaries = neighbourhoods.apply(self.voronoiBoundaryOf)
        self._voronoi.loc[pre_neighbours, 'boundary'] = boundaries

        # re-compute the mapping from grid points to cells, and their distances
        pts = self._grid[self._grid['cell'].isin(pre_neighbours + [s])]
        cells = []
        for _, pt in pts.iterrows():
            cells.append(self.cellContaining(pt.geometry))
        self._grid.loc[pts.index, 'cell'] = cells
        distances = []
        for i in range(len(pts)):
            pt = pts.iloc[i]
            distances.append(self.distanceToSample(pt.geometry, cells[i]))
        self._grid.loc[pts.index, 'distance'] = distances

        # re-compute the weights for all these points
        for (x, y, si, v) in self.iterateWeightsFor(pre_neighbours):
            self._tensor[x, y, si] = v

    def removeSamples(self, ss):
        '''Remove all samples in ss. This is more efficient than removing them
        one by one, and reduces re-computation of weights.'''
        logging.debug(f'Removing cells {ss}')

        # retrieve all cells neighbouring cells to be deleted
        pre_neighbours = set()
        for s in ss:
            pre_neighbours = pre_neighbours.union(set(self.voronoiNeighboursOf(s)))
        pre_neighbours = pre_neighbours.difference(set(ss))
        pre_all = pre_neighbours.union(set(ss))
        pre_boundary = self.voronoiBoundaryOf(pre_all)
        logging.debug(f'Affected remaining cells {pre_neighbours}')

        # remove samples from the data structures
        self._samples.drop(ss, axis=0, inplace=True)            # sample points
        sis = [self._voronoi.index.get_loc(s) for s in ss]
        self._tensor = numpy.delete(self._tensor, sis, axis=2)  # tensor sample planes
        self._voronoi.drop(ss, axis=0, inplace=True)

        # re-compute the cells in the neighbourhood
        # (Need to use an explicit Series because the elements are themselves lists)
        pre_neighbourhood = list(self._samples.loc[pre_neighbours].geometry)
        cells = self.voronoiCells(pre_neighbourhood, pre_boundary)
        self._voronoi.loc[pre_neighbours, ['geometry']] = cells
        neighbourhoods = Series([self.voronoiNeighboursOf(i) + [i] for i in pre_neighbours],
                                index=pre_neighbours)
        self._voronoi.loc[pre_neighbours, 'neighbourhood'] = neighbourhoods
        boundaries = neighbourhoods.apply(self.voronoiBoundaryOf)
        self._voronoi.loc[pre_neighbours, 'boundary'] = boundaries

        # re-compute cell memberships and distances
        pts = self._grid[self._grid['cell'].isin(pre_all)]
        cells = [self.cellContaining(pt) for pt in pts.geometry]
        self._grid.loc[pts.index, 'cell'] = cells
        distances = [self.distanceToSample(pts.iloc[i].geometry, cells[i]) for i in range(len(pts))]
        self._grid.loc[pts.index, 'distance'] = distances

        # re-compute the weights for all these points
        for c in pre_neighbours:
            logging.debug(f'Computing weights for {c}')
            for (x, y, si, v) in self.iterateWeightsFor([c]):
                self._tensor[x, y, si] = v

    def addSample(self, x, y):
        '''Add a sample at the given point to the tensor.'''
        logging.debug(f'Adding sample at ({x}, {y})')

        # check we lie within the boundary
        if not self._boundary.contains(Point(x, y)):
            raise ValueError('New sample point does not lie within boundary')

        # create the new point and its tensor index
        p = Point(x, y)
        s = self._samples.index.max() + 1
        self._samples.loc[s] = dict(x=x, y=y, geometry=p)

        # find the cell containing the new point
        cs = self._voronoi.geometry.intersects(p)
        if len(cs) == 0:
            raise ValueError(f'Can\'t find cell containing ({x} {y})')
        real_cell = cs.index[0]        # if we lie on a boundary, just pick one

        # retrieve the neighbourhood of the new sample point
        pre_neighbours = self.voronoiNeighboursOf(real_cell) + [real_cell]
        pre_coords = list(self._voronoi.loc[pre_neighbours]['centre'])
        pre_boundary = self.voronoiBoundaryOf(pre_neighbours)

        # create a new cell around the sample point, re-computing the others
        new_coords = pre_coords + [p]
        new_voronoi_cells = self.voronoiCells(new_coords, pre_boundary)
        self._voronoi.loc[pre_neighbours, 'geometry'] = new_voronoi_cells[:-1]
        new_cell = new_voronoi_cells[-1]

        # add new cell
        self._voronoi.loc[s] = dict(centre=p,
                                    geometry=new_cell,
                                    neighbourhood=pre_neighbours + [s],
                                    boundary=pre_boundary)

        # re-establish the new cell neighbourhoods and boundaries
        neighbourhoods = Series([self.voronoiNeighboursOf(i) + [i] for i in pre_neighbours],
                                index=pre_neighbours)
        self._voronoi.loc[pre_neighbours, 'neighbourhood'] = neighbourhoods
        boundaries = neighbourhoods.apply(self.voronoiBoundaryOf)
        self._voronoi.loc[pre_neighbours, 'boundary'] = boundaries

        # re-compute the mapping from grid points to cells, and their distances
        pts = self._grid[self._grid['cell'].isin(pre_neighbours)]
        cells = []
        for _, pt in pts.iterrows():
            cells.append(self.cellContaining(pt.geometry))
        self._grid.loc[pts.index, 'cell'] = cells
        distances = []
        for i in range(len(pts)):
            pt = pts.iloc[i]
            distances.append(self.distanceToSample(pt.geometry, cells[i]))
        self._grid.loc[pts.index, 'distance'] = distances

        # expand the tensor to accommodate the new sample
        ss = numpy.zeros((self._tensor.shape[0], self._tensor.shape[1], 1))
        self._tensor = numpy.append(self._tensor, ss, axis=2)

        # re-compute the weights for all these points
        for (x, y, si, v) in self.iterateWeightsFor(pre_neighbours):
            self._tensor[x, y, si] = v

        # return the new sample
        return s


    # ---------- Access ----------

    def __getattr__(self, attr):
        if attr == 'shape':
            return self._tensor.shape
        else:
            raise AttributeError(attr)

    def samplePointsDataFrame(self):
        '''Return the sample points as a DataFrame.'''
        return self._samples

    def cellsDataFrame(self):
        '''Return the Voronoi cells as a DataFrame.'''
        return self._voronoi

    def gridDataFrame(self):
        '''Return the interpolation grid as a DataFrame.'''
        return self._grid

    def boundary(self):
        '''Return the boundary.'''
        return self._boundary

    def tensor(self):
        '''Return the tensor.'''
        return self._tensor

    def samplePoints(self):
        '''Return the sample points.'''
        return list(self._samples['geometry'])

    def cells(self):
        '''Return the Voronoi cells in sample point order.'''
        return list(self._voronoi['geometry'])

    def ys(self):
        '''Return the y-axis (northing) interpolation points.'''
        return self._ys

    def xs(self):
        '''Return the x-axis (easting) interpolation points.'''
        return self._xs

    def weights(self, x, y):
        '''Return a vector of tensor weights at the given point.'''
        return self._tensor[x, y, :]


    # ---------- Applying the tensor ----------

    def __call__(self, samples):
        '''Apply the tensor to the given sample vector to give
        a grid of interpolated points. Equivalent to
        :meth:`apply` with no clipping.

        :param samples: the sample vector
        :returns: a grid'''
        return self.apply(samples)

    def apply(self, samples, clipped = False):
        '''Apply the tensor to the vector of samples, optionally clipping
        the resulting grid to the boundary.'''
        logger.debug('Applying tensor')

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
        then = datetime.now()
        grid = numpy.zeros((self._tensor.shape[0], self._tensor.shape[1]))

        # apply the tensor, optimising for sparseness
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # extract indices of the non-zero elements of each weighting row
                nz = numpy.nonzero(self._tensor[i, j, :])[0]

                # compute the weighted sum
                if len(nz) > 0:
                    # sparse dot product, including only the non-zero elements
                    grid[i, j] = numpy.dot(self._tensor[i, j, nz], samples[nz])

        if clipped:
            # clip the grid to the boundary using a masked array
            mask = numpy.empty(grid.shape)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    x, y = self._xs[i], self._ys[j]
                    mask[i, j] = not self._boundary.contains(Point(x, y))
            grid = numpy.ma.masked_where(mask, grid, copy=False)

        # report time
        dt = (datetime.now()- then).seconds
        logger.info(f'Applied tensor in {dt:.2f}s')

        return grid


    # ---------- I/O to and from NetCDF ----------

    def saveTensor(self, root):
        '''Save information about the tensor to the given NetCDF root.
        This can be overridden by sub-classes if more information needs to be saved.
        The default does nothing.'''
        pass

    def loadTensor(self, root):
        '''Load any extra information about the tensor. The basic data
        structures will already be initialised. The default dsoes nothing.'''
        pass

    def save(self, fn):
        '''Save the tensor to the given file in NetCDF format.

        :param fn: the filename'''

        # create the dataset
        then = datetime.now()
        logger.debug(f'Saving tensor to {fn}')
        root = Dataset(fn, 'w', format='NETCDF4')

        # standard metadata
        now = datetime.now().isoformat()
        root.history = f'Saved {now}'

        # turn the boundary into a list of points, leaving off the last one
        # that simply suplicates the first
        b = self._boundary.exterior.coords[:-1]

        # turn the grid into an array of cell (sample) indices and an array
        # of distances to cell samples
        g = numpy.full((len(self._xs), len(self._ys)), -1, dtype=int)
        d = numpy.full((len(self._xs), len(self._ys)), 0, dtype=float)
        for _, r in self._grid.iterrows():
            g[r['x'], r['y']] = r['cell']
            d[r['x'], r['y']] = r['distance']

        # dimensions
        sample_dim = root.createDimension('sample', len(self._samples))
        boundary_dim = root.createDimension('boundary', len(b))
        grid_x_dim = root.createDimension('grid_x', len(self._xs))
        grid_y_dim = root.createDimension('grid_y', len(self._ys))
        # lat_dim = root.createDimension('lat', len(lat_station))
        # lon_dim = root.createDimension('long', len(lon_station))

        # variables
        boundary_x_var = root.createVariable('boundary_x', 'f4', (boundary_dim.name))
        boundary_x_var.units = 'Boundary longitude (degrees)'
        boundary_y_var = root.createVariable('boundary_y', 'f4', (boundary_dim.name))
        boundary_y_var.units = 'Boundary latitude (degrees)'
        sample_index_var = root.createVariable('sample_index', 'i4', (sample_dim.name))
        sample_index_var.units = 'Sample index (integer)'
        sample_x_var = root.createVariable('sample_x', 'f4', (sample_dim.name))
        sample_x_var.units = 'Sample latitude (degrees)'
        sample_y_var = root.createVariable('sample_y', 'f4', (sample_dim.name))
        sample_y_var.units = 'Sample longitude (degrees)'
        grid_x_var = root.createVariable('grid_x', 'f4', (grid_x_dim.name))
        grid_x_var.units = 'Grid easting (m east of base of UK national grid)'
        grid_y_var = root.createVariable('grid_y', 'f4', (grid_y_dim.name))
        grid_y_var.units = 'Grid northing (m north of base of UK national grid)'
        # lat_var = root.createVariable('lat', 'f4', (station_dim.name))
        # lat_var.units = 'Latitude (degrees)'
        # lon_var = root.createVariable('long', 'f4', (station_dim.name))
        # lon_var.units = 'Longitude (degree)'
        grid_var = root.createVariable('grid', 'i4', (grid_x_dim.name, grid_y_dim.name))
        grid_var.units = 'Voronoi cell containing this grid point (integer)'
        distance_var = root.createVariable('distance', 'f4', (grid_x_dim.name, grid_y_dim.name))
        distance_var.units = 'Distance from this point to sample point (float)'
        tensor_var = root.createVariable('tensor', 'f4', (grid_x_dim.name, grid_y_dim.name, sample_dim.name))
        tensor_var.units = 'Weight assigned to each sample in interpolating point (float)'

        # populate the dataset
        boundary_x_var[:] = list(map(lambda p: p[0], b))
        boundary_y_var[:] = list(map(lambda p: p[1], b))
        sample_index_var[:] = list(self._samples.index)
        sample_x_var[:] = list(self._samples.geometry.apply(lambda p: list(p.coords)[0][0]))
        sample_y_var[:] = list(self._samples.geometry.apply(lambda p: list(p.coords)[0][1]))
        grid_x_var[:] = self._xs
        grid_y_var[:] = self._ys
        grid_var[:, :] = g
        distance_var[:, :] = d
        tensor_var[:, :, :] = self._tensor

        # save any extra data
        self.saveTensor(root)

        # close the file
        root.close()

        # report time
        dt = (datetime.now()- then).seconds
        logger.info(f'Saved tensor to {fn} in {dt:.2f}s')

    @classmethod
    def load(cls, fn, **kwds):
        '''Load a tensor as an instance of the class from the given NetCDF file.
        Any keyword arguments are passed to the constructor for cls.

        :param fn: filename
        :returns: the tensor'''

        # open the dataset
        then = datetime.now()
        cn = cls.__name__
        logger.debug(f'Loading tensor from {fn} (class {cn})')
        root = Dataset(fn, 'r', format='NETCDF4')

        # read the underlying sample points and boundary
        sampleIdx = root['sample_index']
        samplePoints = list(map(lambda p: Point(p[0], p[1]), zip(root['sample_x'], root['sample_y'])))
        df_points = GeoDataFrame({'geometry': samplePoints},
                                 index=sampleIdx)
        boundaryPoints = list(map(lambda p: Point(p[0], p[1]), zip(root['boundary_x'], root['boundary_y'])))
        boundaryPoints += [boundaryPoints[-1]]   # close the sequence
        boundary = Polygon(boundaryPoints)

        # read the grid structure
        xs = list(numpy.asarray(root['grid_x']).astype(float))
        ys = list(numpy.asarray(root['grid_y']).astype(float))
        g = numpy.asarray(root['grid']).astype(int)
        d = numpy.asarray(root['distance']).astype(float)
        df_grid = GeoDataFrame({'x': [i for l in [[j] * len(ys) for j in range(len(xs))] for i in l],
                                'y': list(range(len(list(ys)))) * len(xs),
                                'geometry': [Point(x, y) for (x, y) in product(xs, ys)],
                                'cell': [g[x, y] for (x, y) in product(range(len(xs)), range(len(ys)))],
                                'distance': [d[x, y] for (x, y) in product(range(len(xs)), range(len(ys)))]})

        # create the tensor
        logger.info('Instanciating tensor as {cn}'.format(cn=cls.__name__))
        t = cls(df_points, boundary,
                xs, ys,
                grid=df_grid,
                data=numpy.asarray(root['tensor']).astype(float),
                **kwds)

        # load any additional data into the newly-created tensor
        t.loadTensor(root)

        # close the file
        root.close()

        # report time
        dt = (datetime.now()- then).seconds
        logger.info(f'Loaded tensor from {fn} in {dt:.2f}s')

        return t
