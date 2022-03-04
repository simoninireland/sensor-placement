# Natural nearest neighbour interpolation
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
# along with trhis software. If not, see <http://www.gnu.org/licenses/gpl.html>.


from itertools import product
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy
from geopandas import GeoDataFrame
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from shapely.geometry import shape, Point, Polygon
from shapely.ops import cascaded_union


def nnn_voronoi(df_points, boundary_shape):
    '''Construct a table of natural nearest neighbour Voronoi cells
    and their adjacencies based on a set of samplepo points and a boundary.

    The sample points should be supplied in a `GeoDataFrame` having
    columns `geometry` containing the sample points locations. All the
    samples should lie within the boundary_shape.

    The returned `DataFrame` will contain the `geometry` of each cell,
    it's `centre` (the sample point it surrounds), `sample` the index
    of the sample point in the sample vector, a `neighbourhood` holding
    the indices of the cells that intersect this cell (including the
    index of the cell itself), and a `boundary` holding the outer
    boundary of this set. This is all the topological information
    contained in the Voronoi diagram, plus a link to the way
    sample points are represented as a vector.

    :param df_points: the samples
    :param boundary_shape: the boundary surrounding the samples
    :returns: a dataframe

    '''

    # check that all the sample points lie within the boundary
    if not df_points.geometry.within(boundary_shape).all():
        raise ValueError('At least one point lies on or outside the boundary')

    # create the Voronoi cells
    coords = points_to_coords(df_points.geometry)
    voronoi_cells, voronoi_centres = voronoi_regions_from_coords(coords, boundary_shape)
    cells = list(voronoi_cells.keys())
    df_voronoi = GeoDataFrame({'centre': [df_points.iloc[voronoi_centres[i][0]].geometry for i in cells],
                               'geometry': voronoi_cells.values(),
                               'sample': [voronoi_centres[i][0] for i in cells]})

    # add the neighbourhoods of each cell, their index and overall boundary
    neighbourhoods = []
    boundaries = []
    for i, cell in df_voronoi.iterrows():
        # indices of intersecting neighbour cells
        neighbours = list(df_voronoi[df_voronoi.geometry.touches(cell.geometry)].index) + [i]
        neighbourhoods.append(neighbours)

        # boundary of neighbourhood
        boundaries.append(cascaded_union(df_voronoi.loc[neighbours].geometry))
    df_voronoi['neighbourhood'] = neighbourhoods
    df_voronoi['boundary'] = boundaries

    return df_voronoi


def nnn_geometry(df_points, df_voronoi, xs, ys):
    '''Construct the grid of interpolation points from a set of samples,
    a set of their Voronoi cells, and the sample point axes.

    The returned `DataFrame` will have columns `geometry` for the
    interpolated points, `x` and `y` for the indices of the
    observation along the two axes, and `cell` holding the index of
    the Voronoi cell within which the interpolation point lies.
    An index of -1 is used to represent cells that lie outside the area
    covered by the Voronoi diagram.

    :param df_points: the samples
    :param df_voronoi: the Voronoi cells for these samples
    :param xs: list of x co-ordinates to interpolate at
    :param ys: list of y co-ordinates to interpolate at
    :returns: a dataframe

    '''

    # build the grid
    df_interpoints = GeoDataFrame({'x': [i for l in [[j] * len(ys) for j in range(len(xs))] for i in l],
                                   'y': list(range(len(list(ys)))) * len(xs),
                                   'geometry': [Point(x, y) for (x, y) in product(xs, ys)]})

    # add the cell containing each point
    cells = []
    for _, cell in df_interpoints.iterrows():
        cs = df_voronoi[df_voronoi.geometry.intersects(cell.geometry)]
        if len(cs) == 1:
            cells.append(cs.index[0])
        else:
            cells.append(-1)
    df_interpoints['cell'] = cells

    return df_interpoints


# Original sequential version
#
# def nnn_tensor(df_points, df_voronoi, df_grid):
#     '''Construct the natural nearest neighbour interpolation tensor from a
#     set of samples taken within a boundary and sampled at the given
#     grid of interpolation points.

#     The tensor is a three-dimensional sparse `numpy.array` with axes
#     corresponding to xs, ys, and points, with entries containing the
#     weight given to each sample in interpolating each point.

#     :param df_points: the sample points
#     :param df_voronoi: the Voronoi diagram of the samples
#     :param df_grid: the grid to interpolate onto
#     :returns: a tensor'''

#     # construct the tensor
#     tensor = numpy.zeros((max(df_grid.x) + 1, max(df_grid.y) + 1, len(df_points)))

#     # group the grid points by the real cell they lie within
#     grid_grouped = df_grid.groupby('cell').groups

#     # ignore any points outside of the Voronoi diagram
#     if -1 in grid_grouped.keys():
#         del grid_grouped[-1]

#     # construct the weights
#     for real_cell in grid_grouped.keys():
#         # extract the neighbourhood of Voronoi cells,
#         # the only ones that the cell around this sample point
#         # can intersect and so the only computation we need to do
#         df_real_neighbourhood = df_voronoi.loc[df_voronoi.loc[real_cell].neighbourhood]
#         real_coords = points_to_coords(df_real_neighbourhood.centre)
#         real_boundary_shape = df_voronoi.loc[real_cell].boundary

#         # construct an array that will hold the co-ordinates of all the real points
#         # and the synthetic point
#         synthetic_coords = numpy.append(real_coords, [[0, 0]], axis=0)

#         for pt in grid_grouped[real_cell]:
#             # re-compute the Voronoi cells given the synthetic point
#             p = df_grid.loc[pt]
#             synthetic_coords[-1] = points_to_coords([p.geometry])[0]
#             synthetic_voronoi_cells, synthetic_voronoi_centres = voronoi_regions_from_coords(synthetic_coords, real_boundary_shape)

#             # get the synthetic cell
#             i = [i for i in synthetic_voronoi_centres.keys() if len(synthetic_coords) - 1 in synthetic_voronoi_centres[i]][0]
#             synthetic_cell = synthetic_voronoi_cells[i]

#             # compute the weights
#             synthetic_cell_area = synthetic_cell.area
#             for _, r in df_real_neighbourhood.iterrows():
#                 area = r.geometry.intersection(synthetic_cell).area
#                 if area > 0.0:
#                     tensor[int(p['x']), int(p['y']), int(r['sample'])] = area / synthetic_cell_area

#     return tensor


def nnn_tensor_worker(df_voronoi, df_grid,
                      df_real_neighbourhood, real_coords, real_boundary_shape,
                      points):
    '''Construct the natural nearest neighbour interpolation tensor from a
    set of samples taken within a boundary and sampled at the given
    grid of interpolation points.

    :param df_voronoi: the Voronoi diagram of the samples
    :param df_grid: the grid to interpolate onto
    :param df_real_neighbourhood: the neighbourhood within which to compute
    :param real_coords: the co-ordinates tof sample points
    :param real_boundary_shape: the neighbnourhood boundary
    :param points: the interpolation points
    :returns: a tensor'''

     # construct an array that will hold the co-ordinates of all the real points
    # and the synthetic point
    synthetic_coords = numpy.append(real_coords, [[0, 0]], axis=0)

    # construct the partial tensor
    tensor = []

    for _, p in points.iterrows():
        # re-compute the Voronoi cells given the synthetic point
        synthetic_coords[-1] = points_to_coords([p.geometry])[0]
        synthetic_voronoi_cells, synthetic_voronoi_centres = voronoi_regions_from_coords(synthetic_coords, real_boundary_shape)

        # get the synthetic cell
        i = [i for i in synthetic_voronoi_centres.keys() if len(synthetic_coords) - 1 in synthetic_voronoi_centres[i]][0]
        synthetic_cell = synthetic_voronoi_cells[i]

        # compute the weights
        synthetic_cell_area = synthetic_cell.area
        for _, r in df_real_neighbourhood.iterrows():
            area = r.geometry.intersection(synthetic_cell).area
            if area > 0.0:
                tensor.append((int(p['x']), int(p['y']), int(r['sample']), area / synthetic_cell_area))

    return tensor


def nnn_tensor(df_points, df_voronoi, df_grid, cores=1):
    '''Construct the natural nearest neighbour interpolation tensor from a
    set of samples taken within a boundary and sampled at the given
    grid of interpolation points.

    The tensor is a three-dimensional sparse `numpy.array` with axes
    corresponding to xs, ys, and points, with entries containing the
    weight given to each sample in interpolating each point.

    The operation can be run in parallel on a multicore machine. The degree
    of parallelism is given by cores, which may be:

    - 0 to use the maximum number of available cores
    - +n to use a specific number of cores
    - -n to leave n cores unused

    There is no benefit to using a degree of parallelism greater than
    the number of physical cores on the machine. In cases where there are
    very few cells to compute a smaller amount of parallelism will
    be used anyway.

    :param df_points: the sample points
    :param df_voronoi: the Voronoi diagram of the samples
    :param df_grid: the grid to interpolate onto
    :param cores: (optional) number of cores to use (defaults to 1)
    :returns: a tensor'''

    # compute the nunber of cores to use
    if cores == 0:
        # use all available or the number of cells, whichever is smaller
        cores = min(len(df_points), cpu_count())
    elif cores < 0:
        # use fewer than available, down to a minimum of 1
        cores = min(len(df_points), max(cpu_count() + cores, 1))   # cpu_count() + cores as cores is negative
    else:
        # use the number of cores requested, up to the maximum available,
        # redcuced if there are only a few cells
        cores = min(cores, len(df_points), cpu_count())

    # construct the tensor
    tensor = numpy.zeros((max(df_grid.x) + 1, max(df_grid.y) + 1, len(df_points)))

    # group the grid points by the real cell they lie within
    grid_grouped = df_grid.groupby('cell').groups

    # ignore any points outside of the Voronoi diagram
    if -1 in grid_grouped.keys():
        del grid_grouped[-1]

    # run each cell as its own job
    jobs = []
    for real_cell in grid_grouped.keys():
        # determine the area to compute
        df_real_neighbourhood = df_voronoi.loc[df_voronoi.loc[real_cell].neighbourhood]
        real_coords = points_to_coords(df_real_neighbourhood.centre)
        real_boundary_shape = df_voronoi.loc[real_cell].boundary
        synthetic_points = df_grid.iloc[grid_grouped[real_cell]]

        # create a job record for this cell
        jobs.append((df_real_neighbourhood, real_coords, real_boundary_shape,
                     synthetic_points))

    # run each cell as its own job
    with Parallel(n_jobs=cores) as processes:
        rcs = processes(delayed(lambda j: nnn_tensor_worker(df_voronoi, df_grid, *j))(j) for j in jobs)

        # store the computed weights in the tensor
        for ds in rcs:
            for (x, y, sample, value) in ds:
                tensor[x, y, sample] = value

    return tensor


def apply_tensor(tensor, samples):
    '''Apply an interpolation tensor to a sample vector, giving
    a grid of interpolated points.

    :param tensor: the tensor
    :param samples: the sample vector
    :param boundary_shape: the overall shape of the region
    :returns: a grid of interpolated points

    '''

    # check dimensions
    if len(samples) != tensor.shape[2]:
        raise ValueError('Tensor needs {n} samples, got {m}'.format(n=tensor.shape[2],
                                                                    m=len(samples)))

    # create the grid
    grid = numpy.zeros((tensor.shape[0], tensor.shape[1]))

    # apply the tensor
    sample = 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            grid[x, y] = numpy.dot(tensor[x, y, :], samples)

    return grid


def nnn_masked_grid(grid, boundary_shape, xs, ys):
    '''Mask the grid to the boundary shape.

    :param grid: the grid
    :param boundary_shape: the boundary
    :param xs: the x-axis co-ordinates
    :param ys: the y-axis co-ordinates
    :returns: a grid masked to the boundary shape
    '''

    # generate the mask
    mask = numpy.empty(grid.shape)
    for i in range(len(xs)):
        for j in range(len(ys)):
            x, y = xs[i], ys[j]
            mask[i, j] = not boundary_shape.contains(Point(x, y))

    # return the masked grid
    return numpy.ma.masked_where(mask, grid, copy=False)


def natural_nearest_neighbour(df_points, boundary_shape, xs, ys):
    '''Interpolate samples given by the df_points `DataFrame`
    at positions given by co-ordinates from xs and ys.

    The returned array will have columns `geometry` for the
    interpolated points, `x` and `y` for the indices of the observation
    along the two axes, and `rainfall` for the interpolated rainfall.
    The grid is masked to the boundary shape, so not all points implied
    by the axes will necessarily have values assigned to them.

    :param df_points: the samples
    :param boundary_shape: the boundary surrounding the samples
    :param xs: list of x co-ordinates to interpolate at
    :param ys: list of y co-ordinates to interpolate at
    :returns: a grid of interpolated grid points'''

    # construct the tensor the real Voronoi cells around the sample points
    df_voronoi = nnn_voronoi(df_points, boundary_shape)

    # construct the interpolation grid
    df_grid = nnn_geometry(df_points, df_voronoi, xs, ys)

    # construct the tensor
    tensor = nnn_tensor(df_points, df_voronoi, df_grid)

    # extract the samples
    samples = numpy.array(df_points['rainfall_amount'])

    # apply the tensor to the samples
    grid = apply_tensor(tensor, samples)

    # mask the parts of the grid not lying within the boundary
    masked = nnn_masked_grid(grid, boundary_shape, xs, ys)
    return masked
