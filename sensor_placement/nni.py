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
import numpy
from geopandas import GeoDataFrame
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from shapely.geometry import shape, Point, Polygon
from shapely.ops import cascaded_union


def voronoi_from_samples(df_points, boundary_shape):
    '''Construct a table ot Voronoi cells and their adjacencies based on
    a set of samples and a boundary.''

    The sample points should be supplied in a `GeoDataFrame` having
    columns `geometry` containing the sample points and `rainfall`
    holding the observation at that point. All the samples should
    lie within the boundary_shape.

    The returned `DataFrame` will contain the `geometry` of each cell,
    it's `centre`1 (the sample point it surrounds), a `neighbourhood`
    holding the indices of the cells that intersect this cell (including
    the index of the cell itself), and a `boundary` holding the outer
    boundary of this set.

    :param df_points: the samples
    :param boundary_shape: the boundary surrounding the samples
    :returns: a dataframe'''
    coords = points_to_coords(df_points.geometry)
    voronoi_cells, voronoi_centres = voronoi_regions_from_coords(coords, boundary_shape)
    df_voronoi = GeoDataFrame({'centre': [df_points.iloc[voronoi_centres[i][0]].geometry for i in voronoi_cells.keys()],
                               'geometry': voronoi_cells.values()})

    # add the neighbourhoods of each cell, their index and overall boundary
    neighbourhoods = []
    boundaries = []
    for i, cell in df_voronoi.iterrows():
        neighbours = list(df_voronoi[df_voronoi.geometry.touches(cell.geometry)].index) + [i]
        neighbourhoods.append(neighbours)

        # boundary of neighbourhood
        boundaries.append(cascaded_union(df_voronoi.loc[neighbours].geometry))
    df_voronoi['neighbourhood'] = neighbourhoods
    df_voronoi['boundary'] = boundaries

    return df_voronoi


def interpolation_grid(xs, ys, df_points, df_voronoi):
    '''Construct the grid of interpolation points from a set of samples,
    a set of their voronoi cells, and the sample point axes.

    The returned `DataFrame` will have columns `geometry` for the
    interpolated points, `x` and `y` for the indices of the observation
    along the two axes, and `cell` holding the index of the Voronoi cell
    within which the interpolation point lies.

    :param xs: list of x co-ordinates to interpolate at
    :param ys: list of y co-ordinates to interpolate at
    :param df_points: the samples
    :param df_voronoi: the Voronoi cells for these samples
    :returns: a dataframe'''
    df_interpoints = GeoDataFrame({'x': [i for l in [[j] * len(ys) for j in range(len(xs))] for i in l],
                                   'y': list(range(len(list(ys)))) * len(xs),
                                   'geometry': [Point(x, y) for (x, y) in product(xs, ys)]})
    cells = []
    for _, cell in df_interpoints.iterrows():
        cells.append(df_voronoi[df_voronoi.geometry.intersects(cell.geometry)].geometry.index[0])
    df_interpoints['cell'] = cells

    return df_interpoints


def natural_nearest_neighbour(df_points, boundary_shape, xs, ys):
    '''Interpolate samples given by the df_points `DataFrame`
    at positions given by co-ordinates from xs and ys.

    The sample points should be supplied in a `GeoDataFrame` having
    columns `geometry` containing the sample points and `rainfall`
    holding the observation at that point. All the samples should
    lie within the boundary_shape.

    The returned `DataFrame` will have columns `geometry` for the
    interpolated points, `x` and `y` for the indices of the observation
    along the two axes, and `rainfall` for the interpolated rainfall.

    :param df_points: the samples
    :param boundary_shape: the boundary surrounding the samples
    :param xs: list of x co-ordinates to interpolate at
    :param ys: list of y co-ordinates to interpolate at
    :returns: a dataframe'''

    # check that all the sample points lie within the boundary
    if not df_points.geometry.within(boundary_shape).all():
        raise ValueError('At least one point lies on or outside the boundary')

    # construct the real Voronoi cells around the sample points
    df_voronoi = voronoi_from_samples(df_points, boundary_shape)

    # construct the interpolation grid
    df_interpoints = interpolation_grid(xs, ys, df_points, df_voronoi)

    # perform the interpolation
    interpoints_grouped = df_interpoints.groupby('cell').groups
    interpolated_rainfall = []
    for real_cell in interpoints_grouped.keys():
        # extract the neighbourhood of Voronoi cells,
        # the only ones that the cell around this sample point
        # can intersect and so the only computation we need to do
        df_real_neighbourhood = df_voronoi.loc[df_voronoi.loc[real_cell].neighbourhood]
        real_coords = points_to_coords(df_real_neighbourhood.centre)
        real_boundary_shape = df_voronoi.loc[real_cell].boundary

        # construct an array that will hold the co-ordinates of all the real points
        # and the synthetic point
        synthetic_coords = numpy.array(numpy.append(real_coords, [[0, 0]], axis=0))

        for pt in interpoints_grouped[real_cell]:
            # re-compute the Voronoi cells given the syntheic point
            p = df_interpoints.loc[pt].geometry
            synthetic_coords[-1] = points_to_coords([p])[0]
            synthetic_voronoi_cells, synthetic_voronoi_centres = voronoi_regions_from_coords(synthetic_coords, real_boundary_shape)

            # get the synthetic cell
            i = [i for i in synthetic_voronoi_centres.keys() if len(synthetic_coords) - 1 in synthetic_voronoi_centres[i]][0]
            synthetic_cell = synthetic_voronoi_cells[i]

            # compute the weighted value
            synthetic_cell_area = synthetic_cell.area
            synthetic_rainfall = 0
            total_area = 0
            for _, r in df_real_neighbourhood.iterrows():
                area = r.geometry.intersection(synthetic_cell).area
                total_area += area
                if area > 0.0:
                    obs = df_points[df_points.geometry == r.centre].iloc[0].rainfall
                    synthetic_rainfall += (area / synthetic_cell_area) * obs

            # store synthetic rainfall
            interpolated_rainfall.append(synthetic_rainfall)

    # wrangle the interpolated data into the correct order, drop
    # internal working information, and return
    interpolated_points_in_order = [p for ps in [interpoints_grouped[g] for g in interpoints_grouped.keys()] for p in ps]
    df = df_interpoints.loc[interpolated_points_in_order]
    df['rainfall'] = interpolated_rainfall
    df.drop(columns='cell', inplace=True)
    return df
