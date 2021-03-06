# Interpolation tensor using natural nearest neighbnour interpolation
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
import numpy
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from sensor_placement import Logger, InterpolationTensor


logger = logging.getLogger(Logger)


class NNNI(InterpolationTensor):

    def buildTensor(self):
        # construct the tensor
        then = datetime.now()

        # create tensor matrix
        self._tensor = numpy.zeros((max(self._grid['x']) + 1, max(self._grid['y']) + 1, len(self._samples)))
        logger.debug('Tensor created with shape {s}'.format(s=self._tensor.shape))

        # populate the tensor
        cores = self.numberOfCores()
        if cores== 1:
            self._tensorSeq()
        else:
            self._tensorPar(cores)

        # report time
        dt = (datetime.now() - then).seconds
        logger.info(f'Tensor computed in {dt:.2f}s')

    def _tensorSeq(self):
        '''Construct the natural nearest neighbour interpolation tensor from a
        set of samples taken within a boundary and sampled at the given
        grid of interpolation points.'''
        for (x, y, si, v) in self.iterateWeightsFor():
            self._tensor[x, y, si] = v

    def _tensorPar(self, cores):
        '''Construct the natural nearest neighbour interpolation tensor from a
        set of samples taken within a boundary and sampled at the given
        grid of interpolation points.'''

        # create parallel jobs for each cell
        with Parallel(n_jobs=cores) as processes:
            # run jobs
            real_cells = list(self._grid['cell'].unique())
            if -1 in real_cells:
                real_cells.remove(-1)
            rcs = processes(delayed(lambda c: list(self.iterateWeightsFor([c])))(cell) for cell in real_cells)

            # store the computed weights in the tensor
            for ds in rcs:
                for (x, y, si, v) in ds:
                    self._tensor[x, y, si] = v

    def iterateWeightsFor(self, real_cells = None):
        '''Compute the weights for all points in the given cells (all by default).
        This is an iterator that generates a stream of weights'''

        # group the grid points by the real cell they lie within
        grid_grouped = self._grid.groupby('cell').groups

        # ignore any points outside the Voronoi diagram
        if -1 in grid_grouped.keys():
            del grid_grouped[-1]

        # determine the cells
        if real_cells is None:
            real_cells = grid_grouped.keys()

        # construct the weights
        for real_cell in real_cells:
            logger.debug(f'Computing weights for cell {real_cell}')

            # extract the neighbourhood of Voronoi cells,
            # the only ones that the cell around this sample point
            # can intersect and so the only computation we need to do
            df_real_neighbourhood = self._voronoi.loc[self._voronoi.loc[real_cell].neighbourhood]
            real_coords = list(df_real_neighbourhood.centre)
            real_boundary_shape = self._voronoi.loc[real_cell].boundary

            for pt in grid_grouped[real_cell]:
                # re-compute the Voronoi cells given the synthetic point
                p = self._grid.loc[pt]
                synthetic_coords = real_coords + [p.geometry]
                synthetic_voronoi_cells = self.voronoiCells(synthetic_coords, real_boundary_shape)
                synthetic_cell = synthetic_voronoi_cells[-1]

                # compute the weights
                synthetic_cell_area = synthetic_cell.area
                for id, r in df_real_neighbourhood.iterrows():
                    area = r.geometry.intersection(synthetic_cell).area
                    if area > 0.0:
                        si = self._samples.index.get_loc(id)
                        yield (int(p['x']), int(p['y']), si, area / synthetic_cell_area)
