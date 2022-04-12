# Test natural nearest neighbour
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

import unittest
import numpy
from sensor_placement import *
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon


class NNNITest(unittest.TestCase):

    # ---------- Basics ----------

    def testConstruction(self):
        '''Test we can build the tensor.'''
        boundary = Polygon([Point(0.0, 0.0),
                            Point(0.0, 1.0),
                            Point(1.0, 1.0),
                            Point(1.0, 0.0)])
        df_points = GeoDataFrame([Point(0.25, 0.25),
                                  Point(0.75, 0.25),
                                  Point(0.75, 0.75),
                                  Point(0.25, 0.75)], columns=['geometry'])
        xs = numpy.linspace(0.0, 1.0, num=10)
        ys = numpy.linspace(0.0, 1.0, num=20)

        t = NNNI(df_points, boundary, ys, xs)

        self.assertEqual(t.shape, (20, 10, len(df_points)))
        self.assertEqual(len(t.weights(0, 0)), len(df_points))
        self.assertEqual(t.shape, t._tensor.shape)
        self.assertEqual(t._grid.shape, (len(xs) * len(ys), len(df_points)))

    def testApply(self):
        '''Test we can apply the tensor.'''
        boundary = Polygon([Point(0.0, 0.0),
                            Point(0.0, 1.0),
                            Point(1.0, 1.0),
                            Point(1.0, 0.0)])
        df_points = GeoDataFrame([Point(0.25, 0.25),
                                  Point(0.75, 0.25),
                                  Point(0.75, 0.75),
                                  Point(0.25, 0.75)], columns=['geometry'])
        xs = numpy.linspace(0.0, 1.0, num=10)
        ys = numpy.linspace(0.0, 1.0, num=20)

        t = NNNI(df_points, boundary, ys, xs)

        samples = numpy.asarray([50, 0, 50, 0])
        g = t(samples)

        self.assertEqual(g.shape, (len(ys), len(xs)))
        for i in range(len(ys)):
            for j in range(len(xs)):
                w = t.weights(i, j)
                self.assertFalse((w == 0).all())

    def testApplyPar(self):
        '''Test parallel and sequential return the same answer.'''
        boundary = Polygon([Point(0.0, 0.0),
                            Point(0.0, 1.0),
                            Point(1.0, 1.0),
                            Point(1.0, 0.0)])
        df_points = GeoDataFrame([Point(0.25, 0.25),
                                  Point(0.75, 0.25),
                                  Point(0.75, 0.75),
                                  Point(0.25, 0.75)], columns=['geometry'])
        xs = numpy.linspace(0.0, 1.0, num=10)
        ys = numpy.linspace(0.0, 1.0, num=20)

        t = NNNI(df_points, boundary, ys, xs)
        t_seq = t._tensor.copy()

        t = NNNI(df_points, boundary, ys, xs, cores=2)
        t_par = t._tensor.copy()

        for i in range(len(ys)):
            for j in range(len(xs)):
                for s in range(len(df_points)):
                    self.assertEqual(t_seq[i, j, s], t_par[i, j, s])


    # ---------- Editing----------

    def testRemove(self):
        '''Test we can remove a sample.'''
        boundary = Polygon([Point(0.0, 0.0),
                            Point(0.0, 1.0),
                            Point(1.0, 1.0),
                            Point(1.0, 0.0)])
        df_points = GeoDataFrame([Point(0.25, 0.25),
                                  Point(0.75, 0.25),
                                  Point(0.5, 0.5),
                                  Point(0.75, 0.75),
                                  Point(0.25, 0.75)], columns=['geometry'])
        xs = numpy.linspace(0.0, 1.0, num=10)
        ys = numpy.linspace(0.0, 1.0, num=20)

        t = NNNI(df_points, boundary, ys, xs)

        t.removeSample(2)

        self.assertEqual(t.shape, (20, 10, len(df_points) - 1))
        self.assertCountEqual(set(t._voronoi.loc[0].neighbourhood), set([0, 1, 3, 4]))
        self.assertNotIn(2, set(t._grid['cell']))


if __name__ == '__main__':
    unittest.main()
