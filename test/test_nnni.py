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
from copy import deepcopy
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

        t = NNNI(df_points, boundary, xs, ys)

        self.assertEqual(t.shape, (10, 20, len(df_points)))
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

        t = NNNI(df_points, boundary, xs, ys)

        samples = numpy.asarray([50, 0, 50, 0])
        g = t(samples)

        self.assertEqual(g.shape, (len(xs), len(ys)))
        for i in range(len(xs)):
            for j in range(len(ys)):
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

        t = NNNI(df_points, boundary, xs, ys)
        t_seq = t._tensor.copy()

        t = NNNI(df_points, boundary, xs, ys, cores=2)
        t_par = t._tensor.copy()

        for i in range(len(xs)):
            for j in range(len(ys)):
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

        t = NNNI(df_points, boundary, xs, ys)
        t_orig = deepcopy(t)

        t.removeSample(2)

        # tensor dimensions reduced
        self.assertEqual(t.shape, (len(xs), len(ys), len(df_points) - 1))

        # neighbourhoods correct
        self.assertCountEqual(t._samples.index, [0, 1, 3, 4])
        self.assertCountEqual(set(t._voronoi.loc[0].neighbourhood), set([0, 1, 3, 4]))
        self.assertCountEqual(set(t._voronoi.loc[1].neighbourhood), set([0, 1, 3, 4]))
        self.assertCountEqual(set(t._voronoi.loc[3].neighbourhood), set([0, 1, 3, 4]))
        self.assertCountEqual(set(t._voronoi.loc[4].neighbourhood), set([0, 1, 3, 4]))

        # points in the right cells
        self.assertNotIn(2, set(t._grid['cell']))
        for x in range(int(len(xs) * 0.5)):
            for y in range(int(len(ys) * 0.5)):
                p = t._grid[t._grid['x'] == x]
                q = p[p['y'] == y]
                self.assertEqual(q['cell'].iloc[0], 0)
        for x in range(int(len(xs) * 0.5) + 1, len(xs)):
            for y in range(int(len(ys) * 0.5)):
                p = t._grid[t._grid['x'] == x]
                q = p[p['y'] == y]
                self.assertEqual(q['cell'].iloc[0], 1)
        for x in range(int(len(xs) * 0.5)):
            for y in range(int(len(ys) * 0.5) + 1, len(ys)):
                p = t._grid[t._grid['x'] == x]
                q = p[p['y'] == y]
                self.assertEqual(q['cell'].iloc[0], 4)
        for x in range(int(len(xs) * 0.5 ) + 1, len(xs)):
            for y in range(int(len(ys) * 0.5) + 1, len(ys)):
                p = t._grid[t._grid['x'] == x]
                q = p[p['y'] == y]
                self.assertEqual(q['cell'].iloc[0], 3)

        # boundaries of cells correct
        self.assertTrue(t._voronoi.loc[0]['boundary'].equals(boundary))
        self.assertTrue(t._voronoi.loc[1]['boundary'].equals(boundary))
        self.assertTrue(t._voronoi.loc[3]['boundary'].equals(boundary))
        self.assertTrue(t._voronoi.loc[4]['boundary'].equals(boundary))

        # points all within their cell boundaries
        for x in range(len(t._xs)):
            for y in range(len(t._ys)):
                r = t._grid[t._grid['x'] == x]
                c = r[r['y'] == y]
                cell = c.iloc[0]['cell']
                boundary = t._voronoi.loc[cell].geometry
                self.assertTrue(boundary.intersects(Point(t._xs[x], t._ys[y])))

        # weights in new tensor should always be greater than or equal to those
        # in the original tensor, since we've removed information
        for x in range(len(xs)):
            for y in range(len(ys)):
                for s in range(len(df_points) - 1):
                    s_orig = s + 1 if s >= 2 else s  # skip removed cell
                    self.assertGreaterEqual(t._tensor[x, y, s], t_orig._tensor[x, y, s_orig])


if __name__ == '__main__':
    unittest.main()
