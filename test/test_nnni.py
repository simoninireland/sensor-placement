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


import logging
logging.basicConfig(level=logging.DEBUG)


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
        self.assertEqual(len(t._grid), len(xs) * len(ys))

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

    def testDistance(self):
        '''Test distances in grid.'''
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

        # check distances
        for i in range(len(xs)):
            for j in range(len(ys)):
                g1 = t._grid[t._grid['x'] == i]
                g2 = g1[g1['y'] == j].iloc[0]
                c = g2['cell']
                d = g2['distance']
                p = Point(xs[i], ys[j])
                q = df_points.loc[c].geometry
                ps, qs = list(p.coords)[0], list(q.coords)[0]
                h = numpy.sqrt((qs[0] - ps[0]) ** 2 + (qs[1] - ps[1]) ** 2)
                self.assertEqual(h, d)


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

        # check distances
        for i in range(len(xs)):
            for j in range(len(ys)):
                g1 = t._grid[t._grid['x'] == i]
                g2 = g1[g1['y'] == j].iloc[0]
                c = g2['cell']
                d = g2['distance']
                p = Point(xs[i], ys[j])
                q = df_points.loc[c].geometry
                ps, qs = list(p.coords)[0], list(q.coords)[0]
                h = numpy.sqrt((qs[0] - ps[0]) ** 2 + (qs[1] - ps[1]) ** 2)
                self.assertEqual(h, d)

    def testRemoves(self):
        '''Test we can remove several samples.'''
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

        to_remove = [0, 2]
        t.removeSamples(to_remove)

        # tensor dimensions reduced
        self.assertEqual(t.shape, (len(xs), len(ys), len(df_points) - len(to_remove)))

        # neighbourhoods correct
        self.assertCountEqual(t._samples.index, [1, 3, 4])
        self.assertCountEqual(set(t._voronoi.loc[1].neighbourhood), set([1, 3, 4]))
        self.assertCountEqual(set(t._voronoi.loc[3].neighbourhood), set([1, 3, 4]))
        self.assertCountEqual(set(t._voronoi.loc[4].neighbourhood), set([1, 3, 4]))

        # points in the right cells
        self.assertNotIn(0, set(t._grid['cell']))
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

        # check distances
        for i in range(len(xs)):
            for j in range(len(ys)):
                g1 = t._grid[t._grid['x'] == i]
                g2 = g1[g1['y'] == j].iloc[0]
                c = g2['cell']
                d = g2['distance']
                p = Point(xs[i], ys[j])
                q = df_points.loc[c].geometry
                ps, qs = list(p.coords)[0], list(q.coords)[0]
                h = numpy.sqrt((qs[0] - ps[0]) ** 2 + (qs[1] - ps[1]) ** 2)
                self.assertEqual(h, d)

    def testAdd(self):
        '''Test we can add a sample.'''
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
        t_orig = deepcopy(t)

        new_cell = t.addSample(0.5, 0.5)

        # tensor dimensions increased
        self.assertEqual(t.shape, (len(xs), len(ys), len(df_points) + 1))

        # neighbourhoods correct
        self.assertCountEqual(t._samples.index, set([0, 1, 2, 3, new_cell]))
        self.assertCountEqual(set(t._voronoi.loc[0].neighbourhood), set([0, 1, 3, new_cell]))
        self.assertCountEqual(set(t._voronoi.loc[1].neighbourhood), set([1, 0, 2, new_cell]))
        self.assertCountEqual(set(t._voronoi.loc[2].neighbourhood), set([2, 1, 3, new_cell]))
        self.assertCountEqual(set(t._voronoi.loc[3].neighbourhood), set([3, 0, 2, new_cell]))
        self.assertCountEqual(set(t._voronoi.loc[new_cell].neighbourhood), set([0, 1, 2, 3, new_cell]))
        # boundaries correct
        self.assertTrue(t._voronoi.loc[new_cell]['boundary'].equals(boundary))

        # points all within their cell boundaries
        for x in range(len(t._xs)):
            for y in range(len(t._ys)):
                r = t._grid[t._grid['x'] == x]
                c = r[r['y'] == y]
                cell = c.iloc[0]['cell']
                boundary = t._voronoi.loc[cell].geometry
                self.assertTrue(boundary.intersects(Point(t._xs[x], t._ys[y])))

        # weights in new tensor should always be less than or equal to those
        # in the original tensor, since we've added information
        for x in range(len(xs)):
            for y in range(len(ys)):
                for s in range(len(df_points) - 1):
                    self.assertLessEqual(t._tensor[x, y, s], t_orig._tensor[x, y, s])

        # check distances
        for i in range(len(xs)):
            for j in range(len(ys)):
                g1 = t._grid[t._grid['x'] == i]
                g2 = g1[g1['y'] == j].iloc[0]
                c = g2['cell']
                d = g2['distance']
                p = Point(xs[i], ys[j])
                q = t._samples.loc[c].geometry
                ps, qs = list(p.coords)[0], list(q.coords)[0]
                h = numpy.sqrt((qs[0] - ps[0]) ** 2 + (qs[1] - ps[1]) ** 2)
                self.assertEqual(h, d)

    def testRemoveMappingSingle(self):
        '''Test we can generate the removal map of a single sample.'''
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

        m = t.remapSamplesOnRemoval([1])
        print(m)
        self.assertEqual(len(m), len(df_points) - 1)
        self.assertCountEqual(m, [0, 2, 3, 4])

    def testRemoveMappingMultiple(self):
        '''Test we can generate the removal map of a block of samples.'''
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

        m = t.remapSamplesOnRemoval([1, 3])
        self.assertEqual(len(m), len(df_points) - 2)
        self.assertCountEqual(m, [0, 2, 4])

    def testRemoveMappingMultipleUnorder(self):
        '''Test we can generate the removal map of a block of samples provided out-of-order.'''
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

        m = t.remapSamplesOnRemoval([3, 2])
        self.assertEqual(len(m), len(df_points) - 2)
        self.assertCountEqual(m, [0, 1, 4])

    def testRemoveMappingFirstLast(self):
        '''Test we can generate the removal map of the first and last samples.'''
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

        m = t.remapSamplesOnRemoval([0, 4])
        self.assertEqual(len(m), len(df_points) - 2)
        self.assertCountEqual(m, [1, 2, 3])

    def testRemoveMappingAll(self):
        '''Test we can generate the removal map of all the samples.'''
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

        m = t.remapSamplesOnRemoval([0, 4, 2, 3, 1])
        self.assertEqual(len(m), 0)

    def testRemoveMappingNone(self):
        '''Test we can generate the removal map for none of the samples.'''
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

        m = t.remapSamplesOnRemoval([])
        self.assertEqual(len(m), len(df_points))
        self.assertCountEqual(m, list(range(len(df_points))))

    def testResample(self):
        '''Test we can resample..'''
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

        toRemove = [1, 3]
        ss2 = [0, 2, 4]                        # samples match "after" indices
        m = t.resampleOnRemoval(toRemove)
        self.assertEqual(len(m), len(df_points) - len(toRemove))
        ss3 = numpy.zeros((len(df_points),))
        ss3[m] = ss2
        self.assertCountEqual(ss3, [0, 0.0, 2, 0.0, 4])

    def testResampleUnordered(self):
        '''Test we can resample out-of-order.'''
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

        toRemove = [3, 1]
        ss2 = [0, 2, 4]                        # samples match "after" indices
        m = t.resampleOnRemoval(toRemove)
        self.assertEqual(len(m), len(df_points) - len(toRemove))
        ss3 = numpy.zeros((len(df_points),))
        ss3[m] = ss2
        self.assertCountEqual(ss3, [0, 0.0, 2, 0.0, 4])

    def testResampleNone(self):
        '''Test we can "resample" after no removals.'''
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

        toRemove = []
        ss2 = [0, 1, 2, 3, 4]                    # samples match "after" indices
        m = t.resampleOnRemoval(toRemove)
        self.assertEqual(len(m), len(df_points) - len(toRemove))
        ss3 = numpy.zeros((len(df_points),))
        ss3[m] = ss2
        self.assertCountEqual(ss3, [0, 1, 2, 3, 4])

    def testResampleFirstLast(self):
        '''Test we can resample after removing the first and last points.'''
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

        toRemove = [4, 0]
        ss2 = [1, 2, 3]                        # samples match "after" indices
        m = t.resampleOnRemoval(toRemove)
        self.assertEqual(len(m), len(df_points) - len(toRemove))
        ss3 = numpy.zeros((len(df_points),))
        ss3[m] = ss2
        self.assertCountEqual(ss3, [0.0, 1, 2, 3, 0.0])


if __name__ == '__main__':
    unittest.main()
