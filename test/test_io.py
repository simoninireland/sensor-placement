# Test tensor I/O
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
import os
import logging
from tempfile import NamedTemporaryFile
import numpy
from sensor_placement import *
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon


#logging.basicConfig(level=logging.DEBUG)


class IOTest(unittest.TestCase):

    def setUp( self ):
        '''Set up with a temporary file.'''
        #tf = NamedTemporaryFile()
        #tf.close()
        #self._fn = tf.name
        self._fn = 'test.nc'

    def tearDown( self ):
        '''Delete the temporary file.'''
        try:
            #os.remove(self._fn)
            pass
        except OSError:
            pass

    def testSave(self):
        '''Test we can save a tensor.'''
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
        t.save(self._fn)

    def testLoadAndSave(self):
        '''Test we can load and save a tensor.'''

        # use an asymmetric region and samnple points to check
        # for rotations
        boundary = Polygon([Point(0.0, 0.0),
                            Point(0.0, 1.0),
                            Point(0.5, 1.0),
                            Point(0.5, 0.0)])
        df_points = GeoDataFrame([Point(0.25, 0.25),
                                  Point(0.45, 0.45),
                                  Point(0.45, 0.75),
                                  Point(0.1, 0.3)], columns=['geometry'])
        xs = numpy.linspace(0.0, 0.5, num=10)
        ys = numpy.linspace(0.0, 1.0, num=20)

        # create and save tensor
        t = NNNI(df_points, boundary, xs, ys)
        t.save(self._fn)

        # ... and then load it back
        r = NNNI.load(self._fn, cores=-2)

        # check grid -- need to check individually as there's no
        # assertCountAlmostEqual() function....
        self.assertEqual(len(r.xs()), len(xs))
        self.assertEqual(len(r.ys()), len(ys))
        for i in range(len(xs)):
            self.assertAlmostEqual(r.xs()[i], xs[i], places=5)
        for j in range(len(ys)):
            self.assertAlmostEqual(r.ys()[j], ys[j], places=5)

        # check distances
        for i in range(len(xs)):
            for j in range(len(ys)):
                g1 = r._grid[t._grid['x'] == i]
                g2 = g1[g1['y'] == j].iloc[0]
                c = g2['cell']
                d = g2['distance']
                p = Point(xs[i], ys[j])
                q = df_points.loc[c].geometry
                ps, qs = list(p.coords)[0], list(q.coords)[0]
                h = numpy.sqrt((qs[0] - ps[0]) ** 2 + (qs[1] - ps[1]) ** 2)
                self.assertAlmostEqual(h, d, places=5)

        # apply both tensors to the same sample
        samples = numpy.asarray([50, 0, 50, 0])
        g_t = t(samples)
        g_r = r(samples)

        # check shapes
        self.assertEqual(g_t.shape, (len(xs), len(ys)))
        self.assertEqual(g_r.shape, (len(xs), len(ys)))

        # check interpolated values
        for i in range(len(xs)):
            for j in range(len(ys)):
                self.assertAlmostEqual(g_t[i, j], g_r[i, j], places=5)
