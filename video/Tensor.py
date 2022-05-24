# Scene explaining the tensor representation
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

# See https://help.ceda.ac.uk/article/4442-ceda-opendap-scripted-interactions

import numpy
from manim import *
import shapely
from shapely.geometry import (Point as shapely_Point,
                              MultiPoint as shapely_MultiPoint,
                              Polygon as shapely_Polygon)
from shapely.ops import cascaded_union, nearest_points, voronoi_diagram
from shapely.affinity import translate
from geopandas import GeoDataFrame, GeoSeries
import NNNI


class Tensor(Scene):

    def construct(self):
        # create an NNNI objecvt to re-use the calculations
        nnni = NNNI.NNNI()
        nnni.begin()

        diagram = Group()

        # draw base diagram as created by NNNI.py
        self.next_section('diagram')
        region = Square(side_length=nnni.EDGE)
        diagram.add(region)
        samples = []
        for _, pt in nnni._df_samples.iterrows():
            xy = list(pt.geometry.coords[0])
            p = nnni.ORIGIN + RIGHT * (xy[0] * nnni.EDGE) + UP * (xy[1] * nnni.EDGE)
            samples.append(Dot(p))
        g = VGroup(*samples)
        diagram.add(g)
        cells = []
        for c in nnni._df_voronoi.geometry:
            xys = list(c.exterior.coords)
            vs = map(lambda xy: nnni.ORIGIN + RIGHT * (xy[0] * nnni.EDGE) + UP * (xy[1] * nnni.EDGE), xys)
            poly = Polygon(*vs, stroke_color=PURPLE_A).set_fill(color=PURPLE_E, opacity=0.4)
            cells.append(poly)
        g = VGroup(*cells)
        diagram.add(g)
        self.play(FadeIn(diagram))

        # add the grid lines
        self.next_section('interpolation grid')
        dx = nnni.EDGE / nnni.GRID
        hlines, vlines = [], []
        for i in range(1, nnni.GRID):
            # vertical
            l = Line(start=nnni.ORIGIN + RIGHT * i * dx,
                     end=nnni.ORIGIN + RIGHT * i * dx + UP * nnni.EDGE,
                     color=GOLD)
            vlines.append(l)

            # horizontal
            l = Line(start=nnni.ORIGIN + UP * i * dx,
                     end=nnni.ORIGIN + UP * i * dx + RIGHT * nnni.EDGE,
                     color=GOLD)
            hlines.append(l)
        grid = VGroup(*vlines, *hlines)
        self.play(Create(grid), run_time=3)
        shorten = []
        for i in range(1, nnni.GRID):
            vl = vlines[i - 1]
            shorten.append(vl.animate.put_start_and_end_on(start=nnni.ORIGIN + RIGHT * i * dx + UP * dx,
                                                           end=nnni.ORIGIN + RIGHT * i * dx + UP * (nnni.EDGE - dx)))
            hl = hlines[i - 1]
            shorten.append(hl.animate.put_start_and_end_on(start=nnni.ORIGIN + UP * i * dx + RIGHT * dx,
                                                           end=nnni.ORIGIN + UP * i * dx + RIGHT * (nnni.EDGE - dx)))
        self.play(*shorten)

        self.wait(5)
