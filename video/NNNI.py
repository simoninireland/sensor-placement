# Scene explaining natural nearest-neighbour interpolation
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
from shapely.geometry import (Point as shapely_Point,
                              MultiPoint as shapely_MultiPoint,
                              Polygon as shapely_Polygon)
from shapely.ops import voronoi_diagram
from geopandas import GeoDataFrame


class NNNI(Scene):

    # drawing parameters
    EDGE = 5.0                                            # edge length for interpolation region
    GRID = 10                                             # number of grid points
    ORIGIN = [0, 0, 0]                                    # origin of interpolation region
    BOUNDARY = shapely_Polygon([shapely_Point(0.0, 0.0),  # boundary of region
                                shapely_Point(0.0, 1.0),
                                shapely_Point(1.0, 1.0),
                                shapely_Point(1.0, 0.0)])
    SAMPLES = [shapely_Point(0.15, 0.15),                 # sample points
               shapely_Point(0.45, 0.75),
               shapely_Point(0.75, 0.35)]
    SAMPLE_VALUES = [25, 16, 7]                           # sample values at these points
    SYNTHETIC = shapely_Point(0.5, 0.2)                   # synthetic point

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        rng = numpy.random.default_rng()
        self.ORIGIN = (LEFT + DOWN) * (self.EDGE / 2)

        # samples
        self._df_samples = GeoDataFrame(geometry=self.SAMPLES)
        self._df_samples['rainfall'] = [rng.random() * 20 for _ in range(len(self._df_samples))]

        # Voronoi cells
        voronoi_cells = list(voronoi_diagram(shapely_MultiPoint(self._df_samples.geometry),
                                             self.BOUNDARY))
        cells = []
        for p in self._df_samples.geometry:
            g = [c.intersection(self.BOUNDARY) for c in voronoi_cells if p.within(c)][0]
            cells.append(g)
        self._df_voronoi = GeoDataFrame(dict(centre=self._df_samples.geometry,
                                             geometry=cells))

        # synthetic cell
        df_synthetic = self._df_samples.copy()
        df_synthetic.loc[len(df_synthetic.index)] = {'geometry': self.SYNTHETIC}
        voronoi_cells = list(voronoi_diagram(shapely_MultiPoint(df_synthetic.geometry),
                                             self.BOUNDARY))
        cells = []
        for p in df_synthetic.geometry:
            g = [c.intersection(self.BOUNDARY) for c in voronoi_cells if p.within(c)][0]
            cells.append(g)
        df_synthetic_voronoi = GeoDataFrame(dict(centre=df_synthetic.geometry,
                                                 geometry=cells))

        # find the cell with the synthetic point
        i = df_synthetic_voronoi[df_synthetic_voronoi['centre'] == self.SYNTHETIC].index[0]
        self._synthetic_cell = df_synthetic_voronoi.iloc[i].geometry

        # the overlap with an exemplar "real" cell
        i = self._df_voronoi[self._df_voronoi['centre'] == self.SAMPLES[0]].index[0]
        exemplar = self._df_voronoi.iloc[i].geometry
        self._overlap = self._synthetic_cell.intersection(exemplar)

        # all the overlap areas
        self._overlapAreas = []
        asynth = self._synthetic_cell.area
        s = 0
        for i in range(len(self._df_voronoi)):
            c = self._df_voronoi.iloc[i].geometry
            a = c.intersection(self._synthetic_cell).area
            self._overlapAreas.append(a)
            s += self.SAMPLE_VALUES[i] * (a / asynth)
        self._syntheticvalue = s

    @staticmethod
    def distance(p, q):
        s = 0.0
        for i in range(len(p)):
            s += numpy.abs(p[i] - q[i]) ** 2
        return numpy.sqrt(s)

    def construct(self):
        diagram = Group()

        # draw region
        self.next_section('boundary')
        region = Square(side_length=self.EDGE)
        diagram.add(region)
        self.play(Create(region))
        #self.wait(2)

        # introduce sample points
        self.next_section('add sample points')
        samples = []
        for _, pt in self._df_samples.iterrows():
            xy = list(pt.geometry.coords[0])
            p = self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE)
            samples.append(Dot(p))
        g = VGroup(*samples)
        diagram.add(g)
        self.play(ShowIncreasingSubsets(g, run_time=3))
        #self.wait(2)

        # distances
        self.next_section('show distances and formation of Voronoi cell')
        xy = list(self.SYNTHETIC.coords[0])
        p = self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE)
        measure = Dot(p, color=GREEN)
        self.play(Create(measure))
        lines = []
        lengths = []
        for pt in self._df_samples.geometry:
            # distance arrow
            xy = list(pt.coords[0])
            q = self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE)
            l = DoubleArrow(start=p, end=q, buff=0.05, tip_length=0.2, color=GREEN)
            lines.append(l)

            # length annotation
            length = NNNI.distance(p, q)
            t = DecimalNumber(length, num_decimal_places=2, font_size=32).move_to(q).shift(DOWN * 0.35 + RIGHT * 0.55)
            lines.append(t)
            lengths.append(length)
        distances = VGroup(*lines)
        self.play(Create(distances), run_time=3)
        self.wait(1)
        self.play(Uncreate(measure), Uncreate(distances))

        # add the first Voronoi cell
        cells = []
        self.next_section('first Voronoi cell')
        xys  = list(self._df_voronoi.iloc[0].geometry.exterior.coords)
        vs = map(lambda xy: self.ORIGIN+ RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE), xys)
        firstpoly = Polygon(*vs, stroke_color=GREEN_A).set_fill(color=GREEN_E, opacity=0.4)
        diagram.add(firstpoly)
        self.play(Create(firstpoly))
        #self.wait(3)

        # add the rest of the Voronoi cells
        self.next_section('remaining Voronoi cells')
        for c in self._df_voronoi.iloc[range(1, len(self._df_voronoi))].geometry:
            xys = list(c.exterior.coords)
            vs = map(lambda xy: self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE), xys)
            poly = Polygon(*vs, stroke_color=PURPLE_A).set_fill(color=PURPLE_E, opacity=0.4)
            cells.append(poly)
        g = VGroup(*cells)
        diagram.add(g)
        self.play(Create(g))
        #self.wait(2)

        # remove highlight on first cell
        self.next_section('remove highlight')
        cells.append(firstpoly)
        self.play(firstpoly.animate.set_color(PURPLE_A).set_fill(color=PURPLE_E, opacity=0.4))

        # add the grid lines
        self.next_section('interpolation grid')
        dx = self.EDGE / self.GRID
        lines = []
        for i in range(1, self.GRID):
            # vertical
            l = Line(start=self.ORIGIN + RIGHT * i * dx,
                     end=self.ORIGIN + RIGHT * i * dx + UP * self.EDGE,
                     color=GOLD)
            lines.append(l)

            # horizontal
            l = Line(start=self.ORIGIN + UP * i * dx,
                     end=self.ORIGIN + UP * i * dx + RIGHT * self.EDGE,
                     color=GOLD)
            lines.append(l)
        grid = VGroup(*lines)
        self.play(Create(grid), run_time=3)
        #self.wait(3)

        # add the synthetic cell
        self.next_section('synthetic cell')
        xy = list(self.SYNTHETIC.coords[0])
        p = self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE)
        synth = Dot(p, color=RED)
        diagram.add(synth)
        self.play(Create(synth))
        self.wait(1)
        self.play(FadeOut(grid))
        self.wait(1)
        xys = list(self._synthetic_cell.exterior.coords)
        vs = map(lambda xy: self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE), xys)
        synthpoly = Polygon(*vs, stroke_color=RED_A).set_fill(color=RED_E, opacity=0.6)
        diagram.add(synthpoly)
        self.play(Create(synthpoly))

        # add labels for values
        self.next_section('weight labels')
        labels = []
        for i in range(len(samples)):
            s = samples[i]
            l = MathTex(f'S_{i} = ' + '{v}'.format(v=self.SAMPLE_VALUES[i]), font_size=32).move_to(s.get_center()).shift(UP * 0.5)
            labels.append(l)
        l = MathTex('S_{xy}', font_size=32).move_to(synth.get_center()).shift(RIGHT * 0.5)
        labels.append(l)
        g = Group(*labels)
        diagram.add(g)
        self.play(FadeIn(g))

        # show weights equation
        self.next_section('show weights equation')
        self.play(diagram.animate.shift(LEFT * (self.EDGE / 2)))
        eq = MathTex("S_{xy} = \\sum_s S_s \\times \\frac{area(V_s \\cap V_{xy})}{area(V_{xy})}").next_to(diagram, RIGHT)
        self.play(Write(eq))

        # highlight the overlap
        self.next_section('overlap with underlying real cell')
        xys = list(self._overlap.exterior.coords)
        vs = map(lambda xy: self.ORIGIN + RIGHT * (xy[0] * self.EDGE) + UP * (xy[1] * self.EDGE), xys)
        overlap = Polygon(*vs, stroke_color=ORANGE).set_fill(color=ORANGE, opacity=0.6).shift(LEFT * (self.EDGE / 2))
        diagram.add(overlap)
        self.play(Create(overlap))

        # do calculation
        self.next_section('sample calculation')
        self.play(eq.animate.shift(UP))
        terms = len(self._overlapAreas)
        term = []
        for i in range(terms):
            term.append(MathTex("{s} \\times {a:.2f} {p}".format(s=self.SAMPLE_VALUES[i],
                                                                 a=self._overlapAreas[i],
                                                                 p=" \\, +" if i < terms - 1 else "")))
        head = MathTex("S_{xy}").next_to(eq, DOWN).align_to(eq, LEFT)
        equals = MathTex("=").next_to(head, RIGHT)
        self.play(Write(head), Write(equals))
        for i in range(terms):
            term[i].next_to(equals)
            if i > 0:
                term[i].shift(DOWN * (i / 1.5))
            self.play(Write(term[i]))
        result = MathTex(" = {v:.2f}".format(v=self._syntheticvalue)).next_to(head, RIGHT).shift(DOWN * (len(term) / 1.5))
        self.play(Write(result))


        self.wait(5)
