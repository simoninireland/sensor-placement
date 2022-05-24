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


class Tensor(ThreeDScene):

    def construct(self):
        # create an NNNI object to re-use the calculations
        nnni = NNNI.NNNI()

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

        # add points for optimisation
        self.next_section('points for optimisation')
        self.play(diagram.animate.shift(LEFT * 3),
                  *[vl.animate.shift(LEFT * 3) for vl in vlines],
                  *[hl.animate.shift(LEFT * 3) for hl in hlines])
                  # grid.animate.shift(LEFT * 3))
        text = BulletedList('Weights independent of samples',
                            'Points are independent',
                            'Most weights are zero',
                            font_size=42).shift(UP * 1 + RIGHT * 3.5)
        self.play(Write(text, run_time=5))

        # seperate the grid
        self.next_section('separate interpolation and grid')
        self.play(*[FadeOut(t) for t in text],
                  FadeOut(grid))
        hlines, vlines = [], []
        for i in range(nnni.GRID + 1):
            # vertical
            l = Line(start=RIGHT * i * dx,
                     end=RIGHT * i * dx + UP * nnni.EDGE,
                     color=GOLD).shift(RIGHT * 0.5 + DOWN * (nnni.EDGE / 2))
            vlines.append(l)

            # horizontal
            l = Line(start=UP * i * dx,
                     end=UP * i * dx + RIGHT * nnni.EDGE,
                     color=GOLD).shift(RIGHT * 0.5 + DOWN * (nnni.EDGE / 2))
            hlines.append(l)
        grid = VGroup(*vlines, *hlines)
        self.play(Create(grid))

        # interpolated values
        self.next_section('interpolated values')
        xy = list(nnni.SYNTHETIC.coords[0])
        p =  region.get_corner(DOWN + LEFT) + RIGHT * (xy[0] * nnni.EDGE) + UP * (xy[1] * nnni.EDGE)
        synth = Dot(p, color=RED)
        synthlabel = MathTex("S_{xy}").move_to(p).shift(UP * 0.3, RIGHT * 0.3)
        self.play(FadeIn(synth), FadeIn(synthlabel))
        p = grid.get_corner(DOWN + LEFT) + RIGHT * (xy[0] * nnni.EDGE) + UP * (xy[1] * nnni.EDGE)
        interlabel = MathTex("S_{xy}", font_size=24).move_to(p)
        self.play(Indicate(interlabel, color=RED, scale_factor=3))

        # slide to top-left
        self.next_section('slide to top-left')
        self.play(synth.animate.move_to(region.get_corner(UP + LEFT) + RIGHT * (dx / 2) + DOWN * (dx / 2)),
                  synthlabel.animate.move_to(region.get_corner(UP + LEFT) + RIGHT * (dx / 2) + DOWN * (dx / 2)).shift(UP * 0.3, RIGHT * 0.3),
                  interlabel.animate.move_to(grid.get_corner(UP + LEFT) + RIGHT * (dx / 2) + DOWN * (dx / 2)))

        # fade
        self.next_section('fade out diagram')
        self.play(FadeOut(diagram),
                  FadeOut(synth),
                  FadeOut(synthlabel))
        self.play(grid.animate.shift(LEFT * 3 + DOWN),
                  interlabel.animate.shift(LEFT * 3 + DOWN))

        # explode to tensor
        self.next_section('explode tensor')
        self.move_camera(phi=-numpy.pi / 4)
        self.play(FadeOut(interlabel))
        boxes = []
        labels = []
        for i in range(len(nnni._df_samples)):
            box = Cube(side_length=dx, fill_color=YELLOW_C).move_to(grid.get_corner(UP + LEFT) + RIGHT * (dx / 2) + IN * i * dx * 1.2)
            boxes.append(box)
            l = MathTex('W_{xy' + f'{i}' + '}').next_to(box, LEFT)
            labels.append(l)
            self.play(FadeIn(box), FadeIn(l))

        # explain
        self.next_section('explaining the structure')
        self.move_camera(phi=0.0)
        tensor = Tex("Order-3",  "tensor", arg_separator=' ').shift(UP * 2.5 + RIGHT * 0.5)
        self.play(Write(tensor))
        frame0 = SurroundingRectangle(tensor[0], buff=0.1)
        frame1 = SurroundingRectangle(tensor[1], buff=0.1)
        self.play(Create(frame0))
        self.wait(3)
        self.play(ReplacementTransform(frame0, frame1))
        self.wait(3)
        self.play(FadeOut(frame1))

        # apply
        self.next_section('applying the tensor')
        self.play(FadeOut(tensor),
                  *[b.animate.shift(LEFT * 2 + UP) for b in boxes],
                  *[l.animate.shift(LEFT * 2 + UP) for l in labels],
                  grid.animate.shift(LEFT * 2 + UP))
        samples = IntegerMatrix(map(lambda v: [v], nnni.SAMPLE_VALUES)).shift(RIGHT * 2)
        self.play(Create(samples))
        self.play(FadeOut(grid),
                  *[FadeOut(b)for b in boxes],
                  *[FadeOut(l) for l in labels])
        dotproduct = MathTex("\\cdot")
        weights = MobjectMatrix([labels]).next_to(dotproduct, LEFT)
        self.play(Create(weights),
                  Create(dotproduct),
                  samples.animate.next_to(dotproduct, RIGHT))
        result = MathTex(" = {v:.2f}".format(v=nnni._syntheticvalue)).next_to(samples, RIGHT)
        self.play(Write(result))

        # explain
        self.next_section('explain vector and covector')
        vector = Text('sample vector').move_to(samples.get_center() + DOWN * 2.5)
        vectorbrace = Brace(vector, direction=UP)
        self.play(FadeIn(vector), FadeIn(vectorbrace))
        self.wait(3)
        covector = Text('weight co-vector', color=YELLOW_C).move_to(weights.get_center() + UP * 2)
        covectorbrace = Brace(covector, direction=DOWN, color=YELLOW_C)
        self.play(FadeIn(covector), FadeIn(covectorbrace))


        self.wait(5)
