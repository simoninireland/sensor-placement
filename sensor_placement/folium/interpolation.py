# Data interpolation layer for a Foilum map
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

import folium
import folium.plugins
from sensor_placement.folium import namedLayer


def heatmap(tensor, grid, name=None, min_opacity=0.01, radius=40, blur=40, gradient=None):
    '''Return interpolated rainfall data as a heatmap.'''

    # wrangle the data into heatmap form
    ys, xs = tensor.ys(), tensor.xs()
    rainpoints = []
    mask = grid.mask
    shape = grid.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not mask[i, j]:
                rainpoints.append([ys[i], xs[j], grid[i, j]])

    # create the heatmap
    heatmap = folium.plugins.HeatMap(data=rainpoints,
                                     min_opacity=min_opacity, radius=blur, blur=blur,
                                     gradient=gradient)

    return namedLayer(heatmap, name)
