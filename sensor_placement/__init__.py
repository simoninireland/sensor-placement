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

# old functional interface
#from .nnni import nnn_voronoi, nnn_geometry, nnn_masked_grid, nnn_tensor, apply_tensor, natural_nearest_neighbour

# while debugging, also export the internal implementations
#from .nnni import nnn_tensor_seq, nnn_tensor_par, nnn_tensor_par_worker

# Library-specific logger name
Logger = 'sensor_placement'

# New class interface
from .tensor import InterpolationTensor
from .nnni_tensor import NNNI
