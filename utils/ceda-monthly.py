# Import a year of data from all the UK Met Office rain gauges, exporting as
# a NetCDF4 file in a format similar to that of CEH-GEAR, but intended
# to allow extraction of timeseries across the UK rather than per-station.
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

from sys import argv
from dateparser import parse
from sensor_placement.data import ceda_midas


# Extract year from the command line
if len(argv) not in [2, 3]:
    print('usage: ceda-monthly.py <year> [<filename>]')
    exit(1)
year = int(argv[1])
if len(argv) > 2:
    filename = argv[2]
else:
    filename = f'datasets/ceda_midas_monthly_{year}.nc'


# Pull down the data
ceda_midas(year, filename)
