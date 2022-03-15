# Import daily rainfall totals from the UK EPA tipping buckets
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

# This code use Environment Agency rainfall data from the real-time data API (Beta)

from sys import argv
from dateparser import parse
from sensor_placement.data import uk_epa


# Extract date range from the command line
if len(argv) not in [3, 4]:
    print('usage: epa-daily.py <start> <finish> [<filename>]')
    exit(1)
start = parse(argv[1], settings={'TIMEZONE': 'UTC'})
end = parse(argv[2], settings={'TIMEZONE': 'UTC'})
if len(argv) > 3:
    filename = argv[3]
else:
    startDate = start.date().isoformat()
    endDate = end.date().isoformat()
    filename = f'datasets/epa_daily_{startDate}_{endDate}.nc'


# Pull down the data
uk_epa(start, end, filename)
