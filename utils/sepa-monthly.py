# Import a year of data from all the SEPA rain gauges, exporting as
# a NetCDF4 file in a format similar to that of CEH-GEAR, but intended
# to allow extraction of timeseries for individuaL stations.
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
# along with trhis software. If not, see <http://www.gnu.org/licenses/gpl.html>.

from sys import argv
from datetime import datetime, timedelta
import json
import requests
import numpy
from netCDF4 import Dataset
from pyproj import CRS, Transformer


# extract the year from the command line
if len(argv) not in [2, 3]:
    print('usage: sepr-monthly.py <year> [<filename>]')
    exit(1)
year = int(argv[1])
if len(argv) > 2:
    sepa_monthly_filename = argv[2]
else:
    sepa_monthly_filename = f'datasets/sepa_monthly_{year}.nc'

# data
sepastations_filename = 'datasets/sepa-stations.json'
sepa_monthly_url = 'https://apps.sepa.org.uk/rainfall/api/Month'
monthnames =  ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# the reference date from which CEH-GEAR days are counted
days_base = datetime(year=1800, month=1, day=1)

# map all the station IDs to their locations
with open(sepastations_filename, 'r') as fh:
    sepastations = json.load(fh)
latlons = {int(s['station_no']): (s['station_name'],
                                  s['station_latitude'],
                                  s['station_longitude']) for s in sepastations}

# map the co-ordinates of the stations to a square on the national grid
uk_grid_crs = CRS.from_string('EPSG:27700')            # UK national grid
latlon_crs = CRS.from_string('WGS 84')                 # global lat/lon
proj = Transformer.from_crs(latlon_crs, uk_grid_crs)
uk_grids = {}
es_station = []
ns_station = []
for id in latlons.keys():
    east, north = proj.transform(latlons[id][1], latlons[id][2])
    east = 1000 * int(east / 1000)                     # round to the nearest kilometre
    north = 1000 * int(north / 1000)
    es_station.append(east)
    ns_station.append(north)
    uk_grids[id] = (east, north)

# construct first day on each month (corresponds to CEH-GEAR monthlies)
times = []
for m in range(len(monthnames)):
    firstday = datetime(year=year, month=m + 1, day=1)       # first day of the month
    day = (firstday - days_base).days                        # days since reference date
    times.append(day)

# construct arrays for the time series
id_station = list(latlons.keys())
es_station.sort()
ns_station.sort()
rainfall = numpy.zeros((12, len(id_station)))

# load all the time series
for i in range(len(id_station)):
    # pull the data
    id = id_station[i]
    name = latlons[id][0]
    print(f'Adding {name}')
    url = f'{sepa_monthly_url}/{id}?all=true'
    req = requests.get(url)
    if req.status_code != 200:
       raise Exception('Error downloading dataset {url}: {e}'.format(url=url,
                                                                     e=req.status_code))
    ts = req.json()

    # turn array of dicts into dict keyed by timestamp
    monthlies = {}
    for tv in ts:
        monthlies[tv['Timestamp']] = tv['Value']

    # extract the year's values
    for m in range(len(monthnames)):
        mn = monthnames[m]
        timestamp = f'{mn} {year}'
        if timestamp in monthlies.keys():
            rainfall[m, i] = monthlies[timestamp]
        else:
            print(f'No entry for {timestamp} at station {id}')

# create the NetCDF file and its associated groups and variables
now = datetime.now()
root = Dataset(sepa_monthly_filename, 'w', format='NETCDF4')
root.description = f'SEPA tipping bucket monthly data ({year})'
root.history = f'Created {now}'
station_dim = root.createDimension('station', len(id_station))
x_dim = root.createDimension('x', len(es_station))
y_dim = root.createDimension('y', len(ns_station))
time_dim = root.createDimension('time', len(times))
station_var = root.createVariable('station', 'i4', (station_dim.name))
station_var.units = 'Station number (int)'
x_var = root.createVariable('x', 'f4', (station_dim.name))
x_var.units = 'Easting (m east of base of UK national grid)'
y_var = root.createVariable('y', 'f4', (station_dim.name))
y_var.units = 'North (m north of base of UK national grid)'
time_var = root.createVariable('time', 'f4', (time_dim.name))
time_var.units = 'Days since 1800-1-1'
rainfall_var = root.createVariable('rainfall_amount', 'f4', (time_dim.name, station_dim.name))
rainfall_var.units = 'Monthly total rainfall recorded (kg/m^2)'

# populate the dataset
station_var[:] = id_station
x_var[:] = es_station
y_var[:] = ns_station
time_var[:] = times
rainfall_var[:, :] = rainfall

# store the station names as fixed-width strings
# (see https://unidata.github.io/netcdf4-python/#dealing-with-strings)
names = list(map(lambda ll: ll[0], latlons.values()))
maxlen = max(map(len, names))
nchars_dim = root.createDimension('nchars', maxlen)
names_var = root.createVariable('name', 'S1', (station_dim.name, nchars_dim.name))
names_var._Encoding = 'ascii'
names_var.units = 'Station name (string)'
names = numpy.array(list(map(lambda s: f'{s:<{maxlen}}', names)), dtype=f'<S{maxlen}')
names_var[:] = names

# close the file
root.close()
