# Import a year of data from all the UK Met Office rain gauges, exporting as
# a NetCDF4 file in a format similar to that of CEH-GEAR, but intended
# to allow extraction of timeseries for individuaL stations.
#
# Copyright (C) 2022 Simon Dobson
#
# This file is part of sensor-placement, an experiment exploriung sensor error.
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
from datetime import datetime, timedelta
import dateparser
import csv
import json
import requests
import numpy
from netCDF4 import Dataset
from pyproj import CRS, Transformer


# extract the year from the command line
if len(argv) not in [2, 3]:
    print('usage: ceda-monthly.py <year> [<filename>]')
    exit(1)
year = int(argv[1])
if len(argv) > 2:
    ceda_monthly_filename = argv[2]
else:
    ceda_monthly_filename = f'datasets/ceda_monthly_{year}.nc'

# data
cedastations_filename = 'datasets/ceda-stations.csv'
ceda_daily_dir = 'datasets/ceda/dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202107/'

# the reference date from which CEH-GEAR days are counted
days_base = datetime(year=1800, month=1, day=1)

# extract all the station names and locations from the CSV file
cedastations = dict()
latlons = dict()
with open(cedastations_filename, 'r') as fh:
    r = csv.reader(fh, delimiter=',')
    reading_stations = False
    skip_next_line = False
    for row in r:
        if row[0] == 'data':
            # seen the line that starts the stations
            reading_stations = True
            skip_next_line = True
        elif row[0] == 'end data':
            # end of data
            break
        elif skip_next_line:
            skip_next_line = False
        elif reading_stations:
            id = int(row[0])
            name = row[1]

            # make sure the station has data in the year we're looking for
            if not (year >= int(row[7]) and year <= int(row[8])):
                print(f'No records for {year} at {name}')
                continue

            # map station id and filename
            fn = '{base}/{county}/{id}-{name}/qc-version-1/midas-open_uk-daily-rain-obs_dv-202107_{county}_{id}_{name}_qcv-1_{year}.csv'.format(base=ceda_daily_dir,
                                                                                                                                                id='{id:05d}'.format(id=id),
                                                                                                                                                county=row[3],
                                                                                                                                                name=row[2],
                                                                                                                                                year=year)
            cedastations[id] = fn

            # map station id to location
            latlons[id] = (row[4], row[5])

# map the co-ordinates of the stations to a square on the national grid
uk_grid_crs = CRS.from_string('EPSG:27700')            # UK national grid
latlon_crs = CRS.from_string('WGS 84')                 # global lat/lon
proj = Transformer.from_crs(latlon_crs, uk_grid_crs)
uk_grids = {}
es_station = []
ns_station = []
for id in latlons.keys():
    east, north = proj.transform(*latlons[id])
    east = 1000 * int(east / 1000)                     # round to the nearest kilometre
    north = 1000 * int(north / 1000)
    es_station.append(east)
    ns_station.append(north)
    uk_grids[id] = (east, north)

# construct last day on each month
# see https://stackoverflow.com/questions/42950/how-to-get-the-last-day-of-the-month
times = []
for m in range(12):
    firstday = datetime(year=year, month=m + 1, day=1)       # first day of the month
    nextmonth = firstday.replace(day=28) + timedelta(days=4)
    lastday = nextmonth - timedelta(days=nextmonth.day)      # last day of month
    day = (lastday - days_base).days                         # days since reference date
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

    # read the data
    with open(cedastations[id], 'r') as fh:
        r = csv.reader(fh, delimiter=',')

        reading_measurements = False
        skip_next_line = False
        for row in r:
            if row[0] == 'data':
                # seen the line that starts the stations
                reading_measurements = True
                skip_next_line = True
            elif row[0] == 'end data':
                # end of data
                break
            elif skip_next_line:
                skip_next_line = False
            elif reading_measurements:
                t = dateparser.parse(row[0])
                rainfall[t.month - 1, i] += float(row[9])

# create the NetCDF file and its associated groups and variables
now = datetime.now()
root = Dataset(ceda_monthly_filename, 'w', format='NETCDF4')
root.description = f'UK rainfall data from CEDA monthly data ({year})'
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

# close the file
root.close()
