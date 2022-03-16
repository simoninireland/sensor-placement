# NetCDF file format for raw rainfall observations
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

from datetime import date, datetime
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from shapely.geometry import Point
import numpy


# Reference date 1800-1-1, matched to CEH-GEAR base
days_base = date(year=1800, month=1, day=1)


# Co-ordinate transforms
uk_grid_crs = CRS.from_string('EPSG:27700')            # UK national grid
latlon_crs = CRS.from_string('EPSG:4326')              # global lat/lon
proj = Transformer.from_crs(latlon_crs, uk_grid_crs)


def toNetCDF(fn,
             description,
             source,
             start, end, resolution,
             id_station, names,
             es_station, ns_station, lat_station, lon_station,
             times,
             rainfall):
    '''Create a NetCDF4 file holding raw observations..

    :param fn: filename (None generates an in-memory dataset)
    :param description: text description of the dataset
    :param source: text source, typically the root URL
    :param start: start date for observations
    :param end: end date for observations
    :param resolution: 'daily' or 'monthly'
    :param id_station: array of station identifiers (ints)
    :param names: array of station names
    :param es_station: array of station eastings
    :param ns_station: array of station northings
    :param lat_station: array of station latitudes
    :param lon_station: array of station longitudes
    :param times: array of sample times, in days since 1800-01-01
    :param rainfall: array of observations, keyed by time and station index
    :returns: the dataset'''

    # create the NetCDF file
    if fn is None:
        # in-memory dataset
        root = Dataset('in-memory.nc', 'w', diskless=True, persist=False)
    else:
        # dataset on disc
        root = Dataset(fn, 'w', format='NETCDF4')

    # standard metadata
    now = datetime.now().isoformat()
    root.description = description
    root.history = f'Retrieved {now}'
    root.source = source
    root.start = start.isoformat()
    root.end = end.isoformat()
    root.resolution = resolution

    # dimensions
    station_dim = root.createDimension('station', len(id_station))
    x_dim = root.createDimension('x', len(es_station))
    y_dim = root.createDimension('y', len(ns_station))
    lat_dim = root.createDimension('lat', len(lat_station))
    lon_dim = root.createDimension('long', len(lon_station))
    time_dim = root.createDimension('time', len(times))

    # variables
    station_var = root.createVariable('station', 'i4', (station_dim.name))
    station_var.units = 'Station number (int)'
    x_var = root.createVariable('x', 'f4', (station_dim.name))
    x_var.units = 'Easting (m east of base of UK national grid)'
    y_var = root.createVariable('y', 'f4', (station_dim.name))
    y_var.units = 'North (m north of base of UK national grid)'
    lat_var = root.createVariable('lat', 'f4', (station_dim.name))
    lat_var.units = 'Latitude (degrees)'
    lon_var = root.createVariable('long', 'f4', (station_dim.name))
    lon_var.units = 'Longitude (degree)'
    time_var = root.createVariable('time', 'f4', (time_dim.name))
    time_var.units = 'Days since 1800-1-1'
    rainfall_var = root.createVariable('rainfall_amount', 'f4', (time_dim.name, station_dim.name))
    rainfall_var.units = 'Monthly total rainfall recorded (kg/m^2)'

    # populate the dataset
    station_var[:] = id_station
    x_var[:] = es_station
    y_var[:] = ns_station
    lat_var[:] = lat_station
    lon_var[:] = lon_station
    time_var[:] = times
    rainfall_var[:, :] = rainfall

    # store the station names as fixed-width strings
    # (see https://unidata.github.io/netcdf4-python/#dealing-with-strings)
    maxlen = max(map(len, names))
    nchars_dim = root.createDimension('nchars', maxlen)
    names_var = root.createVariable('name', 'S1', (station_dim.name, nchars_dim.name))
    names_var._Encoding = 'ascii'
    names_var.units = 'Station name (string)'
    names = numpy.array(list(map(lambda s: f'{s:<{maxlen}}', names)), dtype=f'<S{maxlen}')
    names_var[:] = names

    # close the dataset only if not in-memory
    if fn is not None:
        root.close()

    return root


def stations(nc):
    '''Convert a NetCDF dataset to a ``geopandas`` ``GeoDataFrame``. This maps
    only the stations and their locations, not the samples. The order of the
    stations in the DataFrame matches the indices of samples in
    the NetCDF ``rainfall_amount`` variable.

    :param nc: the dataset or filename
    :returns: a dataframe'''
    if not isinstance(nc, Dataset):
        nc = Dataset(nc)

    # create a DataFrame for the dataset
    stations = GeoDataFrame(columns=['name', 'id',
                                     'x', 'y', 'longitude', 'latitude',
                                     'geometry'])

    # populate from the dataset
    for i in range(len(nc['station'])):
        stations.loc[i] = {'id': int(nc['station'][i]),
                           'name': nc['name'][i],
                           'x': nc['x'][i],
                           'y': nc['y'][i],
                           'longitude': nc['long'][i],
                           'latitude': nc['lat'][i],
                           'geometry': Point(nc['long'][i], nc['lat'][i])}

    # use the station id as the DataFrame index
    stations.set_index('id', inplace=True)

    return stations
