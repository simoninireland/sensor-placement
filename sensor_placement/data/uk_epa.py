# Adaptor for UK EPA rainfall API
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
# See https://environment.data.gov.uk/flood-monitoring/doc/rainfall

import requests
from datetime import date, datetime, timedelta
from dateparser import parse
import numpy
from sensor_placement.data import toNetCDF, days_base, proj


# Root URL for the API
root_url = 'http://environment.data.gov.uk/flood-monitoring'

def uk_epa(start, end, fn = None):
    '''Retrieve EPA daily observations betweeen the two date ranges,
    optionally saving the data in a NetCDF4 file.

    :param start: the start date
    :param end: the end date
    :param fn: (optional) the file to create (defaults to in-memory)
    :returns: the dataset'''

    # grab the current list of stations
    url = f'{root_url}/id/stations?parameter=rainfall'
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception('Can\'t get stations: {e}'.format(e=req.status_code))
    ss = req.json()

    # parse-out stations and their positions
    latlons = dict()
    id_station = []
    es_station = []
    ns_station = []
    lat_station = []
    lon_station = []
    for s in ss['items']:
        id = s['stationReference']
        label = s['label']

        # extract the rainfall measure
        measure = None
        for m in s['measures']:
            if m['parameter'] == 'rainfall':
                measure = m['@id']
                break
        if measure is None:
            print('No rainfall measurements at {label}')
            continue

        # get UK grid locations
        if 'lat' not in s.keys() or 'lat' not in s.keys():
            print(f'No location information for {label}')
            continue
        lat, lon = s['lat'], s['long']
        east, north = proj.transform(lat, lon)
        east = 1000 * int(east / 1000)           # round to the nearest kilometre
        north = 1000 * int(north / 1000)

        # record name, postion, and measure key
        latlons[id] = (label, lat, lon, east, north, measure)

        # add to the variable arrays
        id_station.append(id)
        lat_station.append(lat)
        lon_station.append(lon)
        es_station.append(east)
        ns_station.append(north)

    # create array for the measurements
    ndays = (end - start).days + 1
    rainfall = numpy.zeros((ndays, len(id_station)))
    times = []
    for i in range(ndays):
        d = start + timedelta(days=i)
        day = (d.date() - days_base).days
        times.append(day)

    # retrieve all measures in date range
    startDate = start.strftime('%Y-%m-%d')
    endDate = end.strftime('%Y-%m-%d')
    sd = start.date()
    ed = end.date()
    for station in range(len(id_station)):
        # retrieve the measurements
        id = id_station[station]
        label = latlons[id][0]
        measure = latlons[id][5]
        url = f'{measure}/readings?startdate={startDate}&enddate={endDate}'
        req = requests.get(url)
        if req.status_code != 200:
            raise Exception('Can\'t get measure {m} at {l}: {e}'.format(m=measure,
                                                                        l=label,
                                                                        e=req.status_code))
        rs = req.json()

        # add to array against the appropriate day
        print('.', end='', flush=True)
        for m in rs['items']:
            # sometimes there's malformed data
            if 'dateTime' not in m.keys():
                print('No datestamp on reading (ignored)')
                continue
            elif 'value' not in m.keys():
                print('No value for reading (ignored)')
                continue

            # extract the date
            d = parse(m['dateTime']).date()
            if d is None:
                print('Can\'t parse date {d} at {l}'.format(d=m['dateTime'], l=label))
                continue
            elif d < sd:
                print('Date {d} at {l} comes before start'.format(d=m['dateTime'], l=label))
                continue
            elif d > ed:
                print('Date {d} at {l} comes after end'.format(d=m['dateTime'], l=label))
                continue

            # add to day total
            day = (d - sd).days
            rainfall[day, station] += float(m['value'])
    print(';', flush=True)

    # create the file
    return toNetCDF(fn,
                    f'EPA tipping bucket daily data ({startDate} -- {endDate})',
                    root_url,
                    start, end, 'daily',
                    list(range(len(id_station))),                     # force ids to ints
                    list(map(lambda i: latlons[i][0], id_station)),
                    es_station, ns_station, lat_station, lon_station,
                    times,
                    rainfall)
