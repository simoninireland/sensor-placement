# Adaptor for SEPA rainfall data
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

# See https://www2.sepa.org.uk/rainfall/DataDownload

import requests
from datetime import date, datetime, timedelta
from dateparser import parse
import numpy
from sensor_placement.data import toNetCDF, days_base, proj


# Root URL for the API and the monthly endpoint
root_url = 'http://apps.sepa.org.uk/rainfall'
monthly_url = f'{root_url}/api/Month'


# Month names
monthnames =  ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def sepa(year, fn = None):
    '''Retrieve SEPA monthly observations for the given year,
    optionally saving the data in a NetCDF4 file.

    :param year: any date in the desired year
    :param fn: (optional) the file to create (defaults to in-memory)
    :returns: the dataset'''
    session = requests.Session()

    # grab the current list of stations
    url = f'{root_url}/api/Stations?json=true'
    req = session.get(url)
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
    for s in ss:
        id = s['station_no']
        label = s['station_name']

        # get UK grid locations
        lat, lon = s['station_latitude'], s['station_longitude']
        east, north = proj.transform(lat, lon)
        east = 1000 * int(east / 1000)           # round to the nearest kilometre
        north = 1000 * int(north / 1000)

        # record name and postion
        latlons[id] = (label, lat, lon, east, north)

        # add to the variable arrays
        id_station.append(id)
        lat_station.append(lat)
        lon_station.append(lon)
        es_station.append(east)
        ns_station.append(north)

    # create array for the measurements
    rainfall = numpy.zeros((12, len(id_station)))
    times = []
    for m in range(len(monthnames)):
        firstday = datetime(year=year, month=m + 1, day=1).date() # first day of the month
        day = (firstday - days_base).days                         # days since reference date
        times.append(day)

    # retrieve all measures in the year
    for station in range(len(id_station)):
        # retrieve the measurements
        id = id_station[station]
        label = latlons[id][0]
        url = f'{monthly_url}/{id}?all=true'
        req = session.get(url)
        if req.status_code != 200:
            raise Exception('Can\'t get data for {l} from {url}: {e}'.format(url=url,
                                                                             l=label,
                                                                             e=req.status_code))
        ts = req.json()

        # turn array of dicts into dict keyed by timestamp
        monthlies = {}
        for tv in ts:
            monthlies[tv['Timestamp']] = tv['Value']

        # extract the year's values
        print('.', end='', flush=True)
        for m in range(len(monthnames)):
            mn = monthnames[m]
            timestamp = f'{mn} {year}'
            if timestamp in monthlies.keys():
                rainfall[m, station] = monthlies[timestamp]
            else:
                print(f'No entry for {timestamp} at station {id}')
                continue
    print('', flush=True)

    # create the file
    return toNetCDF(fn,
                    f'SEPA tipping bucket monthly data ({year})',
                    root_url,
                    date(year=year, month=1, day=1), date(year=year, month=12, day=31), 'monthly',
                    id_station,
                    list(map(lambda i: latlons[i][0], id_station)),
                    es_station, ns_station, lat_station, lon_station,
                    times,
                    rainfall)
