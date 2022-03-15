# Adaptor for CEDA MIDAS rainfall data
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

# See https://help.ceda.ac.uk/article/4442-ceda-opendap-scripted-interactions

import requests
from datetime import date, datetime, timedelta
from dateparser import parse
from os.path import exists
from io import StringIO
import csv
import numpy
from sensor_placement.data import toNetCDF, days_base, proj


# Root URL, filename pattern, and certificate for the API
root_url = 'https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202107'
ceda_data_filename_pattern = '{base}/{county}/{id}_{name}/qc-version-1/midas-open_uk-daily-rain-obs_dv-202107_{county}_{id}_{name}_qcv-1_{year}.csv'
certificate = './ceda.pem'


def ceda_midas(year, fn = None, cert = None):
    '''Retrieve monthly observations from the CEDA MIDAS dataset for the given year,
    optionally saving the data in a NetCDF4 file.

    :param year: the year
    :param fn: (optional) the file to create (defaults to in-memory)
    :param cert: (optional) path to  CEDA certificate file
    :returns: the dataset'''

    # make sure we have the necessary certificate
    if cert is None:
        cert = certificate
    if not exists(cert):
        raise Exception(f'No certificate file {cert}: do you need to create one?')

    # grab the current list of stations
    url = f'{root_url}/midas-open_uk-daily-rain-obs_dv-202107_station-metadata.csv'
    req = requests.get(url, cert=cert)
    if req.status_code != 200:
        raise Exception('Can\'t get stations: {e}'.format(e=req.status_code))

    # parse-out stations and their positions
    latlons = dict()
    id_station = []
    es_station = []
    ns_station = []
    lat_station = []
    lon_station = []
    with StringIO(req.text) as fh:
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
                label = row[2]

                # make sure the station has data in the year we're looking for
                if not (year >= int(row[7]) and year <= int(row[8])):
                    print(f'No records for {year} at {label}')
                    continue

                # map station id and filename
                dfn = ceda_data_filename_pattern.format(base=root_url,
                                                        id='{id:05d}'.format(id=id),
                                                        county=row[3],
                                                        name=row[2],
                                                        year=year)

                # get UK grid locations
                lat, lon = row[4], row[5]
                east, north = proj.transform(lat, lon)
                east = 1000 * int(east / 1000)           # round to the nearest kilometre
                north = 1000 * int(north / 1000)

                # record name and postion
                latlons[id] = (label, lat, lon, east, north, dfn)

                # add to the variable arrays
                id_station.append(id)
                lat_station.append(lat)
                lon_station.append(lon)
                es_station.append(east)
                ns_station.append(north)

    # construct first day on each month (corresponds to CEH-GEAR monthlies)
    times = []
    for m in range(12):
        firstday = datetime(year=year, month=m + 1, day=1).date() # first day of the month
        day = (firstday - days_base).days                         # days since reference date
        times.append(day)

    # construct arrays for the time series
    rainfall = numpy.zeros((12, len(id_station)))

    # load all the time series
    for station in range(len(id_station)):
        # pull the data
        id = id_station[station]
        label = latlons[id][0]
        url = latlons[id][5]
        req = requests.get(url, cert=cert)
        if req.status_code != 200:
            print('Can\'t get data for {l} from {url}: {e}'.format(url=url,
                                                                   l=label,
                                                                   e=req.status_code))
            continue

        # read the data
        print(label)
        with StringIO(req.text) as fh:
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
                    t = parse(row[0])
                    rainfall[t.month - 1, station] += float(row[9])

    # create the file
    return toNetCDF(fn,
                    f'CEDA MIDAS tipping bucket monthly data ({year})',
                    root_url,
                    date(year=year, month=1, day=1), date(year=year, month=12, day=31), 'monthly',
                    id_station,
                    list(map(lambda i: latlons[i][0], id_station)),
                    es_station, ns_station, lat_station, lon_station,
                    times,
                    rainfall)
