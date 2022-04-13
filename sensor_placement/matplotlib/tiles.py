# Open Street Map tile retrieval
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

# See https://stackoverflow.com/questions/22468108/how-to-read-image-from-stringio-into-pil-in-python

import math
import numpy
import requests_cache
from io import BytesIO
from PIL import Image


# Open Street Map endpoint
#osm = 'http://a.tile.openstreetmap.org/{0}/{1}/{2}.png'
osm = 'http://tile.openstreetmap.org/{0}/{1}/{2}.png'


# Persistent session
session = requests_cache.CachedSession()


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tile(lat_deg, lon_deg, delta_lat, delta_long, zoom):
    xmin, ymax = deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin = deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)

    bbox_ul = num2deg(xmin, ymin, zoom)
    bbox_ll = num2deg(xmin, ymax + 1, zoom)
    bbox_ur = num2deg(xmax + 1, ymin, zoom)
    bbox_lr = num2deg(xmax + 1, ymax +1, zoom)

    cluster = Image.new('RGB',((xmax - xmin + 1) * 256 - 1,(ymax - ymin + 1) * 256 - 1) )
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            url = osm.format(zoom, xtile, ytile)
            img = session.get(url).content
            tile = Image.open(BytesIO(img))
            cluster.paste(tile, box=((xtile - xmin) * 256, (ytile - ymin) * 255))

    return numpy.flip(cluster, axis=0), (bbox_ll[1], bbox_ll[0], bbox_ur[1], bbox_ur[0])
