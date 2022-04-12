# Sample points and other features
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

import folium


def samplePoints(points, rainfall=None, name=None):
    '''Return a layer of sample point markers, optionally with raw data attached.'''

    markers = folium.FeatureGroup(name=name)

    # create the markers
    for i in range(len(points)):
        s = points.iloc[i]
        id = points.index[i]
        name, lon, lat = s['name'], s['longitude'], s['latitude']
        tt = f'{id}: {name} ({lat:.2f}N, {lon:.2f}W)'
        if rainfall is not None:
            tt += ' {r:.2f}mm'.format(r=rainfall[i])
        _ = folium.Marker(location=(lat, lon),
                          tooltip=tt,
                          icon=folium.Icon(color='red', icon='cloud')).add_to(markers)

    return markers
