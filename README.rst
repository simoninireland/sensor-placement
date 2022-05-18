sensor-placement: An experiment in sensor placement and error
=============================================================

.. image:: https://www.gnu.org/graphics/gplv3-88x31.png
    :target: https://www.gnu.org/licenses/gpl-3.0.en.html

Overview
--------

``sensor-placement`` is an experiment in studying the impact that
sensor placement and error have on the interpolation of data. It makes
use of the UK's extensive rainfall gauge network for raw data, and
the well-respected CEH-GEAR interpolated dataset as a baseline reference.


Installation
------------

The master distribution of ``sensor-placement`` is hosted on GitHub. To obtain a
copy, first clone the repo and construct the virtual environment:

::

    git clone git@github.com:simoninireland/sensor-placement.git
    cd sensor-placement
    make env

You then need to obtain the datasets. These are *not* freely
available, but are liberally licensed for research use, especially for
UK academics.

You first need to visit the two data archives and register for
accounts. These are:

- The `Centre for Ecology and Hydrology <https://catalogue.ceh.ac.uk>`_
- The `Centre for Environmental Data Analysis <https://data.ceda.ac.uk>`_

Once you have created usernames and passwords for these sites, copy
the file ``./credentials.example`` to ``./credentials``, and edit it
to insert your values in the appropriate places. Note that this
is a shell script, not Python code. Also be careful to put the
usernames and passwords within single quotes to avoid problems with
special characters.

Don't commit ``./credentials`` to version control! -- it's a breach of
the licensing terms (as well as bad practice) to share your
passwords with others.

You can now download the datasets by running:

::

   make datasets

This downloads about 200MB of data from various sources, so depending
on your internet connection you may want to get at least a coffee, and
possibly dinner, while you wait. You need only perform this step once,
of course.

You can then access the notebooks by running:

::

   make live

which will open a notebook in a web browser. You'll probably want to
start by looking at ``./datacheck.ipynb`` to make sure the data has
come down properly.

If you want to start again for any reason, you can clean-up the build
by running:

::

   make reallyclean

which will delete the virtual environment and the downloaded datasets.


Downloading online datasets
---------------------------

The experiment has code to download datasets from three different
online sources:

- The `CEDA MIDAS collection
  <https://help.ceda.ac.uk/article/4442-ceda-opendap-scripted-interactions>`_,
  consisting of about 150 rain gauges across the UK
- The `SEPA tipping buckets network
  <https://www2.sepa.org.uk/rainfall/DataDownload>`_ of about 280
  stations around Scotland
- The `UK EPA rainfall network
  <https://environment.data.gov.uk/flood-monitoring/doc/rainfall>`_ of
  about 950 rain gauges in England and Wales

These sources can be accessed programmatically using `adaptors
<https://github.com/simoninireland/sensor-placement/tree/main/sensor_placement/data>`_
that generate a common-format NetCDF4 file of observations. There are
also `scripts
<https://github.com/simoninireland/sensor-placement/tree/main/utils>`_
for command-line access, which are used to download sample datasets
during installation.


Author and license
------------------

Copyright (c) 2022, Simon Dobson <simon.dobson@st-andrews.ac.uk>

Licensed under the `GNU General Public Licence v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

This code uses Environment Agency rainfall data from the real-time data API (Beta).
