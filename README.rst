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
the file ./credentials.example to ./credentials, and edit it to insert
your credentials in the appropriate places. Note that this is a shell
script, not Python code. Also be careful to put the usernames and
passwords within single quotes to avoid problems with special characters.

Don't commit ./credentials to version control! -- it's a breach of the
licensing terms to share your credentials with others.

You can now download the datasets by running:

::

   make datasets

This will take a *long* time: how long depends on the speed of your
broadband connection, but the download is over 1.2GB in size, so it's
best to get at least a coffee, and probably dinner, while you wait. You
need only perform this step once, of course.

You can then access the notebooks by running:

::

   make live

which will open a notebook in a web browser.


Author and license
------------------

Copyright (c) 2022, Simon Dobson <simon.dobson@st-andrews.ac.uk>

Licensed under the `GNU General Public Licence v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.
