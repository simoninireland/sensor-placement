# Makefile for sensor placement experiment
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

# ----- Sources -----

# Text
INDEX =
TEXT =

# Notebooks
NOTEBOOKS =  \
	sensor-placement.ipynb \
	datacheck.ipynb \
	simple.ipynb \
	diagrams.ipynb

# Source code
SOURCE = \
	sensor_placement/__init__.py \
	sensor_placement/nnni.py \
	sensor_placement/data/__init__.py \
	sensor_placement/data/raw.py \
	sensor_placement/data/uk_epa.py \
	sensor_placement/data/sepa.py \
	sensor_placement/data/ceda.py \
	sensor_placement/folium/__init__.py \
	sensor_placement/folium/style.py \
	sensor_placement/folium/interpolation.py \
	sensor_placement/folium/samples.py \
	sensor_placement/matplotlib/__init__.py \
	sensor_placement/matplotlib/tiles.py \
	sensor_placement/matplotlib/samples.py \
	sensor_placement/matplotlib/interpolation.py \
	sensor_placement/matplotlib/flowfield.py
SOURCES_TESTS = \
	test/__init__.py \
	test/test_io.py \
	test/test_nnni.py
TESTSUITE = test

# Utilities
UTILS = utils/
SCRIPTS = \
	$(UTILS)/sepa-monthly.py \
	$(UTILS)/ceda-monthly.py \
	$(UTILS)/epa-daily.py


# ----- Data -----

# Data files and directories
DATASETS_DIR = datasets
DIAGRAMS_DIR = diagrams
CREDENTIALS = ./credentials

# CEH-GEAR monthly
CEH_URL = https://catalogue.ceh.ac.uk
CEH_LOGIN_URL = $(CEH_URL)/sso/login
CEH_BASE = /datastore/eidchub/dbf13dd5-90cd-457a-a986-f2f9dd97e93c
CEH_BASE_MONTHLIES = $(CEH_BASE)/GB/monthly
CEH_EXAMPLE_MONTHLY = CEH_GEAR_monthly_GB_2017.nc

# CEDA rain gauges
# see https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202107
CEDA_CA = online_ca_client
CEDA_CA_ROOTS = $(ROOT)/trustroots
CEDA_CERTIFICATE = $(ROOT)/ceda.pem
CEDA_MONTHLY_EXAMPLE_YEAR = 2017
CEDA_MONTHLY_EXAMPLE = ceda_midas_monthly_$(CEDA_MONTHLY_EXAMPLE_YEAR).nc

# SEPA rain gauges
# see https://www2.sepa.org.uk/rainfall/DataDownload
SEPA_MONTHLY_EXAMPLE_YEAR = 2017
SEPA_MONTHLY_EXAMPLE = sepa_monthly_$(SEPA_MONTHLY_EXAMPLE_YEAR).nc

# County boundaries
# see https://ckan.publishing.service.gov.uk/dataset/counties-and-unitary-authorities-december-2018-boundaries-gb-buc
UK_BOUNDARIES_URL = http://geoportal1-ons.opendata.arcgis.com/datasets/28af36afcd764edd8cd62d40bef9181c_0.geojson?outSR={%22latestWkid%22:27700}
UK_BOUNDARIES = UK_BUC.geojson

# Bibliography
BIBLIOGRAPHY = bibliography.bib

# License
LICENSE = LICENSE


# ----- Tools -----

# Root directory
ROOT = $(shell pwd)

# Base commands
PYTHON = python3.7
IPYTHON = ipython
JUPYTER = jupyter
JUPYTER_BOOK = jupyter-book
LATEX = pdflatex
BIBTEX = bibtex
MAKEINDEX = makeindex
SPHINX = sphinx-build
GHP_IMPORT = ghp-import
GHOSTSCRIPT = gs
PIP = pip
VIRTUALENV = $(PYTHON) -m venv
ACTIVATE = . $(VENV)/bin/activate && . $(CREDENTIALS)
RSYNC = rsync
TR = tr
CAT = cat
SED = sed
GIT = git
RM = rm -fr
CP = cp
CHDIR = cd
MKDIR = mkdir -p
ZIP = zip -r
UNZIP = unzip
WGET = wget
ECHO = echo

# Datestamp
DATE = `date`

# Requirements and venv
VENV = venv3
REQUIREMENTS = requirements.txt
DEV_REQUIREMENTS = dev-requirements.txt
KNOWN_GOOD_REQUIREMENTS = known-good-requirements.txt

# pyproj data
PYPROJ_DATA_DIR = `python -c "import pyproj; print(pyproj.datadir.get_data_dir())"`

# Jupyter Book construction
BUILD_DIR = _build
SRC_DIR = src
BOOK_DIR = bookdir
BOOK_BUILD_DIR = $(BOOK_DIR)/$(BUILD_DIR)

# Commands
RUN_SERVER = PYTHONPATH=. $(JUPYTER) notebook
RUN_TESTS = $(PYTHON) -m unittest discover
CREATE_BOOK = $(JUPYTER_BOOK) create $(BOOK_DIR)
BUILD_BOOK = $(JUPYTER_BOOK) build $(BOOK_DIR)
UPLOAD_BOOK = $(GHP_IMPORT) -n -p -f $(BOOK_BUILD_DIR)/html
BUILD_PRINT_BOOK = $(SPHINX) -b latex -c $(ROOT)/latex . $(BUILD_DIR)/latex
BUILD_BIBLIOGRAPHY = $(BIBTEX) $(LATEX_BOOK_STEM)
BUILD_INDEX = $(MAKEINDEX) $(LATEX_BOOK_STEM)
BUILD_EPUB_BOOK = $(SPHINX) -b epub -c $(ROOT)/epub . $(BUILD_DIR)/epub


# ----- Top-level targets -----

# Default prints a help message
help:
	@make usage

# Download datasets
datasets: env $(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY) $(DATASETS_DIR)/$(UK_BOUNDARIES) $(CEDA_CERTIFICATE) $(DATASETS_DIR)/$(CEDA_MONTHLY_EXAMPLE) $(DATASETS_DIR)/$(SEPA_MONTHLY_EXAMPLE)

# CEH interpolated monthlies
$(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY):
	$(ACTIVATE) && $(WGET) --post-data=username=$$CEH_USERNAME\&password=$$CEH_PASSWORD\&success=$(CEH_BASE_MONTHLIES)/$(CEH_EXAMPLE_MONTHLY) -O $(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY) $(CEH_LOGIN_URL)

# UK county and administrative boundaries
$(DATASETS_DIR)/$(UK_BOUNDARIES):
	$(ACTIVATE) && $(WGET) -O $(DATASETS_DIR)/$(UK_BOUNDARIES) $(UK_BOUNDARIES_URL)

# CEDA credentials
# see https://help.ceda.ac.uk/article/4442-ceda-opendap-scripted-interactions
online_ca_client:
	$(GIT) clone https://github.com/cedadev/online_ca_client
	$(CHDIR) online_ca_client/contrail/security/onlineca/client/sh/ && ./onlineca-get-trustroots-wget.sh -U https://slcs.ceda.ac.uk/onlineca/trustroots/ -c $(ROOT)/trustroots -b

$(CEDA_CERTIFICATE): online_ca_client
	$(ACTIVATE) && cd online_ca_client/contrail/security/onlineca/client/sh/ && echo $$CEDA_PASSWORD | ./onlineca-get-cert-wget.sh -U https://slcs.ceda.ac.uk/onlineca/certificate/ -c $(CEDA_CA_ROOTS) -l $$CEDA_USERNAME -S -o $(CEDA_CERTIFICATE)

# CEDA example month
$(DATASETS_DIR)/$(CEDA_MONTHLY_EXAMPLE): $(CEDA_CERTIFICATE)
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) $(UTILS)/ceda-monthly.py $(CEDA_MONTHLY_EXAMPLE_YEAR) $(DATASETS_DIR)/$(CEDA_MONTHLY_EXAMPLE)

# SEPA example month
$(DATASETS_DIR)/$(SEPA_MONTHLY_EXAMPLE):
	$(ACTIVATE) && PYTHONPATH=. $(PYTHON) $(UTILS)/sepa-monthly.py $(SEPA_MONTHLY_EXAMPLE_YEAR) $(DATASETS_DIR)/$(SEPA_MONTHLY_EXAMPLE)

# Run the notebook server
live: env
	$(ACTIVATE) && $(RUN_SERVER)

# Run tests for all versions of Python we're interested in
test: env Makefile
	$(ACTIVATE) && $(RUN_TESTS)

# Build a development venv
.PHONY: env
env: $(VENV) $(DIAGRAMS_DIR) $(DATASETS_DIR)

$(VENV):
	$(VIRTUALENV) $(VENV)
	$(ACTIVATE) && $(PIP) install -U pip wheel
	$(ACTIVATE) && $(PIP) install -r $(REQUIREMENTS)
	$(ACTIVATE) && $(PIP) install -r $(DEV_REQUIREMENTS)

$(DATASETS_DIR):
	$(MKDIR) $(DATASETS_DIR)

$(DIAGRAMS_DIR):
	$(MKDIR) $(DIAGRAMS_DIR)

# Clean up the build
clean:
	$(RM) $(DIAGRAMS_DIR)/*

# Clean up everything, including the venv and the datasets (which are *very* expensive
# to re-download)
reallyclean: clean
	$(RM) $(VENV) $(DATASETS_DIR) $(DIAGRAMS_DIR) $(CEDA_CA) $(CEDA_CA_ROOTS) $(CEDA_CERTIFICATE)


# ----- Usage -----

define HELP_MESSAGE
Editing:
   make live         run the notebook server

Maintenance:
   make env          create a virtual environment
   make datasets     download all the datasets (long!)
   make test         run the unit test suite
   make clean        clean-up the build (mainly the diagrams)
   make reallyclean  delete the venv and all the datasets as well

endef
export HELP_MESSAGE

usage:
	@echo "$$HELP_MESSAGE"
