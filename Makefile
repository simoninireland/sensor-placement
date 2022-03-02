# Makefile for sensor placement experiment
#
# Copyright (C) 2022 Simon Dobson <simon.dobson@st-andrews.ac.uk>
#
# Licensed under the Creative Commons Attribution-Share Alike 4.0
# International License (https://creativecommons.org/licenses/by-sa/4.0/).
#

# ----- Sources -----

# Text
INDEX =
TEXT =

# Notebooks
NOTEBOOKS =  \
	sensor-placement.ipynb

# Source code
SOURCE = \
	sensor_placement/__init__.py \
	sensor_placement/nnni.py

# Utilities
UTILS = utils/
SCRIPTS = \
	$(UTILS)/sepa-monthly.py \
	$(UTILS)/ceda-monthly.py


# ----- Data -----

# Data directory
DATASETS_DIR = datasets
CREDENTIALS = ./credentials

# CEH-GEAR monthly
CEH_URL = https://catalogue.ceh.ac.uk
CEH_LOGIN_URL = $(CEH_URL)/sso/login
CEH_BASE = /datastore/eidchub/dbf13dd5-90cd-457a-a986-f2f9dd97e93c
CEH_BASE_MONTHLIES = $(CEH_BASE)/GB/monthly
CEH_EXAMPLE_MONTHLY = CEH_GEAR_monthly_GB_2017.nc

# SEPA rain gauges
# see https://www2.sepa.org.uk/rainfall/DataDownload
SEPA_URL = https://apps.sepa.org.uk/rainfall
SEPA_STATIONS_URL = $(SEPA_URL)/api/Stations?json=true
SEPA_STATIONS = sepa-stations.json

# SEPA monthly
# see utils/sepa-monthly.py for construction
SEPA_EXAMPLE_MONTHLY = sepa_monthly_2017.nc

# CEDA rain gauges
# see https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202107
CEDA_URL = https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-daily-rain-obs/dataset-version-202107
CEDA_CA = online_ca_client
CEDA_CA_ROOTS = $(ROOT)/trustroots
CEDA_CERTIFICATE = $(ROOT)/ceda.pem
CEDA_STATIONS_URL = $(CEDA_URL)/midas-open_uk-daily-rain-obs_dv-202107_station-metadata.csv
CEDA_STATIONS = ceda-stations.csv

# CEDA monthly
# see utils/ceda-monthly.py for construction
CEDA_EXAMPLE_MONTHLY = ceda_monthly_2017.nc

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
datasets: env $(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY) $(DATASETS_DIR)/$(UK_BOUNDARIES) $(DATASETS_DIR)/$(SEPA_EXAMPLE_MONTHLY) $(DATASETS_DIR)/$(CEDA_EXAMPLE_MONTHLY)

# CEH interpolated monthlies
$(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY):
	$(ACTIVATE) && $(WGET) --post-data=username=$$CEH_USERNAME\&password=$$CEH_PASSWORD\&success=$(CEH_BASE_MONTHLIES)/$(CEH_EXAMPLE_MONTHLY) -O $(DATASETS_DIR)/$(CEH_EXAMPLE_MONTHLY) $(CEH_LOGIN_URL)

# UK county and administrative boundaries
$(DATASETS_DIR)/$(UK_BOUNDARIES):
	$(ACTIVATE) && $(WGET) -O $(DATASETS_DIR)/$(UK_BOUNDARIES) $(UK_BOUNDARIES_URL)

# SEPA rain gauge stations
$(DATASETS_DIR)/$(SEPA_STATIONS):
	$(ACTIVATE) && $(WGET) -O $(DATASETS_DIR)/$(SEPA_STATIONS) $(SEPA_STATIONS_URL)

# SEPA monthly observations
$(DATASETS_DIR)/$(SEPA_EXAMPLE_MONTHLY): $(DATASETS_DIR)/$(SEPA_STATIONS)
	$(ACTIVATE) && $(PYTHON) $(UTILS)/sepa-monthly.py 2017 $(DATASETS_DIR)/$(SEPA_EXAMPLE_MONTHLY)

# CEDA credentials
# see https://help.ceda.ac.uk/article/4442-ceda-opendap-scripted-interactions
online_ca_client:
	git clone https://github.com/cedadev/online_ca_client
	cd online_ca_client/contrail/security/onlineca/client/sh/ && ./onlineca-get-trustroots-wget.sh -U https://slcs.ceda.ac.uk/onlineca/trustroots/ -c $(ROOT)/trustroots -b

$(CEDA_CERTIFICATE): online_ca_client
	$(ACTIVATE) && cd online_ca_client/contrail/security/onlineca/client/sh/ && echo $$CEDA_PASSWORD | ./onlineca-get-cert-wget.sh -U https://slcs.ceda.ac.uk/onlineca/certificate/ -c $(CEDA_CA_ROOTS) -l $$CEDA_USERNAME -S -o $(CEDA_CERTIFICATE)

# CEDA stations
$(DATASETS_DIR)/$(CEDA_STATIONS): $(CEDA_CERTIFICATE)
	$(WGET) --certificate=$(CEDA_CERTIFICATE) -O $(DATASETS_DIR)/$(CEDA_STATIONS) $(CEDA_STATIONS_URL)

# CEDA data archive
$(DATASETS_DIR)/ceda: $(CEDA_CERTIFICATE)
	$(MKDIR) $(DATASETS_DIR)/ceda
	$(CHDIR) $(DATASETS_DIR)/ceda && wget --certificate=$(CEDA_CERTIFICATE) -e robots=off --mirror --no-parent -r $(CEDA_URL)

# CEDA monthly observations
$(DATASETS_DIR)/$(CEDA_EXAMPLE_MONTHLY): $(DATASETS_DIR)/ceda $(DATASETS_DIR)/$(CEDA_STATIONS)
	$(ACTIVATE) && $(PYTHON) $(UTILS)/ceda-monthly.py 2017 $(DATASETS_DIR)/$(CEDA_EXAMPLE_MONTHLY)

# Run the notebook server
live: env
	$(ACTIVATE) && $(RUN_SERVER)

# Build a development venv
.PHONY: env
env: $(VENV)

$(VENV):
	$(VIRTUALENV) $(VENV)
	$(ACTIVATE) && $(PIP) install -U pip wheel
	$(ACTIVATE) && $(PIP) install -r $(REQUIREMENTS)

# Clean up the build
clean:

# Clean up everything, including the venv and the datasets (which are *very* expensive
# to re-download)
reallyclean: clean
	$(RM) $(VENV) $(DATASETS_DIR) $(CEDA_CA) $(CEDA_CA_ROOTS) $(CEDA_CERTIFICATE)


# ----- Usage -----

define HELP_MESSAGE
Editing:
   make live         run the notebook server

Maintenance:
   make env          create a virtual environment
   make datasets     download all the datasets (long!)
   make clean        clean-up the build
   make reallyclean  delete the venv as well

endef
export HELP_MESSAGE

usage:
	@echo "$$HELP_MESSAGE"
