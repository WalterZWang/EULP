# EULP
This repo stores the script to analyze EULP from NREL's dataset

## Structure

``data_example`` contains data example, NREL team needs to reformat the data as the data examples
    * each csv file is a whole year smart meter data for a building
    * 2 columns:: "Datetime" and "value"
    * rows: 1 or 2 years data, at 15 min interval
``environment.yml`` dependece of the environment (Python 3.8)
``time_domain_analysis.py`` python script to be called to do the time domain analysis
``frequence_domain_analysis.py`` python script to be called to do the frequence domain analysis
``utils.py`` utility functions used in ``time_domain_analysis.py`` and ``frequence_domain_analysis``
``result`` contains the analysis results
``fig`` contains analysis plots

## Functions

1. time domain analysis
    * clustering on the time domain, find the optimal number of clusters
    * calculate the distribution of key Load Shape statistics
2. frequency domain analysis


## Steps for analysis

1. Set up the virtual environment, installing the libraries specified in the ``environment.yml``
2. Run ``time_domain_analysis.py`` and ``frequence_domain_analysis.py``
3. Send us the results in ``result`` folder