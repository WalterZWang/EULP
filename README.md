# EULP
This repo stores the script to analyze EULP from NREL's dataset

## Structure

``data_example`` contains data example, NREL team needs to reformat the data as the data examples
    * each csv file is a whole year smart meter data for a building
    * 2 columns:: "Datetime" and "value"
    * rows: 1 or 2 years data, at 15 min interval


## Functions

1. time domain analysis
    * clustering on the time domain, find the optimal number of clusters
    * calculate the distribution of key Load Shape statistics
2. frequency domain analysis
