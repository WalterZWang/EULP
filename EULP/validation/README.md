# EULP
This sub-folder contains the code of time-domain analysis of AMI data from Fort Collins, and compare it with the assumptions used in ComStock 

## Structure

```
├── validation_byWD.ipynb
│
├── validation_byCluster.ipynb
│
├── validation_explore_medium_office.ipynb
│
├── distribution.ipynb
│
├── fig
│   ├── validation
│   │   ├──explore_medium_office
│   │   ├──byWD
│   │   └──byCluster
│   └── distribution
│   │   ├──ecomp_ami_comstock
│   │   └──season
│
├── data
│   ├── ami_time_domain
│   │   └──ami-{*building_type}
│   ├── model_time_domain
│   │   └──model-{*building_type}
│   └── start_time_durations
│
```
For the sake of data security, fig and data are not open sourced in this repo. To run the code, make the subfolder
fig/, data/ first; and then put the data into the subfolder.

``validation_explore_medium_office.py`` use medium size office as an example to explore different validation metrics and approaches

``validation_byWD.ipynb`` compares AMI and Model data by working day and non-working day

``validation_byCluster.ipynb`` compares AMI and Model data by the two clusters identified

``distribution.ipynb`` generate distribution of high load start time and high load duration from the AMI data and explore season variability of the distributions
