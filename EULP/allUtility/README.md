# EULP
This sub-folder contains the code of time-domain analysis of AMI data from multiple utilities 

## Structure

```
├── utils.py
│
├── anaUtility.py
│
├── anaSeason.py
│
├── fig
│   ├── utility
│   ├── season
│
├── data
│   ├── comStock
│   │   ├──weekday_duration.tsv
│   │   └──weekday_start_time.tsv
│   │   └──weekend_duration.tsv
│   │   └──weekend_start_time.tsv
│   ├── weather
│   │   ├──daily_temp_season_by_utility.py
│   └── all_statistics_cluster-{*building type}.csv
│
```
For the sake of data security, fig and data are not open sourced in this repo. To run the code, make the subfolder
fig/, data/ first; and then put the data into the subfolder.


``utils.py`` includes utility functions that will be used by anaUtility.py and anaSeason.py

``anaUtility.py`` compare AMI data from different utilities

``anaSeason.py`` compare AMI data of different seasons
