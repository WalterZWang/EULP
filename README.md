# EULP
This repository hosts LBNL's EULP Python package source code, and a domonstration of using the package.

## Structure

```
|
├── LICENSE
│
├── README.md
│
├── setup.py
│
├── EULP
│   ├── __init__.py
│   ├── LP_metrics.py
│   └── LP_explorations.py
│
├── example
│   ├── data
│   │   ├──sample_1.csv
│   │   └──sample_2.csv
│   ├── example_frequency_domain_workflow.ipynb
│   └── example_time_domain_workflow.ipynb
│
├── results
│   ├── LBNL_Case_1
│   └── NREL_Case_1
│
└── tests

```


``setup.py`` includes package information and tells dependent modules we are about to install.

``EULP`` contains the Python modules of the package
+ ``LP_metrics.py`` module contains a class with methods to calculate load profile metrics (both time-domain and frequency-domain). Key methods include:
    * ``LoadProfileMetrics.scale``: scale the value to a range
    * ``LoadProfileMetrics.get_load_fft``: compute the raw frequency features with a time-series input
    * ``LoadProfileMetrics.get_fft_w_window``: get the frequency feature with a given window and year
    * ``LoadProfileMetrics.method_name``: clustering on the time domain, find the optimal number of clusters
    * ``LoadProfileMetrics.method_name``: calculate the distribution of key Load Shape statistics

+ ``LP_explorations.py`` module contains a class with methods to visualize load profiles
    * ``LoadProfileExplorations.line_plot``: Generates a line plot for a load profile
    * ``LoadProfileExplorations.heatmap``: Generates an annual heat map of a load profile
    * ``LoadProfileExplorations.time_frequenc_plot``: Generates a figure with both time-series line plot and frequency-domain spectrums

+ ``LP_clustering.py`` module contains a class with methods to cluster load profiles (with time- and frequency- features)


``example`` contains:
1. example building [electric load profile](example/data/sample.csv)
    * each csv file is a whole year smart meter data for a building
    * columns: "Datetime" and "Value"
    * rows: 1 or 2 years data, at 15 min interval
2. example jupyter notebooks
    * [``example_time_domain_workflow.ipynb``](example/example_time_domain_workflow.ipynb) demonstrates the capability of time domain analysis of this package
    * [``example_frequency_domain_workflow.ipynb``](example/example_frequency_domain_workflow.ipynb) demonstrates the capability of frequency domain analysis of this package
    * ``time_domain_analysis.py`` demonstrates xyz

``result`` contains the analysis results. For the ease of sharing, please organize the results in sub-directory, for instance:
+ ``LBNL_Case_1``
+ ``NREL_Case_Name``

``tests`` contains test scripts (TBD)


## Installation
We recommend installing the package in a virtual environment. Instructions about creating virtual Python environments:
+ With [Anaconda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)
+ With [venv](https://docs.python.org/3/library/venv.html)

Once the virtual envrionment is installed, change directory to the root path of this repository, then run ```pip install .``` FUTURE: run ```pip install EULP```


## Use
See example workflow in the ``example`` part


## TODO
1. Wrap up the code
2. Test with examples
3. Add license aggreement
4. Package the code and release to PyPi
