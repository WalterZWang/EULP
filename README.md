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
│   ├── example_time_domain_workflow.ipynb
│   └── time_domain_analysis.py
│
├── results
│   ├── LBNL_Case_1
│   │   ├── fig
│   │   └── other
│   └── NREL_Case_1
│
└── tests

```


``setup.py`` includes package information and tells dependent modules we are about to install. We recommend installing the package in a virtual environment. Instructions about creating virtual Python environments:
1. With [Anaconda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)
2. With [venv](https://docs.python.org/3/library/venv.html)

``EULP`` contains the Python modules of the package
+ ``LP_metrics.py`` module contains a class with methods to calculate load profile metrics (both time-domain and frequency-domain). Key methods include:
    + ``method_name``: clustering on the time domain, find the optimal number of clusters
    + ``method_name``: clustering on the time domain, find the optimal number of clusters
    + ``method_name``: calculate the distribution of key Load Shape statistics
+ ``LP_explorations.py`` module contains a class with methods to visualize load profiles

``example`` contains:
1. example building [electric load profile](example/data/sample_1.csv)
    + each csv file is a whole year smart meter data for a building
    + columns: "Datetime" and "Value"
    + rows: 1 or 2 years data, at 15 min interval
2. example jupyter notebooks
    + ``example_time_domain_workflow.ipynb`` demonstrates the capability of time domain analysis of this package
    + ``example_frequency_domain_workflow.ipynb`` demonstrates the capability of frequency domain analysis of this package
    + ``time_domain_analysis.py`` demonstrates xyz

``result`` contains the analysis results

``fig`` contains analysis plots


## Functions

1. time domain analysis
    * clustering on the time domain, find the optimal number of clusters
    * calculate the distribution of key Load Shape statistics
2. frequency domain analysis


## Steps for analysis

1. Set up the virtual environment, installing the libraries specified in the ``environment.yml``
2. Run ``time_domain_analysis.py`` and ``frequence_domain_analysis.py``, see how the functions are used in ``*_example.ipynb``
3. Share with us the results in ``result`` folder


## TODO