from time_domain_analysis import time_domain_analysis, select_cluster_number
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import style
import os
from pathlib import Path


path = "C:/Projects/Load Profiles/AMI/Ft Collins/variability_extraction/ami_by_building"

# Summarize the counts of AMI and modeled data by building type
pairs = []
files = os.listdir(path)
for file in files:
  bldg_id = Path(file).stem
  if 'ami' in bldg_id:
    dat_type = 'ami'
  else:
    dat_type = 'model'

  bldg_type = bldg_id.replace(f'{dat_type}-', '').split('-')[0]
  # print(f'{bldg_type}:{dat_type} from {bldg_id}')
  if not bldg_type:
    print(f'{bldg_type}:{dat_type} from {bldg_id}')
  pairs.append([dat_type, bldg_type])

cts = pd.DataFrame(pairs)
cts.to_csv('summary.csv')
cts.columns = ['data_type', 'bldg_type']
cts['count'] = 1

# Pivot table
vals = ['count']  # Values in Excel pivot table
ags = [np.sum]  # How each of the values will be aggregated, like Value Field Settings in Excel, but applied to all values
idx = ['bldg_type', 'data_type']  # Rows in Excel pivot table
pivot = cts.pivot_table(values=vals, index=idx, aggfunc=ags)
print(pivot)

# Step 1: select the optimal number of clusters
select_cluster_number(path,[2,8], building_type='medium_office')
# check the figures in 'fig' to determine the optimal number of clusters

# step 2: do the clustering with the optimal number of clusters, calculate the key statics and cluster center for each cluster
number_of_clusters = 2
time_domain_analysis(path, number_of_clusters, building_type='medium_office')

# share the csv files in 'result/' with us
