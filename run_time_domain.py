from time_domain_analysis import time_domain_analysis, select_cluster_number
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import style
import os
from pathlib import Path
import botocore
import boto3
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def ListFilesV1(client, bucket, prefix=''):
    """List files in specific S3 URL"""
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix,
                                     Delimiter='/'):
        for content in result.get('Contents', []):
            yield content.get('Key')


s3_client = boto3.client('s3')

bucket_name = 'enduse-bucket2'

data_folder = "/Users/jgonzal2/EULP/data_files/data"
manually_downloaded_data_folder = "/Users/jgonzal2/EULP/data_files/data/manually_downloaded"
metadata_folder = "/Users/jgonzal2/EULP/data_files/metadata"

aws_folders = {'cherryland':'cherryland', 'epb':'EPB', 'fortcollins':'fort_collins', 'horry':'horry', 'maine':'Maine', 'seattle':'seattle ', 'tallahassee':'Tallahassee', 'veic':'veic'}

statusComplete          = 'complete'
statusBadDataExtension  = 'baddata_extension'
statusBadDataLen        = 'baddata_len'
statusBadDataFreq       = 'baddata_freq'
statusBadDataNan        = 'baddata_nan'
statusBadDataInf        = 'baddata_inf'
statusAwsError          = 'aws_client_error'
statusAwsNotFoundError  = 'aws_filenotfound_error'

overwrite_summary = True
download_from_aws = True
keep_data         = True
run_analysis      = False

# Download data from AWS as needed
# Run time domain analysis for each building
# Save results for each building
if download_from_aws:
  if overwrite_summary:
    metadata_files = os.listdir(metadata_folder)

    pairs = []
    for metadata_file in metadata_files:
      metadata = pd.read_csv(metadata_folder + '/' + metadata_file)

      utility_name = metadata_file.split('_')[-1].lower()
      utility_name = utility_name.split('.')[0]

      for index, row in metadata.iterrows():
        bldg_id   = row['BuildingID']
        bldg_type = row['BuildingType']

        if not bldg_type:
          bldg_type = "NONE"

        pairs.append([utility_name, bldg_id, bldg_type, ''])
        
    cts = pd.DataFrame(pairs)
    cts.columns = idx = ['utility_name', 'bldg_id', 'bldg_type', 'status']
    cts['status'] = 'not_started'
    cts.to_csv('time_summary.csv')
    cts['count'] = 1

  else:
    cts = pd.read_csv('/Users/jgonzal2/EULP/time_summary.csv', index_col=0)

  # Pivot table
  vals = ['count']  # Values in Excel pivot table
  ags = [np.sum]  # How each of the values will be aggregated, like Value Field Settings in Excel, but applied to all values
  idx = ['utility_name', 'bldg_id', 'bldg_type']  # Rows in Excel pivot table
  pivot = cts.pivot_table(values=vals, index=idx, aggfunc=ags)
  print(pivot)

  error_files = []

  checkpoint = 0

  for index, row in cts.iterrows():

    # Step 1: select the optimal number of clusters
    # select_cluster_number(path,[2,8], building_type=data_and_bldg_type)
    # check the figures in 'fig' to determine the optimal number of clusters

    utility_name = row['utility_name']
    bldg_id      = row['bldg_id']
    bldg_type    = row['bldg_type']
    
    data_and_bldg_type = f'{utility_name}-{bldg_id}-{bldg_type}'

    # download ami data from aws
    # save to data_path

    s3_file_path = f'{aws_folders[utility_name]}/FilesByBuilding/ami_by_building_com/{bldg_id}.parquet'
    data_path = data_folder + f'/{bldg_id}.parquet'

    if row['status'] != 'not_started':
      print('Skipping ' + data_and_bldg_type)
      continue
    
    else:
      if download_from_aws:
        while True:
          try:
            f = s3_client.download_file(bucket_name, s3_file_path, data_path)

            # sleep to guarantee that the file was downloaded
            time.sleep(0.05)

            if run_analysis:

              building_data_df = pd.read_parquet(data_path)
              building_data_df.index.rename('Datetime', inplace=True)

              # step 2: do the clustering with the optimal number of clusters, calculate the key statics and cluster center for each cluster
              number_of_clusters = 2
              status = time_domain_analysis(data_path, number_of_clusters, building_type=data_and_bldg_type)

              row['status'] = status
              cts.iloc[index] = row

              del building_data_df

            else:
              row['status'] = 'downloaded_not_ran'

            # delete ami data from local machine
            if not keep_data:
              os.remove(data_path)  

            print(f"Completed: {data_and_bldg_type}")

          except botocore.exceptions.ClientError:
              # Check if file exists in AWS
              file_list = ListFilesV1(s3_client, 'enduse-bucket2', prefix=f'{aws_folders[utility_name]}/FilesByBuilding/ami_by_building_com/')
              
              if s3_file_path in file_list:
                print(f"AWS error loading {data_and_bldg_type}")
                row['status'] = statusAwsError
                
              else:
                print(f"AWS file not found in folder {data_and_bldg_type}")
                row['status'] = statusAwsNotFoundError
                
              error_files.append(str(bldg_id) + '.parquet')  
              cts.iloc[index] = row

          except botocore.exceptions.EndpointConnectionError:
            print('Connection Error: Trying to reconnect in 10 seconds')
            time.sleep(10)
            continue

          break
        
    checkpoint+=1

    if checkpoint == 100:
      checkpoint = 0
      cts.to_csv('time_summary.csv')

  cts.to_csv('time_summary.csv')

else:
  data_files = os.listdir(manually_downloaded_data_folder)

  for data_file in data_files:
    if data_file in ['.DS_Store','RID1.parquet','RID2.parquet', 'RID31.parquet', 'RID46.parquet']:
      continue

    building_info = data_file.split('.')[0]
    building_info = building_info.split('-')
    utility_name  = building_info[0]
    bldg_id       = building_info[1]
    bldg_type     = building_info[2]
    
    data_and_bldg_type = f'{utility_name}-{bldg_id}-{bldg_type}'
    data_path = manually_downloaded_data_folder + f'/{data_file}'

    if run_analysis:
      building_data_df = pd.read_parquet(data_path)
      building_data_df.index.rename('Datetime', inplace=True)

      # step 2: do the clustering with the optimal number of clusters, calculate the key statics and cluster center for each cluster
      number_of_clusters = 2
      status = time_domain_analysis(data_path, number_of_clusters, building_type=data_and_bldg_type)

    # delete ami data from local machine
    if not keep_data:
      os.remove(data_path)

    print(f"Completed: {data_and_bldg_type}")

    del building_data_df
