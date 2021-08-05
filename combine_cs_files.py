import pandas as pd
import os

raw_data_folder = '/Users/jgonzal2/EULP/data_files/data/comstock_energy_intensity/raw_data'
output_folder   = '/Users/jgonzal2/EULP/data_files/data/comstock_energy_intensity'

folder_type     = ['by_building_type', 'by_utility', 'by_utility_and_building_type']

region_map =   {'region1':'fortcollins',
                'region2a':'seattle',
                'region2b':'pge',
                'region3a':'maine',
                'region3b':'veic',
                'region3c':'cherryland'}

data_files = os.listdir(raw_data_folder)
data_files = sorted(data_files)

save_count =  {}

for data_file in data_files:
    if '.csv' not in data_file:
        continue

    text    = data_file.split('_')
    region  = text[3]
    utility = region_map[region]

    print((f'{data_file}'))

    dataframe_orig = pd.read_csv(f'{raw_data_folder}/{data_file}')

    dataframe_orig = dataframe_orig[dataframe_orig['enduse']=='total']

    for building_type in dataframe_orig['building_type'].unique():

        dataframe = dataframe_orig[dataframe_orig['building_type']==building_type]

        if building_type not in save_count.keys():
            save_count[building_type] = 0
        if utility not in save_count.keys():
            save_count[utility] = 0
        if f'{building_type}-{utility}' not in save_count.keys():
            save_count[f'{building_type}-{utility}'] = 0

        dataframe['utility'] = utility
        dataframe = dataframe[['timestamp', 'kwh_per_sf', 'building_type','utility']]
        dataframe.rename({'kwh_per_sf':'mean_by_sqft'}, axis=1, inplace=True)

        # save by utility only
        if save_count[utility] == 0:
            dataframe.to_csv(f'{output_folder}/by_utility/{utility}.csv', index=False, header=True)
            save_count[utility] += 1
        else:
            dataframe.to_csv(f'{output_folder}/by_utility/{utility}.csv', mode="a", index=False, header=False)

        # save by building type only
        if save_count[building_type] == 0:
            dataframe.to_csv(f'{output_folder}/by_building_type/{building_type}.csv', index=False, header=True)
            save_count[building_type] += 1
        else:
            dataframe.to_csv(f'{output_folder}/by_building_type/{building_type}.csv', mode="a", index=False, header=False)

        # save by both utility and building type
        if save_count[f'{building_type}-{utility}'] == 0:
            dataframe.to_csv(f'{output_folder}/by_utility_and_building_type/{utility}-{building_type}.csv', index=False, header=True)
            save_count[f'{building_type}-{utility}'] += 1
        else:
            dataframe.to_csv(f'{output_folder}/by_utility_and_building_type/{utility}-{building_type}.csv', mode="a", index=False, header=False)

