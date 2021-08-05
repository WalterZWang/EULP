import pandas as pd
import os

raw_data_folder = '/Users/jgonzal2/EULP/data_files/data/energy_intensity/raw_data'
output_folder   = '/Users/jgonzal2/EULP/data_files/data/energy_intensity'

folder_type     = ['by_building_type', 'by_utility', 'by_utility_and_building_type']

utility_folders = os.listdir(raw_data_folder)
utility_folders = sorted(utility_folders)

save_count =  {}

for utility in utility_folders:
    if utility == '.DS_Store':
        continue

    data_files = os.listdir(f'{raw_data_folder}/{utility}')

    for data_file in data_files:
        if '.csv' not in data_file:
            continue

        text = data_file.split('_')
        building_type_list = text[3:-3]

        if building_type_list[0] == '3xmedian':
            building_type_list = building_type_list[1:]
        elif building_type_list[0] == 'median':
            building_type_list = building_type_list[2:]

        building_type = '_'
        building_type = building_type.join(building_type_list)

        if building_type not in save_count.keys():
            save_count[building_type] = 0
        if utility not in save_count.keys():
            save_count[utility] = 0
        if f'{building_type}-{utility}' not in save_count.keys():
            save_count[f'{building_type}-{utility}'] = 0

        print((f'{utility}-{building_type}'))

        dataframe = pd.read_csv(f'{raw_data_folder}/{utility}/{data_file}')
        dataframe['building_type'] = building_type
        dataframe['utility'] = utility

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

