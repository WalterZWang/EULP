import pandas as pd
import os

results_folder_clustering = "/Users/jgonzal2/EULP/result/clustering"
results_folder_time = "/Users/jgonzal2/EULP/result/time_domain"
output_file    = "/Users/jgonzal2/EULP/result/combined_stats"

building_folders = os.listdir(results_folder_time)
building_folders = sorted(building_folders)

count = len(building_folders)
i = 1

combine_files    = False
merge_dataframes = False
convert_files    = True

save_count = {}

if combine_files:
    # Combine individual csv result files into a few large parquet files
    for folder in building_folders:
        if folder == '.DS_Store':
            continue

        building_info = folder.split('-')
        utility_name  = building_info[0]
        building_id   = building_info[1]
        building_type = building_info[2]

        save_count[building_type] = 0
        save_count[utility_name]  = 0
        save_count[f'{building_type}-{utility_name}'] = 0

    for folder in building_folders:
        if folder == '.DS_Store':
            continue

        print(f'{i}/{count}: {folder}')

        building_info = folder.split('-')
        utility_name  = building_info[0]
        building_id   = building_info[1]
        building_type = building_info[2]

        stats_file   = f'{results_folder_time}/{folder}/statistics.csv'

        dataframe = pd.read_csv(stats_file)

        # save by utility only
        if save_count[utility_name] == 0:
            dataframe.to_csv(f'{output_file}/by_utility/all_statistics-{utility_name}.csv', index=False, header=True)
            save_count[utility_name] += 1
        else:
            dataframe.to_csv(f'{output_file}/by_utility/all_statistics-{utility_name}.csv', mode="a", index=False, header=False)

        # save by building type only
        if save_count[building_type] == 0:
            dataframe.to_csv(f'{output_file}/by_building_type/all_statistics-{building_type}.csv', index=False, header=True)
            save_count[building_type] += 1
        else:
            dataframe.to_csv(f'{output_file}/by_building_type/all_statistics-{building_type}.csv', mode="a", index=False, header=False)

        # save by both utility and building type
        if save_count[f'{building_type}-{utility_name}'] == 0:
            dataframe.to_csv(f'{output_file}/by_utility_and_building_type/all_statistics-{utility_name}-{building_type}.csv', index=False, header=True)
            save_count[f'{building_type}-{utility_name}'] += 1
        else:
            dataframe.to_csv(f'{output_file}/by_utility_and_building_type/all_statistics-{utility_name}-{building_type}.csv', mode="a", index=False, header=False)

        del dataframe

        i += 1

# Merge dataframes with cluster results
if merge_dataframes:
    cluster_folders = sorted(os.listdir(results_folder_clustering))
    output_folders  = sorted(os.listdir(output_file))

    for type_folder in cluster_folders:
        if type_folder == '.DS_Store':
            continue

        for subfolder in sorted(os.listdir(f'{results_folder_clustering}/{type_folder}')):
            if subfolder == '.DS_Store':
                continue

            print(subfolder)

            try:

                cluster_df = pd.read_csv(f'{results_folder_clustering}/{type_folder}/{subfolder}/cluster_result.csv').rename({'data_type':'utility'}, axis=1)

                cluster_cols = [f'{i}_clusters' for i in range(3,30)] + ['building_ID', 'utility','date']

                cluster_df = cluster_df[cluster_cols]

                stats_df  = pd.read_csv(f'{output_file}/{type_folder}/all_statistics-{subfolder}.csv')

                merged = pd.merge(stats_df, cluster_df, on=['building_ID', 'utility','date'])

                merged.to_csv(f'{output_file}/{type_folder}/all_statistics_cluster-{subfolder}.csv')
            
            except FileNotFoundError:
                continue

if convert_files:
    csv_folders = os.listdir(output_file)

    # convert all csv files to parquet files
    for csv_folder in csv_folders:
        if csv_folder == '.DS_Store':
            continue
        for csv_file in os.listdir(f'{output_file}/{csv_folder}'):
            if '.csv' not in csv_file:
                continue
            if csv_file.split('.')[1]=='parquet':
                continue
            dataframe = pd.read_csv(f'{output_file}/{csv_folder}/{csv_file}')
            dataframe['building_ID'] = dataframe['building_ID'].astype(str)
            dataframe.to_parquet(f'{output_file}/{csv_folder}/{csv_file}'.replace('.csv','.parquet'))
            del dataframe