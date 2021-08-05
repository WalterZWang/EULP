import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import style
import os
import time
import math
from pathlib import Path

from lib.time_domain_utils import load_shape_statistic_working_day, find_optimal_cluster_number

style.use("ggplot")

statusComplete          = 'complete'
statusBadDataExtension  = 'baddata_extension'
statusBadDataDate       = 'baddata_date'
statusBadDataLen        = 'baddata_len'
statusBadDataFreq       = 'baddata_freq'
statusBadDataNan        = 'baddata_nan'
statusBadDataZeroes     = 'baddata_zero'
statusBadDataInf        = 'baddata_inf'
statusClusterError      = 'cluster_error'

freq_map = {'H':24, '30T':48, '15T':96}

# threshold for number of allowable errors
baderror_threshold = 0.25

def read_clean_reshape_normalize_data(file, building_type=None):
    status = statusComplete
    bad_data_indicator = [None, None, None, None, None, True, status]
    file_name = Path(file).name

    if Path(file).suffix == '.parquet':
        data = pd.read_parquet(file)
    elif Path(file).suffix == '.csv':
        data = pd.read_csv(file)
    else:
        print(f'Unexpected extension, skipping {file_name}')
        bad_data_indicator[-1] = statusBadDataExtension
        return bad_data_indicator

    # Get the year of the median row, this will be the year for the entire dataset
    yr = data.iloc[[math.ceil(len(data)/2)]].index.year[0]
    data = data[data.index.year == yr]

    if np.count_nonzero(np.isnan(data['total_kWh_excluding_blanks'])) == len(data):
        data['total_kWh_excluding_blanks'] = data.drop('total_kWh_excluding_blanks', axis=1).fillna(0).sum(axis=1)

    data = data['total_kWh_excluding_blanks']

    data = data.loc[f'{yr}-1-1':]
    
    # Ger the new number of rows (should be a multiple of 8760)
    nrows = len(data)

    # find the sampling frequency of the data
    time_interval = int((data.iloc[[math.ceil(nrows/2)+1]].index[0] - data.iloc[[math.ceil(nrows/2)]].index[0]).seconds/60)

    if time_interval == 60:
        freq = "H"
    elif time_interval == 30:
        freq = "30T"
    elif time_interval == 15:
        freq = "15T"
    else:
        bad_data_indicator[-1] = statusBadDataFreq
        return bad_data_indicator

    # make sure the length of the data is at least 75% a full year
    if nrows/(365*freq_map[freq]) < 0.75:
        bad_data_indicator[-1] = statusBadDataLen + f'_{len(data)}'
        return bad_data_indicator

    # data.iloc[0] = 0 # hardcode first row as 0 since all first row values are nan for some reason...
    data = data.values

    # normalize the data by the annual peak load (0.975)
    # normalize the data by the annual base load (0.025)
    data = data / np.nanquantile(data, 0.975)

    nans = np.count_nonzero(np.isnan(data))
    if nans/nrows > baderror_threshold:
        print(f'{nans} NaNs, Skipping {file_name}')
        bad_data_indicator[-1] = statusBadDataNan + f'_{nans}'
        return bad_data_indicator

    zeroes = np.count_nonzero(data==0)
    if zeroes/nrows > baderror_threshold:
        print(f'{nans} Zeroes, Skipping {file_name}')
        bad_data_indicator[-1] = statusBadDataZeroes + f'_{zeroes}'
        return bad_data_indicator

    infs = np.count_nonzero(np.isinf(data))
    if infs > 0:
        print(f'{infs} Infs, Skipping {file_name}')
        bad_data_indicator[-1] = statusBadDataInf + f'_{infs}'
        return bad_data_indicator

    data = data.reshape((-1, int(60/time_interval)*24))

    # clean data: replace row with any nan or all 0s with np.nan
    mask = np.all(np.equal(data, 0), axis=1) | np.any(np.isnan(data), axis=1)
    data[mask] = np.nan

    dates = pd.date_range(start=f'1/1/{yr}', periods=data.shape[0], freq='D', tz='America/Denver')
    days_of_week = dates.day_name()
    holidays = dates.isin(USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max()))

    return [data, days_of_week, dates, holidays, freq, False, status]

def select_cluster_number(path, cluster_range, building_type=None):
    """
    Select the optimal number of clusters, plotting Davies–Bouldin index, Silhouette Coefficient
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """
    ## data pre-processing
    # read in data and convert to pandas dataframe
    files = os.listdir(path)

    group    = path.split('/')[-2]
    subgroup = path.split('/')[-1]

    res_dir = f'/Users/jgonzal2/EULP/result/clustering/{group}/{subgroup}'

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_list = []
    ID_list = []  # give each building an ID
    date_list = []
    dow_list = []
    holiday_list = []
    dat_type_list = []
    bldg_type_list = []
    bad_data_list  = []
    building_mapping = {}  # mapping from csv file to building ID
    i = 0
    for file in files:
        data, days_of_week, dates, holidays, freq, data_bad, status = read_clean_reshape_normalize_data(f"{path}/{file}", building_type)
        if data_bad:
            continue
        data_list.extend(data.tolist())
        # add building ID to each day
        dat_type, bldg_type, bldg_id = Path(file).stem.split('-')
        ID_list.extend([bldg_id for k in range(data.shape[0])])
        date_list.extend(dates)
        dow_list.extend(days_of_week)
        holiday_list.extend(holidays)
        dat_type_list.extend([dat_type for k in range(data.shape[0])])
        bldg_type_list.extend([bldg_type for k in range(data.shape[0])])
        bad_data_list.extend(list(np.all(np.equal(data, 0), axis=1) | np.any(np.isnan(data), axis=1)))
        building_mapping[i] = file
        i += 1

    yr = date_list[math.ceil(len(data)/2)].year

    data = pd.DataFrame(data_list, columns=range(freq_map[freq]), dtype=np.float32)

    ## clustering
    # resample the data to hour-interval
    data_cluster = data.copy()

    index = pd.period_range(f'{yr}-01-01', freq=freq, periods=freq_map[freq])
    data_cluster.columns = index

    data_resample = data_cluster.resample(freq, axis=1).mean()
    data_resample = data_resample.astype("float32")

    # clustering
    clustering = find_optimal_cluster_number(data_resample.fillna(1000))

    for i, cluster_type in enumerate(['kspectral', 'kmeans']):

        if cluster_type == 'kmeans':
            cluster_centers, labels, DBIs, SIs = clustering.select_n_kmeans(cluster_range[0], cluster_range[1])

        else:
            cluster_centers, labels, DBIs, SIs = clustering.select_n_kspectral(cluster_range[0], cluster_range[1])


        cluster_result = pd.DataFrame(labels)
        column_names = {i: f"{i}_clusters" for i in cluster_result.columns}
        cluster_result.rename(columns=column_names, inplace=True)
        cluster_result["building_ID"] = ID_list
        cluster_result["building_type"] = bldg_type_list
        cluster_result["data_type"] = dat_type_list
        cluster_result["date"] = date_list
        cluster_result["day_of_week"] = dow_list
        cluster_result["holiday"] = holiday_list
        cluster_result["bad_data"] = bad_data_list

        i = 0
        for col in data_resample.columns:
            i_str = str(i).rjust(2, '0')
            new_col = f'hour_{i_str}'
            cluster_result[new_col] = data_resample[col]
            i += 1

        cluster_result.to_csv(f"{res_dir}/cluster_result_{cluster_type}.csv")

        ## analysis on the clustering result
        # select the optimal number of clusters
        # Davies-Bouldin Index, the smaller the better
        DBI_df = pd.DataFrame(DBIs, index=["DBI"]).T
        plt.clf()
        DBI_df.plot()
        plt.savefig(f"{res_dir}/Davies-Bouldin_{cluster_type}.png")

        # Silhouette Coefficient, the higher the better
        SI_df = pd.DataFrame(SIs, index=["SI"]).T
        plt.clf()
        SI_df.plot()
        plt.savefig(f"{res_dir}/Silhouette_{cluster_type}.png")
    
def time_domain_analysis(path, number_of_clusters, building_type=None):
    """
    Conduct clustering, calculate key statistics, plot the center and statitics of each cluster
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """

    file = Path(path).name
    path = os.path.dirname(os.path.realpath(path))
    
    building_type = building_type.split('.')[0] # remove the file extension
    building_info = building_type.split('-')

    utility      = building_info[0]
    bldg_id      = building_info[1]
    bldg_type    = building_info[2]

    data_list = []
    ID_list = []  # give each building an ID
    date_list = []
    dow_list = []
    holiday_list = []
    bldg_type_list = []
    bad_data_list = []
    building_mapping = {}  # mapping from csv file to building ID

    data, days_of_week, dates, holidays, freq, data_bad, status = read_clean_reshape_normalize_data(f"{path}/{bldg_id}.parquet", building_type)
    if data_bad:
        return status

    data_list.extend(data.tolist())
    ID_list.extend([bldg_id for k in range(data.shape[0])])
    date_list.extend(dates)
    dow_list.extend(days_of_week)
    holiday_list.extend(holidays)
    bldg_type_list.extend([bldg_type for k in range(data.shape[0])])
    bad_data_list.extend(list(np.all(np.equal(data, 0), axis=1) | np.any(np.isnan(data), axis=1)))
    building_mapping[0] = file

    data = pd.DataFrame(data_list, columns=range(freq_map[freq]), dtype=np.float32)

    # Get the year of the median row, this will be the year for the entire dataset
    yr = date_list[math.ceil(len(data)/2)]

    ## calculate the key statistics for each row (day)
    stats = []

    for i in range(len(data)):
        if bad_data_list[i]:
            stats.append([np.nan for k in range(11)])
        else:
            load_shape = load_shape_statistic_working_day(data.iloc[i], freq_map[freq]/24, 0.2)
            stats.append(load_shape)

    data_load_statistic = pd.DataFrame(
        stats,
        columns=[
            "highLoad",
            "highLoad_SD",
            "baseLoad",
            "baseLoad_SD",
            "Morning Rise Start",
            "High Load Start",
            "High Load Finish",
            "Afternoon Fall Finish",
            "highLoadDuration",
            "riseTime",
            "fallTime",
        ],
    )
    data_load_statistic["Base To Peak Ratio"] = (
        data_load_statistic["baseLoad"] / data_load_statistic["highLoad"]
    )
    data_load_statistic["building_ID"] = ID_list
    data_load_statistic["utility"] = utility
    data_load_statistic["building_type"] = bldg_type_list
    data_load_statistic["date"] = date_list
    data_load_statistic["day_of_week"] = dow_list
    data_load_statistic["holiday"] = holiday_list
    data_load_statistic["bad_data"] = bad_data_list

    # Make the results dirs
    if building_type:
        res_dir = f'result/time_domain/{building_type}'
    else:
        res_dir = f'result/time_domain/all_types'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_load_statistic.to_csv(f"{res_dir}/statistics.csv")

    ## clustering
    # resample the data to hour-interval
    data_cluster = data.fillna(0).copy()
    
    index = pd.period_range(f'{yr}-01-01', freq=freq, periods=freq_map[freq])
    data_cluster.columns = index

    data_resample = data_cluster.resample(freq, axis=1).mean()
    data_resample = data_resample.astype("float32")

    # clustering
    clustering = find_optimal_cluster_number(data_resample)
    try:
        cluster_centers, labels, DBIs, SIs = clustering.select_n(number_of_clusters,number_of_clusters+1)
    except ValueError:
        os.remove(f"{res_dir}/statistics.csv")
        os.rmdir(res_dir)
        return statusClusterError

    centers = pd.DataFrame(data=cluster_centers[number_of_clusters], index=range(0,number_of_clusters))
    centers.index.name = 'cluster_id'
    centers.to_csv(f"{res_dir}/cluster_centers.csv")

    cluster_result = pd.DataFrame(labels)
    column_names = {i: f"{i}_clusters" for i in cluster_result.columns}
    cluster_result.rename(columns=column_names, inplace=True)
    cluster_result["building_ID"] = ID_list

    # plot time series of the center of each cluster
    data_plot_ts = pd.concat(
        [data, cluster_result[f"{number_of_clusters}_clusters"]], axis=1, sort=False
    )


    plt.clf()
    for i in range(number_of_clusters):
        data_plot_ts[data_plot_ts[f"{number_of_clusters}_clusters"] == i].iloc[
            :, :freq_map[freq]
        ].mean().plot(label=f"cluster {i}")
        print(f"cluster {i}: {(data_plot_ts.iloc[:,-1]==i).sum()}")
    plt.xticks(
                np.arange(0, freq_map[freq]+1, freq_map[freq]/4), ["0:00", "6:00", "12:00", "18:00", "24:00"]
            )
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{res_dir}/cluster_center.png")

    # plot statistics of each cluster
    data_plot_stats = pd.concat(
        [data_load_statistic, cluster_result[f"{number_of_clusters}_clusters"]],
        axis=1,
        sort=False,
    )
    field_to_plot = [
        "Morning Rise Start",
        "High Load Start",
        "High Load Finish",
        "Afternoon Fall Finish",
    ]

    for field in field_to_plot:
        plt.clf()
        for i in range(number_of_clusters):
            sns.kdeplot(
                data_plot_stats[data_plot_stats[f"{number_of_clusters}_clusters"] == i][
                    field
                ],
                label=f"cluster_{i}",
                bw=2,
            )

            axes = plt. gca() 
            x_min, x_max = axes.get_xlim() 
            step = (x_max - x_min - max(x_max,x_min)*0.0001)/4.0 

            plt.xticks(
                np.arange(x_min, x_max, step), ["0:00", "6:00", "12:00", "18:00", "24:00"]
            )
        plt.xlabel("Time")
        # plt.ylim(0, 0.15)
        plt.ylabel("Density")
        plt.title(field, fontsize=18)
        plt.savefig(f"{res_dir}/{field}.png")

    # plt.clf()
    # for i in range(number_of_clusters):
    #     data_plot_ts[data_plot_ts[f"{number_of_clusters}_clusters"] == i].iloc[
    #         :, :freq_map[freq]
    #     ].mean().plot(label=f"cluster {i}")
    #     print(f"cluster {i}: {(data_plot_ts.iloc[:,-1]==i).sum()}")
    # plt.xticks(
    #             np.arange(0, 25, 6), ["0:00", "6:00", "12:00", "18:00", "24:00"]
    #         )
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.savefig(f"{res_dir}/cluster_center.png")

    # # plot statistics of each cluster
    # data_plot_stats = pd.concat(
    #     [data_load_statistic, cluster_result[f"{number_of_clusters}_clusters"]],
    #     axis=1,
    #     sort=False,
    # )
    # field_to_plot = [
    #     "Morning Rise Start",
    #     "High Load Start",
    #     "High Load Finish",
    #     "Afternoon Fall Finish",
    # ]
    # for field in field_to_plot:
    #     plt.clf()
    #     for i in range(number_of_clusters):
    #         sns.kdeplot(
    #             data_plot_stats[data_plot_stats[f"{number_of_clusters}_clusters"] == i][
    #                 field
    #             ],
    #             label=f"cluster_{i}",
    #             bw_adjust=2,
    #         )
    #         plt.xticks(
    #             np.arange(0, 25, 6), ["0:00", "6:00", "12:00", "18:00", "24:00"]
    #         )
    #     plt.xlabel("Time")
    #     # plt.ylim(0, 0.15)
    #     plt.ylabel("Density")
    #     plt.title(field, fontsize=18)
    #     plt.savefig(f"{res_dir}/{field}.png")

    return statusComplete


if __name__ == "__main__":
    path = "data_example"
    number_of_clusters = 4
    time_domain_analysis(path, number_of_clusters)
