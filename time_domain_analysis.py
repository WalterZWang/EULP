import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import style
import os
import time
from pathlib import Path

from lib.time_domain_utils import load_shape_statistic_working_day, find_optimal_cluster_number

style.use("ggplot")

def read_clean_reshape_normalize_data(file, building_type=None):
    bad_data_indicator = [None, None, None, None, True]
    file_name = Path(file).name

    # If a building type is specified, limit calculations to that type based on filename
    if building_type:
        if not building_type in file:
            return bad_data_indicator

    if Path(file).suffix == '.parquet':
        data = pd.read_parquet(file)
    elif Path(file).suffix == '.csv':
        data = pd.read_csv(file)
    else:
        print(f'Unexpected extension, skipping {file_name}')
        return bad_data_indicator

    # Check that original data length is 35,040 and skip if not
    if not len(data) == 35040:
        print(f'length of data was {len(data)}, expected 35,040.  Skipping {file_name}')
        return bad_data_indicator

    # Downselect to full days, starting with 1/1 at midnight, ending 12/30.
    # (AMI data is 1/1-12/31 UTC, so 12/31 MST has 7 missing hours of data)
    yr = data.index[48].year
    # print(f'{file_name}')
    # print(f'    before {data.index[0]} to {data.index[-1]}')
    if yr == 2017:
        data = data.loc[f'{yr}-1-1':f'{yr}-12-30']
    # print(f'    after  {data.index[0]} to {data.index[-1]}')
    data = data.values
    if not len(data) >= 34944:
        print(f'length of data was {len(data)}, expected >= 34,944.  Skipping {file_name}')
        return bad_data_indicator

    nans = np.count_nonzero(np.isnan(data))
    if nans > 0:
        print(f'{nans} NaNs, Skipping {file_name}')
        return bad_data_indicator

    # normalize the data by the annual peak load
    data = data / np.nanquantile(data, 1.00)

    nans = np.count_nonzero(np.isnan(data))
    if nans > 0:
        print(f'{nans} NaNs, Skipping {file_name}')
        return bad_data_indicator

    infs = np.count_nonzero(np.isinf(data))
    if infs > 0:
        print(f'{infs} Infs, Skipping {file_name}')
        return bad_data_indicator

    data = data.reshape((-1, 96))
    # clean data: delete the row (day) with na or all 0s
    mask = np.all(np.isnan(data) | np.equal(data, 0), axis=1)
    data = data[~mask]

    dates = pd.date_range(start=f'1/1/{yr}', periods=data.shape[0], freq='D', tz='America/Denver')
    days_of_week = dates.day_name()
    holidays = dates.isin(USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max()))

    return [data, days_of_week, dates, holidays, False]

def select_cluster_number(path, cluster_range, building_type=None):
    """
    Select the optimal number of clusters, plotting Davies–Bouldin index, Silhouette Coefficient
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """
    ## data pre-processing
    # read in data and convert to pandas dataframe
    files = os.listdir(path)

    # Make the results dirs
    if building_type:
        res_dir = f'result/time_domain/{building_type}'
    else:
        res_dir = f'result/time_domain/all_types'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_list = []
    ID_list = []  # give each building an ID
    date_list = []
    dow_list = []
    holiday_list = []
    dat_type_list = []
    bldg_type_list = []
    building_mapping = {}  # mapping from csv file to building ID
    i = 0
    for file in files:
        data, days_of_week, dates, holidays, data_bad = read_clean_reshape_normalize_data(f"{path}/{file}", building_type)
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
        building_mapping[i] = file
        i += 1

    data = pd.DataFrame(data_list, columns=range(96), dtype=np.float32)

    ## clustering
    # resample the data to hour-interval
    data_cluster = data.copy()
    index = pd.period_range("2017-01-01", freq="15T", periods=96)
    data_cluster.columns = index

    data_resample = data_cluster.resample("H", axis=1).mean()
    data_resample = data_resample.astype("float32")

    # clustering
    clustering = find_optimal_cluster_number(data_resample)
    cluster_centers, labels, DBIs, SIs = clustering.select_n(cluster_range[0], cluster_range[1])

    cluster_result = pd.DataFrame(labels)
    column_names = {i: f"{i}_clusters" for i in cluster_result.columns}
    cluster_result.rename(columns=column_names, inplace=True)
    cluster_result["building_ID"] = ID_list
    cluster_result["data_type"] = dat_type_list
    cluster_result["date"] = date_list
    cluster_result["day_of_week"] = dow_list
    cluster_result["holiday"] = holiday_list

    cluster_result.to_csv(f"{res_dir}/cluster_result.csv")

    ## analysis on the clustering result
    # select the optimal number of clusters
    # Davies-Bouldin Index, the smaller the better
    DBI_df = pd.DataFrame(DBIs, index=["DBI"]).T
    plt.clf()
    DBI_df.plot()
    plt.savefig(f"{res_dir}/Davies-Bouldin.png")

    # Silhouette Coefficient, the higher the better
    SI_df = pd.DataFrame(SIs, index=["SI"]).T
    plt.clf()
    SI_df.plot()
    plt.savefig(f"{res_dir}/Silhouette.png")

def time_domain_analysis(path, number_of_clusters, building_type=None):
    """
    Conduct clustering, calculate key statistics, plot the center and statitics of each cluster
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """
    ## data pre-processing
    # read in data and convert to pandas dataframe
    files = os.listdir(path)

    # Make the results dirs
    if building_type:
        res_dir = f'result/time_domain/{building_type}'
    else:
        res_dir = f'result/time_domain/all_types'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_list = []
    ID_list = []  # give each building an ID
    date_list = []
    dow_list = []
    holiday_list = []
    dat_type_list = []
    bldg_type_list = []
    building_mapping = {}  # mapping from csv file to building ID
    i = 0
    for file in files:
        # If a building type is specified, limit calculations to just that type based on filename
        if building_type:
            if not building_type in file:
                continue
        data, days_of_week, dates, holidays, data_bad = read_clean_reshape_normalize_data(f"{path}/{file}", building_type)
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
        building_mapping[i] = file
        i += 1

    data = pd.DataFrame(data_list, columns=range(96), dtype=np.float32)

    ## calculate the key statistics for each row (day)
    stats = []

    for i in range(len(data)):
        load_shape = load_shape_statistic_working_day(data.iloc[i])
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
    data_load_statistic["data_type"] = dat_type_list
    data_load_statistic["date"] = date_list
    data_load_statistic["day_of_week"] = dow_list
    data_load_statistic["holiday"] = holiday_list

    data_load_statistic.to_csv(f"{res_dir}/statistics.csv")

    ## clustering
    # resample the data to hour-interval
    data_cluster = data.copy()
    index = pd.period_range("2018-01-01", freq="15T", periods=96)
    data_cluster.columns = index

    data_resample = data_cluster.resample("H", axis=1).mean()
    data_resample = data_resample.astype("float32")

    # clustering
    clustering = find_optimal_cluster_number(data_resample)
    cluster_centers, labels, DBIs, SIs = clustering.select_n(number_of_clusters,number_of_clusters+1)

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
            :, :96
        ].mean().plot(label=f"cluster {i}")
        print(f"cluster {i}: {(data_plot_ts.iloc[:,-1]==i).sum()}")
    plt.xticks(
                np.arange(0, 97, 24), ["0:00", "6:00", "12:00", "18:00", "24:00"]
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
                bw_adjust=2,
            )
            plt.xticks(
                np.arange(0, 97, 24), ["0:00", "6:00", "12:00", "18:00", "24:00"]
            )
        plt.xlabel("Time")
        # plt.ylim(0, 0.15)
        plt.ylabel("Density")
        plt.title(field, fontsize=18)
        plt.savefig(f"{res_dir}/{field}.png")


if __name__ == "__main__":
    path = "data_example"
    number_of_clusters = 4
    time_domain_analysis(path, number_of_clusters)
