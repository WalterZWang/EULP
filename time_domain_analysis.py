import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import style
import os
import time

from lib.time_domain_utils import load_shape_statistic_working_day, find_optimal_cluster_number

style.use("ggplot")

def select_cluster_number(path, cluster_range):
    """
    Select the optimal number of clusters, plotting Davies–Bouldin index, Silhouette Coefficient
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """
    ## data pre-processing
    # read in data and convert to pandas dataframe
    files = os.listdir(path)

    data_list = []
    ID_list = []  # give each building an ID
    building_mapping = {}  # mapping from csv file to building ID
    i = 0
    for file in files:
        data = pd.read_csv(f"{path}/{file}", index_col="Datetime").values
        # normalize the data by the anually 99% peak load
        data = data / np.nanquantile(data, 0.99)
        data = data.reshape((-1, 96))
        # clean data: delete the row (day) with na or all 0s
        mask = np.all(np.isnan(data) | np.equal(data, 0), axis=1)
        data = data[~mask]
        data_list.extend(data.tolist())
        # add building ID to each day
        ID_list.extend([i for k in range(data.shape[0])])
        building_mapping[i] = file
        i += 1

    data = pd.DataFrame(data_list, columns=range(96), dtype=np.float32)

    ## clustering
    # resample the data to hour-interval
    data_cluster = data.copy()
    index = pd.period_range("2018-01-01", freq="15T", periods=96)
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

    cluster_result.to_csv("result/time_domain/cluster_result.csv")

    ## analysis on the clustering result
    # select the optimal number of clusters
    # Davies-Bouldin Index, the smaller the better
    DBI_df = pd.DataFrame(DBIs, index=["DBI"]).T
    plt.clf()
    DBI_df.plot()
    plt.savefig("fig/time_domain/select_cluster_number/Davies-Bouldin.png")

    # Silhouette Coefficient, the higher the better
    SI_df = pd.DataFrame(SIs, index=["SI"]).T
    plt.clf()
    SI_df.plot()
    plt.savefig("fig/time_domain/select_cluster_number/Silhouette.png")



def time_domain_analysis(path, number_of_clusters):
    """
    Conduct clustering, calculate key statistics, plot the center and statitics of each cluster
    :param path: path to the data, str, example 'data_example'
    :param cluster_range: the range of clusters to be explored, list, example [2,8]
    """
    ## data pre-processing
    # read in data and convert to pandas dataframe
    files = os.listdir(path)

    data_list = []
    ID_list = []  # give each building an ID
    building_mapping = {}  # mapping from csv file to building ID
    i = 0
    for file in files:
        data = pd.read_csv(f"{path}/{file}", index_col="Datetime").values
        # normalize the data by the anually 99% peak load
        data = data / np.nanquantile(data, 0.99)
        data = data.reshape((-1, 96))
        # clean data: delete the row (day) with na or all 0s
        mask = np.all(np.isnan(data) | np.equal(data, 0), axis=1)
        data = data[~mask]
        data_list.extend(data.tolist())
        # add building ID to each day
        ID_list.extend([i for k in range(data.shape[0])])
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

    data_load_statistic.to_csv("result/time_domain/statistics.csv")

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
    plt.savefig("fig/time_domain/cluster_center.png")

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
            plt.xticks(
                np.arange(0, 97, 24), ["0:00", "6:00", "12:00", "18:00", "24:00"]
            )
        plt.xlabel("Time")
        # plt.ylim(0, 0.15)
        plt.ylabel("Density")
        plt.title(field, fontsize=18)
        plt.savefig(f"fig/time_domain/stats/{field}.png")


if __name__ == "__main__":
    path = "data_example"
    number_of_clusters = 4
    time_domain_analysis(path, number_of_clusters)
