import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from matplotlib import style


import multiprocessing
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils import shuffle

def load_shape_statistic_working_day(load, samRate=4, baseLoadDefRatio=0.2):
    '''
    Input
    -------------------------
    load - pandas.series
    samRate  - int, measurements per hour

    Output
    -------------------------
    highLoadDuration: hours
    riseTime,fallTime: hours
    '''
    # load.index = load.index.astype('int')
    quantile = load.quantile([0.025, 0.975])
    quantile['high_load'] = (quantile.loc[0.025]+quantile.loc[0.975])/2
    highLoad_TS = load[load>=quantile.loc['high_load']]
    highLoadTime = highLoad_TS.index.tolist()
    highLoad = highLoad_TS.mean()
    highLoad_SD = highLoad_TS.std()/highLoad_TS.mean()
    baseLoad_TS = load[load<=(quantile.loc[0.025]+baseLoadDefRatio*(quantile.loc[0.975]-quantile.loc[0.025]))]
    baseLoadTime = baseLoad_TS.index.tolist()
    baseLoad = baseLoad_TS.mean()
    if baseLoad == 0:
        baseLoad_SD = 0
    else:
        try:
            baseLoad_SD = baseLoad_TS.std()/baseLoad_TS.mean()
        except:
            baseLoad_SD = 0
    highLoad_start = min(highLoadTime)
    highLoad_end = max(highLoadTime)
    highLoadDuration = (highLoad_end-highLoad_start)/samRate
    baseLoad_morning = max([i for i in baseLoadTime if i<=highLoad_start]+[0])
    baseLoad_evening = min([i for i in baseLoadTime if i>=highLoad_end]+[len(load)])
    riseTime = (highLoad_start - baseLoad_morning)/samRate
    fallTime = (baseLoad_evening - highLoad_end)/samRate
    relativeSD = load[highLoadTime].std()/load[highLoadTime].mean()
    return [highLoad,highLoad_SD,baseLoad,baseLoad_SD,
            baseLoad_morning,highLoad_start,highLoad_end,baseLoad_evening,
            highLoadDuration,riseTime,fallTime]

class find_optimal_cluster_number():
    """
    try k-means using different number of clusters

    Input
    ---------------------------
    data: dataset to be clustered, pd.df, every row is an observation
    ncluster_min,ncluster_max: the range of number of clusters to be tested

    Output
    ---------------------------
    cluster_centers, labels, DBIs

    Example
    ---------------------------
    test = find_optimal_cluster_number(data)
    cluster_centers_h_scale, labels_h_scale, DBIs_h_scale = test.select_n(ncluster_min, ncluster_max)
    """

    def __init__(self, data):
        self.data = data
        self.n_cores = multiprocessing.cpu_count()
        self.cluster_centers = {}
        self.labels = {}
        self.DBIs = {}
        self.SIs = {}

    def cluster(self, n_cluster):
        k_means = KMeans(init='k-means++', n_clusters=n_cluster, n_init=100)
        k_means.fit(self.data)
        cluster_center = np.sort(k_means.cluster_centers_, axis=0)
        label = pairwise_distances_argmin(self.data,cluster_center)
        DBI = metrics.davies_bouldin_score(self.data, label)
        silhouette_avg = metrics.silhouette_score(self.data, label)
        return cluster_center, label, DBI, silhouette_avg

    def select_n(self, ncluster_min, ncluster_max):
        start_time = time.time()
        self.results = Parallel(n_jobs=self.n_cores)(delayed(self.cluster)(n_cluster) for n_cluster in range(ncluster_min, ncluster_max))

        # extract result
        for i in range(ncluster_min, ncluster_max):
            self.cluster_centers[i], self.labels[i], self.DBIs[i], self.SIs[i] = self.results[i-ncluster_min]

        end_time = time.time()
        print(f'Time consumed: {(end_time-start_time)/3600} h')

        return self.cluster_centers, self.labels, self.DBIs, self.SIs
