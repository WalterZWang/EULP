import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font', family='serif')
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14, color='grey')
plt.rc('ytick', labelsize=14, color='grey')
plt.rc('legend', fontsize=16, loc='lower left')
plt.rc('figure', titlesize=18)
plt.rc('savefig', dpi=330, bbox='tight')

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def regress_dist(data, positions, xx):
    x = data['High Load Start'].values
    y = data['High Load Duration'].values
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return f

def read_ami(file_name):
    data = pd.read_csv(file_name, index_col=False,
                       usecols = ['High Load Start', 'highLoadDuration', 'building_ID', \
                                  'date', 'building_type', 'utility', 'holiday'])
    data.rename(columns={'highLoadDuration': 'High Load Duration'}, inplace=True)
    data['date'] = data.apply(lambda x: x['date'].split(' ')[0], axis=1)
    return data

def read_comStock():
    start_time_WD_CS = pd.read_csv("data/comStock/weekday_start_time.tsv", sep='\t')
    start_time_WD_CS.drop(['Option=NA'], axis = 1, inplace = True)
    start_time_WD_CS.set_index('Dependency=building_type', inplace = True)
    
    duration_WD_CS = pd.read_csv("data/comStock/weekday_duration.tsv", sep='\t')
    duration_WD_CS.drop(['Option=missing'], axis = 1, inplace = True)
    
    start_time_NWD_CS = pd.read_csv("data/comStock/weekend_start_time.tsv", sep='\t')
    start_time_NWD_CS.drop(['Option=NA'], axis = 1, inplace = True)
    start_time_NWD_CS.set_index('Dependency=building_type', inplace = True)
    
    duration_NWD_CS = pd.read_csv("data/comStock/weekend_duration.tsv", sep='\t')
    duration_NWD_CS.drop(['Option=NA'], axis = 1, inplace = True)
    
    buildingType_CS = duration_WD_CS['Dependency=building_type'].unique()
    return start_time_WD_CS, duration_WD_CS, start_time_NWD_CS, duration_NWD_CS, buildingType_CS

def generate_samples_ComStock(start_time_all, duration_all, building_type, SampleSize = 10000):
    '''
    start_time_all: marginal distribution of start time, for all building type
    duration_all: distribution of duration conditioned on start_time_all, for all building type
    '''

    startTime_marginal = start_time_all.loc[building_type]
    if startTime_marginal.sum() == 0:
        return []
    
    ## deal with the label, convert from Option=5.75 to float number only
    newIndex = []
    for index in startTime_marginal.index:
        newIndex.append(float(index.split('=')[1]))
    startTime_marginal.index = newIndex

    duration_conditional = duration_all[duration_all['Dependency=building_type'] == building_type]
    duration_conditional.set_index('Dependency=start_time', inplace=True)
    duration_conditional = duration_conditional.drop(['Dependency=building_type'], axis=1)

    samples = []
    for s in duration_conditional.index:
        for d_label in duration_conditional.columns:
            d = float(d_label.split('=')[1])
            sampleNumber = int(duration_conditional.loc[s, d_label] * startTime_marginal[s] * SampleSize)
            while sampleNumber:
                sample = [s, d]
                samples.append(sample)
                sampleNumber -= 1    

    samples = np.array(samples).astype('float')
    samples = pd.DataFrame(data = samples,
                           columns = ['High Load Start', 'High Load Duration'])
    
    return samples