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

from utils import regress_dist, read_ami, read_comStock, generate_samples_ComStock

def seasonComparison(ami, weather, save_fig=False, xmin=0, xmax=12, ymin=0, ymax=20):
    ## Prepare the data
    # Merge data and weather file, get the column season_temp
    data = pd.merge(ami, weather,  how='inner', 
                    left_on=['date','utility'], right_on = ['date','utility'])
    data = data.dropna()
    # Get the column season_time
    season_time = {'Shoulder': [3, 4, 5, 9, 10, 11], 'Summer': [6, 7, 8], 'Winter': [12, 1, 2]}
    season_series = data['month']
    for season, months in season_time.items():
        for month in months:
            season_series = season_series.replace(month, season)
    data['season_time'] = season_series
    
    ## Plot
    building_type = data['building_type'].unique()
    assert building_type.shape[0] == 1
    building_type = building_type[0]

    CS_building = building_type in buildingType_CS
    col_n = 3 + CS_building

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    fig, axs = plt.subplots(4, col_n, sharex=True, sharey=True, figsize=(col_n*2.5+0.5, 11))

    ## Plot AMI data
    # Working day
    data_WD = data[data['holiday']==False]
    for col_i, season in enumerate(data['season_temp'].unique()):
        for row_i, criterion in enumerate(['season_time', 'season_temp']):
            data_plot = data_WD[data_WD[criterion] == season]
            numberOfDays = data_plot.shape[0]
            if numberOfDays>30:
                f_raw_WD = regress_dist(data_plot, positions, xx)
                cfset = axs[row_i, col_i].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
                cset = axs[row_i, col_i].contour(xx, yy, f_raw_WD, colors='k')
                axs[row_i, col_i].clabel(cset, inline=1, fontsize=10)
                if row_i == 0:
                    axs[row_i, col_i].set_title(f"{season}\n\nNumber of Days: {numberOfDays}")
                else:
                    axs[row_i, col_i].set_title(f"Number of Days: {numberOfDays}")
            else:
                if row_i == 0:
                    axs[row_i, col_i].set_title(f"{season}\n\nNumber of Days: {numberOfDays}")
                else:
                    axs[row_i, col_i].set_title(f"Number of Days: {numberOfDays}")            
    # Non-Working day
    data_NWD = data[data['holiday']]
    for col_i, season in enumerate(data['season_temp'].unique()):
        for row_i, criterion in enumerate(['season_time', 'season_temp']):
            data_plot = data_NWD[data_NWD[criterion] == season]
            numberOfDays = data_plot.shape[0]
            if numberOfDays>30:
                f_raw_WD = regress_dist(data_plot, positions, xx)
                cfset = axs[row_i+2, col_i].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
                cset = axs[row_i+2, col_i].contour(xx, yy, f_raw_WD, colors='k')
                axs[row_i+2, col_i].clabel(cset, inline=1, fontsize=10)
                axs[row_i+2, col_i].set_title(f"Number of Days: {numberOfDays}")   
            else:
                axs[row_i+2, col_i].set_title(f"Number of Days: {numberOfDays}")   
                
    # Plot comStock data
    if CS_building:
        samples_WD_ComStock = generate_samples_ComStock(start_time_WD_CS, duration_WD_CS, building_type)
        samples_NWD_ComStock = generate_samples_ComStock(start_time_NWD_CS, duration_NWD_CS, building_type)
        # column 1 - Working day
        if len(samples_WD_ComStock)>0:
            f_raw_WD = regress_dist(samples_WD_ComStock, positions, xx)
            cfset = axs[0, col_n-1].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[0, col_n-1].contour(xx, yy, f_raw_WD, colors='k')
            axs[0, col_n-1].clabel(cset, inline=1, fontsize=10)    
            axs[0, col_n-1].set_title("ComStock\n\nSampled from distribution")
        
        # column 2 - Non-Working day
        if len(samples_NWD_ComStock)>0:
            f_raw_WD = regress_dist(samples_NWD_ComStock, positions, xx)
            cfset = axs[2, col_n-1].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[2, col_n-1].contour(xx, yy, f_raw_WD, colors='k')
            axs[2, col_n-1].clabel(cset, inline=1, fontsize=10)   
            axs[2, col_n-1].set_title("Sampled from distribution")    
        
    for col_i in range(col_n):
        axs[3, col_i].set_xlabel('High Load Start')
        
    axs[0, 0].set_ylabel('Working Days\n\nby Month\n\nHigh Load \nDuration [h]')
    axs[1, 0].set_ylabel('Working Days\n\nby Temperature\n\nHigh Load \nDuration [h]')
    axs[2, 0].set_ylabel('Non-Working Days\n\nby Month\n\nHigh Load \nDuration [h]')
    axs[3, 0].set_ylabel('Non-Working Days\n\nby Temperature\n\nHigh Load \nDuration [h]')

    fig.suptitle(f'{building_type.capitalize()} by Season', fontsize=16)

    if save_fig:
        plt.savefig(f'fig/season/{building_type}.png')
    
    plt.close(fig)

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join

    start_time_WD_CS, duration_WD_CS, start_time_NWD_CS, duration_NWD_CS, buildingType_CS = read_comStock()

    # Get the weather data
    weather = pd.read_csv('data\weather\daily_temp_season_by_utility.csv')
    weather['season_temp'] = weather.apply(lambda x: x['season'].capitalize(), axis=1)
    weather = weather.drop(['season'], axis=1)

    mypath = 'data'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    failed_building_type = []
    for file_name in onlyfiles:
        ami = read_ami(file_name)
        try:
            seasonComparison(ami, weather, save_fig=True)
            print(f'Processed {file_name}')
        except:
            print(f'Error raised when processing {file_name}')
            failed_building_type.append(file_name.split('-')[1].split('.')[0])

    print('Failed Building Type: ', failed_building_type)
    with open('errorSeason.txt', 'w') as f:
        f.write(f'Failed Building Type: {failed_building_type}')