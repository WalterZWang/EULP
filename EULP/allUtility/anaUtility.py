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

def utilityComparison(ami, save_fig=False, xmin=0, xmax=12, ymin=0, ymax=20):
    utilities = ami['utility'].unique()
    building_type = ami['building_type'].unique()
    assert building_type.shape[0] == 1
    building_type = building_type[0]

    col_n = 2
    cs_building = building_type in buildingType_CS
    row_n = utilities.shape[0] + cs_building    

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    fig, axs = plt.subplots(row_n, col_n, sharex=True, sharey=True, figsize=(10, row_n*2.5+0.5))

    # AMI data
    for row_i, utility in enumerate(utilities):
        utility_data = ami[ami['utility']==utility]
        utility_data = utility_data.dropna()
        
        # column 1 - Working day
        data_plot = utility_data[utility_data['holiday']==False]
        numberOfDays = data_plot.shape[0]
        if numberOfDays>30:
            f_raw_WD = regress_dist(data_plot, positions, xx)
            cfset = axs[row_i, 0].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[row_i, 0].contour(xx, yy, f_raw_WD, colors='k')
            axs[row_i, 0].clabel(cset, inline=1, fontsize=10)    
            axs[row_i, 0].set_ylabel(f'{utility.upper()}\n\nHigh Load \nDuration [h]')
        if row_i == 0:
            axs[row_i, 0].set_title(f"Working Day\n\nNumber of Days: {numberOfDays}")
        else:
            axs[row_i, 0].set_title(f"Number of Days: {numberOfDays}")
        
        # column 2 - Non-Working day
        data_plot = utility_data[utility_data['holiday']==True]
        numberOfDays = data_plot.shape[0]
        if numberOfDays > 30:
            f_raw_WD = regress_dist(data_plot, positions, xx)
            cfset = axs[row_i, 1].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[row_i, 1].contour(xx, yy, f_raw_WD, colors='k')
            axs[row_i, 1].clabel(cset, inline=1, fontsize=10)   
        if row_i == 0:
            axs[row_i, 1].set_title(f"Non Working Day\n\nNumber of Days: {numberOfDays}")
        else:
            axs[row_i, 1].set_title(f"Number of Days: {numberOfDays}")

    if cs_building:
        samples_WD_ComStock = generate_samples_ComStock(start_time_WD_CS, duration_WD_CS, building_type)
        samples_NWD_ComStock = generate_samples_ComStock(start_time_NWD_CS, duration_NWD_CS, building_type)
        # column 1 - Working day
        if len(samples_WD_ComStock)>0:
            f_raw_WD = regress_dist(samples_WD_ComStock, positions, xx)
            cfset = axs[row_n-1, 0].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[row_n-1, 0].contour(xx, yy, f_raw_WD, colors='k')
            axs[row_n-1, 0].clabel(cset, inline=1, fontsize=10)    
            axs[row_n-1, 0].set_ylabel('ComStock\n\nHigh Load \nDuration [h]')
            axs[row_n-1, 0].set_title("Sampled from distribution")
        
        # column 2 - Non-Working day
        if len(samples_NWD_ComStock)>0:
            f_raw_WD = regress_dist(samples_NWD_ComStock, positions, xx)
            cfset = axs[row_n-1, 1].contourf(xx, yy, f_raw_WD, cmap='coolwarm')
            cset = axs[row_n-1, 1].contour(xx, yy, f_raw_WD, colors='k')
            axs[row_n-1, 1].clabel(cset, inline=1, fontsize=10)   
            axs[row_n-1, 1].set_title("Sampled from distribution")  
            
    for i in range(2):
        axs[row_n-1, i].set_xlabel('High Load Start')
        axs[row_n-1, i].set_xticks(range(0, 13, 3))
    
    fig.suptitle(f'{building_type.capitalize()} by Utility', fontsize=16)

    if save_fig:
        plt.savefig(f'fig/utility/{building_type}.png')
    
    plt.close(fig)

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join

    start_time_WD_CS, duration_WD_CS, start_time_NWD_CS, duration_NWD_CS, buildingType_CS = read_comStock()

    mypath = 'data'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    failed_building_type = []
    for file_name in onlyfiles:
        ami = read_ami(file_name)
        try:
            utilityComparison(ami, save_fig=True)
            print(f'Processed {file_name}')
        except:
            print(f'Error raised when processing {file_name}')
            failed_building_type.append(file_name.split('-')[1].split('.')[0])

    print('Failed Building Type: ', failed_building_type)
    with open('errorUtility.txt', 'w') as f:
        f.write(f'Failed Building Type: {failed_building_type}')