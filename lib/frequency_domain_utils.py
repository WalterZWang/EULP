import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# from yellowbrick.cluster import KElbowVisualizer

import matplotlib.gridspec as gridspec


def get_fft_values(y_values, T, N, f_s, window=1):
    '''
    Get the frequency and amplitude values
    :param y_values: the time domain y-values
    :param T: period
    :param N: number of data points
    :param f_s: sample frequency
    :return: [frequencies (frequency domain x values)] and [amplitudes (frequency domain x values)]
    '''
    window = np.hanning(len(y_values)) 
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2) # Frequency values (x-axix in the frequency domain)
    fft_values_ = fft(y_values*window) # Real part: amplitude of the signal, Imaginary part: phase of the signal
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2]) # np.abs: get the real part, N//2: Only need the positive part of the result
    return f_values, fft_values

def get_load_fft(df_t, sample_timestep=15):
    '''
    Get the frequency and amplitude values from a time-series dataframe. 
    :param df_t: dataframe with a 'Value' column
    :param sample_timestep: minutes per timestep
    :return: frequency and amplitude values arrays
    '''
    t_n = len(df_t['Value']) * sample_timestep * 60    # Total duration of sample (seconds)
    N = len(df_t['Value'])                             # Total number of sample points
    T = t_n / N                                        # Timestep (seconds)
    f_s = 1/T                                          # Sample frequency (Hz)
    y_values = df_t['Value'].tolist()
    f_values, fft_values = get_fft_values(y_values, T, N, f_s)
    return f_values, fft_values

def get_fft_sum_by_bins(f_values, fft_values, hour_interval_bins):
    ''' 
    Create bins of amplitude sums by hour intervals
    :param f_values: frequency values
    :param fft_values: corresponding amplitude values
    :param hour_interval_bins: a list of hour intervals e.g., [[0.5, 1], [1, 2], [2, 4]]
    :return: a pandas dataframe where each column is the bin name and rows are sum of the amplitude
    '''
    hour_to_sec = 3600
    sec_interval_bins = [[interval_l*hour_to_sec, interval_h*hour_to_sec] for [interval_l, interval_h] in hour_interval_bins]
    hz_freq_bins = [[1/interval_h, 1/interval_l] for [interval_l, interval_h] in sec_interval_bins]
    col_names = [f"{interval_l}hr ~ {interval_h}hr" for [interval_l, interval_h] in hour_interval_bins]
    sum_hz_bins = []
    for i, bounds in enumerate(hz_freq_bins):
        hz_req_l, hz_req_h = bounds
        temp_sum = 0
        for j, hz_value in enumerate(f_values):
            if hz_value>hz_req_l and hz_value<=hz_req_h:
                temp_sum += fft_values[j]
        sum_hz_bins.append(temp_sum)
    df_sum_bins = pd.DataFrame([sum_hz_bins], columns = col_names)
    return df_sum_bins


def normalize_df_col(df, colname, scale_min=0, scale_max=1):
    '''
    Normalize a column of a dataframe to (scale_min, scale_max)
    '''
    arr_values = df[colname].values.reshape((len(df.index), 1))
    scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
    scaler = scaler.fit(arr_values)
    arr_normalized = scaler.transform(arr_values)
    df[colname] = arr_normalized
    return df


def plot_ts_all(df_ts, ls_fft, save_path=None, save_dir=None, str_title=None):
    '''
    Plots both time-series and ferquency spectrum
    '''
    # plot setting
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Time-series and frequency-domian plots: {str_title}", fontsize=12, y=1.05)
    gs = gridspec.GridSpec(2, 1, figure=fig, width_ratios=[12], height_ratios=[6, 6]) 
    gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    
    # Time series plot
    ax = fig.add_subplot(gs[0, 0]) # Entire first row
    ax.plot(df_ts.index, df_ts['Value'])
    ax.set_title('Scaled Time-series Consumption')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Normalized Consumption')
    ax.set_ylim(0, 1)

    # Frequency-domian plot
    f_values, fft_values = ls_fft
    ax1 = fig.add_subplot(gs[1, 0]) # Entire first row
    ax1.scatter(f_values, fft_values, linestyle='-', color='blue', label='')
    ax1.set_title('Frequency-Amplitude')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 0.0006)
    ax1.set_ylim(0, 0.15)

    if save_dir != None:
        plt.savefig(str(f"{os.path.join(save_dir, str_title)}.png"), dpi=80)
        plt.close()
        
    if save_path != None:
        plt.savefig(save_path, dpi=80)    
        plt.close()


def plot_ts_freq(df_ts, ls_fft, figsize=(16, 8), save_path=None, save_dir=None, str_title=None):
    f, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    f.suptitle(f"Time-series and frequency-domian plots: {str_title}", fontsize=12, y=1.02)
    # Time series plot
    axs[0].plot(df_ts.index, df_ts['Value'])
    axs[0].set_title('Scaled Time-series Consumption')
    axs[0].set_xlabel('Datetime')
    axs[0].set_ylabel('Normalized Consumption')
    # axs[0].set_xlim(0, 1.05)
    axs[0].set_ylim(0, 1.05)

    # Frequency-domian plot
    f_values, fft_values = ls_fft
    axs[1].scatter(f_values, fft_values, linestyle='-', color='blue', label='')
    axs[1].set_title('Frequency-Amplitude')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xlim(0.001, 0.0085)
    axs[1].set_ylim(0, 0.02)
    
    if save_dir != None:
        plt.savefig(str(f"{os.path.join(save_dir, str_title)}.png"), dpi=80)
        plt.close()
        
    if save_path != None:
        plt.savefig(save_path, dpi=80)    
        plt.close()


def get_daily_fft_sum_bins(df_ts, hour_interval_bins):
    '''
    Calculate daily spectrum amplitude sum at daily window
    :return: a pandas dataframe with hour bins as columns and spectrum amplitude sums in each bin at each day.
    '''
    for i, single_date in enumerate(np.unique(df_ts.date)):
        df_ts_day = df_ts.loc[df_ts.date == single_date]
        f_values, fft_values = get_load_fft(df_ts_day)
        if i == 0:
            df = get_fft_sum_by_bins(f_values, fft_values, hour_interval_bins)
        else:
            df = df.append(get_fft_sum_by_bins(f_values, fft_values, hour_interval_bins))
    df.index = np.unique(df_ts.date)
    return df

def daily_fft_sum_bins_boxplot(df_fft_bins, title_key=None, save_path=None):
    '''
    Generate boxplot for the spectrum amplitude sum bins
    '''
    plt.figure(figsize=(14,7))
    plt.title(f"Boxplot of Frequency Spectrum Amplitude Sums: {title_key}")
    sns.boxplot(data=pd.melt(df_fft_bins),
                x='variable', 
                y='value')
    plt.ylim([0, 0.3])
    plt.xlabel('Cycle Range (hour)')
    plt.ylabel('Sum of Normalized Spectrum Amplitude')
    if save_path != None:
        plt.savefig(save_path, dpi=200)
        plt.show()

    
def print_hour_interval_bin_info(hour_interval_bins):
    '''
    Print out the hourly bin Hz info
    '''
    print('= '*30)
    for i, hour_interval in enumerate(hour_interval_bins, 1):
        print(f"Bin {i}: {hour_interval[0]} Hz to {hour_interval[1]} Hour --- {round(1/(hour_interval[1]*3600), 5)} Hz to {round(1/(hour_interval[0]*3600), 5)} Hz")
    print('= '*30)