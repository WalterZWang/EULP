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
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


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