import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import silhouette_visualizer

import matplotlib.pyplot as plt
import seaborn as sns

class LoadProfileMetrics:
    DEFAULT_INTERVAL_M = 15 # Default time-interval in minutes
    DEFAUL_YEAR = 2015      # Default year of data to use
    DEFAULT_BINS = [
        [0.5, 0.75],
        [0.75, 1],
        [1, 1.25],
        [1.25, 1.5],
        [1.5, 1.75],
        [1.75, 2],
        [2, 4],
        [4, 8],
        [8, 12]
    ]
    
    def __init__(self, df_ts, area=None):
        self.df_ts = df_ts

    def __str__(self):
        start_time = self.df_ts['Datetime'].iloc[0]
        end_time = self.df_ts['Datetime'].iloc[-1]
        return f"{'-'*50}\nLoad profile between {start_time} and {end_time}\n{'-'*30}\nTop 10 rows:\n{self.df_ts.head(10)}\n{'-'*30}\nSummary:\n {self.df_ts.describe()}\n{'-'*50}"

    def scale(self, colname='Value', scale_min=0, scale_max=1):
        '''
        Scale a column of a dataframe to (scale_min, scale_max)
        '''
        df = self.df_ts.copy(deep=True)
        arr_values = df[colname].values.reshape((len(df.index), 1))
        scaler = preprocessing.MinMaxScaler(feature_range=(scale_min, scale_max))
        scaler = scaler.fit(arr_values)
        arr_scaled = scaler.transform(arr_values)
        df[colname] = arr_scaled
        self.df_scaled = df
        

    @staticmethod
    def get_fft_w_window(df_ts, window='D', year=DEFAUL_YEAR, get_bins=False, day_type='weekday', bins=DEFAULT_BINS):
        '''
        Calculate the fft for the building with specified window size
        :window: the window 
        :return: a pandas dataframe with fft bins grouped by specified window
        '''
        df_ts['Datetime'] = pd.to_datetime(df_ts['Datetime'])
        df_ts['date'] = df_ts['Datetime'].dt.date

        if day_type == 'weekday':
            df_ts = df_ts.loc[df_ts['Datetime'].dt.dayofweek < 5]
        elif day_type == 'weekend':
            df_ts = df_ts.loc[df_ts['Datetime'].dt.dayofweek >= 5]

        if year != 'All':
            df_ts = df_ts.loc[df_ts['Datetime'].dt.year == year]
        if get_bins:
            # Group fft features into bins
            v_valid_dates = []
            for i, str_date in enumerate(np.unique(df_ts['date'])):
                df_window = df_ts[df_ts['date'] == (str_date)]
                f_values, fft_values = LoadProfileMetrics.get_load_fft(df_window)
                try:
                    if i == 0:
                        df_fft = LoadProfileMetrics.get_fft_sum_by_bins(f_values, fft_values)
                    else:
                        df_fft = df_fft.append(LoadProfileMetrics.get_fft_sum_by_bins(f_values, fft_values))
                    v_valid_dates.append(str_date)
                except:
                    pass
            df_fft.insert(0, column='Date', value=v_valid_dates)
        else:
            # Get every fft feature
            values = []
            for i, str_date in enumerate(np.unique(df_ts['date'])):
                df_window = df_ts[df_ts['date'] == (str_date)]
                f_values, fft_values = LoadProfileMetrics.get_load_fft(df_window)
                row_values = [str(str_date)] + list(fft_values)
                try:
                    if len(values) == 0:
                        col_names = ['Date'] + [round(freq, 5) for freq in f_values]
                        values = [row_values]
                    else:
                        values.append(row_values)
                except:
                    pass
            df_fft = pd.DataFrame(values, columns=col_names)
        return df_fft

        
    @staticmethod
    def get_load_fft(df_t, sample_timestep=DEFAULT_INTERVAL_M):
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
        f_values, fft_values = LoadProfileMetrics.get_fft_values(y_values, T, N, f_s)
        return f_values, fft_values

    @staticmethod
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
    
    @staticmethod
    def get_fft_sum_by_bins(f_values, fft_values, hour_interval_bins=DEFAULT_BINS):
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

    @staticmethod
    def daily_fft_sum_bins_boxplot(df_fft_bins, title_key=None, save_path=None):
        '''
        Generate boxplot for the spectrum amplitude sum bins
        '''
        plt.figure(figsize=(14,7))
        plt.title(f"Boxplot of Frequency Spectrum Amplitude Sums: {title_key}")
        sns.boxplot(data=pd.melt(df_fft_bins),
                    x='variable', 
                    y='value')
        plt.ylim([0, 0.85])
        plt.xlabel('Cycle Range (hour)')
        plt.ylabel('Sum of Normalized Spectrum Amplitude')
        if save_path != None:
            plt.savefig(save_path, dpi=200)
            plt.show()
            
    @staticmethod
    def prepare_kmeans_data(df_features, standardize_features=True):
        scaler = preprocessing.StandardScaler()
        if standardize_features:
            out = scaler.fit_transform(df_features)
        else:
            out = df_features.to_numpy()
        return out
    
    
    @staticmethod
    def kmeans_elbow_plot(cluster_data, k_max=25):
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2,k_max))
        visualizer.fit(cluster_data) # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure
        

    @staticmethod
    def kmeans_silhouette_plot(cluster_data, k):
        # Use the quick method and immediately show the figure
        silhouette_visualizer(KMeans(k, random_state=42), cluster_data, colors='yellowbrick')

        