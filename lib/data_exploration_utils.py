import os
import shutil
import datetime
import glob
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import cv2
from keras import models
from keras import layers
from keras import applications
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def get_all_file_paths(dir_lookup, str_file_type, building_type=None):
    files = []
    for f in glob.glob(f'{dir_lookup}/*.{str_file_type}'):
        if building_type:
            if not building_type in f:
                continue

        files.append(os.path.abspath(f))

    return files

def clean_pge_df_ts(csv_path, year=None):
    '''
    Extract and clean PG&E time-series
    :csv_path: path to the raw PG&E time series CSV
    '''
    # Return the normalized value
    bad_data_indicator = [None, True]
    if Path(csv_path).suffix == '.parquet':
        df_t = pd.read_parquet(csv_path)
    elif Path(csv_path).suffix == '.csv':
        df_t = pd.read_csv(csv_path, parse_dates=True)
    else:
        print(f'Unexpected extension, skipping {csv_path}')
        return bad_data_indicator

    # if year != None:
    #     df_t['Datetime'] = pd.to_datetime(df_t['Datetime'])
    #     df_t = df_t[df_t['Datetime'].dt.year == year]
    #     if len(df_t.index) == 0: return df_t

    # Check that original data length is 35,040 and skip if not
    if not len(df_t) >= 0.75*35040:
        print(f'length of data was {len(df_t)}, expected at least {0.75*35040}.  Skipping {csv_path}')
        return bad_data_indicator

    # df_t = df_t.set_index(pd.DatetimeIndex(df_t['Datetime']))
    # df_t = df_t.drop(columns =['Datetime'])
    df_t = df_t.fillna(0.0)
    df_t = normalize_df_col(df_t, 'total_kWh_excluding_blanks')
    df_t['date'] = pd.DatetimeIndex(df_t.index).date

    # Downselect to full days, starting with 1/1 at midnight, ending 12/30.
    # (AMI data is 1/1-12/31 UTC, so 12/31 MST has 7 missing hours of data)
    yr = df_t.index[3000].year
    if yr == 2017:
        df_t = df_t.loc[f'{yr}-1-1':f'{yr}-12-30']
    # print(f'    after  {df_t.index[0]} to {df_t.index[-1]}')

    if not len(df_t) >= 34944:
        print(f'length of data was {len(df_t)}, expected >= 34,944.  Skipping {csv_path}')
        return bad_data_indicator

    return [df_t, False]


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

def generate_heatmap(df_ts, save_path=None):
    '''
    This function takes a pandas datatime series as input,
    and generate a heatmap where the x-axis is the day index and the y-axis is the timestamps of a day.
    '''
    groups = df_ts.groupby(pd.Grouper(freq='D')) # Group by day
    print(f'groups = {len(groups)}')
    df_plot = pd.DataFrame()
    for i, (name, group) in enumerate(groups):
        # print(f'{name} has {len(group.values)} values')
        if not len(group.values) == 96:
            # print(f'{name} has {len(group.values)} *****')
            # print(group)
            continue
        try:
            df_plot[i+1] = group.values
        except:
            pass

    # plt.figure(figsize=(5,5))
    plt.matshow(df_plot, interpolation=None, aspect='auto', fignum=1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if save_path != None:
        plt.savefig(save_path, dpi=200)
        # plt.show()
        # plt.close()


def generate_ts_html(df_t, str_title, dir_save):
    '''
    This function create a html to visualize time-series data with plotly
    '''
    fig = px.line(df_t, x=df_t.index, y='total_kWh_excluding_blanks', range_x=['2016-01-01','2017-12-31'])
    fig.update_layout(
        title={
            'text': f"Normalized consumption for {str_title}",
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Datetime",
        yaxis_title="total_kWh_excluding_blanks",
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.write_html(f"{dir_save}/{str_title}.html")


def model_vgg16_conv_base():
    '''
    This function build a pre-trained (on ImageNet) VGG16 model's convolution base
    '''
    # Use VGG16 as conv base
    conv_base = applications.VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.layers[0].trainable = False
    return model

def get_conv_base_features(img_in, model_conv_base):
    '''
    This function extracts the features from a pre-trained CNN model's convolution base
    :img_in: image file
    :model_conv_base: conv base of a keras sequential model
    :return: a numpu array of the features
    '''
    im = cv2.imread(img_in)
    im = cv2.resize(im,(224,224))
    img = preprocess_input(np.expand_dims(im.copy(), axis=0))
    conv_base_feature = model_conv_base.predict(img)
    conv_base_feature_np = np.array(conv_base_feature) # to np array so that KMeans can read
    return conv_base_feature_np