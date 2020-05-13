import os
import shutil
import datetime

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


def get_all_file_paths(dir_lookup, str_file_type):
    return [os.path.join(dir_lookup, f) for f in os.listdir(dir_lookup) if f.endswith(str_file_type)]


def clean_pge_df_ts(csv_path, year=None):
    ''' 
    Extract and clean PG&E time-series
    :csv_path: path to the raw PG&E time series CSV
    '''
    # Return the normalized value
    df_t = pd.read_csv(csv_path, parse_dates=True)
    if year != None:
        df_t['Datetime'] = pd.to_datetime(df_t['Datetime'])
        df_t = df_t[df_t['Datetime'].dt.year == year]
        if len(df_t.index) == 0: return df_t
    
    df_t = df_t.set_index(pd.DatetimeIndex(df_t['Datetime']))
    df_t = df_t.drop(columns =['Datetime'])
    df_t = normalize_df_col(df_t, 'Value')
    df_t['date'] = pd.DatetimeIndex(df_t.index).date
    return df_t


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
    df_plot = pd.DataFrame()
    for i, (name, group) in enumerate(groups):
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
        plt.show()
        plt.close()


def generate_ts_html(df_t, str_title, dir_save):
    '''
    This function create a html to visualize time-series data with plotly
    '''
    fig = px.line(df_t, x=df_t.index, y='Value', range_x=['2015-01-01','2015-12-31'])
    fig.update_layout(
        title={
            'text': f"Normalized consumption for {str_title}",
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Datetime",
        yaxis_title="Value",
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