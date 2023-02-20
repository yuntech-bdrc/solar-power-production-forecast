#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load package
import os
import time
import datetime
import calendar
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import pickle


# In[2]:


# load sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn_rvm import EMRVC
from sklearn_rvm import EMRVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb


# In[3]:


import pandas as pd
import numpy as np
import datetime
import time
import import_ipynb
import read_CWB_3H as cwb
import read_OWM_3H as owm
import read_TCW_1H as twc
# import get_msm_data as getmsm
# import get_power
## 在線使用設置##############
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[ ]:





# In[4]:


msm = pd.read_csv('./MSM_data/save/solar_汙水廠_dswrfpred.csv')
msm = msm.drop_duplicates(['TIME_TO_INTERVAL'], keep="last")
msm = msm.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)
msm[11450:11500]
mask = (msm['TIME_TO_INTERVAL'] == '2022-08-12 02:00:00')
msm = msm[mask]
msm['Radiation(MSMv4)[1d]'].values[0]


# In[5]:


def all_data(latitude,longitude):
    # 抓取歷史資料
    # 晴空輻射資料
    sky_radiation = pd.read_csv('clear_sky_data/solar_汙水廠_ClearSkyRadiation.csv')
    sky_radiation['TIME_TO_INTERVAL'] = pd.to_datetime(sky_radiation['TIME_TO_INTERVAL'])
    # 歷史輻射
    cwb_rad_data = pd.read_csv('Observation_CWB/467490.csv')
    #整理CWB氣象局觀測資料
    cwb_rad_data = bulid_cwb_radiation(cwb_rad_data)
    cwb_rad_data = cwb_rad_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last")
    cwb_rad_data['TIME_TO_INTERVAL'] = pd.to_datetime(cwb_rad_data['TIME_TO_INTERVAL'])
    cwb_rad_data = cwb_rad_data[['TIME_TO_INTERVAL', 'Radiation']]
    data = pd.merge(sky_radiation,cwb_rad_data,on='TIME_TO_INTERVAL',how='outer')
    # 歷史彰師大資料
    NCUE = pd.read_csv('power_data/solar_汙水廠_history.csv')
    NCUE['TIME_TO_INTERVAL'] = pd.to_datetime(NCUE['TIME_TO_INTERVAL'])
    data = pd.merge(data,NCUE,on='TIME_TO_INTERVAL',how='outer')
    # 中興大學資料
    msm = pd.read_csv('./MSM_data/save/solar_汙水廠_dswrfpred.csv')
    msm = msm.drop_duplicates(['TIME_TO_INTERVAL'], keep="last")
    msm = msm.sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    msm['TIME_TO_INTERVAL'] = pd.to_datetime(msm['TIME_TO_INTERVAL'])
    data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
    msm1 = msm[~msm['Radiation(MSMv4)[1d]'].isna()][['TIME_TO_INTERVAL','Radiation(MSMv4)[1d]']]
    msm2 = msm[~msm['Radiation(MSMv4)[2d]'].isna()][['TIME_TO_INTERVAL','Radiation(MSMv4)[2d]']]

    data['Radiation(MSM)'] = np.nan
#     data['Radiation(MSM)'] = data.apply(lambda x: put_msm_data(x, cwb_rad_data, 'Radiation'), axis=1)
    data['Radiation(MSM)'] = data.apply(lambda x: put_msm_data(x, msm, 'Radiation(MSMv4)[1d]','Radiation(MSMv4)[2d]','Radiation(MSMv4)[3d]','Radiation(MSMv4)[4d]'), axis=1)
    data = data.sort_values(by='TIME_TO_INTERVAL')
    return data


# In[6]:


# def put_msm_data(row, cwb_rad_data, feature):
#     r = cwb_rad_data['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])
#     if (len(cwb_rad_data[r][feature]) > 0):
#         return cwb_rad_data[r][feature].values[0]
#     else:
#         return row['Radiation(MSM)']


# In[7]:


def put_msm_data(row, msm, feature, feature_2,feature_3,feature_4):
    r = msm['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])
    if ((len(msm[r][feature]>0)) and (msm[r][feature].isnull().values.any() == False) ):
        return msm[r][feature].values[0]/1000
    
    elif((len(msm[r][feature_2]>0)) and (msm[r][feature_2].isnull().values.any() == False) ):
        return msm[r][feature_2].values[0]/1000
    
    elif((len(msm[r][feature_3]>0)) and (msm[r][feature_3].isnull().values.any() == False) ):
        return msm[r][feature_3].values[0]/1000
    
    elif((len(msm[r][feature_4]>0) and msm[r][feature_4].isnull().values.any() == False) ):
        return msm[r][feature_4].values[0]/1000
    
    else:
        return row['Radiation(MSM)']


# In[8]:


#CWB氣象局觀測資料整理
def bulid_cwb_radiation(rad_raw):
    rad_raw['GloblRad'] = rad_raw['GloblRad'].apply(lambda x:rename_str(x))
    rad_raw['Precp'] = rad_raw['Precp'].apply(lambda x:rename_str(x))
    rad_raw['WDGust'] = rad_raw['WDGust'].apply(lambda x:rename_str(x))
    rad_raw['WSGust'] = rad_raw['WSGust'].apply(lambda x:rename_str(x))
    rad_raw['WD'] = rad_raw['WD'].apply(lambda x:rename_str(x))
    rad_raw['WS'] = rad_raw['WS'].apply(lambda x:rename_str(x))
    rad_raw['RH'] = rad_raw['RH'].apply(lambda x:rename_str(x))
    rad_raw['Td dew point'] = rad_raw['Td dew point'].apply(lambda x:rename_str(x))
    rad_raw['Temperature'] = rad_raw['Temperature'].apply(lambda x:rename_str(x))
    rad_raw['SeaPres'] = rad_raw['SeaPres'].apply(lambda x:rename_str(x))
    rad_raw['Visb'] = rad_raw['Visb'].apply(lambda x:rename_str(x))
    rad_raw['UVI'] = rad_raw['UVI'].apply(lambda x:rename_str(x))
    rad_raw['Cloud Amount'] = rad_raw['Cloud Amount'].apply(lambda x:rename_str(x))
    #1kwh=3.6mj
#     1w = 0.0036mj
    rad_raw['GloblRad'].astype(float)
    rad_raw['GloblRad'] = rad_raw['GloblRad']/3.6
#     rad_raw['GloblRad'] = rad_raw['GloblRad']/0.0036
    rad_raw['ObsTime'] = rad_raw['ObsTime']-1
    rad_raw['TIME_TO_INTERVAL'] = rad_raw.apply(lambda raw:'{} {:02d}:00:00'.format(raw['date'], raw['ObsTime']), axis=1)
    rad_raw = rad_raw[['TIME_TO_INTERVAL', 'date', 'ObsTime', 'GloblRad', 
                       'Precp', 'WDGust', 'WSGust', 
                       'WD', 'WS', 'RH', 'Td dew point', 'Temperature', 
                       'SeaPres', 'Visb', 'UVI', 'Cloud Amount']]
    rad_raw = rad_raw.rename(columns={'ObsTime':'Hour', 'GloblRad':'Radiation', 'date':'Date'})
    return rad_raw

def rename_str(X):
    if X=='/':
        return np.nan
    elif X=='X':
        return np.nan
    elif X=='...':
        return np.nan
    elif X=='T':
        return np.nan
    else:
        return float(X)


# In[9]:


# data merged into 3 hour units
def merge_data_to_3_hour(raw, add=[]):
    group_by_3h = ['TIME_TO_INTERVAL', 'Date', 'Hour' ]+add
    data_3h = raw.copy()
    data_3h['TIME_TO_INTERVAL'] = pd.to_datetime(data_3h['TIME_TO_INTERVAL'])
    data_3h['Date'] = data_3h['TIME_TO_INTERVAL'].dt.date
    data_3h['Hour'] = data_3h['TIME_TO_INTERVAL'].dt.hour
    data_3h['TIME_TO_INTERVAL'] = pd.to_datetime(data_3h.apply(
        lambda row: '{} {:02d}:00:00'.format(row['Date'], (row['Hour'])//3*3), axis=1))
    data_3h['Hour'] = data_3h['TIME_TO_INTERVAL'].dt.hour
    data_3h = data_3h.groupby(group_by_3h).mean().reset_index()
    return data_3h
# data merged into 1 hour units
def merge_data_to_1_hour(raw):
    data_1h = raw.copy()
    data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(data_1h['TIME_TO_INTERVAL'])
    data_1h['Date'] = data_1h['TIME_TO_INTERVAL'].dt.date
    data_1h['Hour'] = data_1h['TIME_TO_INTERVAL'].dt.hour
    data_1h['Minute'] = data_1h['TIME_TO_INTERVAL'].dt.minute
    data_1h['TIME_TO_INTERVAL_1h'] = pd.to_datetime(data_1h.apply(
        lambda row: '{} {:02d}:{:02d}:00'.format(row['Date'], (row['Hour']), row['Minute']), axis=1))
    data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(data_1h.apply(
        lambda row: '{} {:02d}:{:02d}:00'.format(row['Date'], (row['Hour'])//3*3, row['Minute']), axis=1))
    return data_1h


# In[10]:


weather_data = pd.read_csv('CWB.3H/save/CWB.3H.Merge.Multiple.csv')
weather_data


# In[11]:


def calculate_similar_day(start, data_1h, data_3h, plant_info, latitude, longitude):
    sdv3 = {}
    sdv3['CWB'] = int(plant_info['CWB'][0])
    sdv3['IBM'] = int(plant_info['TWC'][0])
    sdv3['OWM'] = int(plant_info['OWM'][0])
    # 氣象局預報
    keys = ['CWB', 'IBM', 'OWM']
    for key in keys:
        print(sdv3['CWB'])
        if key == 'CWB':
            weather_data = pd.read_csv('CWB.3H/save/CWB.3H.Merge.Multiple.csv')
        elif key == 'IBM':
            weather_data = pd.read_csv('WeatherChannel.1H/save/IBM.1H.Merge.Multiple(merge).csv')
        else:
            weather_data = pd.read_csv('OpenWeatherMap.3H/save/OWM.3H.Merge.Multiple(merge).csv')
            weather_data['TIME_TO_INTERVAL'] = pd.to_datetime(weather_data['TIME_TO_INTERVAL'])+datetime.timedelta(hours=1)
            weather_data['TIME_TO_INTERVAL'] = pd.to_datetime(weather_data['TIME_TO_INTERVAL'])


        weather_data = build_weather_service_data(weather_data, longitude, latitude, key)
        weather_data = weather_data[['TIME_TO_INTERVAL', 'TimeAhead', 'WeatherType']]
    #     print(2)
        weather_data_24h_ahead = weather_data[weather_data['TimeAhead'].eq(24)].reset_index(drop=True)  
        weather_data = weather_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last", ignore_index=True)
        
        #將WeatherType欄位填入天氣預報類型，若該時間沒預報資料則以晴天未預設
        data_3h['WeatherType'] = data_3h.apply(
            lambda x: apply_weather_type(start, data_1h, key, x, weather_data), axis=1)
        data_3h['WeatherType(pred)[1]'] = data_3h.apply(
            lambda x: apply_weather_type(start, data_1h, key, x, weather_data_24h_ahead), axis=1)
        data_3h['Alpha[1]'] = 1
        data_3h[f'WeatherType({key})'] = data_3h['WeatherType']
        data_3h[f'WeatherType(pred)({key})'] = data_3h['WeatherType(pred)[1]']
        
        #透過天氣類型來計算相似日的輻射值
        data_1h['ClearSkyIndex'] = data_1h.apply(lambda row: ClearSkyIndex(row), axis=1)
        data_1h[f'Radiation(SDv3)({key})'] = data_1h.apply(lambda row: similar_day_radiation_v3(row, data_1h, data_3h, sdv3[key]), axis=1)
        data_1h[f'Radiation(SDv3)({key})'] = (data_1h[f'Radiation(SDv3)({key})']*data_1h['ClearSkyRadiation'])
        data_1h[f'Radiation(today)({key})'] = data_1h.apply(lambda row: similar_today_radiation_v3(row, data_1h, data_3h, sdv3[key]), axis=1)
        data_1h[f'Radiation(today)({key})'] = (data_1h[f'Radiation(today)({key})']*data_1h['ClearSkyRadiation'])
        # print(4)
    #將三小時區間變成1小時區間
    data_3h = pd.concat([data_3h]*3)
    data_3h = data_3h.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)
    for i in range(len(data_3h)):
        data_3h.loc[i, 'TIME_TO_INTERVAL'] = data_3h.loc[i]['TIME_TO_INTERVAL'] + datetime.timedelta(hours=i % 3)

    data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(data_1h.apply(
        lambda row: '{} {:02d}:00:00'.format(row['Date'], (row['Hour'])), axis=1))

    return data_1h, data_3h


# In[12]:


def ClearSkyIndex(row):
    data_1h = row.copy()
    if(data_1h['ClearSkyRadiation'] > 0):
        data_ClearSky = data_1h['Radiation']/data_1h['ClearSkyRadiation']
    else:
        data_ClearSky = 0
    return data_ClearSky


# In[13]:


#將各案場的WeatherType改成相同天氣類型
def build_weather_service_data(raw, longitude, latitude, service='CWB'):
    data = raw.copy()
    
    # read file and rename
    if(service=='CWB'):
        data['Location'] = data['LocationName'] + data['CityName']
    elif(service=='IBM'):
        data['Location'] = data['Name']
    elif(service=='OWM'):
        data['Location'] = data['Name']
        
    data = data.drop_duplicates(['TIME_TO_INTERVAL', 'Location', 'TimeAhead'], keep="last")
    
    # select data by hour
    data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
    if(service=='OWM'):
        data['TIME_TO_INTERVAL'] = data['TIME_TO_INTERVAL']+ datetime.timedelta(hours=1)
    
    data['Hour'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.hour
#     data = data[data['Hour'].isin(range(6, 17+1))]
    
    if(service=='CWB'):
        data['WeatherType'] = data['WeatherType'].replace({
            '午後短暫雷陣雨': '短暫陣雨或雷雨',
            '有雨': '陣雨',
            '短暫雨': '短暫陣雨',
            '午後短暫陣雨': '短暫陣雨'})
        data['WeatherType'] = data['WeatherType'].replace({
            '短暫陣雨或雷雨': '陰',
            '短暫陣雨': '陰',
            '陣雨或雷雨': '陣雨'
        })
        
        pass
    elif(service=='OWM'):
# #         能見度
        data['WeatherType'] = data['WeatherType'].replace({
            '晴，少雲':'晴',
            '多雲':'晴',
            '陰，多雲':'晴',
            '小雨':'晴',
        })
        #rename
        data['WeatherType'] = data['WeatherType'].replace({
            '小雪':'陣雨',
            '大雨':'陰',
            '中雨':'多雲',
        })

        pass
    
    # find the recent forecast point, then get forecast data from recent point
    locus = data.drop_duplicates(['Location'], keep="last").reset_index(drop=True)
    #計算距離案場最近的點
    recent = get_recent_target(longitude, latitude, locus)
    print(recent)
    data = data[data['Location'].eq(recent)].reset_index(drop=True)
    
    # info output
    print('服務預報點：', recent)
    print(data.drop_duplicates(['WeatherType'], keep="last")['WeatherType'])
    data = data.sort_values(by=['TIME_TO_INTERVAL', 'TimeAhead'])
    return data 


# In[14]:


from math import radians, cos, sin, asin, sqrt
# calculate distance based on latitude and longitude
def geodistance(lon_a, lat_a, lon_b, lat_b):
    lon_a, lat_a, lon_b, lat_b = map(radians, [lon_a, lat_a, lon_b, lat_b])
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    a = sin(dlat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dlon/2)**2
    dis = 2*asin(sqrt(a))*6371*1000
    return dis
def get_recent_target(longitude, latitude, locus, column='Location'):
    # initialization
    # recent_point: 表示距離輸入 plant 最近的 "預報點"，最終會被回傳
    # shortest_dist: 表示表示距離輸入 station 最近 "預報點" 的距離，初始值設很大是為了避免一直寫入
    recent_point = 'Not Found.'
    shortest_dist = 1000*1000

    # go through the search list
    for i in range(len(locus)):
        current = locus.loc[i]
        dist = geodistance(
            longitude, latitude, 
            float(current['Longitude']), float(current['Latitude']))

        # if the current distance is shorter than the historical shortest distance
        # then use current point replace recent point
        if(dist < shortest_dist):
            shortest_dist = dist
            recent_point = current[column]
    # end of search, return
    return recent_point


# In[15]:


# apply weather type from forecast data(cwb, weather.com, open weather map).
def apply_weather_type(start, data_1h, key, row, weather_forecast, default='晴'):
    ''' apply weather type from forecast data(cwb, weather.com, open weather map).
    '''
    if pd.to_datetime(row['TIME_TO_INTERVAL']) >= start:
        
        mask = weather_forecast['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])
        if(mask.any()):
#             print(f">= start: {start}, {row['TIME_TO_INTERVAL']}, {weather_forecast[mask]['WeatherType'].values[0]}")
            return weather_forecast[mask]['WeatherType'].values[0]
        else:
            print(f'Not Found.>= start: {start}', row['TIME_TO_INTERVAL'])
              #若不存在取代為nan
    #         return np.nan
            return '晴'
    else:
        mask = data_1h['TIME_TO_INTERVAL'].eq(pd.to_datetime(row['TIME_TO_INTERVAL']))
        if(mask.any()):
#             print(f"< start: {start}, {row['TIME_TO_INTERVAL']}, {data_1h[mask][f'WeatherType({key})'].values[0]}")
            return data_1h[mask][f'WeatherType({key})'].values[0]
        else:
            print(f'Not Found. < start: {start}', row['TIME_TO_INTERVAL'])
              #若不存在取代為nan
    #         return np.nan
            return '晴'


# In[16]:


def similar_day_radiation_v3(row, raw_1h, raw_3h, limit=10):
    # init
    data_1h = raw_1h.copy()
    data_3h = raw_3h.copy()

    # 篩選出這筆小時單位資料，在三小時單位的位置(至多1筆)
    mask = data_3h['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])
#     print(row['TIME_TO_INTERVAL'])
#     print(data_3h[mask])
    row_3h = data_3h[mask].iloc[0]

    # 篩選出所有三小時單位資料當中，和目前資料相同時間段(3小時單位)的資料
    mask = data_3h['Hour'].eq(row_3h['Hour'])
    date_in_same_zone = data_3h[mask].reset_index(inplace=False, drop=True)

    # 篩選出三小時資料當中，早於當前時間的資料
    mask = (date_in_same_zone['TIME_TO_INTERVAL']<row_3h['TIME_TO_INTERVAL'])
    date_in_same_zone = date_in_same_zone[mask]

    # 取得相同時間帶內，同樣天氣類型的資料
    target_weather = row_3h['WeatherType(pred)[1]']
    date_with_same_weather = date_in_same_zone[date_in_same_zone['WeatherType'].eq(target_weather)]
    date_with_same_weather = date_with_same_weather.reset_index(inplace=False, drop=True)
    date_with_same_weather = date_with_same_weather['TIME_TO_INTERVAL'].tolist()

    # 找到相似日的日期後，切換回 1 小時單位取樣
    mask1 = data_1h['TIME_TO_INTERVAL'].isin(date_with_same_weather)
    mask2 = data_1h['Hour'].eq(row['Hour'])
    available_data = data_1h[mask1 & mask2]

    # 反轉資料時間順序，以利取得相對目標日而言較近的資料
    # 移除有缺值的日子，不要被計入
    available_data = available_data.iloc[::-1].reset_index(inplace=False, drop=True)
#     available_data = available_data[~available_data['Radiation'].isna()]
    available_data = available_data[~available_data['Radiation'].isna()]

    # 取用最佳化的相似日數量，最佳化的數字必須由使用者提供
    available_data = available_data[:limit]
#     print(available_data)
    if(len(available_data)>0):
        available_data = available_data['ClearSkyIndex'].mean()
        return available_data
    else:
        if row['ClearSkyRadiation']!=0:
            return row['Radiation(MSM)']/row['ClearSkyRadiation']
        else:
            return 0   


# In[17]:


def similar_today_radiation_v3(row, raw_1h, raw_3h, limit=10):
    # init
    data_1h = raw_1h.copy()
    data_3h = raw_3h.copy()

    # 篩選出這筆小時單位資料，在三小時單位的位置(至多1筆)
    mask = data_3h['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])
#     print(row['TIME_TO_INTERVAL'])
#     print(data_3h[mask])
    row_3h = data_3h[mask].iloc[0]

    # 篩選出所有三小時單位資料當中，和目前資料相同時間段(3小時單位)的資料
    mask = data_3h['Hour'].eq(row_3h['Hour'])
    date_in_same_zone = data_3h[mask].reset_index(inplace=False, drop=True)

    # 篩選出三小時資料當中，早於當前時間的資料
    mask = (date_in_same_zone['TIME_TO_INTERVAL']<row_3h['TIME_TO_INTERVAL'])
    date_in_same_zone = date_in_same_zone[mask]

    # 取得相同時間帶內，同樣天氣類型的資料
    target_weather = row_3h['WeatherType']
    date_with_same_weather = date_in_same_zone[date_in_same_zone['WeatherType'].eq(target_weather)]
    date_with_same_weather = date_with_same_weather.reset_index(inplace=False, drop=True)
    date_with_same_weather = date_with_same_weather['TIME_TO_INTERVAL'].tolist()

    # 找到相似日的日期後，切換回 1 小時單位取樣
    mask1 = data_1h['TIME_TO_INTERVAL'].isin(date_with_same_weather)
    mask2 = data_1h['Hour'].eq(row['Hour'])
    available_data = data_1h[mask1 & mask2]

    # 反轉資料時間順序，以利取得相對目標日而言較近的資料
    # 移除有缺值的日子，不要被計入
    available_data = available_data.iloc[::-1].reset_index(inplace=False, drop=True)
#     available_data = available_data[~available_data['Radiation'].isna()]
    available_data = available_data[~available_data['Radiation'].isna()]

    # 取用最佳化的相似日數量，最佳化的數字必須由使用者提供
    available_data = available_data[:limit]
#     print(available_data)
    if(len(available_data)>0):
        available_data = available_data['ClearSkyIndex'].mean()
        return available_data
    else:
        if row['ClearSkyRadiation']!=0:
            return row['Radiation(MSM)']/row['ClearSkyRadiation']
        else:
            return 0   


# In[18]:


def data_merge(start, data_1h, data_3h):
    data_1h = data_1h[['TIME_TO_INTERVAL','Power',
                       'Radiation', 'ClearSkyRadiation',
                       'Radiation(SDv3)(CWB)','Radiation(SDv3)(IBM)',
                       'Radiation(SDv3)(OWM)','Radiation(MSM)',
                       'Radiation(today)(CWB)','Radiation(today)(IBM)',
                       'Radiation(today)(OWM)'
                      ]]
    data_3h = data_3h[['TIME_TO_INTERVAL', 
                       'WeatherType(CWB)', 'WeatherType(pred)(CWB)',
                       'WeatherType(IBM)', 'WeatherType(pred)(IBM)',
                       'WeatherType(OWM)', 'WeatherType(pred)(OWM)',
                      ]]

    data = pd.merge(data_1h, data_3h, on=['TIME_TO_INTERVAL'], how='outer').sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    data = data.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep="last")
    data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
    data = data.sort_values(by='TIME_TO_INTERVAL')

    return data


# In[19]:


def Weather(data):
    #讀取資料
    CWB_weather_data = pd.read_csv('./CWB.3H/save/CWB.3H.Merge.Multiple.csv')
    IBM_weather_data = pd.read_csv('./WeatherChannel.1H/save/IBM.1H.Merge.Multiple(merge).csv')
    OWM_weather_data = pd.read_csv('./OpenWeatherMap.3H/save/OWM.3H.Merge.Multiple(merge).csv')
    #抓取最近距離
    CWB_weather_data = get_recent(CWB_weather_data, longitude, latitude, 'CWB')
    IBM_weather_data = get_recent(IBM_weather_data, longitude, latitude, 'IBM')
    OWM_weather_data = get_recent(OWM_weather_data, longitude, latitude, 'OWM')
    #抓出想要欄位，並重新命名
    CWB_weather_data = CWB_weather_data[['TIME_TO_INTERVAL', 'ApparentTemperature(pred)', 'Temperature(pred)','RelativeHumidity(pred)']]
    CWB_weather_data = CWB_weather_data.rename(columns={'ApparentTemperature(pred)':'ApparentTemperature(pred)[CWB]',
                                                        'Temperature(pred)':'Temperature(pred)[CWB]',
                                                        'RelativeHumidity(pred)':'RelativeHumidity(pred)[CWB]'})
    IBM_weather_data = IBM_weather_data[['TIME_TO_INTERVAL', 'FeelsLikeTemperature(pred)', 'Temperature(pred)','RelativeHumidity(pred)']]
    IBM_weather_data = IBM_weather_data.rename(columns={'FeelsLikeTemperature(pred)':'FeelsLikeTemperature(pred)[IBM]',
                                                        'Temperature(pred)':'Temperature(pred)[IBM]',
                                                        'RelativeHumidity(pred)':'RelativeHumidity(pred)[IBM]'})
    OWM_weather_data = OWM_weather_data[['TIME_TO_INTERVAL', 'FeelsLikeTemperature(pred)', 'Temperature(pred)','RelativeHumidity(pred)']]
    OWM_weather_data = OWM_weather_data.rename(columns={'FeelsLikeTemperature(pred)':'FeelsLikeTemperature(pred)[OWM]',
                                                        'Temperature(pred)':'Temperature(pred)[OWM]',
                                                        'RelativeHumidity(pred)':'RelativeHumidity(pred)[OWM]'})
    OWM_weather_data['TIME_TO_INTERVAL'] = pd.to_datetime(OWM_weather_data['TIME_TO_INTERVAL'])+datetime.timedelta(hours=1)
    #刪除重複值(取最後一筆，表示預測資料)
    CWB_weather_data = CWB_weather_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last", ignore_index=True)
    IBM_weather_data = IBM_weather_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last", ignore_index=True)
    OWM_weather_data = OWM_weather_data.drop_duplicates(['TIME_TO_INTERVAL'], keep="last", ignore_index=True)
    #抓取想要的時間點
    CWB_mask = (CWB_weather_data['TIME_TO_INTERVAL'] >= '2021-04-01')
    IBM_mask = (IBM_weather_data['TIME_TO_INTERVAL'] >= '2021-04-01')
    OWM_mask = (OWM_weather_data['TIME_TO_INTERVAL'] >= '2021-04-01')

    CWB_weather_data = CWB_weather_data[CWB_mask]
    IBM_weather_data = IBM_weather_data[IBM_mask]
    OWM_weather_data = OWM_weather_data[OWM_mask]
    #因資料是每三小時一筆，改成每小時一筆
    CWB_weather_data_1h = pd.concat([CWB_weather_data]*3)
    CWB_weather_data_1h = CWB_weather_data_1h.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)
    IBM_weather_data_1h = IBM_weather_data
    IBM_weather_data_1h = IBM_weather_data_1h.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)
    OWM_weather_data_1h = pd.concat([OWM_weather_data]*3)
    OWM_weather_data_1h = OWM_weather_data_1h.sort_values(by=['TIME_TO_INTERVAL']).reset_index(drop=True)

    CWB_weather_data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(CWB_weather_data_1h['TIME_TO_INTERVAL'])
    for i in range(len(CWB_weather_data_1h)):
        CWB_weather_data_1h.loc[i, 'TIME_TO_INTERVAL'] = CWB_weather_data_1h.loc[i]['TIME_TO_INTERVAL'] + datetime.timedelta(hours=i % 3)

    IBM_weather_data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(IBM_weather_data_1h['TIME_TO_INTERVAL'])
    for i in range(len(IBM_weather_data_1h)):
        IBM_weather_data_1h.loc[i, 'TIME_TO_INTERVAL'] = IBM_weather_data_1h.loc[i]['TIME_TO_INTERVAL'] + datetime.timedelta(hours=i % 3)

    OWM_weather_data_1h['TIME_TO_INTERVAL'] = pd.to_datetime(OWM_weather_data_1h['TIME_TO_INTERVAL'])
    for i in range(len(OWM_weather_data_1h)):
        OWM_weather_data_1h.loc[i, 'TIME_TO_INTERVAL'] = OWM_weather_data_1h.loc[i]['TIME_TO_INTERVAL'] + datetime.timedelta(hours=i % 3)

    merge_weather = pd.merge(CWB_weather_data_1h, IBM_weather_data_1h, on=['TIME_TO_INTERVAL'], how='outer').sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    merge_weather = pd.merge(merge_weather, OWM_weather_data_1h, on=['TIME_TO_INTERVAL'], how='outer').sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    merge_weather

    data = pd.merge(data, merge_weather, on=['TIME_TO_INTERVAL'], how='outer').sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    return data


# In[20]:


def get_recent(raw, longitude, latitude, service='CWB'):
    data = raw.copy()
    # read file and rename
    if(service=='CWB'):
        data['Location'] = data['LocationName'] + data['CityName']
    elif(service=='IBM'):
        data['Location'] = data['Name']
    elif(service=='OWM'):
        data['Location'] = data['Name']
        
    data = data.drop_duplicates(['TIME_TO_INTERVAL', 'Location', 'TimeAhead'], keep="last")
    
    # find the recent forecast point, then get forecast data from recent point
    locus = data.drop_duplicates(['Location'], keep="last").reset_index(drop=True)
    recent = get_recent_target(longitude, latitude, locus)
    print(recent)
    data = data[data['Location'].eq(recent)].reset_index(drop=True)
    
    # info output
    print('服務預報點：', recent)
    data = data.sort_values(by=['TIME_TO_INTERVAL', 'TimeAhead'])
    return data 


# In[21]:


plant_info = pd.read_csv('Plant_Info_Baoshan.csv')
plant_info = plant_info.loc[1:1].reset_index(drop=True)
# 案場資料
latitude = plant_info['Latitude'][0]
longitude = plant_info['Longitude'][0]
#抓取彰師大發電量歷史資料、觀測站歷史資料、晴空輻射歷史資料
merge_data = all_data(latitude,longitude)
print(merge_data.tail(100))
merge_data['TIME_TO_INTERVAL'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL'])
merge_data['Hour'] = merge_data['TIME_TO_INTERVAL'].dt.hour
merge_data['Date'] = merge_data['TIME_TO_INTERVAL'].dt.date
#抓取全部時間資料
merge_data = merge_data[merge_data['Hour'].isin(range(0,24))]
start_date = '2021-05-03'
start_date = pd.to_datetime(start_date)
mask = (merge_data['TIME_TO_INTERVAL'] >= start_date)
merge_data = merge_data[mask]
merge_data = merge_data.reset_index(drop=True)
#分成三小時版和一小時版
data_3h = merge_data_to_3_hour(merge_data)
data_1h = merge_data_to_1_hour(merge_data) 
#依據天氣類型去計算相似日的輻射值
data_1h, data_3h = calculate_similar_day(start_date, data_1h, data_3h, plant_info, latitude, longitude)
#將預測的相似輻射值和真實的歷史資料合併data
print('data_1h:',data_1h)
print('data_3h:',data_3h)
#合併3小時區間和1小時區間的資料
data = data_merge(start_date, data_1h, data_3h)
#加入天氣類型(ex：體感溫度,溫度,濕度)
data  = Weather(data)


# In[22]:


data_1h.tail(200)


# In[23]:


data_3h.tail(100)


# In[24]:


data.tail(250)


# In[25]:


data.to_csv('./Dataset/solar_汙水廠(history).csv', index=None)


# # pred_power(3to010)

# In[26]:


merge_raw = pd.read_csv(f'Dataset/solar_汙水廠(history).csv', low_memory=False)


# In[ ]:





# In[27]:


merge_data


# In[28]:


data.head()


# In[29]:


data_3h


# In[30]:


data_1h.head(50)


# In[31]:


merge_data
test ='2021-05-03'
test = pd.to_datetime(test)
merge_data['TIME_TO_INTERVAL'] = pd.to_datetime(merge_data['TIME_TO_INTERVAL'])
mask = (merge_data['TIME_TO_INTERVAL'] >= test)
merge_data = merge_data[mask]
merge_data.head(30)


# In[ ]:




