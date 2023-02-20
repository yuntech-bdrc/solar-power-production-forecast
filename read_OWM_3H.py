#!/usr/bin/env python
# coding: utf-8

# # 資料來源
# ##### https://openweathermap.org/forecast5

# In[1]:


import os
import json
import glob
import pandas as pd
import numpy as np
import calendar
import datetime
from datetime import timedelta


# In[2]:


# 儲存在 Windows 和 Ubuntu 的檔案名稱不同，在計算下載時間的時候會錯誤
# 透過程式將 2 種命名格式調整至固定格式
def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]
def rebuild_crawler_time(string):
    try:
        new_string = string.replace(".txt", "")
        new_string = string.replace(".csv", "")
        new_string = new_string.replace('%3A', ':')
        new_string = new_string.replace('_', ':')
        new_string = new_string[len(new_string)-19:len(new_string)-0]
#         print(new_string)
        if(new_string.count(':')>2):
            new_string = replacer(new_string, " ", new_string.find(':'))
        new_string = pd.to_datetime(new_string)
#         print(new_string)
    except:
        new_string = ''
    return new_string


# # 打包 3 小時預報資料
# ### 請先備份並下載原始預報資料

# In[3]:


# def aggregate_data(target_date, directory='.\\OpenWeatherMap.3H'):
#     array = []

#     #讀取資料夾
#     for folder in os.listdir(directory):
#     #     print(folder)
#         #跳過
#         if(folder=='Save'):
#             continue

#         #讀取csv 
#         for filename in glob.glob(f'{directory}\\{folder}\\*.csv'):
#             filetime = rebuild_crawler_time(filename.split('\\')[3])
#             start, end = target_date, target_date+datetime.timedelta(days=1)
#             if end>filetime>start:
#                 try:
# #                     print(filetime)
#                     temp = pd.read_csv(filename)
#                     temp['CrawlerTime'] = filetime
#                     array.append(temp)
#                 except:
#             #             print(filename)
#                     pass

#     package = pd.concat(array, axis=0, ignore_index=True)
#     package.to_csv(f'{directory}/Save/OWM.3H.Raw(new).csv', index=None)
#     return package


# In[4]:


def aggregate_data2(start_date, end_date, directory='.\\OpenWeatherMap.3H'):
    array = []

    #讀取資料夾
    for folder in os.listdir(directory):
    #     print(folder)
        #跳過
        if(folder=='Save'):
            continue

        #讀取csv 
        for filename in glob.glob(f'{directory}\\{folder}\\*.csv'):
            filetime = rebuild_crawler_time(filename.split('\\')[3])
#             start, end = target_date, target_date+datetime.timedelta(days=1)
            if end_date>filetime>start_date:
                try:
#                     print(filetime)
                    temp = pd.read_csv(filename)
                    temp['CrawlerTime'] = filetime
                    array.append(temp)
                except:
            #             print(filename)
                    pass

    package = pd.concat(array, axis=0, ignore_index=True)
    package.to_csv(f'{directory}/Save/OWM.3H.Raw(new).csv', index=None)
    return package


# In[5]:


# tdate = pd.to_datetime('2022-04-18')
# forecast = aggregate_data(tdate)


# # 整理 3 小時預報資料
# ### 請先將個別的原始檔案彙整成 1 個檔案，再執行本程式
# ### 包含時間轉換、重新命名、去除重覆資料.....等

# In[6]:


from datetime import datetime
def unix_to_datetime(unix):
    unix = int(unix)
    return datetime.utcfromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S')
def uct_to_local_time(utc_time):
    os.environ['TZ']='Asia/Taipei'
    d = datetime.strptime(utc_time,"%Y-%m-%d %H:%M:%S")
    utc_ts = calendar.timegm(d.utctimetuple())
    return datetime.fromtimestamp(utc_ts).replace(microsecond=d.microsecond)


# In[7]:


# def rebuild_crawler_time(string):
#     new_string = string.replace(".csv", "")
#     new_string = new_string[len(new_string)-26:len(new_string)-7].replace("_", " ")
#     t1, t2 = new_string[:10], new_string[10:].replace('-', ':')
#     try:
#         new_string = pd.to_datetime(t1+t2)
#     except:
#         new_string = ''
#     return new_string


# In[8]:


def sort_data_3h():
    # 讀取原始預報資料：剛從原始檔案彙整起來，尚未處理過的檔案
    file = 'OpenWeatherMap.3H/Save/OWM.3H.Raw(new).csv'
    package = pd.read_csv(file, low_memory=False)

    # 原始檔案的變數名稱有很多縮寫，為避免未來看不懂，故更改成較完整的名稱
    # num = 預測天數的序號(可以拿來排序預測週期的長短)
    package = package.rename(columns={
        'id': 'ID', 'name': 'Name',
        'dt': 'TIME_TO_INTERVAL',
        'lat': 'Latitude', 'lon': 'Longitude',
        'feels_like': 'FeelsLikeTemperature(pred)',
        'temp': 'Temperature(pred)',
        'temp_min': 'MaxTemperature(pred)',
        'temp_max': 'MinTemperature(pred)',
        'pop': 'PoP(pred)', 
        'rain': 'Rain(pred)',
        'wind_speed': 'WindSpeed(pred)',
        'wind_deg': 'WindDirection(pred)',
        'humidity': 'RelativeHumidity(pred)',
        'pressure': 'Pressure(pred)',
        'sea_level': 'AtmosphericPressureSea(pred)',
        'grnd_level': 'AtmosphericPressureGround(pred)',
        'visibility': 'Visibility(pred)',
        'weather_description': 'WeatherType',
        'weather_main': 'WeatherType(main)',
        'weather_id': 'WeatherType(index)'
    }, inplace=False)
    package = package.drop(['Unnamed: 0', 'temp_kf', 'snow', 'dt_txt'], axis=1, errors='ignore')

    # 透過 "uct_to_local_time()" 轉換成相同時區
    package['TIME_TO_INTERVAL'] = package['TIME_TO_INTERVAL'].apply(lambda x: uct_to_local_time(unix_to_datetime(x)))
    package['TIME_TO_INTERVAL'] = pd.to_datetime(package['TIME_TO_INTERVAL'])
    package['CrawlerTime'] = pd.to_datetime(package['CrawlerTime'])
    package = package.sort_values(by=['ID', 'TIME_TO_INTERVAL', 'CrawlerTime'], inplace=False)
    package = package.reset_index(inplace=False, drop=True)

    # Convert temperature unit
    package['Temperature(pred)'] = (package['Temperature(pred)']-273.15)
    package['MaxTemperature(pred)'] = (package['MaxTemperature(pred)']-273.15)
    package['MinTemperature(pred)'] = (package['MinTemperature(pred)']-273.15)
    package['FeelsLikeTemperature(pred)'] = (package['FeelsLikeTemperature(pred)']-273.15)

    # 早期有根據天泰案場的經緯度，抓幾個特定案場的預報，但是後來取消了
    # 為了資料的一致性，要過濾掉這些案場早期蒐集的預報資料
    package = package[~package['ID'].isnull()]
    package = package[~package['ID'].isin(['TCG01', 'THCY01', 'TJI01', 'TZQ01'])]

    # 移除重覆的資料(1)： "TIME_TO_INTERVAL"(資料時間) & "CrawlerTime" 相同(預測時間)
    drop = package.copy()
    drop = drop.sort_values(by=['ID', 'TIME_TO_INTERVAL', 'CrawlerTime'])
    drop = drop.drop_duplicates(['ID', 'TIME_TO_INTERVAL', 'CrawlerTime'], keep="last")
    drop = drop.reset_index(inplace=False, drop=True)
    print(len(package)-len(drop))

    file = 'OpenWeatherMap.3H/Save/OWM.3H.Merge.Raw(new).csv'
    drop.to_csv(file, index=False)
    return drop


# In[9]:


# drop = sort_data_3h()


# # 整合新舊資料 & 填充缺值
# ### 將本次整理的 3 小時預報和先前的合併
# ### 同時，根據預測時間，將資料分為 24 小時內和 24 小時前的預報

# In[10]:


# 根據提前預報的時間，將資料分成:「 24小時前 」,「 24小時內」
def build_multiple_lead_time_data(data):
    forecast = data.copy()
    forecast['TIME_TO_INTERVAL'] = pd.to_datetime(forecast['TIME_TO_INTERVAL'])
    forecast['CrawlerTime'] = pd.to_datetime(forecast['CrawlerTime'])
    forecast['TimeAhead'] = forecast['TIME_TO_INTERVAL'] - forecast['CrawlerTime']
    forecast['DayAhead'] = forecast['TIME_TO_INTERVAL'].dt.date - forecast['CrawlerTime'].dt.date
    forecast.sort_values(by=['ID', 'TIME_TO_INTERVAL', 'TimeAhead'], inplace=True)
    forecast = forecast.reset_index(inplace=False, drop=True)
    
    # 01. 24 小時內
    merge0 = forecast.copy()
    merge0 = merge0.drop_duplicates(['ID', 'TIME_TO_INTERVAL'], keep="first")  
    merge0['TimeAhead'] = 0
    print('merge0' ,len(merge0))

    # 02. 24 小時前
    merge24 = forecast.copy()
    merge24 = merge24[merge24['DayAhead']>=timedelta(days=1)]
    merge24 = merge24.drop_duplicates(['ID', 'TIME_TO_INTERVAL'], keep="first")  
    merge24['TimeAhead'] = 24
    print('merge24', len(merge24))
    
    build = pd.concat([merge24, merge0], axis=0, ignore_index=True)
    build = build.drop(['DayAhead'], axis=1)
    return build


# In[11]:


# 使用「 24小時內 」的預測資料，填補「 24小時前 」資料的缺值
def fill_multiple_lead_time_data(data):
    forecast = data.copy()
    forecast['TIME_TO_INTERVAL'] = pd.to_datetime(forecast['TIME_TO_INTERVAL'])
    forecast['CrawlerTime'] = pd.to_datetime(forecast['CrawlerTime'])
    forecast = forecast.sort_values(by=['ID', 'TIME_TO_INTERVAL', 'CrawlerTime'], inplace=False)
    forecast = forecast.reset_index(inplace=False, drop=True)
    
    # 01.
    merge0 = forecast.copy()
    merge0 = merge0.drop_duplicates(['ID', 'TIME_TO_INTERVAL'], keep="last")  
    merge0['TimeAhead'] = 0
    print('merge0' ,len(merge0))
    
    # 02.
    merge24 = forecast.copy()
    merge24 = merge24[merge24['TimeAhead'].eq(24)]
    filling = merge0[~merge0['TIME_TO_INTERVAL'].isin(merge24['TIME_TO_INTERVAL'].tolist())]
    merge24 = pd.concat([merge24, filling]).reset_index(inplace=False, drop=True)
    merge24['TimeAhead'] = 24
    print('merge24', len(merge24))
    
    build = pd.concat([merge24, merge0], axis=0, ignore_index=True)
    build = build.drop_duplicates(['ID', 'TIME_TO_INTERVAL', 'TimeAhead'], keep="last")
    return build


# In[12]:


def merge_data_24(drop):
    # 標記「 24小時內 」和「 24小時前 」的預報
    forecast = drop.copy()
    forecast = build_multiple_lead_time_data(forecast)

    # 整合新資料與舊資料
    history = pd.read_csv(f'OpenWeatherMap.3H/Save/OWM.3H.Merge.Multiple(merge).csv')
    history['TIME_TO_INTERVAL'] = pd.to_datetime(history['TIME_TO_INTERVAL'])
    merge = pd.concat([forecast, history], axis=0, ignore_index=True)
    # merge = forecast.copy()
    merge['TIME_TO_INTERVAL'] = pd.to_datetime(merge['TIME_TO_INTERVAL'])
    merge['CrawlerTime'] = pd.to_datetime(merge['CrawlerTime'])
    merge = merge.sort_values(by=['ID', 'TIME_TO_INTERVAL', 'CrawlerTime'], inplace=False)
    merge = merge.drop_duplicates(['TIME_TO_INTERVAL', 'ID', 'TimeAhead'], keep="last")

    # 標記「 24小時內 」和「 24小時前 」的預報
    merge = fill_multiple_lead_time_data(merge)
    print(f'history: {len(forecast)}, merge: {len(merge)}')

    file = 'OpenWeatherMap.3H/Save/OWM.3H.Merge.Multiple(new).csv'
    forecast.to_csv(file, index=False)
    file = 'OpenWeatherMap.3H/Save/OWM.3H.Merge.Multiple(merge).csv'
    merge.to_csv(file, index=False)


# In[ ]:





# In[13]:


# start,end = pd.to_datetime('2022-08-12'),pd.to_datetime('2023-01-19')
# forecast = aggregate_data2(start,end)
# drop = sort_data_3h()
# merge_data_24(drop)


# In[14]:


# merge_data_24(drop)


# In[15]:


# drop


# In[16]:


# start_date, end_date = pd.to_datetime('2022-04-20'), pd.to_datetime('2022-04-21')
# forecast = aggregate_data2(start_date,end_date)
# drop = sort_data_3h()
# merge_data_24(drop)


# In[17]:


# pd.read_csv('./OpenWeatherMap.3H/Save/OWM.3H.Merge.Multiple(merge)2.csv')


# In[ ]:




