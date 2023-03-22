#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
from influxdb import InfluxDBClient
databasename = ['MG1']
client = InfluxDBClient('120.107.146.56', 8086, 'ncue01', 'Q!A@Z#WSX', 'MG1') 


# In[2]:


## 在線使用設置##############
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[3]:


#轉為小時單位
def bulid_hour_data(data_raw):
    data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
    data_raw['Hour'] = data_raw['TIME_TO_INTERVAL'].dt.hour
    data_raw['Date'] = data_raw['TIME_TO_INTERVAL'].dt.date
    data_raw = data_raw.groupby(['Date', 'Hour']).mean().reset_index()
    data_raw['TIME_TO_INTERVAL'] = data_raw.apply(lambda raw:'{} {:02d}:00:00'.format(raw['Date'], int(raw['Hour'])), axis=1)
    return data_raw


# In[4]:


#抓取資料庫全部資料
def get_power():
    tablename = 'MG1_PV'
    # result = client.query(f'SELECT * FROM {tablename} where Time >= 2022-05-20') 
    result = client.query(f'SELECT * FROM {tablename}') 
    data =  list(result.get_points())
    data = pd.DataFrame(data)

    data = data.rename(columns={'Time':'TIME_TO_INTERVAL'})
    data = bulid_hour_data(data)
    data.to_csv('Dataset/merge_power_data(hour).csv')


# In[5]:


#根據時間抓取該時資料
def get_power_date(date):
    tablename = 'MG1_PV'
    result = client.query(f"SELECT * FROM {tablename} where Time >= '{date}' - 8h") 
#     result = client.query(f"SELECT * FROM {tablename} where Time >= '{date}'") 
#     result = client.query(f'SELECT * FROM {tablename}') 
    data =  list(result.get_points())
    data = pd.DataFrame(data)
    data = data.rename(columns={'Time':'TIME_TO_INTERVAL'})
    data = bulid_hour_data(data)
    data = data[['kP','TIME_TO_INTERVAL']]
    data.set_axis(['Power','TIME_TO_INTERVAL'], axis='columns', inplace=True)
    return data
#     data.to_csv('Dataset/merge_power_data(hour).csv')


# In[6]:


def new_power(row, datas):
    data = datas[datas['TIME_TO_INTERVAL'].eq(row['TIME_TO_INTERVAL'])]
    if data.any(axis=None):
        return data['kP'].values[0]
    else:
        return row['Power']

# merge_raw['Power'] = merge_raw.apply(lambda x:new_power(x, data), axis=1)


# # 每小時爬取一次

# In[ ]:


while(True):
    localtime = time.localtime()
    result = time.strftime("%M:%S", localtime)
    if(result<='02:00'):
        start_time = time.time()
        # 設定要產生的開始與結束日期
        day = datetime.datetime.today()
        day = pd.to_datetime(day, format='%Y%m%d')
        start = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))-datetime.timedelta(days=1)
        data = get_power_date(start)
        old = pd.read_csv(f'./power_data/solar_plant_history.csv', low_memory=False)
        d = pd.concat([old, data], axis=0, ignore_index=True)
        d = d.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
        d.to_csv(f'./power_data/solar_plant_history.csv', index=None)
        print('okok')
        end_time = time.time()
        finish = end_time - start_time
        print(finish)
        time.sleep(3600-finish)
    else:
        m,s = result.strip().split(":")
        start_time = int(m)*60+int(s)
        time.sleep(3600-start_time)


# In[ ]:


# start = '2022-01-01'
# data = get_power_date(start)
# old = pd.read_csv(f'./power_data/solar_plant_history.csv')
# d = pd.concat([old, data], axis=0, ignore_index=True)
# d = d.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
# d.to_csv(f'./power_data/solar_plant_history.csv', index=None)


# In[ ]:


# merge_power_data = pd.read_csv('./power_data/merge_power_data(hour).csv')
# merge_power_data = merge_power_data[['kP','TIME_TO_INTERVAL']]
# merge_power_data.set_axis(['Power','TIME_TO_INTERVAL'], axis='columns', inplace=True)
# merge_power_data


# In[ ]:


# solar_allhour = pd.read_csv('./power_data/solar_plant(allhour).csv')
# solar_allhour = solar_allhour[['Power','TIME_TO_INTERVAL']]
# solar_allhour


# In[ ]:


# merge_data = pd.concat([merge_power_data,solar_allhour],join='outer')
# merge_data = merge_data.drop_duplicates(keep='first',inplace=False)
# merge_data = merge_data.sort_values(by=['TIME_TO_INTERVAL'])
# merge_data = merge_data.reset_index(drop=True,inplace=False)
# merge_data.to_csv('./power_data/solar_plant_history.csv',index=False)


# In[ ]:


# merge_data


# In[ ]:




