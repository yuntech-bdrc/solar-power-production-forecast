#!/usr/bin/env python
# coding: utf-8

# # 晴空輻射
# ### 晴空輻射是在特定時間的理論上輻射值，其假設天空無雲
# ### 根據物理公式推算得出

# In[ ]:


import pandas as pd
from pvlib.location import Location
import datetime
import time


# In[ ]:


#CSR抓取
def calculate_csr(lat, lon, start, end):
    # 根據案場資料導入經緯度資訊
    tus = Location(lat, lon, 'Asia/Taipei')
    # 設定資料的取樣間隔，可以設定 5 分鐘, 1 小時, 1 日
    times = pd.date_range(start=start, end=end, freq='5min', tz=tus.tz)
    # 呼叫套件 pvlib.location 產生晴空輻射
    data = tus.get_clearsky(times).reset_index()
    data['TIME_TO_INTERVAL'] = data['index']
    data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.tz_localize(None)
    # 僅保留晴空輻射(ghi)，移除其他欄位
    data = data.drop(columns=['index', 'dni', 'dhi'])
    #rename
    data = data.rename(columns={'ghi':'ClearSkyRadiation'})
    #轉換時間格式
    data['TIME_TO_INTERVAL'] = pd.to_datetime(data['TIME_TO_INTERVAL'])
    data = bulid_hour_data(data)
    return data


# In[ ]:


def bulid_hour_data(data_raw):
    data_raw['Hour'] = data_raw['TIME_TO_INTERVAL'].dt.hour
    data_raw['Date'] = data_raw['TIME_TO_INTERVAL'].dt.date
    data_raw = data_raw.groupby(['Date', 'Hour']).mean().reset_index()
    data_raw['TIME_TO_INTERVAL'] = data_raw.apply(lambda raw:'{} {:02d}:00:00'.format(raw['Date'], raw['Hour']), axis=1)
    data_raw = data_raw.drop(['Hour', 'Date'], axis=1)
    data_raw = data_raw.sort_values(by='TIME_TO_INTERVAL').reset_index(drop=True)
    data_raw['TIME_TO_INTERVAL'] = pd.to_datetime(data_raw['TIME_TO_INTERVAL'])
    return data_raw


# In[ ]:


plant_info = pd.read_csv(f'Plant_Info_Baoshan.csv', low_memory=False)
plant_info
plant_no = 1
plant_id = plant_info.iloc[plant_no]['ID']
plant_name = plant_info.iloc[plant_no]['Name']
capacity = float(plant_info.iloc[plant_no]['Capacity'])
latitude, longitude = float(plant_info.iloc[plant_no]['Latitude']), float(plant_info.iloc[plant_no]['Longitude'])
print("plant: %s, %s" % (plant_id, plant_name))


# # 每天凌晨0點爬取一次

# In[ ]:


while(True):
    hour = pd.to_datetime(datetime.datetime.today()).hour
    if hour == 0:
        start_time = time.time()
        # 設定要產生的開始與結束日期
        day = datetime.datetime.today()
        day = pd.to_datetime(day, format='%Y%m%d')
        start = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day)+' 00:00:00')+datetime.timedelta(days=1)
        end = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day)+' 23:00:00')+datetime.timedelta(days=1)
        csr = calculate_csr(latitude, longitude, start, end)
        csr['ClearSkyRadiation'] = csr['ClearSkyRadiation']/1000
        # 新舊資料合併
        old = pd.read_csv(f'./clear_sky_data/{plant_id}_ClearSkyRadiation.csv')
        d = pd.concat([old, csr], axis=0, ignore_index=True)
        d = d.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
        d.to_csv(f'./clear_sky_data/{plant_id}_ClearSkyRadiation.csv', index=None)
        print('okok')
        end_time = time.time()
        finish = end_time - start_time
        print(finish)
        time.sleep(86400-finish)
    else:
        localtime = time.localtime()
        finish = time.strftime("%H:%M:%S", localtime)
        h,m,s = finish.strip().split(":")
        finish_time = int(h)*3600+int(m)*60+int(s)
        time.sleep(86400-finish_time)


# In[ ]:


# # 儲存檔案
# start, end = '2021-04-01 00:00:00', '2022-08-13 23:00:00'
# csr = calculate_csr(latitude, longitude, start, end)
# csr['ClearSkyRadiation'] = csr['ClearSkyRadiation']/1000
# csr.to_csv(f'./clear_sky_data/{plant_id}_ClearSkyRadiation.csv', index=None)
# csr.tail(10)


# In[ ]:




