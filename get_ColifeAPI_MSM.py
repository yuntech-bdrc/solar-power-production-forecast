#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request as req
import json 
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from datetime import timedelta
import datetime
import certifi
import time


# In[2]:


plant_info = pd.read_csv('Plant_Info_Baoshan.csv')
plant_on = 1
plant_latlon = [plant_info.loc[plant_on]['Latitude'], plant_info.loc[plant_on]['Longitude']]
plant_name = plant_info.loc[plant_on]['Name']
plant_id = plant_info.loc[plant_on]['ID']
print(plant_id,plant_name)
plant_info


# In[3]:


plant_info = plant_info.loc[1:1].reset_index()
# plant_info


# # 使用中興大學提供之API抓取MSM預測日照

# 
# - prate: 降水(mm/h)
# - tmp2m: 2m氣溫(K)
# - q2m: 2m比濕(g/g)
# - rh2m: 2m相對濕度(%)
# - pres: 氣壓(Pa)
# - ws10m: 10m風速(m/s)
# - u10m:  10m東西向風速(m/s)
# - v10m: 10m南北向風速(m/s)
# - dswrf: 太陽輻射(W/m2)
# - dlwrf: 大氣長波輻射(W/m2)
# - hpblsfc: 混合層高(m)
# - apcpsfc: apcpsfc(mm/h) (降水量)
# - blftx: blftx(K)  (最佳提升指數)
# - cape: cape(m) (對流可用位能)
# - clwmr: clwmr(kg/m3) (雲水混合比)
# - lftx: lftx(K)  (表面提升指數)
# - lhf: 蒸發熱(W/m2)
# - shf: 可感熱(W/m2)
# - slp: 海平面壓力(Pa)
# - sm105: sm105(fraction) (10-200 厘米地下土壤體積水分)
# - sm5: sm5(fraction)
# - tmp105: tmp105(K) (10-200 厘米地下溫度)
# - tmp5: tmp5(K)
# 

# In[4]:


def dtna(x):
    if x == '':
        x=0
    return x

#MSM模型,測量方式,抓取氣象預報資訊,經緯度,案場名稱,起始時間,結束時間
#抓取msm預測資料
def get_data(model, level, variable, lat, lon, plant_id, start, end):
    dates = [d.strftime('%Y-%m-%d')for d in pd.date_range(start, end)]
    data = pd.DataFrame()
    for date in dates:
#         print(date)
#         print(model)
#         print(level)
#         print(variable)
#         print(lat)
#         print(lon)
        url = f'https://pm25.colife.org.tw/windy/data/timeline_fast.php?expDate={date}&model={model}&level={level}&variable={variable}&domain=&lat={lat}&lng={lon}'
        print(url)
        with req.urlopen(url, cafile=certifi.where()) as res:
            data=json.load(res)
        # hour = list(range(0,24))*8

        data = pd.DataFrame(data).reset_index()
#         data['TIME_TO_INTERVAL'] = pd.to_datetime(data['index'], format='%Y%m%d')
        data['TIME_TO_INTERVAL'] = data['index'].apply( lambda row: f"{row[:4]}-{row[4:6]}-{row[6:8]} {row[8:]}:0:00")
        data['data'] = data['data'].apply(lambda x:dtna(x))
        data['data'] = data['data'].astype(np.float32) 
        data['hour'] = pd.to_datetime(data['TIME_TO_INTERVAL']).dt.hour
#         data.to_csv(f'./Dataset/Tientai/MSM/{plant_id}/{plant_id}_{date}.csv', index=None)
        data.to_csv(f'MSM_data/{plant_id}_{date}.csv', index=None)
        print(date)
#         print(data)
        data = pd.concat([data,data], axis=0)
        
    return data
    


# ## 儲存API抓取之MSM預測日照
# #### (每天預測包含過去4天預測未來一日和當天預測未來四天)
# #### MSM觀測日照為MSM未來一日之預測日照(過去4天，抓取緊抓取當天，也就是昨天預測今天的預測值)
# #### MSM預測日照包含未來96小時(未來4天)

# In[5]:


#整理抓取下來的資料
def msm_pred_packaging(plantid, start, end, msm_var, target):
#     start, end = '2021-04-10', '2021-06-30'
    dates = [d.strftime('%Y-%m-%d')for d in pd.date_range(start, end)]
    pid = plantid
#     for pid in plantid:
    #未來一天的預測資料(24小時)
    obsDatas = pd.DataFrame()
    #未來四天的預測資料(96小時)
    predDatas = pd.DataFrame()
    for i in range(len(dates)):
        #依照日期抓取資料夾中新預測MSM數據
#         newdata = pd.read_csv(f'./Dataset/Tientai/MSM/{plant_id}/{plant_id}_{date}.csv')
        newdata = pd.read_csv(f'MSM_data/{pid}_{dates[i]}.csv')
        newdata['date'] = pd.to_datetime(newdata['TIME_TO_INTERVAL']).dt.date
        #抓取包含當天在內的預測值(過去四天)
        obsData = newdata[~(pd.to_datetime(newdata['TIME_TO_INTERVAL'])>pd.to_datetime(f'{dates[i]} 23:59:59'))]
        #抓去未來四天的預測值
        predData = newdata[(pd.to_datetime(newdata['TIME_TO_INTERVAL'])>pd.to_datetime(f'{dates[i]} 23:59:59'))]
        #依照未來預測值與過去預測值進行合併
        obsDatas = pd.concat([obsDatas,obsData], axis=0)
        predDatas = pd.concat([predDatas,predData], axis=0)
    obsDatas = obsDatas.reset_index(drop=True)
    predDatas = predDatas.reset_index(drop=True)
    obsDatas['Date'] = pd.to_datetime(obsDatas['date'])
#     obsDatas = obsDatas[pd.to_datetime(obsDatas['expDate']) == obsDatas['Date']]
    #obsDatas_ForecastBeforeDays抓出過去一天的
    obsDatas['ForecastBeforeDays'] = pd.to_datetime(obsDatas['date']) - pd.to_datetime(obsDatas['expDate'])
    obsDatas = obsDatas[obsDatas['ForecastBeforeDays'].eq(datetime.timedelta(days=-1))]
    
    #predDatas_ForecastBeforeDays抓出未來四天的值
    predDatas['Radiation(MSMv4)'] = predDatas['data']
    predDatas['expDate'] = pd.to_datetime(predDatas['expDate'])
    predDatas['Date'] = pd.to_datetime(predDatas['date'])
    predDatas['ForecastBeforeDays'] = predDatas['Date'] - predDatas['expDate']
    variable = ['Radiation(MSMv4)']
    columns = ['TIME_TO_INTERVAL', 'Date']
    shift_days = 4
    # 依照提前預測天數，更新要篩選的欄位
    for column in variable:
        for i in range(shift_days):
            columns.append(f'{column}[{i+1}d]')

        # 計算提前預測的天數
    merge = pd.DataFrame(columns=['TIME_TO_INTERVAL', 'Date'])
    merge['TIME_TO_INTERVAL'] = pd.to_datetime(merge['TIME_TO_INTERVAL'])
    merge['Date'] = pd.to_datetime(merge['Date'])
    predDatas['TIME_TO_INTERVAL'] = pd.to_datetime(predDatas['TIME_TO_INTERVAL'])
    for i in range(shift_days):
        # 篩選提前預測 i 天的資料
        shift = predDatas[predDatas['ForecastBeforeDays'].eq(datetime.timedelta(days=i+1))]

    #         重新命名變數，避免合併的時候衝突
        for column in variable:
            # 篩選提前預測 i 天的資料
            shift = shift.rename(columns={column: f'{column}[{i+1}d]'})
#         print(merge.dtypes)
#         print(shift.dtypes)
        merge = pd.merge(shift, merge, on=['TIME_TO_INTERVAL', 'Date'], how='outer')
    predDatas = merge.copy()
    predDatas = predDatas[columns]
#     print(predDatas)
#     #全部預測資料合併obsDatas(過去一天)、predDatas(未來四天)
#     obsDatas.to_csv(f'MSM_data/save/{pid}_{msm_var}obs(0814).csv', index=None)
#     #抓取
#     predDatas.to_csv(f'MSM_data/save/{pid}_{msm_var}pred(0814).csv', index=None)
    old = pd.read_csv(f'./MSM_data/save/solar_汙水廠_dswrfpred.csv')
    d = pd.concat([old, predDatas], axis=0, ignore_index=True)
    d = d.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
    d.to_csv(f'./MSM_data/save/solar_汙水廠_dswrfpred.csv', index=None)
    
    old_2 = pd.read_csv(f'./MSM_data/save/solar_汙水廠_dswrfobs.csv')
    d_2 = pd.concat([old_2, obsDatas], axis=0, ignore_index=True)
    d_2 = d_2.drop_duplicates(subset=['TIME_TO_INTERVAL'], keep='last')
    d_2.to_csv(f'./MSM_data/save/solar_汙水廠_dswrfobs.csv', index=None)



    return predDatas


# In[6]:


# msm_var_name = ['dswrf']
# model = 'msm.nc.v3'
# level = 'sfc'
# variable = 'dswrf'
# # target = 'Taipower'
# target = 'Baoshan'
# latlng = plant_latlon
# # start, end = '2020-01-01', '2021-12-30'
# # start, end = '2021-04-25', '2022-05-17'
# start, end = '2022-11-21', '2022-11-23'
# for plant in range(len(plant_info)):
#     plant_id = plant_info.loc[plant]['ID']
#     plant_Latitude = plant_info.loc[plant]['Latitude']
#     plant_Longitude = plant_info.loc[plant]['Longitude']
#     for var in msm_var_name:
#         print(plant_id, var)
#         data = get_data(model, level, var, plant_Latitude, plant_Longitude, plant_id, start, end)
#         d = msm_pred_packaging(plant_id, start, end, var, target)


# # 每天中午12點爬取一次

# In[ ]:


msm_var_name = ['dswrf']
model = 'msm.nc.v3'
level = 'sfc'
variable = 'dswrf'
target = 'Baoshan'
latlng = plant_latlon
plant_id = plant_info.loc[0]['ID']
plant_Latitude = plant_info.loc[0]['Latitude']
plant_Longitude = plant_info.loc[0]['Longitude']
while(True):
    hour = pd.to_datetime(datetime.datetime.today()).hour
    if hour == 12:
        start_time = time.time()
        # 設定要產生的開始與結束日期
        day = datetime.datetime.today()
        day = pd.to_datetime(day, format='%Y%m%d')
        start = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))-datetime.timedelta(days=1)
        end = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))-datetime.timedelta(days=1)
        for var in msm_var_name:
            print(plant_id, var)
            data = get_data(model, level, var, plant_Latitude, plant_Longitude, plant_id, start, end)
            d = msm_pred_packaging(plant_id, start, end, var, target)
        print('okok')
        end_time = time.time()
        finish = end_time - start_time
        print(finish)
        time.sleep(86400-finish)
    else:
        localtime = time.localtime()
        finish = time.strftime("%H:%M:%S", localtime)
        h,m,s = finish.strip().split(":")
        h=int(h)
        if(h>12):
            h=h-12
        else:
            h=h-12+24
        finish_time = int(h)*3600+int(m)*60+int(s)
        time.sleep(86400-finish_time)


# In[ ]:


d[:50]


# In[ ]:




