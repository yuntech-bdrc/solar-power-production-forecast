#!/usr/bin/env python
# coding: utf-8

# # 取得最近觀測站

# In[ ]:


import urllib.request as req
import json 
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from datetime import timedelta
import requests
from bs4 import BeautifulSoup

import time
import datetime


# In[ ]:


from math import radians, cos, sin, asin, sqrt
# calculate distance based on latitude and longitude
#計算最近的案場資料
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
            current['Longitude'], current['Latitude'])

        # if the current distance is shorter than the historical shortest distance
        # then use current point replace recent point
        if(dist < shortest_dist):
            shortest_dist = dist
            recent_point = current[column]
    # end of search, return
    return recent_point


# # 抓取CWB觀測資料
# https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp

# In[ ]:


stations = pd.read_csv('Observatory_CWB.csv')
stations = stations[stations['Location'] == '北區']
stations = stations.reset_index(drop=True)


# In[ ]:


stations


# In[ ]:


def crawler_obs_data(start, end):
    dates = [d.strftime('%Y-%m-%d')for d in pd.date_range(start, end)]
    for i in range(len(stations)):
        staID = stations.loc[i]['ID']
        staName = stations.loc[i]['Name']
        print(staID)
        data = pd.DataFrame()
        for date in dates:
            print(date)
            url = f'https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station={staID}&stname={staID}&datepicker={date}&altitude=83.04m'
            print(url)
            headers = {'user-agent': 'Mozilla/5.0 (Windows NT 20.0; Win64; x64) AppleWebKit/538.36 (KHTML, like Gecko) Chrome/86.0.3809.132 Safari/537.36'}
            html = requests.get(url, headers=headers)
            html.encoding = 'utf-8'
            soup = BeautifulSoup(html.text, 'html.parser')
            # find no data page
            error = soup.find("label", class_="imp")

            form =[]

            # title
            titles = soup.find_all("th")
            # titles = titles[11:28]
            titles = titles[28:46]
            strtitle=[]
            for title in titles:
                title = title.contents
                title=title[0] #+title[2]+title[4]
                strtitle.append(title)

            # parameter
            soup = soup.tbody
            tmps = soup.find_all("tr")
            tmps = tmps[3:]
            for tmp in tmps:
                tmp = tmp.find_all("td")
                parameter =[]
                for strtmp in tmp:
                    strtmp = ''.join(filter(lambda x: (x.isdigit() or x == '.'  or x == 'T'), strtmp.string))
                    parameter.append(strtmp)
                form.append(parameter)

            form = pd.DataFrame(form, columns=strtitle)
            form['date'] = date
            data = pd.concat([data,form], axis=0)
            data = data.reset_index(drop=True)
        old = pd.read_csv(f'./Observation_CWB/{staID}.csv')
        d = pd.concat([old, data], axis=0, ignore_index=True)
        d = d.drop_duplicates(subset=['ObsTime', 'date'], keep='last')
        d.to_csv(f'./Observation_CWB/{staID}.csv', index=None)


# # 每天中午12點爬取一次

# In[ ]:


while(True):
    hour = pd.to_datetime(datetime.datetime.today()).hour
    if hour == 12:
        start_time = time.time()
        day = datetime.datetime.today()
        day = pd.to_datetime(day, format='%Y%m%d')
        day = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))
        start = day-datetime.timedelta(days=1)
        end = day-datetime.timedelta(days=1)
        crawler_obs_data(start, end)
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


# start, end = '2022-08-14', '2022-08-14'
# crawler_obs_data(start, end)


# In[ ]:


# while(True):
#     today = datetime.datetime.today()
#     hour = pd.to_datetime(datetime.datetime.today()).hour
#     if hour == 12:
#         day = today-datetime.timedelta(days=1)
#         day = pd.to_datetime(day, format='%Y%m%d')
#         day = pd.to_datetime(str(day.year)+'-'+str(day.month)+'-'+str(day.day))

#         staID = stations.loc[0]['ID']
#         data = pd.read_csv(f'./Dataset/Observation_CWB/{staID}.csv')
#         if pd.to_datetime(data['date'][-1:].values[0])==day:
#             pass
#         else:
#             start, end = day, day
#             crawler_obs_data(start, end)
# #     time.sleep(43200)
#     time.sleep(3600)


# In[ ]:




