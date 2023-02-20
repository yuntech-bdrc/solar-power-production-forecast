#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import datetime
import numpy as np
import pandas as pd
import threading
import requests
import json
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()


# In[ ]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from os.path import basename
from email.mime.application import MIMEApplication
from email.utils import COMMASPACE, formatdate


# In[ ]:


#編輯郵件內容
def send_mail_info(folder):
    content = MIMEMultipart()
    content["subject"] = "file" #郵件主題 
    content["from"] = "m11023032@gemail.yuntech.edu.tw" #自己的郵件地址 
    content["to"] = "m11023032@gemail.yuntech.edu.tw" #傳送到哪裡  
    content.attach(MIMEText(f"the file `{folder}`"))
    send_mail(content)   
#傳送郵件
def send_mail(content):
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:
        try:
            smtp.ehlo() #確認伺服器是否有回應
            smtp.starttls()#TLS郵件加密模式
            smtp.login("m11023032@gemail.yuntech.edu.tw", "aa0932684719")
            smtp.send_message(content)
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)


# In[ ]:


folder = 'Crawler.OpenWeatherMap.3H'


# In[ ]:


def replaceText(oldString):
    newString = oldString
    newString = newString.replace(" ", "_")
    newString = newString.replace(":", "-")
    newString = newString.replace(".", "_")
    return newString


# In[ ]:


def buildUrl(lat, lon):
    apiKey = "f7569384672b5c27516508cb78254e69"
    language = "zh_tw"
    url = "https://api.openweathermap.org/data/2.5/forecast?lat=%s&lon=%s"%(lat, lon)
    url += "&lang=%s&exclude=current&appid=%s"%(language, apiKey)
    return url


# In[ ]:


#將json檔形式拆成dataframe形式
def hourData(res_json, df_hour):
    dfOut_hour = df_hour.copy()
    # 拆解二維資料
    weather_id, weather_main, weather_description = [], [], []
    temp, feels_like, temp_min, temp_max = [], [], [], []
    pressure, sea_level, grnd_level, humidity, temp_kf = [], [], [], [], []
    clouds, wind_speed, wind_deg, rain, sys = [], [], [], [], []
    for i in range(len(res_json['list'])):
        # weather
        weather = res_json['list'][i]['weather'][0]
        weather_id.append(weather['id'])
        weather_main.append(weather['main'])
        weather_description.append(weather['description'])
        # main
        main = res_json['list'][i]['main']
        temp.append(main['temp'])
        feels_like.append(main['feels_like'])
        temp_min.append(main['temp_min'])
        temp_max.append(main['temp_max'])
        pressure.append(main['pressure'])
        sea_level.append(main['sea_level'])
        grnd_level.append(main['grnd_level'])
        humidity.append(main['humidity'])
        temp_kf.append(main['temp_kf'])
        # clouds
        cloudslist = res_json['list'][i]['clouds']
        clouds.append(cloudslist['all'])
        # wind
        wind = res_json['list'][i]['wind']
        wind_speed.append(wind['speed'])
        wind_deg.append(wind['deg'])
        # rain
        try:
            rainlist = res_json['list'][i]['rain']
            rain.append(rainlist['3h'])
        except:
            rain.append('')
        # sys
        syslist = res_json['list'][i]['sys']
        sys.append(syslist['pod'])
    # 插入分解完的二維資料
    dfOut_hour['weather_id'] = np.array(weather_id)
    dfOut_hour['weather_main'] = np.array(weather_main)
    dfOut_hour['weather_description'] = np.array(weather_description)
    dfOut_hour['temp'] = np.array(temp)
    dfOut_hour['feels_like'] = np.array(feels_like)
    dfOut_hour['temp_min'] = np.array(temp_min)
    dfOut_hour['temp_max'] = np.array(temp_max)
    dfOut_hour['pressure'] = np.array(pressure)
    dfOut_hour['sea_level'] = np.array(sea_level)
    dfOut_hour['grnd_level'] = np.array(grnd_level)
    dfOut_hour['humidity'] = np.array(humidity)
    dfOut_hour['temp_kf'] = np.array(temp_kf)
    dfOut_hour['clouds'] = np.array(clouds)
    dfOut_hour['wind_speed'] = np.array(wind_speed)
    dfOut_hour['wind_deg'] = np.array(wind_deg)
    dfOut_hour['rain'] = np.array(rain)
    dfOut_hour['sys'] = np.array(sys)
    dfOut_hour = dfOut_hour.drop(['main', 'weather', 'clouds', 'wind', 'sys'], axis=1)
    # 新增位址資訊
    dfOut_hour['lat'] = res_json['city']['coord']['lat']
    dfOut_hour['lon'] = res_json['city']['coord']['lon']
    return dfOut_hour


# In[ ]:


def getData(town_info, directory):
    get_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(':', '%3A')
    for i in range(len(town_info)):  # LOCATION_DF
        # 取得案場經緯度
        row = town_info.loc[i]
        lat, lon = row['LAT'], row['LON']
        try:
            res = requests.get(buildUrl(lat, lon), verify=False, timeout=5)
            res_json = json.loads(res.text)
            res.close()
            # JSON 整理 & 儲存
            #將欄位轉成dataframe
            df_3hour = pd.DataFrame(res_json['list'])
            dfOut_3hour = hourData(res_json, df_3hour)
            # 新增地點資訊
            dfOut_3hour['id'] = row['PLANT_ID']
            dfOut_3hour['name'] = row['PLANT_NAME']
            if not os.path.exists(f'{directory}/{row["PLANT_ID"]}'):
                os.mkdir(f'{directory}/{row["PLANT_ID"]}')
            # 匯出
            dfOut_3hour.to_csv(f'{directory}/{row["PLANT_ID"]}/{get_time}.csv')
            time.sleep(0.01)
        except:
            send_mail_info(folder)
            pass


# In[ ]:


# 如果目標資料夾不存在，則新建該資料夾
directory = "OpenWeatherMap.3H"
if not os.path.exists(directory):
    os.mkdir(directory)


# In[ ]:


town_info = pd.read_csv('Township_Coordinates_Taiwan.csv')
town_info.head(4)


# In[ ]:


# 彰化縣彰化市,500,120.5694208,24.07532909
for i in range(len(town_info)):
    Observatory = town_info.loc[i:i]
    if Observatory['PLANT_NAME'].values[0] == '彰化縣彰化市':
        target = Observatory
print(target)
target = target.reset_index()


# # 在每小時02分之前爬取一次

# In[ ]:


while(True):
    localtime = time.localtime()
    result = time.strftime("%M:%S", localtime)
    if(result<='02:00'):
        start_time = time.time()
        #輸入地址和目錄
        getData(target, directory)
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




