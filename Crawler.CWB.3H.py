#!/usr/bin/env python
# coding: utf-8

# https://opendata.cwb.gov.tw/dataset/forecast/F-D0047-001

# In[ ]:


import io
import os
import time
import datetime
import json
import requests
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


# In[ ]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from os.path import basename
from email.mime.application import MIMEApplication
from email.utils import COMMASPACE, formatdate


# 登入帳號

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


folder = 'CWB.3H'


# In[ ]:


getTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(getTime)
# time.sleep(3600)


# In[ ]:


statIDs = ['F-D0047-017']


# In[ ]:


# # 如果目標資料夾不存在，則新建該資料夾
# for name in statIDs:
#     if not os.path.exists(f'{folder}/{name}'):
#         os.mkdir(f'{folder}/{name}')


# # 在每小時02分之前爬取一次

# In[ ]:


while(True):
    localtime = time.localtime()
    result = time.strftime("%M:%S", localtime)
    if(result<='02:00'):
        start_time = time.time()
        getTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(':', '%3A')
        authCode = 'CWB-E89A9E94-16F6-4EBE-A690-3950827C2C11'
        #print(getTime)
        for statID in statIDs:
            try:
                url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/{}?Authorization={}'.format(statID, authCode)

                header = {}

                r = requests.get(url, headers=header)
                res = r.text

                with io.open('{}/{}/{}.txt'.format(folder, statID, getTime), 'w') as outfile:
        #             print('save {}/{}/{}.txt'.format(folder, statID, getTime))
                    outfile.write(res)
                time.sleep(0.01)
            except:
                send_mail_info('err: '+folder)
                pass
        print(getTime, '------------------------')
        end_time = time.time()
        finish = end_time - start_time
        print(finish)
        time.sleep(3600-finish)
    else:
        m,s = result.strip().split(":")
        start_time = int(m)*60+int(s)
        time.sleep(3600-start_time)


# In[ ]:




