#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df_X = pd.read_csv(r'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save\OWM.3H.Merge.Multiple(merge).csv', header=None)

df_X


# In[3]:


# import pandas as pd

# df_X = pd.read_csv(r'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save\OWM.3H.Merge.Multiple(merge)50.csv', header=None)
# df_X


# In[4]:


# df_2 = pd.DataFrame(df_X["TIME_TO_INTERVAL"], columns = idx)
print(df_X[0].values)
time_all = df_X[0].values
df_datatime = pd.date_range(start=time_all[2], periods=12, freq='15min')
print(df_datatime)
# df_X["TIME_TO_INTERVAL"]
# df_tt = pd.DataFrame(pd.to_datetime(df_X["TIME_TO_INTERVAL"]))
# print(df_tt)


# In[5]:


#整個時間欄
time_all = df_X[0].values
print(df_X.iloc[0,:].to_frame().T)
df_X1 = df_X.iloc[0,:].to_frame().T
for j in range(1,15450):
    #每隔15分鐘產生一個時間，共產生12個
    df_datatime = pd.date_range(start=time_all[j], periods=12, freq='15min')
    print(time_all[j])
    #第j行
    df_X_original =df_X.loc[j,:]


    for i in range(1,12):
        #將就的替換成新的 ex:2020-06-01 20:00:00 -> 2020-06-01 20:15:00
        df_X_new = df_X_original.replace(str(df_datatime[0]), str(df_datatime[i]))
        if i==1 and j ==1:
            #to_frame().T為轉置
            df_X1 = pd.concat([df_X1,df_X_new.to_frame().T],axis=0, ignore_index=True)
        else:
            df_X1 = pd.concat([df_X1,df_X_new.to_frame().T],axis=0, ignore_index=True)
        
# df_X1 = df_X1.sort_values(by=0)
df_X1


# In[6]:


df_X1.to_csv(f'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save15m\OWM.15M.Merge.Multiple(merge)2.csv', index=None,header=False)


# In[7]:


# df_X = pd.read_csv(r'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save15m\OWM.15M.Merge.Multiple(merge).csv', header=None)

# df_X = df_X.drop_duplicates(keep='first', inplace=False)  # 刪除重複


# In[9]:


df_X = pd.read_csv(r'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save\OWM.3H.Merge.Multiple(merge).csv', header=None)
df_X2 = pd.concat([df_X,df_X1],axis=0, ignore_index=True)
df_X2


# In[12]:


df_X2.to_csv(f'C:\研究所\發電量\太陽能交接檔案\OpenWeatherMap.3H\Save15m\OWM.15M.Merge.Multiple(merge).csv', index=None,header=False)


# In[ ]:





# In[ ]:




