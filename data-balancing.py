
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split 
from sklearn import preprocessing


# In[4]:

df = pd.read_csv('expanded-data.csv') 


# In[5]:

df.dtypes


# In[6]:

df = df[df.brand_id.str.contains('\\\\""') == False]
df = df[df.age_range.str.contains('\\\\""') == False]
df = df[df.gender.str.contains('\\\\""') == False]


# In[7]:

df[df['gender'].str.contains('\\\\""')]


# In[8]:

df_features= df.ix[:,df.columns != "label"]
df_labels=df.ix[:,df.columns=="label"]
df_features_train,df_features_test,df_labels_train,df_labels_test = train_test_split(df_features,df_labels,test_size=0.3)
print(pd.value_counts(df_labels_test['label']))
print(pd.value_counts(df_labels_train['label']))


# In[9]:

df.isnull().values.any()


# In[10]:

df_dp = df.dropna(axis=0, how='any')
df_dp.isnull().values.any()


# In[11]:

df_dp.head()


# In[12]:

df_features= df_dp.ix[:,df_dp.columns != "label"]
df_labels=df_dp.ix[:,df_dp.columns=="label"]
df_features_train,df_features_test,df_labels_train,df_labels_test = train_test_split(df_features,df_labels,test_size=0.3)
print(pd.value_counts(df_labels_test['label']))
print(pd.value_counts(df_labels_train['label']))


# In[13]:

#调用smote
os = SMOTE(random_state=0) 
os_data_x,os_data_y=os.fit_sample(df_features_train.values,df_labels_train.values.ravel())
os_test_x,os_test_y=os.fit_sample(df_features_test.values,df_labels_test.values.ravel())


# In[14]:

os_x,os_y=os.fit_sample(df_features.values,df_labels.values.ravel())


# In[15]:

from __future__ import division 


# In[16]:

columns = df_features_train.columns
os_data_x = pd.DataFrame(data=os_data_x,columns=columns )
print(len(os_data_x))
os_data_y= pd.DataFrame(data=os_data_y,columns=["label"])
# check description of data
print("length of oversampled data is ",len(os_data_x))
print("Number of normal customer",len(os_data_y[os_data_y["label"]==0]))
print("Number of repeated customer",len(os_data_y[os_data_y["label"]==1]))
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["label"]==0])/len(os_data_x))
print("Proportion of Repeated data in oversampled data is ",len(os_data_y[os_data_y["label"]==1])/len(os_data_x))


# In[17]:

col = df_features_test.columns
os_test_x = pd.DataFrame(data=os_test_x,columns=col )
print(len(os_test_x))
os_test_y= pd.DataFrame(data=os_test_y,columns=["label"])


# In[18]:

cols = df_features.columns
os_x = pd.DataFrame(data=os_x,columns=cols )
print(len(os_x))
os_y= pd.DataFrame(data=os_y,columns=["label"])


# In[20]:

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(os_x)
df_normalized = pd.DataFrame(np_scaled, columns=cols)
df_normalized


# In[81]:

newtraindata=pd.concat([os_data_x,os_data_y],axis=1)
#newtestdata=pd.concat([os_test_x,os_test_y],axis=1)
newtraindata.to_csv('balanced_train.csv',sep=',')
#newtestdata.to_csv('balance_test.csv',sep=',')


# In[83]:

newtestdata=pd.concat([os_test_x,os_test_y],axis=1)
newtestdata.to_csv('balance_test.csv',sep=',')


# In[21]:

newdata=pd.concat([df_normalized,os_y],axis=1)
newdata.to_csv('balance_data.csv',sep=',')


# In[ ]:



