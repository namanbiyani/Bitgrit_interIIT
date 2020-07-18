#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense


# In[2]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
y_test=pd.read_csv("Y_test.csv")
train['span']=train['span']/30
test['span']=test['span']/30
train=train.fillna(0)
test=test.fillna(0)


# In[3]:


test.head(5)
ids=test['id']
ids


# In[4]:


y_val=y_test.drop(['id'],axis=1)
y=pd.DataFrame(train['target'])
y=y.append(y_val,ignore_index=True)
y


# In[5]:


x_val=test.drop(['id'],axis=1)
X_train=train.drop(['id','target'],axis=1)
# print(np.shape(x_val))
# print(np.shape(X_train))
X_train=pd.concat([x_val, X_train], ignore_index=True)
# X_train.append(x_val,ignore_index=True)
# # x_val
# # X_new
# X_new



# In[6]:


# Y_train=torch.tensor(y.iloc[0:4177].values).resize_((4177,1))
# X_test=torch.tensor(x_test.values)


# In[7]:


model = Sequential()
model.add(Dense(1805, input_dim=361, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))


# In[8]:


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
print(np.shape(X_train))
print(np.shape(y))


# In[ ]:


model.fit(X_train,y,epochs=1000,batch_size=64)


# In[10]:


Y_test=model.predict(x_val)
Y_test


# In[11]:


final=pd.DataFrame()
final['id']=test['id']
lis=[]
for i in Y_test.tolist():
    lis.append(i[0])
final['target']=lis


# In[12]:


print(final)


# In[13]:


# final.sort_values("id", axis = 0, ascending = True, 
#                  inplace = True, na_position ='last')


# In[14]:


final.to_csv('Final.csv',sep=',',index=False)


# In[ ]:




