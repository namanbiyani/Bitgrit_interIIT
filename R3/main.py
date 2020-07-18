#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 00:09:06 2019

@author: deepank
"""
import torch
import sys
if torch.cuda.is_available():
    import torch.cuda as t
import torch.nn as nn
import numpy as np
import adabound
import pandas as pd
import random
import torch.nn.functional as F
import math
#from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


train=pd.read_csv(r'/home/deepank/Downloads/Round3/train.csv')
test=pd.read_csv(r'/home/deepank/Downloads/Round3/test.csv')
date=train['Date']
x_train = train
x_train['Date'] = pd.to_datetime(x_train['Date'], errors='coerce')
#Hot encoding
x_train['Season'] = x_train['Date'].dt.month
x_train['Season2'] = x_train['Date'].dt.month
x_train['Season3'] = x_train['Date'].dt.month
x_train['Season']=x_train['Season'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[1,1,1,0,0,0,0,0,0,0,0,1])#
x_train['Season2']=x_train['Season2'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[0,0,0,1,1,1,1,0,0,0,0,0])#
x_train['Season3']=x_train['Season3'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[0,0,0,0,0,0,0,1,1,1,1,0])#
x_train['E1'] = x_train['EventTarget'].replace([0,1,2],[0,1,0])
x_train['E2'] = x_train['EventTarget'].replace([0,1,2],[0,0,1])
x_train['EventTarget']=x_train['EventTarget'].replace([0,1,2],[1,0,0])


x_test = test
x_test['Date'] = pd.to_datetime(x_test['Date'], errors='coerce')
x_test['Season'] = x_test['Date'].dt.month
x_test['Season2'] = x_test['Date'].dt.month
x_test['Season3'] = x_test['Date'].dt.month
x_test['Season']=x_test['Season'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[1,1,1,0,0,0,0,0,0,0,0,1])#
x_test['Season2']=x_test['Season2'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[0,0,0,1,1,1,1,0,0,0,0,0])#
x_test['Season3']=x_test['Season3'].replace([1,2,3,4,5,6,7,8,9,10,11,12],[0,0,0,0,0,0,0,1,1,1,1,0])#

x_test['E1'] = x_test['EventTarget'].replace([0,1,2],[0,1,0])
x_test['E2'] = x_test['EventTarget'].replace([0,1,2],[0,0,1])
x_test['EventTarget']=x_test['EventTarget'].replace([0,1,2],[1,0,0])

A=x_train[x_train['Procedure']=='A'].drop(['n_Procedure'],inplace=False,axis=1)
A_target=x_train[x_train['Procedure']=='A']['n_Procedure']
B=x_train[x_train['Procedure']=='B'].drop(['n_Procedure'],inplace=False,axis=1)
B_target=x_train[x_train['Procedure']=='B']['n_Procedure']
C=x_train[x_train['Procedure']=='C'].drop(['n_Procedure'],inplace=False,axis=1)
C_target=x_train[x_train['Procedure']=='C']['n_Procedure']

Test=t.DoubleTensor(x_test.drop(['Date'],axis=1).values)
X_Full=t.DoubleTensor(A.drop(['Date','Procedure'],inplace=False,axis=1).values)
Y_Full=t.DoubleTensor(A_target.values)


class Model_Bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.REG1=nn.Sequential(
                nn.Linear(2,10),
                nn.BatchNorm1d(10),
                GELU(),
                nn.Linear(10,6),
                nn.BatchNorm1d(6),
                GELU(),
                nn.Linear(6,1)) 
        self.REG2=nn.Sequential(
                nn.Linear(3,10),
                nn.BatchNorm1d(10),
                GELU(),
                nn.Linear(10,6),
                nn.BatchNorm1d(6),
                GELU(),
                nn.Linear(6,1)) 
        self.REG3=nn.Sequential(
                nn.Linear(3,10),
                nn.BatchNorm1d(10),
                GELU(),
                nn.Linear(10,6),
                nn.BatchNorm1d(6),
                GELU(),
                nn.Linear(6,1)) 
        self.REG4=nn.Sequential(
                nn.Linear(3,1),
                GELU())
    def forward(self, x):
        x = torch.cat((self.REG1(x[:,[0,2,3]]),self.REG2(x[:,[1,7,8]]),self.REG3(x[:,[4,5,6]])),1)
        x=self.REG4(x)
        #x=self.REG1(x[:,0:2])
        return x
    
criterion = torch.nn.MSELoss()
model=Model_Bit().cuda()

#writer = SummaryWriter('/home/deepank/runs')
optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
a_pred=[]
lis=[i for i in range(0,326)]
ls= random.sample(lis, int(326*90/100))

for epoch in range(1,501):
    optimizer.zero_grad()
    y_pred = model(X_Full[[ls]])
    loss = criterion(y_pred, Y_Full[[ls]])
    y_val = model(X_Full)
    loss_val=criterion(y_val,Y_Full)
    loss.backward()
    optimizer.step()
    if epoch % 10==0:
        ls= random.sample(lis, int(326*90/100))
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())),'value_val ',np.exp(-np.sqrt(loss_val.item())))
    #torch.save(model.state_dict(), '/home/deepank/Downloads/BITGRIT/model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.pth')
    T=model(Test)
    a_pred=T.cpu().detach().numpy()
    
X_Full=t.DoubleTensor(B.drop(['Date','Procedure'],inplace=False,axis=1).values)
Y_Full=t.DoubleTensor(B_target.values)

model2=Model_Bit().cuda()
optimizer = adabound.AdaBound(model2.parameters(), lr=1e-3, final_lr=0.1)
b_pred=[]
lis=[i for i in range(0,326)]
ls= random.sample(lis, int(326*90/100))

for epoch in range(1,501):
    optimizer.zero_grad()
    y_pred = model2(X_Full[[ls]])
    loss = criterion(y_pred, Y_Full[[ls]])
    y_val = model2(X_Full)
    loss_val=criterion(y_val,Y_Full)
    loss.backward()
    optimizer.step()
    if epoch % 10==0:
        ls= random.sample(lis, int(326*90/100))
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())),'value_val ',np.exp(-np.sqrt(loss_val.item())))
    #torch.save(model.state_dict(), '/home/deepank/Downloads/BITGRIT/model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.pth')
    T=model2(Test)
    b_pred=T.cpu().detach().numpy()
 
X_Full=t.DoubleTensor(C.drop(['Date','Procedure'],inplace=False,axis=1).values)
Y_Full=t.DoubleTensor(C_target.values)
    
model3=Model_Bit().cuda()
optimizer = adabound.AdaBound(model3.parameters(), lr=1e-3, final_lr=0.1)
c_pred=[]
lis=[i for i in range(0,326)]
ls= random.sample(lis, int(326*90/100))

for epoch in range(1,501):
    optimizer.zero_grad()
    y_pred = model3(X_Full[[ls]])
    loss = criterion(y_pred, Y_Full[[ls]])
    y_val = model3(X_Full)
    loss_val=criterion(y_val,Y_Full)
    loss.backward()
    optimizer.step()
    if epoch % 10==0:
        ls= random.sample(lis, int(326*90/100))
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())),'value_val ',np.exp(-np.sqrt(loss_val.item())))
    #torch.save(model.state_dict(), '/home/deepank/Downloads/BITGRIT/model_norm2_'+str(epoch)+'_'+str(np.exp(-np.sqrt(loss_val.item())))+'.pth')
    T=model3(Test)
    c_pred=T.cpu().detach().numpy()
 
q=[]
for i in range(0,len(a_pred)):
    q.append(a_pred[i][0])
    q.append(b_pred[i][0])        
    q.append(c_pred[i][0])
print(q)
    
df=pd.DataFrame(q)
print(df)
df.to_csv('try.csv', index=False, header=False)    
print('Done')    
