import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import adabound
from datetime import datetime
torch.set_default_tensor_type(torch.DoubleTensor)

train=pd.read_csv(r'/home/deepank/Downloads/Bitgrit/softbank_comp1_data/competition_data/train_text.csv')
test=pd.read_csv(r'/home/deepank/Downloads/Bitgrit/softbank_comp1_data/competition_data/test.csv')
#to be seen 30 or 31 or mean+std?
train['span']=train['span']/30
test['span']=test['span']/30
train=train.fillna(0)
test=test.fillna(0)

y=train['target']
x_test=test.drop(['id'],axis=1)
train=train.drop(['id','target'],axis=1)

X_train=torch.tensor(train.iloc[0:4000,0:361].values)
X_train_val=torch.tensor(train.iloc[4000:,0:361].values)
X_text=torch.tensor(train.iloc[:,361:].values)
Y_train=torch.tensor(y.iloc[0:4000].values).resize_((4000,1))
Y_train_val=torch.tensor(y.iloc[4000:].values).resize_((177,1))
X_test=torch.tensor(x_test.values)

class Model_Bit(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.REG=nn.Sequential(
                nn.Linear(378,1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(256,128),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(128,64),
                nn.BatchNorm1d(64),
                nn.Dropout(p=0.2),
                nn.ELU(),
                nn.Linear(64,1))
        
    def forward(self, x):
        x = self.REG(x)
        return x

U,S,V = torch.svd(torch.t(X_text))
PCA_X = torch.mm(X_text,U[:,:17])
X_train=torch.cat((X_train,PCA_X[:4000,:]),dim=1)
X_train_val=torch.cat((X_train_val,PCA_X[4000:,:]),dim=1)

class Model_PCA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.REG_PCA=nn.Sequential(
                nn.Linear(361,722),
                nn.ELU(),
                nn.Linear(722,256),
                nn.ELU(),
                nn.Linear(256,128),
                nn.ELU(),
                nn.Linear(128,17))
        
    def forward(self, x):
        x = self.REG_PCA(x)
        return x



model_PCA = Model_PCA()
optimizer_PCA= adabound.AdaBound(model_PCA.parameters(), lr=0.001,final_lr=0.01)
model_PCA.train()

criterion = torch.nn.MSELoss()
model = Model_Bit()
optimizer = adabound.AdaBound(model.parameters(), lr=0.001,final_lr=0.01)
model.train()
print('Starting PCA calculation')
for epoch in range(1,4001):
    optimizer_PCA.zero_grad()
    y_pca = model_PCA(X_train[:,:361])
    loss_pca = criterion(y_pca, X_train[:,361:])
    y_val_pca = model_PCA(X_train_val[:,:361])
    loss_pca_val=criterion(y_val_pca,X_train_val[:,361:])
    if epoch % 20==0:
        print(' epoch: ', epoch,' loss: ', loss_pca.item(),' valscore :',loss_pca_val.item(),' value: ', np.exp(-np.sqrt(loss_pca.item())))
    loss_pca.backward()
    optimizer_PCA.step()
torch.save(model_PCA.state_dict(), '/home/deepank/Downloads/model_PCA'+str(np.exp(-np.sqrt(loss_pca.item())))+'.pth')
final_pca=model_PCA(X_test)
X_test=torch.cat((X_test,final_pca),dim=1)

print('Starting Model Training')

for epoch in range(1,30001):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    y_val = model(X_train_val)
    loss_val=criterion(y_val,Y_train_val)
    if epoch % 20==0:
        print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())))
    loss.backward()
    optimizer.step()
    if epoch%1000==0:
        torch.save(model.state_dict(), '/home/deepank/Downloads/model'+str(np.exp(-np.sqrt(loss.item())))+'.pth')
a=30001
a=int(input('Want to continue?'))
while a!=0:
    for epoch in range(a,a+1000):
        optimizer.zero_grad
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)
        y_val = model(X_train_val)
        loss_val=criterion(y_val,Y_train_val)
        if epoch % 20==0:
            print(' epoch: ', epoch,' loss: ', loss.item(),' valscore :',loss_val.item(),' value: ', np.exp(-np.sqrt(loss.item())))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), '/home/deepank/Downloads/model'+str(np.exp(-np.sqrt(loss.item())))+'.pth')
    a=int(input('Want to continue?'))    
    
Y_test=model(X_test)

#saving to a csv file
ans=Y_test.detach().numpy()
final=pd.DataFrame()
final['id']=test['id']
lis=[]
for i in ans.tolist():
    lis.append(i[0])
final['target']=lis
now = datetime.now()
current_time = now.strftime("%d_%H_%M_%S")
final.to_csv(current_time+'_.csv',sep=',',index=False)
print('Done')