import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import data
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
filename_train = "Training.csv"
filename_test = "Test_X.csv"
training_set = np.loadtxt(filename_train,delimiter = ",")   # load the training set 
test_set = np.loadtxt(filename_test,delimiter = ",") # load the features for the test set.

X_train = training_set[:9216,0:40] # features for the 10,000 training examples
Y_train = training_set[:9216,40] # labels for the 10,000 training examples
X_test = test_set # features for the 1,000 test examples
X_valid = training_set[9216:,0:40]
Y_valid = training_set[9216:,40]
'''
Put your training procedure in the following blank part.
'''






class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(40,128), 
            nn.Dropout(0.2),
            nn.Sigmoid(), 
            
            nn.Linear(128,16),
            nn.Dropout(0.2),
            nn.Sigmoid(),
            
            nn.Linear(16,1),
            
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
neural = NN().cuda()


x_train = torch.Tensor(X_train)
train_mean = np.zeros(40)
train_std = np.zeros(40)
for i in range(40):
    train_mean[i] = X_train[:,i].mean()
    train_std[i] = X_train[:,i].std()
    
y_train = torch.Tensor(Y_train).reshape(-1,1)
x_test = torch.Tensor(X_test)
x_valid = torch.Tensor(X_valid)
# Normalization
for i in range(40):
    x_test[:,i] = (x_test[:,i]-train_mean[i])/train_std[i]
    x_train[:,i] = (x_train[:,i]-train_mean[i])/train_std[i]
    x_valid[:,i] = (x_valid[:,i]-train_mean[i])/train_std[i]
y_valid = torch.Tensor(Y_valid).reshape(-1,1)

class Trainset(Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._len = x.shape[0]

    def __getitem__(self, item):  # 每次循环的时候返回的值
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

class Testset(Dataset):
    def __init__(self, x):
        self._x = x
        
        self._len = x.shape[0]

    def __getitem__(self, item):  # 每次循环的时候返回的值
        return item,self._x[item]

    def __len__(self):
        return self._len

batch_size = Y_train.shape[0]
train_loader = DataLoader(Trainset(x_train,y_train),
                            batch_size=batch_size,
                            shuffle=True)
test_loader = DataLoader(Testset(x_test),
                            batch_size=1,
                            shuffle=False)
valid_loader = DataLoader(Trainset(x_valid,y_valid),
                            batch_size=Y_valid.shape[0],
                            shuffle=False)

#损失函数（交叉熵）
loss_fn = nn.L1Loss()
#学习率
learning_rate = 0.0001
#优化器（Adam）

optimizer = torch.optim.Adam(neural.parameters(), lr=learning_rate)
#训练轮数
epoch = 100000



losslst = []
for i in range(epoch):
    epoch_loss = 0
    
    neural.train()
    for trainx,trainy in train_loader:
        trainx = trainx.cuda()
        trainy = trainy.cuda()
        outputs = neural(trainx) #前向传播

        
        
        loss = loss_fn(outputs,trainy) #计算损失函数
        loss = loss.mean()

        optimizer.zero_grad() #在进行下一次传播前将梯度清零
        loss.backward() #反向传播
        optimizer.step() #更新参数
        epoch_loss += loss.item()
    
    losslst.append(epoch_loss*batch_size/X_train.shape[0])
    
    neural.eval()
    if i % 100 == 0:
        
        print("训练轮数：{}，loss：{}".format(i, epoch_loss*batch_size/X_train.shape[0]))
        plt.plot(np.arange(len(losslst)),np.array(losslst))
        plt.savefig('loss.png')

    if i % 1000 == 0:
        
        for validx,validy in valid_loader:
            validx = validx.cuda()
            validy = validy.cuda()
            print("valid loss: {} ".format(loss_fn(neural(validx),validy).mean())) 
            file = open('valid.csv',"w")
            np.savetxt(file,np.concatenate((neural(validx).detach().cpu().numpy(),validy.detach().cpu().numpy()),axis=1),delimiter=',',fmt='%f')
        
        
        torch.save(neural,'model.pth')
        print('model has been saved')
        result = torch.zeros(x_test.shape)

        for i,testx in test_loader:
            testx = testx.cuda()
            result[i] = neural(testx).cpu()

        f = open('result.csv',"w")
        np.savetxt(f,result[:,0].detach().numpy(),delimiter=',',fmt='%f')
        print('result has been generated')
   









'''
Do not change the following lines, except for providing your answers and file name.
'''
Y_hat = result[:,0] # This is your estimated probabilities for the labels being 1, and its size is (1000,1)
filename = "xxx.csv" # Modify your file name 
np.savetxt(filename,Y_hat.detach().numpy(),delimiter = ",",fmt="%.8f") # Do not change this line, and send your results to TA before the deadline.




