from LoadData import CustomDataset,load_data_2np
from Unet import UNet50 as UN50
import torch
import time
import numpy as np
from torch.utils.data import DataLoader,Dataset

#Load in data
X_train, Y_train, X_valid, Y_valid, X_trans, Y_trans, X_test, Y_test = load_data_2np(hot_encoding=True, valid_perc=0.1, train_perc=0.1,show=0,CAD_perc=0,background=True)
DS_train = CustomDataset(X_train, Y_train,one_hot=True)
DS_valid = CustomDataset(X_train, Y_train,one_hot=True)
DS_test = CustomDataset(X_test,Y_test,one_hot=True)

#perform preprocessing:
# DS_train.gray_gamma()
# DS_valid.gray_gamma()
# DS_test.gray_gamma()

#load in model
net = UN50.UNet50(n_classes=9)
net.double()

#creat lossfunction and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)

#define hyper parameters
net.train()
NUM_EPOCHS = 5
check_at = 1
#training loop
for epoch in range(NUM_EPOCHS):
    DS_train.transforms()
    DL_train = DataLoader(DS_train,batch_size=8,shuffle=True)

    for i,data in enumerate(DL_train):
        input = data[0].type(torch.DoubleTensor)
        target = data[1].type(torch.DoubleTensor)
        #train the network
        net.train()
        optimizer.zero_grad()
        output = net(input)
        print(output.shape)
        print(target.shape)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

    #remove the transforms again
    DS_train.remove_transforms()





