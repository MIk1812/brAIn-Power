from LoadData import CustomDataset,load_data_2np
from Unet import UNet50 as UN50
import torch
import time
import numpy as np
from torch.utils.data import DataLoader,Dataset
# define some helper functions for later:
def IoU_score(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


#Load in data
X_train, Y_train, X_valid, Y_valid, X_trans, Y_trans, X_test, Y_test = load_data_2np(hot_encoding=True, valid_perc=0.1, train_perc=0.1,show=0,CAD_perc=0,background=True)
DS_train = CustomDataset(X_train, Y_train,one_hot=True)
DS_valid = CustomDataset(X_train, Y_train,one_hot=True)
DS_test = CustomDataset(X_test,Y_test,one_hot=True)

#perform preprocessing:
DS_train.gray_gamma_enhanced()
DS_valid.gray_gamma_enhanced()
DS_test.gray_gamma_enhanced()

DL_valid = DataLoader(DS_valid)
#load in model
net = UN50.UNet50(n_classes=9,rgb=False)
net.double()

#creat lossfunction and optimizer
criterion = UN50.DiceLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)

#define hyper parameters
net.train()
NUM_EPOCHS = 2
check_at = 2

#define list to stor intermediat results
valid_iter = []
valid_loss = []
valid_iou = []

train_iter = []
train_loss = []
train_iou = []
#training loop
for epoch in range(NUM_EPOCHS):
    DS_train.transforms()
    DL_train = DataLoader(DS_train,batch_size=4,shuffle=True)

    net.train()
    for i,data in enumerate(DL_train):
        input = data[0].type(torch.DoubleTensor)
        target = data[1].type(torch.DoubleTensor)
        #train the network
        optimizer.zero_grad()
        output = net(input)
        print('befor')
        loss = criterion(output,target)
        print('check loss: ',loss)
        loss.backward()
        optimizer.step()

        #store the results
        train_iter.append(epoch*len(DL_train)+i)
        train_loss.append(loss)

    #remove the transforms again
    DS_train.remove_transforms()

    #validate
    if epoch%check_at==0:
        net.eval()
        L = 0
        for j,valid_data in enumerate(DL_valid):
            input = valid_data[0].type(torch.DoubleTensor)
            target = valid_data[1].type(torch.DoubleTensor)
            output = net(input)
            loss = criterion(output, target)
            L += loss
            valid_iou.append(IoU_score(target.detach().numpy(),output.detach().numpy()))
        valid_iter.append(epoch)
        valid_loss.append(L/j)

        print(f'At epoch {epoch} Training loss is at {train_loss[-1]}')
        print(f'At epoch {epoch} Validation loss is at {valid_loss[-1]} and the IoU is at {valid_iou[-1]}%')




