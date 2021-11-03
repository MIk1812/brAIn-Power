import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib
import cv2
import time
import torch
from torch.utils.data import DataLoader,Dataset

# transforms an image from a random range to range 0-1
def normalise(img):
    img = (img - np.min(img)) / (-np.min(img) + np.max(img))
    return img

def load_data_2np(folder='car_segmentation_2021',percentage=0.01,hot_encoding=False):
    """
    :param folder: str: relative path to your car_segmentation_2021 folder
    :param percentage: float: how many in percentage yopu want to load
    :return: input and target
    """

    #get list of all clean datafiles
    datalst = glob.glob(f'{folder}\\clean_data\\clean_data\\*')
    num_pict = int(len(datalst)*percentage)

    X = np.zeros((num_pict,3,256,256))
    if hot_encoding:
        Y = np.zeros((num_pict,9,256,256))
    else:
        Y = np.zeros((num_pict, 1, 256, 256))

    # read all data and assign to the right arrays
    for i in range(num_pict):
        img = np.load(datalst[i])
        imgx,imgy = img[0:3],img[-1]
        X[i,:,:,:] = imgx[:,:,:]

        # print(normalise(imgx.reshape(256,256,3)).max(),normalise(imgx.reshape(256,256,3)).min())
        # cv2.imshow('', normalise(imgx).transpose([2,1,0]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if hot_encoding:
            for j in range(9):
                imgy.astype('int64')
                Y[i, j, :, :] = np.where(imgy==j,1,0)

        else:
            Y[i, :, :, :] = imgy


    return X,Y



class CustomDataset(Dataset):
    def __init__(self,data,labels):
        self.labels = labels
        self.data = data
        print('line59',self.data.min(),self.data.max())
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data,label


def load_data_2tensor(folder='car_segmentation_2021',percentage=0.01,hot_encoding=False,batch_size=8,shuffle=False):
    X,target = load_data_2np(folder,percentage=percentage,hot_encoding=hot_encoding)
    DS = CustomDataset(X,target)
    DL = DataLoader(DS, batch_size=batch_size ,shuffle=shuffle)
    return DL




if __name__ == '__main__':
    # start a timer to see execution time
    start_time = time.time()

    # This function loads in the data, makes batches and return it as a DataLoader object used in the training loops
    DL = load_data_2tensor(folder='car_segmentation_2021',percentage=1,hot_encoding=True,batch_size=8,shuffle=False)
    example_feat , example_labels  = next(iter(DL))

    print(f'feature batch shape: {example_feat.shape}')
    print(f'label batch shape: {example_labels.shape}')

    # img = example_feat[0].numpy().transpose([2, 1, 0])
    # imgC1,imgC2,imgC3 = normalise(img)[:,:,0],normalise(img)[:,:,1],normalise(img)[:,:,2]
    # img[:,:,0],img[:,:,1],img[:,:,2] = imgC3,imgC2,imgC1
    #
    # cv2.imshow('',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    print("--- %s seconds ---" % (time.time() - start_time))
