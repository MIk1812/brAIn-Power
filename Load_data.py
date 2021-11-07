import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib
import cv2
import time
import random
import torch
from torch.utils.data import DataLoader,Dataset


# transforms an image from a random range to range 0-1
def normalise(img):
    img = (img - np.min(img)) / (-np.min(img) + np.max(img))
    return img



def load_data_2np(folder='car_segmentation_2021',hot_encoding=False,test_perc=0.01,valid_perc=0.1,train_perc=1.0):
    """
    :param folder: str: relative path to your car_segmentation_2021 folder
    :param percentage: float: how many in percentage yopu want to load
    :param hot_encoding: if set to true target will be onehotencoded instead of containing numbers from 0-8
    :return: input and target
    """

    def assign_to_array(num_pict, datalst):
        X = np.zeros((num_pict, 3, 256, 256))
        if hot_encoding:
            Y = np.zeros((num_pict, 9, 256, 256))
        else:
            Y = np.zeros((num_pict, 1, 256, 256))

        # read all data and assign to the right arrays
        for i in range(num_pict):
            img = np.load(datalst[i])
            imgx, imgy = img[0:3], img[-1]
            X[i, :, :, :] = imgx[:, :, :]

            if hot_encoding:
                for j in range(9):
                    imgy.astype('int64')
                    Y[i, j, :, :] = np.where(imgy == j, 1, 0)

            else:
                Y[i, :, :, :] = imgy
        return X,Y
    #get list of all clean datafiles
    datalst = glob.glob(f'{folder}\\clean_data\\clean_data\\*')
    num_test = int(len(datalst) * test_perc)
    num_valid = int(len(datalst) * valid_perc)
    num_train = min(int(len(datalst)-num_test-num_valid),int(len(datalst)*train_perc))


    #get list of *_a.npy files
    actual_lst = []
    for idx,name in enumerate(datalst):
        name = name.split('_')
        if name[-1]=="a.npy":
            actual_lst.append(datalst[idx])
            datalst.remove(datalst[idx])
        if len(actual_lst) >= num_test:
            assert 'maximum percentage is 1.89% for testdata'
            break

    #randomise the data list in order to have a varied validationlst
    random.seed(456)
    random.shuffle(datalst)

    # assign the pictures to the right arrays
    X_train,Y_train = assign_to_array(num_pict=num_train,datalst=datalst[0:num_train])
    X_valid,Y_valid = assign_to_array(num_pict=num_valid,datalst=datalst[num_train::])
    X_test,Y_test = assign_to_array(num_pict=num_test,datalst=actual_lst)

    #normalise the vectors
    mu1,mu2,mu3 = X_train[0].mean(),X_train[1].mean(),X_train[2].mean()
    sigma1,sigma2,sigma3 = X_train[0].std(),X_train[1].std(),X_train[2].std()

    X_train[0],X_train[1],X_train[2] = (X_train[0]-mu1+0.5)/np.sqrt(sigma1),(X_train[1]-mu2+0.5)/np.sqrt(sigma2),(X_train[2]-mu3+0.5)/np.sqrt(sigma3)

    return X_train,Y_train,X_valid,Y_valid ,X_test,Y_test


class CustomDataset(Dataset):
    def __init__(self,data,labels):
        self.labels = labels
        self.data = data
        self.grayed=False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data,label

    def grayscale(self):
        self.grayed = True
        img_arr = self.data
        self.data = np.zeros([self.data.shape[0],self.data.shape[2],self.data.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = normalise(img_arr[i])
            img = 0.299*img_arr[i,0]+0.587*img_arr[i,1]+0.114*img_arr[i,2]
            img = normalise(img)
            self.data[i]=img

    def gray_gamma(self,gamma=1.2,show=False):
        """
        this functions raises every individual pixel to a desired power. positive gammas enlarges contrast at brighter parts, lower gammas enlar contrast for the darker parts
        :param gamma: the exponent
        :param show: default is false if true it will show every picture to evaluate it yourself, shut the program manually after seen the desired amount.

        """
        self.grayed = True

        img_arr = self.data
        self.data = np.zeros([self.data.shape[0],self.data.shape[2],self.data.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = normalise(img_arr[i])
            img = 0.299*img_arr[i,0]+0.587*img_arr[i,1]+0.114*img_arr[i,2]
            img = normalise(img)
            img = img**(gamma)
            self.data[i]=img
            if show:
                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def gray_gamma_enhanced(self, show=False, method='log',a=-1):
        """
        this functions raises every individual pixel to a desired power. positive gammas enlarges contrast at brighter parts, lower gammas enlar contrast for the darker parts
        Difference with normal gamma is that this functions determines a gamma for every image based on their avarage pixel values
        a is used to scale the gammas. negative a means that average dark picture will will get increased contrast lighter part end vice versa.

        :param a: gamma = a*gamma+b (with condition that b=1-1/2*a) only used if method is linear
        :param method is a string can be linear or log if log the arg a is neglected and gamma is defined to get the average to 0.5
        :param show: default is false if true it will show every picture to evaluate it yourself, shut the program manually after seen the desired amount.

        """
        self.grayed = True

        method = method.lower()
        b = 1 - (0.5 * a)
        img_arr = self.data
        self.data = np.zeros([self.data.shape[0], self.data.shape[2], self.data.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = normalise(img_arr[i])
            img = 0.299 * img_arr[i, 0] + 0.587 * img_arr[i, 1] + 0.114 * img_arr[i, 2]
            img = normalise(img)
            mu = img.mean()
            if method=='linear':
                gamma = a*mu+b
            elif method == 'log':
                gamma = np.log(0.5)/np.log(mu)
            else:
                raise Exception(f'method must be \'log\' or \'linear\' instead of{method}')
            img = img**(gamma)
            self.data[i] = img
            if show:
                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def gray_log(self,show=False):
        """
        This will increase contrast of the darker colors.
        :param show: default is false if true it will show every picture to evaluate it yourself, shut the program manually after seen the desired amount.

        """
        self.grayed = True

        img_arr = self.data
        self.data = np.zeros([self.data.shape[0], self.data.shape[2], self.data.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = normalise(img_arr[i])
            img = 0.299 * img_arr[i, 0] + 0.587 * img_arr[i, 1] + 0.114 * img_arr[i, 2]
            img = normalise(img)
            img = np.log(1+img)/(np.log(1+img.max()))
            self.data[i] = img
            if show:
                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def get_edges(self,show=False,merged=False):
        # elementwise multiplication followed by summation
        def apply_kernel(img, kernel):
            img = cv2.filter2D(img,-1,kernel)
            return img


        # if data is not yet grayscaled do this
        if not self.grayed:
            self.grayscale()

        #define kernels to detect edges
        kernel_l2r =np.array([[-2,0,2],
                              [-2,0,2],
                              [-2,0,2]])
        kernel_r2l = np.array([[2,0,-2],
                               [2,0,-2],
                               [2,0,-2]])
        kernel_t2b = np.array([[-2,-2,-2],
                               [0,0,0],
                               [2,2,2]])
        kernel_b2t = np. array([[2,2,2],
                                [0,0,0],
                                [-2,-2,-2]])
        #define new data array to hold the grey image and the four edge detected images for decent use with dataloader
        if merged:
            new_dat = np.zeros([self.data.shape[0], 2, self.data.shape[-2], self.data.shape[-1]])
        else:
            new_dat = np.zeros([self.data.shape[0],5,self.data.shape[-2],self.data.shape[-1]])

        img_arr = self.data
        for i in range(len(img_arr)):
            for idx, kernel in enumerate([kernel_l2r, kernel_r2l, kernel_t2b, kernel_b2t]):
                if merged:
                    img = img_arr[i]
                    img = apply_kernel(img, kernel)
                    img = normalise(img)
                    img_arr[i] = img
                else:
                    img = img_arr[i]
                    img = apply_kernel(img,kernel)
                    img = normalise(img)

                    new_dat[i,idx+1] = img

                if show:
                    cv2.imshow('', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if merged:
                new_dat[i,1] = img_arr[i]

        self.data = new_dat

def load_data_2tensor(folder='car_segmentation_2021',hot_encoding=False,test_perc=0.01,valid_perc=0.1,batch_size=8,shuffle=False):
    # X,target = load_data_2np(folder,percentage=percentage,hot_encoding=hot_encoding)
    X_train,Y_train,X_valid,Y_valid,X_test,Y_test = load_data_2np(folder,hot_encoding,test_perc,valid_perc,train_perc=0.02)
    DS_train = CustomDataset(X_train,Y_train)
    DS_valid = CustomDataset(X_train, Y_train)
    DS_test = CustomDataset(X_train, Y_train)
    DL_train = DataLoader(DS_train, batch_size=batch_size ,shuffle=shuffle)
    DS_valid = DataLoader(DS_valid)
    DS_test = DataLoader(DS_test)

    return DL_train,DS_valid,DS_test


def test_pics(example_feat):

    for i in range(8):
        img = example_feat[i].numpy().transpose([2, 1, 0])
        imgC1, imgC2, imgC3 = normalise(img)[:, :, 0], normalise(img)[:, :, 1], normalise(img)[:, :, 2]
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = imgC1, imgC2, imgC3
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # start a timer to see execution time
    start_time = time.time()

    # This function loads in the data, makes batches and return it as a DataLoader objects used in the training loops
    # DL_train,_,_ = load_data_2tensor(folder='car_segmentation_2021',hot_encoding=True,batch_size=8,shuffle=False)
    # example_feat , example_labels  = next(iter(DL_train))
    #
    # print(f'feature batch shape: {example_feat.shape}')
    # print(f'label batch shape: {example_labels.shape}')
    #
    # test_pics(example_feat)


    #or do it yourself:
    #load in the data as np arrays (to just check it use a low train_perc to reduce runtime)
    X_train,Y_train,X_valid,Y_valid,X_test,Y_test = load_data_2np(hot_encoding=True, test_perc=0.01, valid_perc=0.1, train_perc=0.05)

    #change the data into a dataset object in order to use DataLoader functionality
    DS_train = CustomDataset(X_train,Y_train)
    DS_train.gray_gamma_enhanced()
    DS_train.get_edges(show=True,merged=False)
    DL_train = DataLoader(DS_train,batch_size=8,shuffle=True)





    print("--- %s seconds ---" % (time.time() - start_time))
