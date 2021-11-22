import numpy as np
import glob

import numpy.random
from matplotlib import pyplot as plt
import matplotlib
import cv2
import time
import random
import torch
from torch.utils.data import DataLoader,Dataset
import os
import platform



# transforms an image from a random range to range 0-1
def feature_scaling(img):
    img = (img - np.min(img)) / (-np.min(img) + np.max(img))
    return img

#correct the colors when loading in the data
def color_correction(img):
    img_new = np.zeros(img.shape)
    img_new[0,:,:] = img[0,:,:]+ 0.485 / 0.229
    img_new[1,:,:] = img[1,:,:] + 0.456 / 0.224
    img_new[2,:,:] = img[2,:,:]+ 0.406 / 0.225
    return img_new

def one_hot2_2d(label):
    for i in range(9):
        label[i] = label[i]*i

    new_label = np.sum(label,axis=0)
    new_label.astype(np.uint8)
    end_label = np.zeros((1,label.shape[-2],label.shape[-1]))
    end_label[0] = new_label
    return end_label

#get all transformation functions


def scale(img,label):
    if img.shape[0] == 3:
        img = img.transpose([2, 1, 0])
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape

    if label.shape[0]==9:
        label = one_hot2_2d(label)
    label = label[0]
    scaling_factor = random.uniform(0.85,1.15)
    res = cv2.resize(img, (int(scaling_factor * rows), int(scaling_factor * cols)), interpolation=cv2.INTER_LINEAR)
    res_label = cv2.resize(label, (int(scaling_factor * rows), int(scaling_factor * cols)), interpolation=cv2.INTER_LINEAR)
    newimg = np.zeros(img.shape)
    newlabel = np.zeros(label.shape)


    if res.shape[1]<newimg.shape[1]:
        newimg[0:res.shape[0],0:res.shape[1]] = res
        newlabel[0:res.shape[0],0:res.shape[1]] = res_label
    else:
        newimg = res[0:newimg.shape[0],0:newimg.shape[1]]
        newlabel = res_label[0:newimg.shape[0],0:newimg.shape[1]]

    newlabel = np.round(newlabel)
    if newimg.shape[-1]==3:
        return newimg.transpose([2, 1, 0]),newlabel

    else:
        return newimg,newlabel


def translate(img,label):
    if img.shape[0] == 3:
        img = img.transpose([2, 1, 0])
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
    if label.shape[0] == 9:
        label = one_hot2_2d(label)
    label = label[0]
    dx = random.uniform(1, 90)
    dy = random.uniform(1, 90)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    newlabel = cv2.warpAffine(label, M, (cols, rows))

    if dst.shape[-1]==3:
        return dst.transpose([2, 1, 0]),newlabel
    else:
        return dst,newlabel

def rotate(img,label):

    if img.shape[0] == 3:
        img = img.transpose([2, 1, 0])
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape

    if label.shape[0] == 9:
        label = one_hot2_2d(label)
    label = label[0]
    theta = random.uniform(0,360)

    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), theta, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    newlabel = cv2.warpAffine(label, M, (cols, rows))

    if dst.shape[-1] == 3:
        return dst.transpose([2, 1, 0]),newlabel
    else:
        return dst,newlabel

def shear(img,label):
    if img.shape[0]==3:
        img = img.transpose([2, 1, 0])
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
    if label.shape[0] == 9:
        label = one_hot2_2d(label)
    label = label[0]
    # pt11 = [int(rows/3),int(cols/2)]
    # center = [int(rows/2),int(cols/2)]
    # pt13 = [int(2*rows / 3), int(cols / 2)]
    # dx = random.uniform(-20,20)
    # dy = random.uniform(-20,20)

    # pt21 = [pt11[0]+dx,pt11[1]+dy]
    # pt23 = [pt13[0] - dx, pt13[1] - dy]
    #
    # pts1 = np.float32([pt11, center, pt13])
    # pts2 = np.float32([pt21, center, pt23])
    shx = random.uniform(-0.3,0.3)
    shy = random.uniform(-0.3,0.3)

    M = np.float32([[1, shx, 0],
                    [shy, 1, 0],
                    [0, 0, 1]])
    # M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    newlabel = cv2.warpPerspective(label, M, (cols, rows))

    if dst.shape[-1]==3:
        return dst.transpose([2, 1, 0]),newlabel
    else:
        return dst,newlabel

def one_hot_encode(img=None,imgs=None):
    if img is None:
        num_pict = imgs.shape[0]
        Y = np.zeros((num_pict,9, 256, 256))
        imgs.astype('int8')
        for i in range(num_pict):
            for j in range(9):
                Y[i, j, :, :] = np.where(imgs[i] == j, 1, 0)
    elif imgs is None:
        Y = np.zeros(( 1, 256, 256))
        for j in range(9):
            img.astype('int8')
            Y[0, j, :, :] = np.where(img == j, 1, 0)
    else:
        raise Exception('only one of the two may be given')

    return Y
def load_data_2np(hot_encoding=False,valid_perc=0.1,train_perc=1.0,CAD_perc=0.2,show=0,background=True):
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
            X[i, :, :, :] = color_correction(imgx[:, :, :])

            if hot_encoding:
                for j in range(9):
                    imgy.astype('int8')
                    Y[i, j, :, :] = np.where(imgy == j, 1, 0)
            else:
                Y[i, :, :, :] = imgy
        return X,Y

    # get OS
    OS = platform.system().lower()
    #get list of all clean datafiles\
    folder = os.getenv('DEEP')
    if OS == 'windows':
        if background:
            train_folder_real = folder + '\\with_background\\train_real\\'
            test_folder = folder + '\\with_background\\test\\'
            train_folder_cad = folder + '\\with_background\\train_CAD\\'
            train_folder_trans = folder + '\\with_background\\train_transforms\\'
        else:
            train_folder_real = folder + '\\without_background\\train_real\\'
            test_folder = folder + '\\without_background\\test\\'
            train_folder_cad = folder + '\\without_background\\train_CAD\\'
            train_folder_trans = folder + '\\without_background\\train_transforms\\'
    elif OS == 'darwin':
        if background:
            train_folder_real = folder + '/with_background/train_real/'
            test_folder = folder + '/with_background/test/'
            train_folder_cad = folder + '/with_background/train_CAD/'
            train_folder_trans = folder + '/with_background/train_transforms/'
        else:
            train_folder_real = folder + '/without_background/train_real/'
            test_folder = folder + '/without_background/test/'
            train_folder_cad = folder + '/without_background/train_CAD/'
            train_folder_trans = folder + '/without_background/train_transforms/'
    else:
        raise Exception("Your OS is not accepted.")

    testlst = glob.glob(f'{test_folder}*')
    trainlst_real = glob.glob(f'{train_folder_real}*')
    trainlst_cad = glob.glob(f'{train_folder_cad}*')
    trainlst_trans = glob.glob(f'{train_folder_trans}*')

    # randomise the data list in order to have a varied validationlst
    random.seed(456)
    random.shuffle(testlst)
    random.shuffle(trainlst_real)
    random.shuffle(trainlst_cad)
    random.shuffle(trainlst_trans)

    datalst = trainlst_real+trainlst_cad[0:int(CAD_perc*len(trainlst_cad))]
    datalst = datalst[0:int(train_perc*len(datalst))]

    num_valid = int(len(datalst) * valid_perc)
    num_train = int(len(datalst) - num_valid)

    # assign the pictures to the right arrays
    X_train,Y_train = assign_to_array(num_pict=num_train,datalst=datalst[0:num_train])
    X_valid,Y_valid = assign_to_array(num_pict=num_valid,datalst=datalst[num_train::])
    X_trans,Y_trans = assign_to_array(num_pict=len(trainlst_trans),datalst=trainlst_trans)
    X_test,Y_test = assign_to_array(num_pict=len(testlst),datalst=testlst)

    for i in range(show):
        img = X_train[i].transpose([2,1,0])
        img = feature_scaling(img)
        cv2.imshow('loading', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return X_train,Y_train,X_valid,Y_valid, X_trans,Y_trans ,X_test,Y_test


class CustomDataset(Dataset):
    def __init__(self,data,labels,transform_arr=None,transform_labels=None,one_hot=False):
        if transform_arr is None:
            self.trans = False
            self.trans_labels = False
        else:
            self.trans = transform_arr
            self.trans_labels = transform_labels
        self.labels = labels
        self.data = data
        self.grayed=False
        self.one_hot = one_hot

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data,label

    def grayscale(self):
        self.grayed = True
        img_arr = np.zeros([self.data.shape[0]+self.trans.shape[0]]+list(self.data.shape[1::]))
        img_arr[0:self.data.shape[0]] = self.data.copy()
        img_arr[self.data.shape[0]::] = self.trans.copy()
        self.data = np.zeros([self.data.shape[0],self.data.shape[2],self.data.shape[3]])
        self.trans = np.zeros([self.trans.shape[0],self.trans.shape[2],self.trans.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = feature_scaling(img_arr[i])
            img = 0.299*img_arr[i,0]+0.587*img_arr[i,1]+0.114*img_arr[i,2]
            img = feature_scaling(img)
            if i<self.data.shape[0]:
                self.data[i]=img
            else:
                self.trans[i-self.data.shape[0]] = img

    def gray_gamma(self,gamma=1.2,show=0):
        """
        this functions raises every individual pixel to a desired power. positive gammas enlarges contrast at brighter parts, lower gammas enlar contrast for the darker parts
        :param gamma: the exponent
        :param show: default is false if true it will show every picture to evaluate it yourself, shut the program manually after seen the desired amount.

        """
        self.grayed = True

        img_arr = np.zeros([self.data.shape[0] + self.trans.shape[0]] + list(self.data.shape[1::]))
        img_arr[0:self.data.shape[0]] = self.data.copy()
        img_arr[self.data.shape[0]::] = self.trans.copy()
        self.data = np.zeros([self.data.shape[0], self.data.shape[2], self.data.shape[3]])
        self.trans = np.zeros([self.trans.shape[0], self.trans.shape[2], self.trans.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = feature_scaling(img_arr[i])
            img = 0.299*img_arr[i,0]+0.587*img_arr[i,1]+0.114*img_arr[i,2]
            img = feature_scaling(img)
            img = img**(gamma)
            if i<self.data.shape[0]:
                self.data[i]=img
            else:
                self.trans[i-self.data.shape[0]] = img

            if i<show:
                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def gray_gamma_enhanced(self, show=0, method='log',a=-1):
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

        img_arr = np.zeros([self.data.shape[0] + self.trans.shape[0]] + list(self.data.shape[1::]))
        img_arr[0:self.data.shape[0]] = self.data.copy()
        img_arr[self.data.shape[0]::] = self.trans.copy()
        self.data = np.zeros([self.data.shape[0], self.data.shape[2], self.data.shape[3]])
        self.trans = np.zeros([self.trans.shape[0], self.trans.shape[2], self.trans.shape[3]])
        for i in range(len(img_arr)):
            img_arr[i] = feature_scaling(img_arr[i])
            img = 0.299 * img_arr[i, 0] + 0.587 * img_arr[i, 1] + 0.114 * img_arr[i, 2]
            img = feature_scaling(img)
            mu = img.mean()
            if method=='linear':
                gamma = a*mu+b
            elif method == 'log':
                gamma = np.log(0.5)/np.log(mu)
            else:
                raise Exception(f'method must be \'log\' or \'linear\' instead of{method}')
            img = img**(gamma)

            if i<self.data.shape[0]:
                self.data[i]=img
            else:
                self.trans[i-self.data.shape[0]] = img
            if i<show:
                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow('', self.labels[i,0])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    def gray_log(self,show=0):
        """
        This will increase contrast of the darker colors.
        :param show: default is false if true it will show every picture to evaluate it yourself, shut the program manually after seen the desired amount.

        """
        self.grayed = True

        img_arr = np.zeros([self.data.shape[0] + self.trans.shape[0]] + list(self.data.shape[1::]))
        img_arr[0:self.data.shape[0]] = self.data.copy()
        img_arr[self.data.shape[0]::] = self.trans.copy()
        self.data = np.zeros([self.data.shape[0], self.data.shape[2], self.data.shape[3]])
        self.trans = np.zeros([self.trans.shape[0], self.trans.shape[2], self.trans.shape[3]])

        for i in range(len(img_arr)):
            img_arr[i] = feature_scaling(img_arr[i])
            img = 0.299 * img_arr[i, 0] + 0.587 * img_arr[i, 1] + 0.114 * img_arr[i, 2]
            img = feature_scaling(img)
            img = np.log(1+img)/(np.log(1+img.max()))
            if i < self.data.shape[0]:
                self.data[i] = img
            else:
                self.trans[i - self.data.shape[0]] = img

            if i<show:
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
            new_dat = np.zeros([self.data.shape[0]+self.trans.shape[0], 2, self.data.shape[-2], self.data.shape[-1]])
        else:
            new_dat = np.zeros([self.data.shape[0]+self.trans.shape[0],5,self.data.shape[-2],self.data.shape[-1]])


        img_arr = np.zeros([self.data.shape[0] + self.trans.shape[0]] + list(self.data.shape[1::]))
        img_arr[0:self.data.shape[0]] = self.data.copy()
        img_arr[self.data.shape[0]::] = self.trans.copy()
        self.data = np.zeros([self.data.shape[0], self.data.shape[1], self.data.shape[2]])
        self.trans = np.zeros([self.trans.shape[0], self.trans.shape[1], self.trans.shape[2]])

        for i in range(len(img_arr)):
            for idx, kernel in enumerate([kernel_l2r, kernel_r2l, kernel_t2b, kernel_b2t]):
                if merged:
                    img = img_arr[i]
                    img = apply_kernel(img, kernel)
                    img = feature_scaling(img)
                    img_arr[i] = img
                else:
                    img = img_arr[i]
                    img = apply_kernel(img,kernel)
                    img = feature_scaling(img)

                    new_dat[i,idx+1] = img

                if i<show:
                    cv2.imshow('', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if merged:
                new_dat[i,1] = img_arr[i]


        self.data = new_dat[0:self.data.shape[0]]
        self.trans = new_dat[self.data.shape[0]::]

    def transforms(self,p_rot=0.1,p_trans=0.1,p_zoom=0.1,p_shear=0.1,show=0):

        num_rot = int(p_rot * self.data.shape[0])
        num_trans = int(p_trans * self.data.shape[0])
        num_zoom = int(p_zoom * self.data.shape[0])
        num_shear = int(p_shear * self.data.shape[0])

        new_dat = np.zeros([self.data.shape[0]+num_rot+num_trans+num_zoom+num_shear]+list(self.data.shape[1::]))
        new_lab = np.zeros([self.data.shape[0] + num_rot + num_trans + num_zoom + num_shear] + [1]+list(self.labels.shape[-2::]))

        if self.one_hot:
            new_lab2 = np.zeros(
                [self.data.shape[0] + num_rot + num_trans + num_zoom + num_shear]  + list(self.labels.shape[1::]))

        # get a list of indexes for which we want a transformation this must be random.
        idx_dat = list(range(self.data.shape[0]))
        random.shuffle(idx_dat)
        idxlst_rot = idx_dat[0:num_rot]
        idxlst_trans = idx_dat[num_rot:num_rot+num_trans]
        idxlst_zoom = idx_dat[num_rot+num_trans:num_rot+num_trans+num_zoom]
        idxlst_shear = idx_dat[num_rot+num_trans+num_zoom:num_rot+num_trans+num_zoom+num_shear]

        # helper function to show each transform
        def local_show_func(i,show,img):
            if i < show:
                if img.shape[0] == 1:
                    cv2.imshow('', img[0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif len(img.shape) == 2:
                    cv2.imshow('', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                else:
                    print(img.shape)
                    cv2.imshow('', img.transpose([2, 1, 0]))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # get the transformations of those pictures and append them in newdat
        for i,idx in enumerate(idxlst_rot):
            new_dat[i],new_lab[i] = rotate(self.data[idx],self.labels[idx])

            local_show_func(i,show,new_dat[i])
            local_show_func(i, show, new_lab[i])

        for j,idx in enumerate(idxlst_trans):
            new_dat[i+j+1],new_lab[i+j+1] = translate(self.data[idx],self.labels[idx])
            local_show_func(j,show,new_dat[i+j])
            local_show_func(j, show, new_lab[i+j])

        for k,idx in enumerate(idxlst_zoom):
            new_dat[i+j+k+2],new_lab[i+j+k+2] = scale(self.data[idx],self.labels[idx])
            local_show_func(k, show, new_dat[i+j+k])
            local_show_func(k, show, new_lab[i+j+k])

        for l,idx in enumerate(idxlst_shear):
            new_dat[i+j+k+l+3],new_lab[i+j+k+l+3] = shear(self.data[idx],self.labels[idx])
            local_show_func(l, show, new_dat[i+j+k+l])
            local_show_func(l, show, new_lab[i+j+k+l])


        new_dat[i+j+k+l+4::] = self.data.copy()
        self.data = new_dat.copy()
        if self.one_hot:
            new_lab2[::i+j+k+l+4] = one_hot_encode(imgs=new_lab[::i+j+k+l+4])
            new_lab2[i+j+k+l+4::] = self.labels.copy()
            self.labels = new_lab2.copy()
        else:
            new_lab[i+j+k+l+4::] = self.labels.copy()
            self.labels = new_lab.copy()
        self.amount_transforms = i+j+k+l+4

    def remove_transforms(self):
        self.labels = self.labels[self.amount_transforms::]
        self.data = self.data[self.amount_transforms::]
        self.amount_transforms = 0

    def add_transforms(self,p_trans=None,number_trans=None):
        if self.trans is not False:
            index_list = list(range(self.trans.shape[0]))
            random.shuffle(index_list)

            #get the desired amount of transformations
            if p_trans is None and number_trans is None:
                Exception('p_trans or number_trans should be set to a number')
                amount = 0
            elif p_trans is not None:
                amount = int(p_trans * len(index_list))
            else:
                amount = number_trans

            # check that amount is lower than the length of the list
            amount = min(amount,len(index_list))

            # get random transformed images and add them tyo the self.data variable
            newdat = np.zeros([amount+self.data.shape[0]]+list(self.data.shape[1::]))
            newlab = np.zeros([amount+self.labels.shape[0]]+list(self.labels.shape[1::]))
            index_list = index_list[0:amount]
            for iter,idx in enumerate(index_list):
                newdat[iter] = self.trans[idx]
                newlab[iter] = self.trans_labels[idx]

            newdat[amount::] = self.data.copy()
            newlab[amount::] = self.labels.copy()
            self.data = newdat.copy()
            self.labels = newlab.copy()
            self.amount_transforms = amount
        else:
            raise Exception('To use this function you need to give a transforms array as an input.')


def test_pics(example_feat):

    for i in range(8):
        img = example_feat[i].numpy().transpose([2, 1, 0])
        imgC1, imgC2, imgC3 = feature_scaling(img)[:, :, 0], feature_scaling(img)[:, :, 1], feature_scaling(img)[:, :, 2]
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = imgC1, imgC2, imgC3
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # start a timer to see execution time
    start_time = time.time()


    #or do it yourself:
    #load in the data as np arrays (to just check it use a low train_perc to reduce runtime)
    X_train, Y_train, X_valid, Y_valid, X_trans, Y_trans, X_test, Y_test = load_data_2np(hot_encoding=True, valid_perc=0.1, train_perc=0.1,show=0,CAD_perc=0,background=True)

    print(X_train.shape,Y_train.shape)

    #change the data into a dataset object in order to use DataLoader functionality
    DS_train = CustomDataset(X_train,Y_train,X_trans,Y_trans,one_hot=True)

    DS_train.gray_gamma_enhanced(show=0)
    # DS_train.transforms()
    DS_train.add_transforms(number_trans=100)
    DS_train.remove_transforms()
    DS_train.get_edges(show=True,merged=False)
    DL_train = DataLoader(DS_train,batch_size=8,shuffle=True)
    # begin = time.perf_counter()
    # for i in range(50):
    #     # DS_train.transforms(p_trans=0.05,p_zoom=0.05,p_shear=0.05,p_rot=0.05)
    #     DS_train.add_transforms()
    #     DL_train = DataLoader(DS_train,batch_size=8,shuffle=True)
    #     DS_train.remove_transforms()
    # end = time.perf_counter()
    # print(f'it took {end-begin} seconds to run') #98.27 seconds to run this with in loop transforms and 74 seconds to load in the transforms beforehand

    print("--- %s seconds ---" % (time.time() - start_time))
