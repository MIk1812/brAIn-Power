import glob
import os

from LoadData import scale,rotate,translate,shear
import numpy as np
import cv2
import time
import random
def getnoback(fname):

    dat = np.zeros((13, 256, 256))
    img = cv2.imread(fname)
    img = np.array(img)
    fname = fname.split('\\')

    label = np.load(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\clean_data' + '\\' +
                    fname[-1].split('.')[0] + '.npy')

    dat[0:3] = img.transpose([2, 1, 0])
    dat[-1] = label[-1]
    return dat


raws = glob.glob(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\correct_raw\*')

# get only non aug pics
testing = []
temp_datalst = raws.copy()
for idx,name in enumerate(temp_datalst):
    name = name.split('_')
    if name[-1]=="a.jpg":# and int(name[-2][-2::])<52:
        name[-2] = name[-2].split('\\')
        if int(name[-2][-1])<=52:
            testing.append(temp_datalst[idx])
            raws.remove(temp_datalst[idx])

    for jdx,j in enumerate(name):
        if j[-3::]=='aug':
            raws.remove(temp_datalst[idx])


# store lat bit of file names
test_fnames = []
for fname in testing:
    fname = fname.split('\\')[-1].split('.')[0]+'.npy'
    test_fnames.append(fname)

train_fnames = []
for fname in raws:
    fname = fname.split('\\')[-1].split('.')[0]+'.npy'
    train_fnames.append(fname)

# get opel and doors last bit:
flst = glob.glob(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\clean_data\*')
OPEL_DOOR_lst = []
for fname in flst:
    file = fname.split('\\')[-1].split('_')[0]
    if file == "DOOR" or file == "OPEL":
        OPEL_DOOR_lst.append(fname.split('\\')[-1])


# #place test pics in the right files
# for idx,ftest in enumerate(test_fnames):
#     img_arr_with_b = np.load(f'C:\\Users\\oscar\\PycharmProjects\\brAIn-Power\\car_segmentation_2021\\clean_data\\{ftest}')
#     fbtest = ftest.split('.')[0]+'.png' #no background pics are in jpg format so adjust foldermap
#     img_arr_without_b = getnoback(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\output\\'+fbtest)
#     np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\with_background\test\\'+ftest,img_arr_with_b)
#     np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\without_background\test\\'+ftest,img_arr_without_b)

# # place normal train pics in right folders
#     for idx, ftrain in enumerate(train_fnames):
#         img_arr_with_b = np.load(
#             f'C:\\Users\\oscar\\PycharmProjects\\brAIn-Power\\car_segmentation_2021\\clean_data\\{ftrain}')
#         fbtrain = ftrain.split('.')[0] + '.png'  # no background pics are in jpg format so adjust foldermap
#         img_arr_without_b = getnoback(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\output\\' + fbtrain)
#         np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\with_background\train_real\\'+ftrain,img_arr_with_b)
#         np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\without_background\train_real\\'+ftrain,img_arr_without_b)


# # place cad train pics in right folders
# for idx, fcad in enumerate(OPEL_DOOR_lst):
#     try:
#         img_arr_with_b = np.load(
#             f'C:\\Users\\oscar\\PycharmProjects\\brAIn-Power\\car_segmentation_2021\\clean_data\\{fcad}')
#         fbcad = fcad.split('.')[0] + '.png'  # no background pics are in jpg format so adjust foldermap
#         img_arr_without_b = getnoback(r'C:\Users\oscar\PycharmProjects\brAIn-Power\car_segmentation_2021\output\\' + fbcad)
#         np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\with_background\train_CAD\\'+fcad,img_arr_with_b)
#         np.save(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\without_background\train_CAD\\'+fcad,img_arr_without_b)
#     except:
#         print('check',fcad)

#
# train_real = glob.glob(r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\with_background\train_real\*')
#
# funcs = [shear,translate,rotate,scale]
# func_names = ['shear','translate','rotate','scale']
# for i,fname in enumerate(train_real):
#     for idx, fun in enumerate(funcs):
#         dat = np.load(fname)
#         img = dat[0:3]
#         label = dat[-1]
#         img,label = fun(img, label)
#
#         dat[0:3] = img
#         dat[-1] = label
#         fname2 = fname.split('.')
#         newfname = r'C:\Users\oscar\PycharmProjects\brAIn-Power\correct_data\with_background\train_transforms\\' + fname2[0].split('\\')[-1]+f'_{func_names[idx]}.npy'
#         np.save(newfname,dat)
#
#     if i%20==0:
#         print(f'were at  of {1/len(train_real)*(i+1)*100} percent')
#
#
#
#
