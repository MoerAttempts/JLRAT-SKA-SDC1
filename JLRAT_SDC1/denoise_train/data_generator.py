# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# no need to run this code separately


import glob
#import os
import cv2
import numpy as np
#from multiprocessing import Pool
# from fitsData_generator import *

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from SDC1_prediction.SKA_pre_constant import *

# patch_size, stride = 41, 10
patch_size, stride = 101, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
# batch_size = 128
# batch_size=128
batch_size = 16

#
class Source:
    def __init__(self, para=''):
        self.id = 0
        self.ra_core = 0.0
        self.dec_core = 0.0
        self.ra_cent = 0.0
        self.dec_cent = 0.0
        self.flux = 0.0
        self.core_frac = 0.0
        self.bmaj = 0.0
        self.bmin = 0.0
        self.pa = 0.0
        self.size = 0
        self.type = 0
        self.selection = 0
        self.x = 0.0
        self.y = 0.0
        if len(para) > 0:
            self.set_para(para)

    def set_para(self, para):
        self.id = int(para[0])
        self.ra_core = para[1]
        self.dec_core = para[2]
        self.ra_cent = para[3]
        self.dec_cent = para[4]
        self.flux = para[5]
        self.core_frac = para[6]
        self.bmaj = para[7]
        self.bmin = para[8]
        self.pa = para[9]
        self.size = para[10]
        self.type = para[11]
        self.selection = para[12]
        self.x = para[13]
        self.y = para[14]


def sdc_load_catalog(filename='TrainingSet_B1_v2.txt',skiprows=18):
    print('load catalog')
    dir ='D:\pyProjects\\'
    filein = dir + filename
    trainLog = np.loadtxt(filein, skiprows=skiprows)
    # col idx 12 is source  appear or not
    source_appear=trainLog[:,12]
    source_remain=np.nonzero(source_appear)[0]
    trainLog_remain=trainLog[source_remain,:]

    return trainLog_remain


def sdc_load_image(filename):
    # filein = dir + filename
    hdul = fits.open(filename)

    hdr = hdul[0].header
    # print(repr(hdr))
    data = np.squeeze(hdul[0].data)*1e4

    # scale ,how to scale 0 -1?
    # data=data*1e4
    hdul.close()
    hdr['NAXIS'] = 2
    hdr.remove('NAXIS4')
    hdr.remove('NAXIS3')
    hdr.remove('CDELT4')
    hdr.remove('CDELT3')
    hdr.remove('CRVAL4')
    hdr.remove('CRVAL3')
    hdr.remove('CRPIX4')
    hdr.remove('CRPIX3')
    hdr.remove('CTYPE4')
    hdr.remove('CTYPE3')

    return data, hdr

# isReverse : Flase --> asending, week --> strong
#           :  True --> descend , strong --> week
def sdc_extra_sour(catalog, row=0,isReverse=True):
    ind = np.argsort(catalog[:, 5])  # asending
    if isReverse:
        ind = ind[::-1]  # reverse
    source = Source(catalog[ind[row]])
    return source

def sdc_write_img(fileout, data, hdr):
    hdu = fits.PrimaryHDU(data, header=hdr)
    # hdu.writeto(fileout,overwrite=True)
    hdu.writeto(fileout, clobber=True)

    print(fileout)

def sdc_image_patches(catalog,filename_high,filename_low, size=[41, 41], num=[10,100],channels=1):



    data_high, hdr1 = sdc_load_image(filename=filename_high)

    data_low, hdr2 = sdc_load_image(filename=filename_low)

    wcs_high = WCS(hdr1)
    wcs_low = WCS(hdr2)

    patches_high = []
    patches_low=[]


    for i in range(num[0],num[1]):
        # isReverse : Flase --> asending, week --> strong
        #           :  True --> descend , strong --> week
        # isReverse True is Strong Source, False is Week sources
        source = sdc_extra_sour(catalog, row=i,isReverse=True)
        pos = [source.x, source.y]

        cut_fits_high = Cutout2D(data_high, position=pos, size=size, wcs=wcs_high)
        if LOG_TYPE == 1:
            cut_img_data_high = np.log1p(cut_fits_high.data)
        else:
            cut_img_data_high = np.log10(cut_fits_high.data + 1)
        # base = 6
        # cut_img_data_high = np.log(cut_fits_high.data + 1) / np.log(base)
        cut_img_data_high = np.delete(cut_img_data_high, -1, 0)
        cut_img_data_high = np.delete(cut_img_data_high, -1, 1)
        # cut_img_data_high-=WHOLE_MEAN
        cut_img_data_high = (cut_img_data_high - WHOLE_MEAN)
        # if np.max(cut_img_data_high) >alow_max:
        #     continue

        if channels is not 1:
            cut_img_data_high_3cha = np.stack((cut_img_data_high, cut_img_data_high, cut_img_data_high), axis=-1)
            patches_high.append(cut_img_data_high_3cha)
        else:
            patches_high.append(cut_img_data_high)

        cut_fits_low = Cutout2D(data_low, position=pos, size=size, wcs=wcs_low)
        # print('mini ', np.min(cut_fits_low.data))
        if LOG_TYPE == 1:
            cut_img_data_low = np.log1p(cut_fits_low.data)
        else:
            cut_img_data_low = np.log10(cut_fits_low.data + 1)
        # cut_img_data_low = np.log(cut_fits_low.data + 1) / np.log(base)
        cut_img_data_low = np.delete(cut_img_data_low, -1, 0)
        cut_img_data_low = np.delete(cut_img_data_low, -1, 1)
        # cut_img_data_low-=WHOLE_MEAN
        cut_img_data_low = (cut_img_data_low - WHOLE_MEAN)
        if channels is not 1:
            cut_img_data_low_3cha = np.stack((cut_img_data_low, cut_img_data_low, cut_img_data_low), axis=-1)
            patches_high.append(cut_img_data_low_3cha)
        else:
            patches_low.append(cut_img_data_low)

        # print('patch shape',np.shape(cut_img_data_low))



        # print('exatract patches',i)
    print('finish patches extracton')
    print(np.shape(patches_high))
    return patches_high, patches_low

# #####

def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches , generate many patches from one image
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                #patches.append(x)        
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
                
    return patches

def datagenerator(cata_name,filename_high,filename_low, skiprows,data_dir='D:\pyProjects\\',  channels=1):
    
    # file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    filename_high=data_dir + filename_high
    filename_low=data_dir + filename_low
    print('load high file',filename_high)
    print('load low file', filename_low)

    catalog = sdc_load_catalog(filename=cata_name,skiprows=skiprows)
    print('load catalog file with skip rows: ',cata_name,skiprows)
    cata_dim1,dim2=np.shape(catalog)

    maxIdx=cata_dim1//(batch_size*20)
    # initrialize
    # data = []
    # generate patches
    # 400 img # (596,40,40) patches
    # for i in range(len(file_list)):
    #     patch = gen_patches(file_list[i])
    #     data.append(patch)
    #     if verbose:
    #         print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')  #
    if BAND_WIDTH == 'B5':
        patches_high, patches_low = sdc_image_patches(catalog, filename_high, filename_low,
                                                      size=[patch_size, patch_size], num=[0, 1310],
                                                      channels=channels)
    # fit patches
    else:
        fileIdx = np.random.randint(low=1, high=maxIdx)
        patches_high, patches_low = sdc_image_patches(catalog, filename_high, filename_low,
                                                      size=[patch_size, patch_size],
                                                      num=[(batch_size * 20) * (fileIdx - 1),
                                                           (batch_size * 20) * fileIdx],
                                                      channels=channels)











    # data = np.array(data, dtype='uint8')
    patches_high= np.array(patches_high, dtype='float32')
    patches_low= np.array(patches_low, dtype='float32')
    # print('data shape:',np.shape(data))
    patches_high = patches_high.reshape((patches_high.shape[0],patches_high.shape[1],patches_high.shape[2],channels))
    patches_low = patches_low.reshape((patches_low.shape[0],patches_low.shape[1],patches_low.shape[2],channels))
    # print('len data',len(data))
    # delete ,make mod==0
    discard_n = len(patches_high)-len(patches_high)//batch_size*batch_size
    patches_high = np.delete(patches_high,range(discard_n),axis = 0)
    patches_low = np.delete(patches_low,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return patches_high,patches_low

if __name__ == '__main__':   

    data = datagenerator()
    

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
