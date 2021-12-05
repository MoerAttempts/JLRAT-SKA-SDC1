import os
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from astropy.modeling.models import Ellipse2D
from skimage.draw import ellipse_perimeter
# dir ='F:\\tensorFlowProjects\\'

pix_scale = 1.67847000000E-04 * 3600 #0.6042
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


def sdc_load_catalog(filename='TrainingSet_B1_v2_ML.txt'):  # B1TrueLabel   ,TrainingSet_B1_v2
    filein = dir + filename
    trainLog = np.loadtxt(filein, skiprows=18)
    # col idx 12 is source  appear or not
    source_appear=trainLog[:,12]
    source_remain=np.nonzero(source_appear)[0]
    trainLog_remain=trainLog[source_remain,:]

    return trainLog_remain


def sdc_load_image(filename='SKAMid_B1_1000h_v3.fits'):
    # filein = dir + filename
    hdul = fits.open(filename)

    hdr = hdul[0].header
    # print(repr(hdr))
    data = np.squeeze(hdul[0].data)*1e4
    # print('in sdc_load_image min is ', np.min(data))
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

def sdc_load_LogImage(filename='SKAMid_B1_1000h_v3.fits'):
    # filein = dir + filename

    hdul = fits.open(filename)

    hdr = hdul[0].header
    # print(repr(hdr))
    data = np.squeeze(hdul[0].data)
    # print('in sdc_load_image min is ', np.min(data))
    hdul.close()
    # hdr['NAXIS'] = 2
    # hdr.remove('NAXIS4')
    # hdr.remove('NAXIS3')
    # hdr.remove('CDELT4')
    # hdr.remove('CDELT3')
    # hdr.remove('CRVAL4')
    # hdr.remove('CRVAL3')
    # hdr.remove('CRPIX4')
    # hdr.remove('CRPIX3')
    # hdr.remove('CTYPE4')
    # hdr.remove('CTYPE3')

    return data, hdr

# isReverse : Flase --> asending, week --> strong
#           :  True --> descend , strong --> week
def sdc_extra_sour(catalog, row=0,isReverse=True):
    ind = np.argsort(catalog[:, 5])  # asending
    if isReverse:
        ind = ind[::-1]  # reverse
    source = Source(catalog[ind[row]])
    # print('Line:{}; Id:{}; Flux:{}'.format(ind[row], source.id, source.flux))
    x_coor=catalog[:, 13]
    y_coor=catalog[:, -1]
    # print('x min:{}  x max:{}  y min:{} y max :{}'.format(np.min(x_coor),np.max(x_coor),np.min(y_coor),np.max(y_coor)))
    return source


def sdc_write_img(fileout, data, hdr):
    hdu = fits.PrimaryHDU(data, header=hdr)
    # hdu.writeto(fileout,overwrite=True)
    hdu.writeto(fileout, clobber=True)

    # print(fileout)


# find local and global rectangle coordinate
def extractRec(source,local_center):

    rr, cc = ellipse_perimeter(int(np.round(local_center[0])), int(np.round(local_center[1])), int(np.round(source.bmaj/pix_scale)+1),  int(np.round(source.bmin/pix_scale)+1),np.deg2rad(source.pa))

    right_down_x=int(np.max(rr))
    right_down_y=int(np.max(cc))
    left_top_x=int(np.min(rr))
    left_top_y=int(np.min(cc))
    return left_top_x,left_top_y,right_down_x,right_down_y





# a     Semimajor axis
# b     Semiminor axis
# angle Angle of the ellipse (in degrees)
def calculateEllipse(x, y, a, b, angle, steps):
    beta = -angle * (np.pi / 180)
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)

    alpha = np.linspace(0, 360, steps) * (np.pi / 180)
    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta)
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta)
    print('x size', X.shape)
    return X, Y

# -----------------------------------------------------
def sdc_image_cut(ratio=0.5, size=[201, 201], num=[0,10],isScale=False):
    catalog = sdc_load_catalog()
    data, hdr = sdc_load_image()
    print('data ************* min', np.min(data))
    wcs = WCS(hdr)



    for i in range(num[0],num[1]):
        # isReverse : False --> asending, week --> strong
        #           :  True --> descend , strong --> week
        # isReverse True is Strong Source, False is Week sources
        source = sdc_extra_sour(catalog, row=i,isReverse=True)
        pstr='source id %d and num %d '% (source.id,i)
        print(pstr)
        # pos_x=source.x-(size[0]/2)
        # if pos_x<1:
        #     pos_x=1
        #     print('x edge')
        # pos_y=source.y-(size[1]/2)
        # if pos_y<1:
        #     pos_y=1
        #     print('y edge')

        # PA is clockwise from the x-wise direction
        # if source.pa<=180 and source.pa>=0:
        #     angle=180-source.pa
        # elif source.pa >180 and source.pa<=360:
        #     angle = 360-source.pa
        # elif source.pa<0:
        #     angle = source.pa*-1

        # pix_scale = 1.67847000000E-04 * 3600
        # el_x,el_y=calculateEllipse(51, 51, source.bmaj,source.bmin, angle-90, 36*2)
        # print('pa',source.pa)
        # print('angle',angle)
        # print('maj  min',source.bmaj,source.bmin)
        # print('maj  min in pix', int(source.bmaj/pix_scale), int(source.bmin/pix_scale))
        pos = [source.x, source.y]
        # print('source pos', pos)
        cut_img = Cutout2D(data, position=pos, size=size, wcs=wcs,mode='strict')
        cut_data =np.log10(cut_img.data+1)
        max_v=np.max(cut_data)
        print('max value',max_v)
        print('min value after cut==========================',np.min(cut_data))


        center_coordinates = (int(size[0]/2),int(size[1]/2))

        axesLength = (int(np.round(source.bmaj/pix_scale))+1, int(np.round(source.bmin/pix_scale))+1)
        print('center',center_coordinates)
        print('axesl',axesLength)
        startAngle = 0
        endAngle = 360
        #  color in BGR
        # color = (np.max(cut_data), np.max(cut_data), np.max(cut_data))
        color=(int(np.ceil(max_v)),int(np.ceil(max_v)),int(np.ceil(max_v)))
        thickness = 1

        testimage = cv2.imread('D:\\SKA1cha\\FRCNN\MWCNN\\testsets\\Set14\\baboon.bmp')
        # cv2.imshow('2', (cut_data/np.max(cut_data)))
        dis_image = cv2.ellipse(np.flipud(cut_data), center_coordinates, axesLength,source.pa, startAngle, endAngle,color ,thickness)
        # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('image', dis_image)
        # cv2.imshow('2', dis_image)






        # rec
        x1,y1,x2,y2=extractRec(source,center_coordinates)
        # print('p1 p2',x1,y1,x2,y2)
        start_point = (x1, y1)
        end_point = (x2, y2)
        rec_image = cv2.rectangle(dis_image, start_point, end_point, color, thickness)
        #
        # ellipse_area=np.pi*(source.bmaj/pix_scale/2+1)*(source.bmin/pix_scale/2+1)
        # ellipse_rate=float(ellipse_area)/(float(np.abs(x1-x2)*np.abs(y1-y2)))
        # print('source rec area', x1,y1,x2,y2,(np.abs(x1-x2)*np.abs(y1-y2)))
        # print('ellipse area',ellipse_area)
        # print('ellipse rate',ellipse_rate)



        dis_height =size[0]*2
        cv2.namedWindow('1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('1', dis_height, dis_height)
        cv2.imshow('1',rec_image)



        if isScale:   cut_data=cut_data/np.max(cut_data)

        # print('img shape',cut_img.shape)
        scal_cut_data = cut_data / np.max(cut_data)
        # Img = np.stack((scal_cut_data,) * 3, axis=-1)
        # # im = Image.fromarray(Img)
        # print(Img.shape)
        # elx=el_x.astype(int)
        # ely = el_y.astype(int)
        # Img[elx,ely]=[255,0,0]
        # flux_max = np.max(cut_data)
        #
        # lab_data = np.zeros(cut_data.shape, dtype=np.int)
        # ind = (cut_data > flux_max * ratio)
        # lab_data[(cut_data > flux_max * ratio)] = 1
        # plt.figure(1)
        # plt.imshow(Img, origin='lower')
        # plt.show()

        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(cut_data, origin='lower', interpolation='none', cmap='Greys_r')
        # e2 = mpatches.Ellipse(center_coordinates, source.bmaj/pix_scale,source.bmin/pix_scale, angle, edgecolor='red',facecolor='none')
        # print('ellipse',e2)
        # ax.add_patch(e2)
        # plt.show()


        mat=cv2.UMat.get(rec_image )
        # log1p() and exmp1()
        # print('max log1p',np.max(np.log1p(mat)))
        # print('min log1p',np.min(np.log1p(mat)))
        plt.imshow(np.log1p(mat), origin='upper') #'upper' lower
        plt.show()



        #    plt.subplot(projection=cut_img.wcs)
        #    plt.imshow(cut_data,origin='lower')
        #    plt.imshow(lab_data,origin='lower')
        #    plt.contour(cut_data)
        #    plt.show()


        # fileout_img='B1_V8_SPLIT\\id_%d_%d_Image_SKAMid_B1_8h_v3.fits'%(i,source.id)
        # fileout_img = 'B1_V8_SPLIT_week\\id_%d_%d_Image_SKAMid_B1_8h_v3.fits' % (i, source.id)
        # fileout_img = 'B1_V100_SPLIT\\id_%d_%d_Image_SKAMid_B1_100h_v3.fits' % (i, source.id)
        # fileout_img = 'D:\SKA1cha\FRCNN\dncnn_keras\data\Test\Set12\\id_%d_%d_Image_SKAMid_B1_8h_v3.fits' % (i, source.id)
        # fileout_jpg = 'B1_V1000_SPLIT\\id_%d_%d_Image_SKAMid_B1_1000h_v3.jpg' % (i, source.id)
        # fileout_img = 'B1_V1000_SPLIT_week\\id_%d_%d_Image_SKAMid_B1_1000h_v3.fits' % (i, source.id)

        # hdr.update(cut_img.wcs.to_header())
        # sdc_write_img(fileout_img, cut_img.data, hdr)
        # scipy.misc.imsave(fileout_jpg, Image.Img)
        # im.save(fileout_jpg)

        # sdc_write_img(fileout_lab, lab_data, hdr)


if __name__ == '__main__':
    ratio = 1/3.0
    size = [181,181]
    num = [0,100]
    sdc_image_cut(ratio=ratio,size=size,num=num,isScale=False)
