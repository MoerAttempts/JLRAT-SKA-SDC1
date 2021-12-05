
import numpy as np
from astropy.io import fits
from SKA_dataChallenge.SKA_pre_constant import *

"""
for flux calibration import
"""
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5



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

def sdc_load_fits_image(filename='SKAMid.fits',isScale=True,isDispHead=False):
    hdul = fits.open(filename)
    hdr = hdul[0].header

    if isScale:
        data = np.squeeze(hdul[0].data)*SKA_IMG_SACLE
    else:
        data = np.squeeze(hdul[0].data)

    if isDispHead:
        print(repr(hdr))
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
    hdul.close()
    return data, hdr

def sdc_load_catalog(filename='TrainingSet_B2_v2_ML.txt',skiprows=1,isSelection=True):
    # filein = dir + filename
    trainLog = np.loadtxt(filename, skiprows=skiprows)
    if isSelection:
        # col idx 12 is source  appear or not
        source_appear=trainLog[:,12]
        # source_remain=np.nonzero(source_appear)[0]
        source_remain=np.nonzero(source_appear)
        trainLog_remain=trainLog[source_remain]
        return trainLog_remain
    else:
        return trainLog