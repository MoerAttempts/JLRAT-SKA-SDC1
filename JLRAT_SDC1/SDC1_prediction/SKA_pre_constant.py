

BAND_WIDTH='B1'
# integration hours for 1000,100,8; format is string
integrationH='1000'

print('constants in Band',BAND_WIDTH, 'integration',integrationH)
# for generating catalog
EACH_SOURCE_FITTING_PARA_NUM=10

# pixel source distance scale in SKA1 challenge
# for example, source in astronomy bmaj/PIX2ASTRO_SCALE = pixel.bmaj
# pixel.bmaj * PIX2ASTRO_SCALE = source bmaj in astronomy

OBJ_THRES = 0.50


# for small img shape
LOCAL_IMG_SHAPE = [320, 320]

print('slice size',LOCAL_IMG_SHAPE)


EXTRE_MIN=-999999999999

SKA_IMG_SACLE=1e4

BOUNDARY_VALUE = 4

# norm data type
LOG_TYPE = 1

if BAND_WIDTH is 'B1':
    # BMAJ_in_pixel=  BMAJ/(CDELT2*3600)
    PIX2ASTRO_SCALE = 1.67847000000E-04 * 3600


    # for B1 image scale
    SKA_IMG_SACLE=1e4

    PXLs_in_BEAM = 6.981494759877087

    if integrationH == '1000':
        # 1000H log1p normalization
        MAXMIN_RANGE = 8.675326
        WHOLE_MEAN = 0.00012073634
        MAX_VALUE = 8.478551

        # v2, estimated from denoising model
        BK_SIGMA = 0.0003773196367546916










