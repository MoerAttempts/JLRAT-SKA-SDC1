from SDC1_prediction.read_cha_fits_data import *
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

from fit_binary_img import *
from SKA_pre_constant import *

from deno_predc import *

import tensorflow as tf
import warnings

if __name__ == '__main__':
    print('Begins to Processing', BAND_WIDTH, ' data ....... ')
    # construct string
    folder_path = 'D:\\\pyProjects\\'
    fits_path = folder_path + 'SKAMid_' + BAND_WIDTH + '_'+integrationH+'h_v3.fits'

    # Read fits image and then scaled
    global_img, galobal_hdr = sdc_load_fits_image(filename=fits_path, isScale=True)
    global_wcs = WCS(galobal_hdr)

    fit_souce_obj = fit_binary_img()


    global_img_shape = np.shape(global_img)
    print('global_img_shape',global_img_shape)

    its_full_num = int(global_img_shape[0] // (LOCAL_IMG_SHAPE[0]//2))

    cols_name = ['RA_core', 'DEC_core', 'RA_centroid', 'DEC_centroid', 'FLUX', 'Corefrac', 'BMAJ',
                 'BMIN', 'PA', 'CLASS', 'score', 'dim1', 'dim2', 'x1', 'y1', 'x2', 'y2', 'ID']
    whole_catalog_pd = pd.DataFrame(columns=cols_name)

    if (its_full_num - 1) * LOCAL_IMG_SHAPE[0] - global_img_shape[0] < 0 or (its_full_num - 1) * LOCAL_IMG_SHAPE[0] - \
            global_img_shape[0] > 80:
        warnings.warn('iteration number with problem')
        # assert 'iteration number with problem'

    smallImg_w = LOCAL_IMG_SHAPE[0]
    hugeImg_w = global_img_shape[0]

    full_its_idx = 0
    progress_bar = tf.keras.utils.Progbar(target=its_full_num * its_full_num, verbose=1)

    for dim1_itr in range(0, its_full_num):
        for dim2_itr in range(0, its_full_num):
            centers_x = dim2_itr * int(smallImg_w//2) + int(smallImg_w // 2)
            centers_y = dim1_itr * int(smallImg_w//2) + int(smallImg_w // 2)
            if dim2_itr == its_full_num - 1:
                centers_x = global_img_shape[1] - int(smallImg_w / 2) - 1
            if dim1_itr == its_full_num - 1:
                centers_y = global_img_shape[0] - int(smallImg_w / 2) - 1

            # xrange = [centers_x + int(smallImg_w / 2 - 1), centers_x - int(smallImg_w / 2)]
            # yrange = [centers_y + int(smallImg_w / 2 - 1), centers_y - int(smallImg_w / 2)]
            if centers_x + int(smallImg_w / 4 - 1) > hugeImg_w or centers_y + int(
                    smallImg_w / 4 - 1) > hugeImg_w or centers_x - int(smallImg_w / 4) < 0 or centers_y - int(
                smallImg_w / 4) < 0:

                continue

            # extract local image to prediction
            local_fits_img = Cutout2D(global_img, position=[centers_x, centers_y], size=LOCAL_IMG_SHAPE, wcs=global_wcs,
                                      mode='partial')

            glb_index = len(whole_catalog_pd.index)
            local_img_catalog = fit_souce_obj.process_local_img(local_fits_img,
                                                                local_fits_img.wcs, dim1_itr,
                                                                dim2_itr, glb_index, area_thres=10)

            if local_img_catalog is not local_img_catalog.empty:
                whole_catalog_pd = pd.concat([whole_catalog_pd, local_img_catalog], ignore_index=True)
                print('  whole_catalog shape', whole_catalog_pd.shape)


            disp_str = '    Detection dim2: %d / %d   dim1: %d / %d   Percentage: %3.3f %%' % (
                dim2_itr, its_full_num - 1, dim1_itr, its_full_num - 1,
                float(full_its_idx / (its_full_num * its_full_num)) * 1e2)
            progress_bar.update(current=full_its_idx)
            print(disp_str, end='')

            # update full index
            full_its_idx += 1




    result_name = BAND_WIDTH + '_' + integrationH + 'h_append.csv'
    fmt = ['%-.8f', '%-.8f', '%-.8f', '%-.8f', '%-.5E', '%-.8f', '%-3.3f', '%-3.3f', '%-3.3f', '%-5d', '%.5f']
    whole_catalog_pd.to_csv(result_name, sep=' ', index=False, float_format=fmt)
    print(result_name, whole_catalog_pd.shape)





