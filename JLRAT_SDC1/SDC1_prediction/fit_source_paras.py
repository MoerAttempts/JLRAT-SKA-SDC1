"""
Copyright 2019-2030 yulei(yulei@nao.cas.cn)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

"""
This file is used to fitting SKA data challenge one source parameters.
Currently, we only fit 10 parameters for each source, not include SIZE
"""

from prediction_model_train import models
from skimage import filters
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from SDC1_prediction.read_cha_fits_data import *
import os
from SDC1_prediction.deno_predc import *
from SDC1_prediction.SKA_pre_constant import *
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import pandas as pd
from gaussfitter.gaussfitter import gaussfit
from skimage.draw import ellipse, ellipse_perimeter,rectangle_perimeter

import cv2
import matplotlib.pyplot as plt

class fit_source_paras:

    """
    build full model with combine Dncnn denoise model and multi-scale detection model
    """

    def __init__(self, dncnnModel_path, SKA_Model_path):

        # build full model

        dncnn_model = load_model(dncnnModel_path, compile=False)
        for layer in dncnn_model.layers:
            print(layer.name)

            if layer.name  == 'concatenate_1':
                print('im here')
                layer.name = layer.name + 'n'
        dncnn_model.summary()

        # print('-3',dncnn_model.layers[-3].name)




        SKA_model = models.load_model(SKA_Model_path, backbone_name='vgg16')
        SKA_model.summary()



        # if BAND_WIDTH == 'B5':
        #     """
        #     model.outputs must be a list ,
        #     model.output can be a tensor or a list, which depends on how to design the model.output
        #     """
        #     output_test = [
        #         SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[0],
        #         SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[1],
        #         SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[2],
        #         dncnn_model.output]
        #
        #     self.SKA_SFM = Model(inputs=dncnn_model.input, outputs=output_test, name='SKA_SFM')
        # else:
        #     self.SKA_SFM = Model(input=dncnn_model.input,
        #                              output=SKA_model(
        #                                  concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output])),
        #                              name='SKA_SFM')
        # output_test = [
        #     SKA_model(concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output])), dncnn_model.output]
        # self.SKA_SFM = Model(input=dncnn_model.input,
        #                      output=output_test,
        #                      name='SKA_SFM')

        # self.SKA_SFM = Model(input=dncnn_model.input,
        #                      output=SKA_model(
        #                          concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output])),
        #                      name='SKA_SFM')

        output_test = [
            SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[0],
            SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[1],
            SKA_model(inputs=concatenate([dncnn_model.output, dncnn_model.input, dncnn_model.output]))[2],
            dncnn_model.output]
        self.SKA_SFM = Model(inputs=dncnn_model.input, outputs=output_test, name='SKA_SFM')



    def convert_pixl2_astroCoord(self, local_wcs, local_core_idx, local_cen_idx, flux_inPxl, core_frac,
                                 pixel_BMAJ, pixel_BMIN, pa, class_id, source_score, dim1, dim2, x1, y1, x2, y2, ID):

        globFromlocal_core_RA_Dec = pixel_to_skycoord(local_core_idx[0], local_core_idx[1], local_wcs)
        globFromlocal_cen_RA_Dec = pixel_to_skycoord(local_cen_idx[0], local_cen_idx[1], local_wcs)

        # return [float(globFromlocal_core_RA_Dec[0]), float(globFromlocal_core_RA_Dec[1]), float(globFromlocal_cen_RA_Dec[0]),
        #         float(globFromlocal_cen_RA_Dec[1]), float(flux_inPxl/SKA_IMG_SACLE/PXLs_in_BEAM), float(core_frac), float(pixel_BMAJ * PIX2ASTRO_SCALE),
        #         float(pixel_BMIN * PIX2ASTRO_SCALE), float(pa), int(class_id),float(source_score),float(source_area)]
        return ['{ra_co:.8f}'.format(ra_co=globFromlocal_core_RA_Dec.ra.deg),
                '{dec_co:.8f}'.format(dec_co=globFromlocal_core_RA_Dec.dec.deg),
                '{ra_ce:.8f}'.format(ra_ce=globFromlocal_cen_RA_Dec.ra.deg),
                '{dec_ce:.8f}'.format(dec_ce=globFromlocal_cen_RA_Dec.dec.deg),
                '{flx:.5E}'.format(flx=flux_inPxl / SKA_IMG_SACLE / PXLs_in_BEAM),
                '{cof:.8f}'.format(cof=core_frac),
                '{maj:.3f}'.format(maj=pixel_BMAJ * PIX2ASTRO_SCALE),
                '{min:.3f}'.format(min=pixel_BMIN * PIX2ASTRO_SCALE),
                '{pa:.3f}'.format(pa=pa), '{cls:d}'.format(cls=class_id),
                '{score:.8f}'.format(score=source_score),
                '{d1:d}'.format(d1=dim1),
                '{d2:d}'.format(d2=dim2),
                '{x1:d}'.format(x1=x1),
                '{y1:d}'.format(y1=y1),
                '{x2:d}'.format(x2=x2),
                '{y2:d}'.format(y2=y2),
                '{ID:d}'.format(ID=ID)
                ]


    # log10 and normalization input fits image data
    # the input fits image data already with appropriate scale
    def pre_localImg_asInput(self, local_img_data):
        # normalized original fits data*SKA_IMG_SACLE
        local_img_norm = local_img_data

        # make log in local
        if BAND_WIDTH == 'B5':
            local_img_norm = np.log10(local_img_norm + 1)
        else:
            local_img_norm = np.log1p(local_img_norm)
        # normalization
        local_img_norm = (local_img_norm - WHOLE_MEAN) / MAXMIN_RANGE

        # used to adjust detection bbox
        self.shape = np.shape(local_img_data)

        # make img data to tensor
        # return to_tensor(local_img_norm)
        return local_img_norm



    def process_local_prediction(self, local_fits_img, local_wcs, dim1_idx,dim2_idx,glb_index,isDisp=False):

        local_img_data = local_fits_img.data
        # cut out global x,y values
        each_ymin = local_fits_img.ymin_original
        # each_ymax = local_fits_img.ymax_original
        each_xmin = local_fits_img.xmin_original
        # each_xmax = local_fits_img.xmax_original

        # record source parameters
        cols_name = ['RA_core', 'DEC_core', 'RA_centroid', 'DEC_centroid', 'FLUX', 'Corefrac', 'BMAJ',
                     'BMIN', 'PA', 'CLASS', 'score', 'dim1', 'dim2', 'x1', 'y1', 'x2', 'y2', 'ID']
        self.local_cat_pd = pd.DataFrame(columns=cols_name)



        # bdim1, bdim2 is used to check boundary source
        bdim1, bdim2 = np.shape(local_img_data)

        # each small image whole mask
        smallImg_mask = np.zeros((bdim1,bdim2), np.uint8)

        """
        local_img_norm is the real input for next steps
        """
        local_img_norm = self.pre_localImg_asInput(local_img_data)



        boxes, scores, labels, output_denoiseImg = self.SKA_SFM.predict_on_batch(
            to_tensor(local_img_norm))[:4]
        self.denoiseImg = from_tensor(output_denoiseImg)
        smallImg_mask[(local_img_norm > 3.0 * BK_SIGMA)] = 1


        scs_loc=scores
        dectObj_num = len(scs_loc[scs_loc >= OBJ_THRES])
        if dectObj_num < 1:
            if isDisp:
                print('detection None')
            return self.local_cat_pd
        else:
            if isDisp:
                print('analysis detection parameters')
        catalog_idx = 0

        # analysis each detection
        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            if score <= OBJ_THRES:
                continue
            # each bbox
            pre_box = np.round(box).astype(int)
            # x1,y1,x2,y2
            if pre_box[0] < BOUNDARY_VALUE or pre_box[1] < BOUNDARY_VALUE or pre_box[2] > bdim2 - BOUNDARY_VALUE or pre_box[3] > bdim1 - BOUNDARY_VALUE :
                continue

            # check bbox size , too small .. give up ..
            # if (pre_box[3] - pre_box[1]) <= 4 or (pre_box[2] - pre_box[0]) <= 4:
            #     # print('based on current background threshold, no binary pixels on with this detected source')
            #     continue


            each_source_mask = np.zeros((bdim1,bdim2), np.uint8)
            each_source_mask[pre_box[1]:pre_box[3], pre_box[0]:pre_box[2]] = smallImg_mask[pre_box[1]:pre_box[3],
                                                                             pre_box[0]:pre_box[2]]

            # check whether a binary image of the source is empty, based on the sigma background
            each_source_pix_num = len(each_source_mask[each_source_mask > 0])
            if BAND_WIDTH != 'B5' and each_source_pix_num < 4:
                # print('based on current background threshold, no binary pixels on with this detected source')
                continue

            # fit gaussian
            # B1  use norm img to test, others need to use denoise img
            # extract source image
            s_x1 = pre_box[0] - 1
            s_y1 = pre_box[1] - 1
            s_x2 = pre_box[2] + 1
            s_y2 = pre_box[3] + 1
            source_img = self.denoiseImg[s_y1:s_y2 + 1, s_x1:s_x2 + 1]
            # source_img = local_img_norm[s_y1:s_y2 + 1, s_x1:s_x2 + 1]
            height, amp, xfit, yfit, xwid, ywid, angle = gaussfit(source_img)
            if xwid >= bdim2 // 2 or ywid >= bdim1 // 2 or xwid == 0 or ywid == 0:
                continue
            # convert to
            dim1_e_s, dim2_e_s = ellipse(int(np.around(yfit + s_y1)), int(np.around(xfit + s_x1)),
                                         int(np.around(xwid * 1.5)),
                                         int(np.around(ywid * 1.5)),
                                         rotation=np.deg2rad(-angle))
            # denoise_img[dim1_e_s, dim2_e_s] = np.max(denoise_img)
            if len(dim1_e_s) == 0:
                continue
            x_min_rect = np.min(dim2_e_s)
            x_max_rect = np.max(dim2_e_s)
            y_min_rect = np.min(dim1_e_s)
            y_max_rect = np.max(dim1_e_s)

            if x_min_rect < 4 or y_min_rect < 4 or x_max_rect > bdim2 - 4 or \
                    y_max_rect > bdim1 - 4 or len(dim1_e_s) == 0:
                continue

            bk_num = len(smallImg_mask[smallImg_mask[dim1_e_s, dim2_e_s]] == 1)
            # source_msk_num = len(each_source_mask[each_source_mask == 1])


            each_source_flux_pixel = np.sum(local_img_data[dim1_e_s,dim2_e_s])
            BMAJ_pixl = xwid*3
            BMIN_pixl = ywid*3
            sou_PA = angle

            local_orignal_img = np.zeros(np.shape(local_img_data))
            local_orignal_img[pre_box[1]:pre_box[3], pre_box[0]:pre_box[2]] = local_img_data[pre_box[1]:pre_box[3],
                                                                              pre_box[0]:pre_box[2]]

            source_core_idx = np.unravel_index(np.argmax(local_orignal_img, axis=None), local_orignal_img.shape)
            source_core_xy = [source_core_idx[1], source_core_idx[0]]
            each_source_Corefraction = local_img_data[source_core_idx] / each_source_flux_pixel
            source_cen_xy = [xfit + s_x1, yfit + s_y1]

            cu_idx = len(self.local_cat_pd.index)
            self.local_cat_pd.loc[cu_idx, :] = self.convert_pixl2_astroCoord(local_wcs,
                                                                             source_core_xy, source_cen_xy,
                                                                             each_source_flux_pixel,
                                                                             each_source_Corefraction, BMAJ_pixl,
                                                                             BMIN_pixl, sou_PA, label + 1,
                                                                             float(score), dim1_idx, dim2_idx,
                                                                             pre_box[0] + each_xmin,
                                                                             pre_box[1] + each_ymin,
                                                                             pre_box[2] + each_xmin,
                                                                             pre_box[3] + each_ymin, glb_index + cu_idx)



        return self.local_cat_pd
