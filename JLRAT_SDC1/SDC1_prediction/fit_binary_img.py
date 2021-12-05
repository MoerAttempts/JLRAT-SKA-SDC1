
from skimage.morphology import disk, dilation,remove_small_objects
from skimage.measure import label, regionprops

from SDC1_prediction.SKA_pre_constant import *
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

import pandas as pd
from gaussfitter.gaussfitter import gaussfit
from skimage.draw import ellipse, ellipse_perimeter,rectangle_perimeter
import numpy as np

class fit_binary_img:

    def __init__(self):
        # self.original_img = original_img
        print('may load denoise model for paras fitting')


    def pre_localImg_asInput(self, local_img_data):
        # normalized original fits data*SKA_IMG_SACLE
        local_img_norm = local_img_data
        # make log in local
        local_img_norm = np.log1p(local_img_norm)
        # normalization
        local_img_norm = (local_img_norm - WHOLE_MEAN) / MAXMIN_RANGE

        return local_img_norm

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

    def process_local_img(self, local_fits_img, local_wcs, dim1_idx,dim2_idx,glb_index, area_thres):

        # record source parameters
        cols_name = ['RA_core', 'DEC_core', 'RA_centroid', 'DEC_centroid', 'FLUX', 'Corefrac', 'BMAJ',
                     'BMIN', 'PA', 'CLASS', 'score', 'dim1', 'dim2', 'x1', 'y1', 'x2', 'y2', 'ID']
        self.local_cat_pd = pd.DataFrame(columns=cols_name)

        local_img_data = local_fits_img.data
        # cut out global x,y values
        each_ymin = local_fits_img.ymin_original
        each_xmin = local_fits_img.xmin_original

        local_img_norm = self.pre_localImg_asInput(local_img_data)
        binary_img = np.zeros(np.shape(local_img_norm))
        binary_img[local_img_norm > 3.0 * BK_SIGMA] = 1

        # set your threshold
        if np.sum(binary_img) < 6:
            return self.local_cat_pd

        binary_img = binary_img.astype(bool)

        # remove small objects
        remove_small_objects(binary_img, min_size=area_thres-1, connectivity=2, in_place=True)

        label_image = label(binary_img)

        dim_1, dim_2 = np.shape(local_img_data)

        for region in regionprops(label_image):

            if region.area >= area_thres:
                # ra,dec space range
                # c x , r y
                y1, x1, y2, x2 = region.bbox
                # remove edges sources
                if x1 < 4 or y1 < 4 or y2 > dim_1 - 3 or x2 > dim_2 - 3:
                    continue


                source_img = local_img_norm[y1:y2 + 1, x1:x2 + 1]
                height, amp, xfit, yfit, xwid, ywid, angle = gaussfit(source_img)
                if xwid >= dim_2 // 2 or ywid >= dim_1 // 2 or xwid == 0 or ywid == 0:
                    continue
                # convert to
                dim1_e_s, dim2_e_s = ellipse(int(np.around(yfit + y1)), int(np.around(xfit + x1)),
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

                if x_min_rect < 4 or y_min_rect < 4 or x_max_rect > dim_2 - 4 or \
                        y_max_rect > dim_1 - 4 or len(dim1_e_s) == 0:
                    continue

                each_source_mask = np.zeros((dim_1, dim_2), np.uint8)
                each_source_mask[y1:y2, x1:x2] = binary_img[y1:y2, x1:x2]



                each_source_flux_pixel = np.sum(local_img_data[dim1_e_s, dim2_e_s])
                BMAJ_pixl = xwid * 3
                BMIN_pixl = ywid * 3
                sou_PA = angle


                # find max point, used to core ra,dec evaluation
                local_orignal_img = local_img_data.copy()
                local_orignal_img[each_source_mask != 1] = EXTRE_MIN

                source_core_idx = np.unravel_index(np.argmax(local_orignal_img, axis=None), local_orignal_img.shape)
                source_core_xy = [source_core_idx[1], source_core_idx[0]]
                each_source_Corefraction = local_img_data[source_core_idx] / each_source_flux_pixel
                source_cen_xy = [xfit + x1, yfit + y1]

                cu_idx = len(self.local_cat_pd.index)
                self.local_cat_pd.loc[cu_idx, :] = self.convert_pixl2_astroCoord(local_wcs,
                                                                                 source_core_xy, source_cen_xy,
                                                                                 each_source_flux_pixel,
                                                                                 each_source_Corefraction, BMAJ_pixl,
                                                                                 BMIN_pixl, sou_PA,  1,
                                                                                 float(-1.0), dim1_idx, dim2_idx,
                                                                                 x1 + each_xmin,
                                                                                 y1 + each_ymin,
                                                                                 x2 + each_xmin,
                                                                                 y2 + each_ymin,
                                                                                 glb_index + cu_idx)

        return self.local_cat_pd