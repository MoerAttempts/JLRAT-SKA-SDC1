"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

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


import keras
import numpy as np
from keras.utils import get_file
import tensorflow as tf
from . import predictModel
from . import Backbone
from ..utils.image import preprocess_image

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, \
    Activation,Dropout,Multiply
from keras import backend as K

class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def jsfm(self, *args, **kwargs):
        return vgg_jsfm(*args, backbone=self.backbone, **kwargs)

    # def download_imagenet(self):
    #     """ Downloads ImageNet weights and returns path to weights file.
    #     Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
    #     """
    #     if self.backbone == 'vgg16':
    #         resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
    #         checksum = '6d6bbae143d832006294945121d1f1fc'
    #     elif self.backbone == 'vgg19':
    #         resource = keras.applications.vgg19.vgg19.WEIGHTS_PATH_NO_TOP
    #         checksum = '253f8cb515780f3b799900260a226db6'
    #     else:
    #         raise ValueError("Backbone '{}' not recognized.".format(self.backbone))
    #
    #     return get_file(
    #         '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
    #         resource,
    #         cache_subdir='models',
    #         file_hash=checksum
    #     )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg', 'vggstyle']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def base_layer(input_tensor=None, trainable=False):
    filter_szie1 = [128, 256, 256, 256, 256]
    # Determine proper input shape
    input_shape = (None, None, 2)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    X_shortcut = x1
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv3')(x1)
    x1 = keras.layers.Add()([X_shortcut, x1])
    x1 = Activation('relu', name='block1_addAct1')(x1)
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv4')(x1)
    X_shortcut = x1
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv5')(x1)
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv6')(x1)
    x1 = keras.layers.Add()([X_shortcut, x1])
    x1 = Activation('relu', name='block1_addAct2')(x1)
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='block1_conv7')(x1)
    # x1_ac = Activation('sigmoid', name='block1_sig')(x1)
    # x1 = Multiply(name='block1_mul')([x1, x1_ac])

    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)
    # x1 = Conv2D(filter_szie1[1], (3, 3), activation='relu', strides=(2, 2), padding='same', name='block1_conv8')(x1)

    # Block 2
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    X_shortcut = x2
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv2')(x2)
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv3')(x2)
    x2 = keras.layers.Add()([X_shortcut, x2])
    x2 = Activation('relu', name='block2_addAct1')(x2)
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv4')(x2)
    X_shortcut = x2
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv5')(x2)
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv6')(x2)
    x2 = keras.layers.Add()([X_shortcut, x2])
    x2 = Activation('relu', name='block2_addAct2')(x2)
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='block2_conv7')(x2)
    # x2_ac = Activation('sigmoid', name='block2_sig')(x2)
    # x2 = Multiply(name='block2_mul')([x2, x2_ac])

    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)
    # x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', strides=(2, 2), padding='same', name='block2_conv8')(x2)
    #

    # Block 3

    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    X_shortcut = x2
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv2')(x3)
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv3')(x3)
    x3 = keras.layers.Add()([X_shortcut, x3])
    x3 = Activation('relu', name='block3_addAct1')(x3)
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv4')(x3)
    X_shortcut = x3
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv5')(x3)
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv6')(x3)
    x3 = keras.layers.Add()([X_shortcut, x3])
    x3 = Activation('relu', name='block3_addAct2')(x3)
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv7')(x3)
    # x3_ac = Activation('sigmoid', name='block3_sig')(x3)
    # x3 = Multiply(name='block3_mul')([x3, x3_ac])

    # x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)
    # x3 = Conv2D(filter_szie1[1], (3, 3), activation='relu', strides=(2, 2), padding='same', name='block3_conv8')(x3)

    # Block 4

    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    # X_shortcut = x4
    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv2')(x4)
    # # x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block3_conv3')(x3)
    # x4 = keras.layers.Add()([X_shortcut, x4])
    # x4 = Activation('relu', name='block4_addAct1')(x4)
    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv4')(x4)
    # X_shortcut = x4
    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv5')(x4)
    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv6')(x4)
    # x4 = keras.layers.Add()([X_shortcut, x4])
    # x4 = Activation('relu', name='block4_addAct2')(x4)
    # x4 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='block4_conv7')(x4)



    return keras.models.Model(inputs=input_tensor, outputs=x3, name='jfs_backbone')



def vgg_jsfm(num_classes, backbone='vgg', inputs=None, modifier=None, **kwargs):
    """ Constructs a  model using a vgg style backbone.

    Args
        num_classes: Number of classes to predict.
        backbone:  backbone style .
        inputs: The inputs to the network (defaults to a Tensor of shape (320, 320, 2)).
        modifier: A function handler which can modify the backbone before using it.

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:

        inputs = keras.layers.Input(shape=(320, 320, 2))

    # create the vgg backbone
    if backbone == 'vgg':
        vgg = base_layer(input_tensor=inputs, trainable=True)
        layer_names = ['block2_conv7', "block3_conv7"]
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model

    layer_outputs = [vgg.get_layer(name).output for name in layer_names]
    return predictModel.pred_sdc1(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
