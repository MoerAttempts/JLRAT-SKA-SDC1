"""
Copyright 2019 LeeDongYeun (https://github.com/LeeDongYeun/)

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
import tensorflow as tf
import keras_resnet
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, AveragePooling2D, Activation, UpSampling2D, \
    GlobalAveragePooling2D, Reshape


from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from ..utils.anchors import *
from . import assert_training_model


def tensor_shape(tensor):
    return getattr(tensor, '_shape_val')

def upsample_add(tensors):
    _, h, w, _ = tensors[1].shape

    h = int(h)
    w = int(w)
    # up = tf.image.resize_bilinear(tensors[0], size=(h, w))
    up = tf.image.resize(tensors[0], size=(h, w))
    out = up + tensors[1]

    return out


def upsample_add_output_shape(input_shapes):
    shape = list(input_shapes[1])

    return shape


def Conv(inputs, filters, kernel_size, strides, padding, name='conv'):

    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_conv')(inputs)
    bn = keras_resnet.layers.BatchNormalization(freeze=False, name=name+'_BN')(conv)
    relu = keras.layers.ReLU(name=name)(bn)

    return relu

def create_pool_pyramid(stage, input_size=(None, None, 256), name='PPD'):

    input=keras.layers.Input(input_size)
    filter_szie1 = [256, 256, 256, 256, 256]
    output_feature=filter_szie1[0]//2

    # x1 = Conv2D(filter_szie1[0], (1, 1), activation='relu', padding='same', name='pfp'+str(stage)+'block1_conv0')(input)
    X_shortcut = input
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block1_conv1')(input)
    x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block1_conv2')(x1)
    x1 = keras.layers.Add()([X_shortcut, x1])
    x1 = Activation('relu', name='ppd'+str(stage)+'block1_ac')(x1)
    # x1 = Conv2D(filter_szie1[0], (3, 3), activation='relu', padding='same', name='pfp'+str(stage)+'block1_conv3')(x1)

    # Block 2
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='ppd'+str(stage)+'block1_pool')(x1)
    X_shortcut = x2
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block2_conv1')(x2)
    x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block2_conv2')(x2)
    x2 = keras.layers.Add()([X_shortcut, x2])
    x2 = Activation('relu', name='ppd'+str(stage)+'block2_ac')(x2)
    # x2 = Conv2D(filter_szie1[1], (3, 3), activation='relu', padding='same', name='pfp'+str(stage)+'block2_conv3')(x2)
    # size_buffer.append([int(x2.shape[2])] * 2)


    # Block 3
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='ppd'+str(stage)+'block2_pool')(x2)
    X_shortcut = x3
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block3_conv1')(x3)
    x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block3_conv2')(x3)
    x3 = keras.layers.Add()([X_shortcut, x3])
    x3 = Activation('relu', name='ppd'+str(stage)+'block3_ac')(x3)
    # x3 = Conv2D(filter_szie1[2], (3, 3), activation='relu', padding='same', name='pfp'+str(stage)+'block3_conv3')(x3)
    # size_buffer.append([int(x3.shape[2])] * 2)
    # m2 = keras.layers.UpSampling2D(size=(2, 2), name='block3_Up')(x3)



    # m2 = keras.layers.Add()([m2, m1])
    # m2 = Conv2D(filter_szie1[3], (3, 3), activation='relu', strides=(2, 2), padding='same', name='block3_dconv4')(m2)
    # m2 = Conv2D(filter_szie1[3], (3, 3), activation='relu', strides=(2, 2), padding='same', name='block3_dconv5')(m2)


    # Block 4
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name='ppd'+str(stage)+'block3_pool')(x3)
    X_shortcut = x4
    x4 = Conv2D(filter_szie1[3], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block4_conv1')(x4)
    x4 = Conv2D(filter_szie1[3], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block4_conv2')(x4)
    x4 = keras.layers.Add()([X_shortcut, x4])
    x4 = Activation('relu', name='ppd'+str(stage)+'block4_ac')(x4)
    # x4 = Conv2D(filter_szie1[3], (3, 3), activation='relu', padding='same', name='pfp'+stage+'block4_conv3')(x4)

    # Block 5
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='ppd'+str(stage)+'block4_pool')(x4)
    X_shortcut = x5
    x5 = Conv2D(filter_szie1[4], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block5_conv1')(x5)
    x5 = Conv2D(filter_szie1[4], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block5_conv2')(x5)
    x5 = keras.layers.Add()([X_shortcut, x5])
    x5 = Activation('relu', name='ppd'+str(stage)+'block5_ac')(x5)
    x5= Conv2D(filter_szie1[4], (3, 3), activation='relu', padding='same', name='pfp'+str(stage)+'block5_conv3')(x5)


    x6 = AveragePooling2D((2, 2), strides=(2, 2), name='ppd'+str(stage)+'block5_pool')(x5)
    X_shortcut = x6
    x6 = Conv2D(filter_szie1[4], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block6_conv1')(x6)
    x6 = Conv2D(filter_szie1[4], (3, 3), activation='relu', padding='same', name='ppd'+str(stage)+'block6_conv2')(x6)  #same
    x6 = keras.layers.Add()([X_shortcut, x6])
    x6 = Activation('relu', name='ppd'+str(stage)+'block6_ac')(x6)
    c6 = x6

    c5 = Conv(c6, filters=filter_szie1[4], kernel_size=(3, 3), strides=(1, 1), padding='same',
              name=name + "_" + str(stage) + '_c5')
    c5 = UpSampling2D(size=(2, 2))(c5)
    c5 = keras.layers.Add()([c5, x5])

    # c5=x5

    c4 = Conv(c5, filters=filter_szie1[4], kernel_size=(3, 3), strides=(1, 1), padding='same',
              name=name + "_" + str(stage) + '_c4')
    # c4 = keras.layers.Lambda(lambda x: tf.image.resize(x, size=size_buffer[3]),
    #                          name=name + "_" + str(stage) + '_upsample_add4')(c4)
    c4 = UpSampling2D(size=(2, 2))(c4)
    c4 = keras.layers.Add()([c4, x4])
    # c4 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add4')([c4, f4])

    c3 = Conv(c4, filters=filter_szie1[4], kernel_size=(3, 3), strides=(1, 1), padding='same',
              name=name + "_" + str(stage) + '_c3')
    # c3 = keras.layers.Lambda(lambda x: tf.image.resize(x, size=size_buffer[2]),
    #                          name=name + "_" + str(stage) + '_upsample_add3')(c3)
    c3 = UpSampling2D(size=(2, 2))(c3)
    c3 = keras.layers.Add()([c3, x3])
    # c3 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add3')([c3, f3])

    c2 = Conv(c3, filters=filter_szie1[4], kernel_size=(3, 3), strides=(1, 1), padding='same',
              name=name + "_" + str(stage) + '_c2')
    # c2 = keras.layers.Lambda(lambda x: tf.image.resize(x, size=size_buffer[1]),
    #                          name=name + "_" + str(stage) + '_upsample_add2')(c2)
    c2 = UpSampling2D(size=(2, 2))(c2)
    c2 = keras.layers.Add()([c2, x2])
    # c2 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add2')([c2, f2])

    c1 = Conv(c2, filters=filter_szie1[4], kernel_size=(3, 3), strides=(1, 1), padding='same',
              name=name + "_" + str(stage) + '_c1')
    # c1 = keras.layers.Lambda(lambda x: tf.image.resize(x, size=size_buffer[0]),
    #                          name=name + "_" + str(stage) + '_upsample_add1')(c1)
    c1 = UpSampling2D(size=(2, 2))(c1)
    c1 = keras.layers.Add()([c1, x1])
    # c1 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add1')([c1, f1])


    o1 = Conv(c1, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o1')
    o2 = Conv(c2, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o2')
    o3 = Conv(c3, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o3')
    o4 = Conv(c4, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o4')
    o5 = Conv(c5, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o5')
    o6 = Conv(c6, filters=output_feature, kernel_size=(1, 1), strides=(1, 1), padding='valid',
              name=name + "_" + str(stage) + '_o6')

    outputs = [o1, o2, o3, o4, o5, o6]
    return keras.models.Model(inputs=input, outputs=outputs, name=name+ "_" + str(stage))

# Concatenate different size feature maps
def LinkUnit_1(input_size_1=(None, None, 256), input_size_2=(None, None, 256), feature_size_1=256, feature_size_2=256,
          name='link_1'):
    C4 = keras.layers.Input(input_size_1)
    C5 = keras.layers.Input(input_size_2)

    F4 = Conv(C4, filters=feature_size_1, kernel_size=(3, 3), strides=(1, 1), padding='same', name='F4')
    F5 = Conv(C5, filters=feature_size_2, kernel_size=(1, 1), strides=(1, 1), padding='same', name='F5')

    F5 = keras.layers.UpSampling2D(size=(2, 2), name='F5_Up')(F5)

    outputs = keras.layers.Concatenate(name=name)([F4, F5])

    return keras.models.Model(inputs=[C4, C5], outputs=outputs, name=name)


# control and Concatenate feature maps
def LinkUnit_2(stage, base_size=(None, None, 512), cpd_size=(None, None, 128), feature_size=128, name='link_2'):
    base = keras.layers.Input(base_size)
    cpd = keras.layers.Input(cpd_size)


    outputs = Conv(base, filters=feature_size, kernel_size=(1, 1), strides=(1, 1), padding='same', name=name+"_"+str(stage) + '_base_feature')
    outputs = keras.layers.Concatenate(name=name+"_"+str(stage))([outputs, cpd])
    return keras.models.Model(inputs=[base, cpd], outputs=outputs, name=name+"_"+str(stage))


# downsampling feature maps and output feature pyramid
def CPD(stage, input_size=(None, None, 256), feature_size=256, name="CPD"):
    output_features = feature_size // 2



    inputs = keras.layers.Input(input_size)
    # level one
    level = 1
    f1 = inputs  #320
    f2 = Conv(f1, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f2')#160
    f3 = Conv(f2, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f3')#80
    f4 = Conv(f3, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f4')#40
    f5 = Conv(f4, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f5')#20
    f6 = Conv(f5, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f6')#10
    # define a Lambda function to compute upsample_blinear
    level = 2
    c6 = f6

    c5 = Conv(c6, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same',name=name + "_" + str(stage) + '_c5')
    c5 = UpSampling2D(size=(2, 2))(c5)
    c5 = keras.layers.Add()([c5, f5])

    # c5 =f5


    c4 = Conv(c5, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c4')

    c4 = UpSampling2D(size=(2, 2))(c4)
    c4 = keras.layers.Add()([c4, f4])


    c3 = Conv(c4, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c3')

    c3 = UpSampling2D(size=(2, 2))(c3)
    c3 = keras.layers.Add()([c3, f3])


    c2 = Conv(c3, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c2')

    c2 = UpSampling2D(size=(2, 2))(c2)
    c2 = keras.layers.Add()([c2, f2])


    c1 = Conv(c2, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c1')

    c1 = UpSampling2D(size=(2, 2))(c1)
    c1 = keras.layers.Add()([c1, f1])


    # level three:using 1 * 1 kernel to make it smooth
    level = 3

    o1 = Conv(c1, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o1')
    o2 = Conv(c2, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o2')
    o3 = Conv(c3, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o3')
    o4 = Conv(c4, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o4')
    o5 = Conv(c5, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o5')
    o6 = Conv(c6, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o6')

    outputs = [o1, o2, o3, o4, o5, o6]

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name + "_" + str(stage))


def _concatenate_features(features):
    transposed = np.array(features).T
    transposed = np.flip(transposed, 0)

    concatenate_features = []
    for features in transposed:
        concat = keras.layers.Concatenate()([f for f in features])
        concatenate_features.append(concat)

    return concatenate_features



def _create_feature_pyramid(base_feature, stage=6):
    features = []

    inputs = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(base_feature)

    cpd = CPD(1)

    outputs = cpd(inputs)
    max_output = outputs[0]
    features.append(outputs)

    for i in range(2, stage + 1):
        link2 = LinkUnit_2(i - 1)
        inputs = link2([base_feature, max_output])
        if np.mod(i, 2) == 0:
            pool_fpy = create_pool_pyramid(i)
            outputs = pool_fpy(inputs)
        else:
            cpd = CPD(i)
            outputs = cpd(inputs)

        max_output = outputs[0]
        features.append(outputs)

    feature_pyramid = _concatenate_features(features)
    feature_pyramid.reverse()
    return feature_pyramid


def _calculate_input_sizes(concatenate_features):
    input_size = []
    for features in concatenate_features:
        size = (int(features.shape[1]), int(features.shape[2]), int(features.shape[3]))
        input_size.append(size)
    return input_size


def SE_block(input_size, compress_ratio=16, name='SE_block'):

    inputs = keras.layers.Input(input_size)




    pool = keras.layers.GlobalAveragePooling2D()(inputs)
    reshape = keras.layers.Reshape((1, 1, input_size[-1]))(pool)
    # control channel numbers form input_size[2] to input_size//compress_ratio, which will reduce the computation
    fc1 = keras.layers.Conv2D(filters=input_size[-1]// compress_ratio, kernel_size=1, strides=1, padding='valid',
                              activation='relu', name=name+'_fc1')(reshape)
    # use sigmoid scale the weight value to [0 1] and restore back to the channel numbers to  input_size[2]
    fc2 = keras.layers.Conv2D(filters=input_size[-1], kernel_size=1, strides=1, padding='valid', activation='sigmoid',
                              name=name+'_fc2')(fc1)

    reweight = keras.layers.Multiply(name=name+'_reweight')([inputs, fc2])
    return keras.models.Model(inputs=inputs, outputs=reweight, name=name)






def FCCWM(input_pyramid, compress_ratio=16, name='FCCWM'):
    inputs = []
    outputs = []


    for i in range(len(input_pyramid)):
        each_input = input_pyramid[i]
        each_size=getattr(each_input, '_keras_shape')
        filters = each_size[-1]
        input_size = (None, None, filters)
        _input = keras.layers.Input(input_size)
        # _input=input_pyramid[i]

        se_block = SE_block(input_size, compress_ratio=compress_ratio, name='SE_block_' + str(i))

        _output = se_block(_input)

        inputs.append(_input)
        outputs.append(_output)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()

    return model


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)


    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()

    return model


def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors, pyramid_feature_size=640)),
        ('classification', default_classification_model(num_classes, num_anchors, pyramid_feature_size=640))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def pred_sdc1(inputs, backbone_layers, num_classes, num_anchors=None, submodels=None, name='pred_sdc1'):
    if num_anchors is None:
        num_anchors = AnchorParameters_vgg.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C1, C2 = backbone_layers
    _, h4, w4, f4 = C1.shape
    _, h5, w5, f5 = C2.shape

    link1 = LinkUnit_1(feature_size_1=256, feature_size_2=256)

    base_feature = link1([C1, C2])

    # this is relative with default_submodels parameters  stage*128
    feature_pyramid = _create_feature_pyramid(base_feature, stage=5)


    fccwm = FCCWM(feature_pyramid)

    outputs = fccwm(feature_pyramid)

    pyramids = __build_pyramid(submodels, outputs)


    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)

# make prediction model
def JSFM_bbox(model=None, nms=True, class_specific_filter=True, name='JSFM-bbox', anchor_params=None, **kwargs):
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters_vgg

    # create prediction model
    if model is None:
        model = pred_sdc1(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    _, h, w, f = model.input.shape
    anchor_params = AnchorParameters_vgg
    # anchor_params.dispParas()


    feature_layer = model.get_layer("FCCWM")
    features = feature_layer.get_output_at(1)
    anchors = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])
    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)