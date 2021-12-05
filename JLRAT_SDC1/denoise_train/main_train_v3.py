# -*- coding: utf-8 -*-

import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract, Softmax, Add, Multiply, Reshape, \
    concatenate,MaxPooling2D,AveragePooling2D,UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
import keras.backend as K
from SDC1_prediction.SKA_pre_constant import *


## Params
parser = argparse.ArgumentParser()
parser.add_argument('--band', default='560', type=str, help='choose the band')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--depth', default=1000, type=int, help='depth')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()

save_dir = os.path.join('models', args.band + '_' + 'depth' + str(args.depth))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def ResBlock(inputs, filters, kernel_size, strides, layer_count, use_bnorm=False):
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer='Orthogonal',
               padding='same',
               name='conv' + str(layer_count))(inputs)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)

    layer_count += 1
    xc = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer='Orthogonal',
               padding='same',
               name='conv' + str(layer_count))(x)
    layer_count += 1
    xc = Activation('relu', name='relu' + str(layer_count))(xc)
    for i in range(3):
        layer_count += 1
        xc = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                    padding='same',
                    use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            xc = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(xc)
        layer_count += 1
        xc = Activation('relu', name='relu' + str(layer_count))(xc)
    layer_count += 1
    xc = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer='Orthogonal',
                padding='same',
                name='conv' + str(layer_count))(xc)
    x = Add()([x, xc])
    return x

def covBlock(inputs, filters, kernel_size, strides, layer_count, use_bnorm=False):
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer='Orthogonal',
               padding='same',
               name='conv' + str(layer_count))(inputs)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    for i in range(3):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                    padding='same',
                    use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            x = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(xc)
        layer_count += 1
        xc = Activation('relu', name='relu' + str(layer_count))(xc)
    layer_count += 1
    xc = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer='Orthogonal',
                padding='same',
                name='conv' + str(layer_count))(xc)
    x = Add()([x, xc])
    return x

def Conv(inputs, filters, kernel_size, strides, padding, name='conv'):

    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_conv')(inputs)
    # bn = keras_resnet.layers.BatchNormalization(freeze=False, name=name+'_BN')(conv)
    relu=Activation('relu', name='relu' + name)(conv)

    return relu


# The second branch middle without conv layer, only dilated conv layes
# this is our version
def SDC1deNet(depth, filters=64, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise

    # =========== branch two ================
    layer_count += 1
    y = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    y = Activation('relu', name='relu' + str(layer_count))(y)

    # for i in range(5):
    #     layer_count += 1
    #     y = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
    #                use_bias=False, name='conv' + str(layer_count))(y)
    #     if use_bnorm:
    #         layer_count += 1
    #         # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
    #         y = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(y)
    #     layer_count += 1
    #     y = Activation('relu', name='relu' + str(layer_count))(y)

    for i in range(5):
        layer_count += 1
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same',
                   name='conv' + str(layer_count))(y)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            y = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(y)
        layer_count += 1
        y = Activation('relu', name='relu' + str(layer_count))(y)

    # layer_count += 1
    # y = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
    #            name='conv' + str(layer_count))(inpt)
    # layer_count += 1
    # y = Activation('relu', name='relu' + str(layer_count))(y)

    for i in range(5):
        layer_count += 1
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same',
                   name='conv' + str(layer_count))(y)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            y = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(y)
        layer_count += 1
        y = Activation('relu', name='relu' + str(layer_count))(y)

    # for i in range(5):
    #     layer_count += 1
    #     y = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
    #                use_bias=False, name='conv' + str(layer_count))(y)
    #     if use_bnorm:
    #         layer_count += 1
    #         # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
    #         y = BatchNormalization(axis=3, momentum=0.1, epsilon=0.0001, name='bn' + str(layer_count))(y)
    #     layer_count += 1
    #     y = Activation('relu', name='relu' + str(layer_count))(y)
    layer_count += 1
    y = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(y)
    layer_count += 1
    y = Activation('relu', name='relu' + str(layer_count))(y)

    layer_count += 1
    y = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(y)
    layer_count += 1
    y = Subtract(name='subtract' + str(layer_count))([inpt, y])  # input - noise

    o = concatenate([x, y], axis=-1)

    layer_count += 1
    z = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same',
               name='conv' + str(layer_count))(o)  # gray is 1 color is 3
    z = Subtract()([inpt, z])
    model = Model(inputs=inpt, outputs=z)

    # model = Model(inputs=inpt, outputs=x)

    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.hdf5'))  # get name list of all .hdf5 files
    # file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            # print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr / 10
    elif epoch <= 80:
        lr = initial_lr / 20
    else:
        lr = initial_lr / 20
    log('current learning rate is %2.8f' % lr)
    return lr


def train_datagen(catalog_name, skiprows, epoch_num=5, batch_size=128, channels=1):
    while (True):
        n_count = 0
        if n_count == 0:
            patches_high, patches_low = dg.datagenerator(catalog_name, filename_high, filename_low,
                                                         skiprows,
                                                         data_dir='D:\pyProjects\\',
                                                         channels=channels)
            # use all data max value as scalar

            print('patch length', len(patches_high))

            assert len(patches_high) % args.batch_size == 0, \
                log(
                    'make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            # print('len of xs',len(xs))

            patches_high = patches_high.astype('float64') / MAXMIN_RANGE
            patches_low = patches_low.astype('float64') / MAXMIN_RANGE




            indices = list(range(patches_low.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)  # shuffle
            for i in range(0, len(indices), batch_size):
                # ground true
                batch_x = patches_high[indices[i:i + batch_size]]

                batch_y = patches_low[indices[i:i + batch_size]]


                yield batch_y, batch_x


# define loss
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))


# for B1-B5
BAND_WIDTH_MAIN = 'B1'
skiprows = 18

filename_high = 'SKAMid_' + BAND_WIDTH_MAIN + '_1000h_v3.fits'
filename_low = 'SKAMid_' + BAND_WIDTH_MAIN + '_8h_v3.fits'
cata_name = 'TrainingSet_' + BAND_WIDTH_MAIN + '_v2.txt'


if __name__ == '__main__':
    print('Tranin bandwidth', BAND_WIDTH_MAIN)

    # model selection
    img_channel_num = 1
    model = SDC1deNet(depth=17, filters=64, image_channels=img_channel_num, use_bnorm=True)

    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = load_model(os.path.join(save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)

    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                   verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit_generator(
        train_datagen(catalog_name=cata_name, skiprows=skiprows, batch_size=args.batch_size, channels=img_channel_num),
        steps_per_epoch=1000, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
        callbacks=[checkpointer, csv_logger, lr_scheduler])


















