from keras.applications.vgg16 import VGG16
#from keras.applications.resnet import ResNet50
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import preprocess_input

import tensorflow as tf
import numpy as np
import keras
from keras.models import *
import keras.layers as layers
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense
from keras.layers import  Dropout, Cropping2D, BatchNormalization,  Activation, UpSampling2D, Flatten
from keras.optimizers import *
from pylab import plt
from focal_loss import binary_focal_loss as focal_loss


class UNET:
    '''to build a network from keras, without dropout, BN, or anythin improvement'''
    def __init__(self, INPUT_SHAPE, class_num):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.class_num = class_num
        self.OUTPUT=[INPUT_SHAPE[0], INPUT_SHAPE[1], class_num]
        self.model = None
        self.short_cut = []
        self.do_BN = False
        self.do_dropout=False
        self.do_transpose = False

        self.INPUT=None # used for transfer learning
        self.deconv_results = [] # used for transfer learning
        self.transfer_model=None

    def conv_block(self, x, filters, size, name, times=2):

        for i in range(times):
            x = Conv2D(filters, size, padding='SAME', name=name+'_conv{}'.format(i+1))(x)
            x = self.BN(x)
            x = Activation('relu', name=name+'_relu{}'.format(i))(x)
        return x

    def deconv_block(self, x, filters, size, name, conv_times=2):
        x = self.Deconv(x, filters)
        x = self.BN(x)
        x = Activation('relu', name=name+'_relu')(x)
        x = Concatenate()([self.short_cut.pop(), x])

        for i in range(conv_times):
            x = Conv2D(filters, size, padding='same')(x)
            x = self.BN(x)
            x = Activation('relu', name=name+'_relu{}'.format(i+1))(x)


        return x

    def build_network(self, do_compile=True, use_focal_loss=False, alpha=0.9):
        inputs = Input(self.INPUT_SHAPE, name='input')

        x = self.conv_block(inputs, 16, 3, name='Block1', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool1')(x)

        x = self.conv_block(x, 32, 3, name='Block2', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool2')(x)

        x = self.conv_block(x, 64, 3, name='Block3', times=2)
        x = self.Drop(x)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool3')(x)

        x = self.conv_block(x, 128, 3, name='Block4', times=2)
        x = self.Drop(x)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool4')(x)

        x = self.conv_block(x, 256, 3, name='Block5', times=2)
        x = self.Drop(x)

        # thanks for detect-cell-edge-use-unet i9n github
        x = self.deconv_block(x, 128, 3, 'deBlock1')
        x = self.Drop(x)	

        x = self.deconv_block(x, 64, 3, 'deBlock2')
        x = self.Drop(x)	

        x = self.deconv_block(x, 32, 3, 'deBlock3')
        x = self.Drop(x)

        x = self.deconv_block(x, 32, 3, 'deBlock4')

        if do_compile:
            if not use_focal_loss:
                x = Conv2D(self.class_num, 3, padding='same')(x)
                x = Activation('softmax', name='output_softmax')(x)
                adam = Adam()
                model = Model(input=inputs, output=x)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                x = Conv2D(1, 3, padding='same')(x)
                x = Activation('sigmoid', name='output_sigmoid')(x)
                adam = Adam()
                model = Model(input=inputs, output=x)
                model.compile(optimizer=adam, loss=[focal_loss(alpha=alpha, gamma=0)], metrics=['accuracy'])

        self.model = model
        return model

    def BN(self, x,trainable=True):
        '''do BN when self.do_BN is True'''
        if self.do_BN:
            x = BatchNormalization(axis=-1, trainable=trainable)(x)
        else:
            x = x
        return  x

    def Drop(self, x):
        '''do dropout when self.do_dropout is True'''
        if self.do_dropout:
            shape = x.shape.as_list()
            noise_shape = [1]*(len(shape)-1)+[shape[-1]]
            x = Dropout(0.5, noise_shape=noise_shape)(x)
        else:
            x = x
        return x

    def Deconv(self, x, filters, trainable=True):
        if not self.do_transpose:
            x = UpSampling2D(size=[2, 2])(x)
            x = Conv2D(filters, 4, padding='same', trainable=trainable)(x)
        else:
            x = Conv2DTranspose(filters, 4, strides=[2,2], padding='SAME', trainable=trainable)(x)
        return x

    def get_features(self, image, layer_name):
        for i in self.model.layers:
            if i.name == layer_name:
                new_model = Model(inputs=self.model.inputs, outputs=i.output)
                features_map = new_model.predict(image)
                return features_map

    def plot_feature_image(self, image, layer_name):

        map = self.get_features(image, layer_name)
        n,w,h,c = map.shape
        assert n==1
        plt.figure(1)

        n_x = int(c**0.5)
        n_y = int(c/n_x)+1
        for i in range(c):
            _, ax = plt.subplot(n_x, n_y, i+1)
            plt.imshow(map[0,:,:,i].T, cmap='gray')
            ax.set_axis_off()
            ax.set_title('chn:%s'%(i+1))
        plt.show()

class UNET_RESNET(UNET):
    '''
    it is a very complex network, I design a resudial part for it, in convolutional Black, I add another two branches to it
    '''

    def conv_block(self, x, filters, size, name, times=2, trainable=True):

        # pre-process
        x = Conv2D(filters, 1, padding='SAME', name=name + '_conv0', trainable=trainable)(x)
        x = self.BN(x, trainable=trainable)
        x = Activation('relu', name=name + '_relu0')(x)

        x1 = x
        # dense block
        for i in range(times):
            x1 = Conv2D(filters, size, padding='SAME', name=name+'_conv{}'.format(i+1), trainable=trainable)(x1 )
            x1 = self.BN(x1, trainable=trainable)
            x1 = Activation('relu', name=name+'_relu{}'.format(i+1))(x1)

        # vertical
        x2 = x
        x2 = Conv2D(filters, [1, 9], padding='SAME', name=name + '_conv_vec', trainable=trainable)(x2)
        x2 = self.BN(x2, trainable=trainable)
        x2 = Activation('relu', name=name + '_relu_vec')(x2)

        # horizontal
        x3 = x
        x3 = Conv2D(filters, [9, 1], padding='SAME', name=name + '_conv_hon', trainable=trainable)(x3)
        x3 = self.BN(x3, trainable=trainable)
        x3 = Activation('relu', name=name + '_relu_hon')(x3)

        x4 = Concatenate()([x1,x2,x3])
        x4 = self.Drop(x4)
        x4 = Conv2D(filters, 1, padding='SAME', name=name + '_merge', trainable=trainable)(x4)
        x4 = self.BN(x4, trainable=trainable)
        x4 = Activation('relu', name=name + '_relu_end')(x4)

        # res
        x = layers.add([x,x4])

        return x

    def deconv_block(self, x, filters, size, name, conv_times=2,trainable=True):
        # upsample
        x = self.Deconv(x, filters, trainable=trainable)
        x = self.BN(x, trainable=trainable)
        x = Activation('relu', name=name + '_relu_de')(x)
        x = Concatenate()([self.short_cut.pop(), x])
        self.deconv_results.append(x)

        x = self.conv_block(x, filters, size, name, times=conv_times, trainable=trainable)

        return x

    def build_network(self, do_compile=True, use_focal_loss=False, alpha=0.9):
        inputs = Input(self.INPUT_SHAPE, name='input')
        self.INPUT = inputs
        
        chns = [32,64,128,256,512,256,128,64,64]

        chns.reverse()
        x = self.conv_block(inputs, chns.pop(), 3, name='Block1', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool1')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block2', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool2')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block3', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool3')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block4', times=2)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool4')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block5', times=2)

        # thanks for detect-cell-edge-use-unet i9n github
        x = self.deconv_block(x, chns.pop(), 3, 'deBlock1')

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock2')

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock3')

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock4')

        if do_compile:
            if not use_focal_loss:
                x = Conv2D(self.class_num, 3, padding='same')(x)
                x = Activation('softmax', name='output_softmax')(x)
                adam = Adam()
                model = Model(input=inputs, output=x)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                x = Conv2D(1, 3, padding='same')(x)
                x = Activation('sigmoid', name='output_sigmoid')(x)
                adam = Adam()
                model = Model(input=inputs, output=x)
                model.compile(optimizer=adam, loss=[focal_loss(alpha=alpha, gamma=0)], metrics=['accuracy'])

        self.model = model
        return model

    def build_transfer_network(self):

        inputs = Input(self.INPUT_SHAPE, name='input')
        self.INPUT = inputs
        chns = [32,64,128,256,512,256,128,64,64]
        # chns = [16,32,64 ,128,256,128,64,32,32]
        chns.reverse()
        x = self.conv_block(inputs, chns.pop(), 3, name='Block1', times=2, trainable=True)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool1')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block2', times=2, trainable=True)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool2')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block3', times=2, trainable=False)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool3')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block4', times=2, trainable=False)
        self.short_cut.append(x)

        x = MaxPooling2D(name='pool4')(x)

        x = self.conv_block(x, chns.pop(), 3, name='Block5', times=2, trainable=False)

        # thanks for detect-cell-edge-use-unet i9n github
        x = self.deconv_block(x, chns.pop(), 3, 'deBlock1', trainable=False)

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock2', trainable=False)

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock3', trainable=True)

        x = self.deconv_block(x, chns.pop(), 3, 'deBlock4', trainable=True)

        
        x = Conv2D(1, 3, padding='same')(x)
        x = Activation('sigmoid', name='output_sigmoid')(x)
        adam = Adam(lr = 1e-4)
        model = Model(input=inputs, output=x)
        model.compile(optimizer=adam, loss=[focal_loss(alpha=0.9, gamma=0)], metrics=['accuracy'])

        self.model = model
        return model
  