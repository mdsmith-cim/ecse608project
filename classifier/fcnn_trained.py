import time
import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided
from itertools import product

import theano
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer, ReshapeLayer, DimshuffleLayer, \
    concat, TransformerLayer, SliceLayer, ElemwiseSumLayer, InverseLayer, Conv2DLayer, PadLayer
from lasagne.layers import dropout, get_output, get_all_params, get_output_shape, batch_norm, Upscale2DLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne.nonlinearities import rectify, softmax, linear
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.updates import nesterov_momentum
from lasagne.init import GlorotUniform, Constant
from lasagne.regularization import regularize_layer_params, l2, l1, regularize_network_params
from scipy.misc import imresize

__author__ = 'ajesson'

class FCNN(object):

    def __init__(self, batch_size=1, activation=rectify, num_classes=21,
                 filter_size=(3, 3), num_channels=3, patch_size=(224, 224)):

        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.filter_size = filter_size
        self.patch_size = patch_size
        self.extraction_step = (1, 1, 32, 32)
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.batch_size = batch_size

        self.network = None
        self.hidden = None
        self.convolutional_layers = None
        self.dense_layers = None
        self.softmax = None

        self.activation = activation

        self.build_net()

    def predict(self, x):

        # mu = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        mu = np.array([103.939, 116.779, 123.68], dtype=np.float32)

        x = np.float32(x) - mu

        x = np.expand_dims(np.rollaxis(x, 2, 0), axis=0)
        xs, ys = x.shape[2:]
        xp = 96 + (32 - xs % 32)/2 + 1
        yp = 96 + (32 - ys % 32)/2 + 1

        x = np.pad(x, ((0, 0), (0, 0), (xp, xp), (yp, yp)), mode='symmetric')

        prediction = get_output(self.network, deterministic=True)
        predict_fn = theano.function([self.input_var], prediction)

        patches = np.float32(self.extract_patches(x, self.extraction_step))

        predicted = np.zeros((patches.shape[0], self.num_classes) + self.extraction_step[2:], dtype=np.float32)

        for i in range(patches.shape[0]):

            predicted[i] = predict_fn(patches[i:i+1])

        out = self.reconstruct_from_patches(predicted, (1, self.num_classes) + x.shape[2:])

        return out[:, :, xp:-xp, yp:-yp]

    def transform(self, x):

        prediction = get_output(self.dense_layers[-1], deterministic=True)
        predict_fn = theano.function([self.input_var], prediction)

        return predict_fn(x)

    def build_net(self):

        input_layer = InputLayer(shape=(1, self.num_channels) + self.patch_size,
                                 input_var=self.input_var)

        c00 = Conv2DDNNLayer(input_layer,
                             num_filters=64,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c01 = Conv2DDNNLayer(c00,
                             num_filters=64,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        p0 = MaxPool2DDNNLayer(c01,
                               pool_size=2)

        c10 = Conv2DDNNLayer(p0,
                             num_filters=128,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c11 = Conv2DDNNLayer(c10,
                             num_filters=128,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        p1 = MaxPool2DDNNLayer(c11,
                               pool_size=2)

        c20 = Conv2DDNNLayer(p1,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c21 = Conv2DDNNLayer(c20,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c22 = Conv2DDNNLayer(c21,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        p2 = MaxPool2DDNNLayer(c22,
                               pool_size=2)

        c30 = Conv2DDNNLayer(p2,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c31 = Conv2DDNNLayer(c30,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c32 = Conv2DDNNLayer(c31,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c33 = Conv2DDNNLayer(c32,
                             num_filters=21,
                             filter_size=(1, 1),
                             nonlinearity=linear,
                             W=GlorotUniform())

        c33_slice = SliceLayer(c33, indices=slice(12, -12), axis=2)
        c33_slice = SliceLayer(c33_slice, indices=slice(12, -12), axis=3)

        p3 = MaxPool2DDNNLayer(c32,
                               pool_size=2)

        c40 = Conv2DDNNLayer(p3,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c41 = Conv2DDNNLayer(c40,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c42 = Conv2DDNNLayer(c41,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c43 = Conv2DDNNLayer(c42,
                             num_filters=21,
                             filter_size=(1, 1),
                             nonlinearity=linear,
                             W=GlorotUniform())

        c43_slice = SliceLayer(c43, indices=slice(5, -5), axis=2)
        c43_slice = SliceLayer(c43_slice, indices=slice(5, -5), axis=3)

        p4 = MaxPool2DDNNLayer(c42,
                               pool_size=2)

        c50 = Conv2DDNNLayer(dropout(p4, p=self.dropout_rate),
                             num_filters=4096,
                             filter_size=(7, 7),
                             nonlinearity=self.activation,
                             W=GlorotUniform(gain='relu'))

        c51 = Conv2DDNNLayer(dropout(c50, p=self.dropout_rate),
                             num_filters=4096,
                             filter_size=(1, 1),
                             nonlinearity=self.activation,
                             W=GlorotUniform(gain='relu'))

        c52 = Conv2DDNNLayer(c51,
                             num_filters=21,
                             filter_size=(1, 1),
                             nonlinearity=linear,
                             W=GlorotUniform())

        c52_up = PadLayer(Upscale2DLayer(c52, 2), 1)

        sum_54 = ElemwiseSumLayer((c52_up, c43_slice))

        # sum_54_up = Upscale2DLayer(sum_54, 2)
        # b_up_0 = np.zeros((2, 3), dtype='float32')
        # b_up_0[0, 0] = 1
        # b_up_0[1, 1] = 1
        # b_up_0 = b_up_0.flatten()
        # l_loc_54 = DenseLayer(sum_54, num_units=6, W=Constant(0), b=b_up_0, nonlinearity=None)
        # l_loc_54.add_param(l_loc_54.W, l_loc_54.W.get_value().shape, trainable=False)
        # l_loc_54.add_param(l_loc_54.b, l_loc_54.b.get_value().shape, trainable=False)
        # sum_54_up = TransformerLayer(sum_54, l_loc_54, 0.5)

        sum_54_up = Conv2DLayer(sum_54,
                                num_filters=21,
                                filter_size=(4, 4),
                                stride=(2, 2),
                                W=Constant(1./16.))
        sum_54_up = InverseLayer(sum_54_up,
                                 sum_54_up)

        sum_543 = ElemwiseSumLayer((sum_54_up, c33_slice))

        # sum_543_up = Upscale2DLayer(sum_543, 8)
        b_up_1 = np.zeros((2, 3), dtype='float32')
        b_up_1[0, 0] = 1
        b_up_1[1, 1] = 1
        b_up_1 = b_up_1.flatten()
        l_loc_543 = DenseLayer(sum_543, num_units=6, W=Constant(0), b=b_up_1, nonlinearity=None)
        l_loc_543.add_param(l_loc_543.W, l_loc_543.W.get_value().shape, trainable=False)
        l_loc_543.add_param(l_loc_543.b, l_loc_543.b.get_value().shape, trainable=False)
        sum_543_up = TransformerLayer(sum_543, l_loc_543, 0.125)

        # sum_543_up = Conv2DLayer(PadLayer(sum_543, 28),
        #                          num_filters=21,
        #                          filter_size=(16, 16),
        #                          stride=(8, 8),
        #                          W=Constant(1./256.))
        # sum_543_up = InverseLayer(sum_543_up,
        #                           sum_543_up)

        shape = get_output_shape(sum_543_up)
        shuffle = DimshuffleLayer(sum_543_up, pattern=(0, 2, 3, 1))

        reshape = ReshapeLayer(shuffle,
                               shape=(np.prod(np.array(shape)[2:])*shape[0], shape[1]))

        # self.softmax = batch_norm(DenseLayer(dropout(reshape, p=0.),
        #                           num_units=self.num_classes,
        #                           nonlinearity=softmax))
        self.softmax = NonlinearityLayer(reshape, nonlinearity=softmax)

        self.network = ReshapeLayer(DimshuffleLayer(self.softmax, pattern=(1, 0)),
                                    shape=(shape[0], self.num_classes) + shape[2:])

    def extract_patches(self, arr, extraction_step=1):

        patch_shape = (1, arr.shape[1]) + self.patch_size

        # extraction_step = (1, 1) + tuple(np.array(arr.shape[2:] - self.patch_size)/self.f)
        if isinstance(extraction_step, numbers.Number):
            extraction_step = tuple([extraction_step] * arr_ndim)
        patch_strides = arr.strides

        slices = [slice(None, None, st) for st in extraction_step]
        indexing_strides = arr[slices].strides

        patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                               np.array(extraction_step)) + 1

        shape = tuple(list(patch_indices_shape) + list(patch_shape))
        strides = tuple(list(indexing_strides) + list(patch_strides))

        out = as_strided(arr, shape=shape, strides=strides)

        return out.reshape((int(np.prod(np.array(patch_indices_shape))), arr.shape[1]) + self.patch_size)

    def reconstruct_from_patches(self, patches, image_size):

        i_h, i_w = image_size[2:]
        p_h, p_w = patches.shape[2:]

        img = np.zeros(image_size)
        n = np.zeros(image_size)

        n_h = i_h - p_h - 96 + 1
        n_w = i_w - p_w - 96 + 1

        # st_h = (i_h - p_h)/self.f
        # st_w = (i_w - p_w)/self.f

        for p, (i, j) in zip(patches,
                             product(range(96, n_h, p_h),
                                     range(96, n_w, p_w))):
            img[0, :, i:i + p_h, j:j + p_w] += p
            n[0, :, i:i + p_h, j:j + p_w] += 1

        return img/(n + 1e-9)
