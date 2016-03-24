import time
import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided
from itertools import product

import theano
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer, ReshapeLayer, DimshuffleLayer, concat, MaxPool2DLayer, InverseLayer
from lasagne.layers import dropout, get_output, get_all_params, get_output_shape, Layer, Upscale2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.updates import nesterov_momentum
from lasagne.init import GlorotUniform, Orthogonal
from lasagne.regularization import regularize_layer_params, l2, l1
from scipy.misc import imresize

__author__ = 'ajesson'

class FCNN(object):

    def __init__(self, conv_nodes=[64], fc_nodes=[256, 128], learning_rate=0.001,
                 num_epochs=10, dropout_rate=0.5, batch_size=1,
                 learning_rate_decay=0.93, activation=rectify, validate_pct=0.1, momentum=0.9,
                 l1_penalty=0., l2_penalty=5e-4, num_classes=20, f=2,
                 filter_size=(3, 3), num_channels=4, patch_size=(224, 224),
                 verbose=False):

        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.filter_size = filter_size
        self.patch_size = patch_size
        self.r = len(conv_nodes)*(filter_size[0]/2)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv_nodes = conv_nodes
        self.fc_nodes = fc_nodes
        self.f = f

        self.learning_rate = theano.shared(np.asarray(learning_rate, dtype=theano.config.floatX))
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        self.validate_pct = validate_pct

        self.network = None
        self.hidden = None
        self.convolutional_layers = None
        self.dense_layers = None
        self.softmax = None
        self.train_fn = None
        self.val_fn = None
        self.decay_learning_rate = theano.function(inputs=[],
                                                   outputs=self.learning_rate,
                                                   updates={self.learning_rate:
                                                            self.learning_rate * learning_rate_decay})

        self.activation = activation
        self.verbose = verbose

        self.l2 = []
        self.build_net()
        # self.l1 = regularize_layer_params(self.softmax, l1) * l1_penalty
        # self.l2_out = regularize_layer_params(self.softmax, l2) * 1e-4
        self.build_train_fn()

    def predict(self, x):

        prediction = get_output(self.network, deterministic=True)
        predict_fn = theano.function([self.input_var], prediction)

        return imresize(predict_fn(x)[0], x.shape[-2:], interp='nearest')

    def transform(self, x):

        prediction = get_output(self.dense_layers[-1], deterministic=True)
        predict_fn = theano.function([self.input_var], prediction)

        return predict_fn(x)

    def fit(self, x, y, idx_train=None, idx_test=None):

        validate_flag = idx_train is not None

        if validate_flag:

            self.build_validate_fn()

        else:

            idx_train = range(x.shape[0])


        print("Starting training...")

        for epoch in range(self.num_epochs):

            start_time = time.time()
            self.run_epoch(x[idx_train], y[idx_train])

            if validate_flag:

                self.run_epoch_validate(x[idx_test], y[idx_test])

            self.decay_learning_rate()

            if self.verbose:

                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))

    def build_train_fn(self):

        prediction = get_output(self.softmax, deterministic=False)

        loss = categorical_crossentropy(prediction, self.target_var)
        loss = aggregate(loss) + self.l1 + self.l2 + self.l2_out

        params = get_all_params(self.network, trainable=True)

        updates = nesterov_momentum(loss, params, learning_rate=self.learning_rate, momentum=self.momentum)

        self.train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)

    def build_validate_fn(self):

        prediction = get_output(self.softmax, deterministic=True)
        loss = categorical_crossentropy(prediction, self.target_var)
        loss = aggregate(loss)
        acc = categorical_accuracy(prediction, self.target_var)
        acc = aggregate(acc)

        self.val_fn = theano.function([self.input_var, self.target_var], [loss, acc])

    def run_epoch(self, x, y, mask):

        train_err = 0
        train_batches = 0
        for batch in self.iterate_minibatches(x, y, self.batch_size, shuffle=True):
            inputs, targets = batch
            train_err += self.train_fn(inputs, targets)
            train_batches += 1

        if self.verbose:

            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    def run_epoch_validate(self, x_val, y_val, mask_val):

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in self.iterate_minibatches(x_val, y_val, self.batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        if self.verbose:

            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

    def build_net(self):

        input_layer = InputLayer(shape=(self.batch_size, self.num_channels, None, None),
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

        p4 = MaxPool2DDNNLayer(c42,
                               pool_size=2)

        c50 = Conv2DDNNLayer(dropout(p4, p=self.dropout_rate),
                             num_filters=4096,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c51 = Conv2DDNNLayer(dropout(c50, p=self.dropout_rate),
                             num_filters=4096,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        self.l2 = regularize_layer_params([c50, c51], l2) * 5e-4

        c52 = Conv2DDNNLayer(c51,
                             num_filters=1000,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        feature_layer = concat((p2,
                                Upscale2DLayer(p3, scale_factor=2),
                                Upscale2DLayer(c52, scale_factor=4)))

        shape = get_output_shape(feature_layer)

        shuffle = DimshuffleLayer(feature_layer, pattern=(0, 2, 3, 1))

        reshape = ReshapeLayer(shuffle,
                               shape=(np.prod(np.array(shape)[2:]), shape[1]))

        self.softmax = DenseLayer(dropout(reshape, p=self.dropout_rate),
                                  num_units=self.num_classes,
                                  nonlinearity=softmax)

        self.network = ReshapeLayer(DimshuffleLayer(self.softmax, pattern=(1, 0)),
                                    shape=(self.batch_size, self.num_classes) + shape[2:])

    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

            t_tmp = targets[excerpt]

            t_out = imresize(t_tmp[0, 0], 0.125, interp='nearest')

            yield inputs[excerpt], np.ravel(t_out)

    def iterate_minibatches_test(inputs, batchsize, shuffle=False):

        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    def extract_patches(self, arr):

        patch_shape = (1, arr.shape[1]) + self.patch_size

        extraction_step = (1, 1) + tuple(np.array(arr.shape[2:] - self.patch_size)/self.f)

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

        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1

        st_h = (i_h - p_h)/self.f
        st_w = (i_w - p_w)/self.f

        for p, (i, j) in zip(patches,
                             product(range(0, n_h, st_h),
                                     range(0, n_w, st_w))):
            img[0, :, i:i + p_h, j:j + p_w] += p
            n[0, :, i:i + p_h, j:j + p_w] += 1

        return img/(n + 1e-9)
