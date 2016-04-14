import time
import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided
from itertools import product

import theano
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer, ReshapeLayer, DimshuffleLayer, \
    concat, MaxPool2DLayer, InverseLayer, SliceLayer
from lasagne.layers import dropout, get_output, get_all_params, get_output_shape, Layer, Upscale2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import aggregate, categorical_crossentropy, categorical_accuracy
from lasagne.updates import nesterov_momentum
from lasagne.init import GlorotUniform, Orthogonal
from lasagne.regularization import regularize_layer_params, l2, l1, regularize_network_params
from scipy.misc import imresize

__author__ = 'ajesson'

class FCNN(object):

    def __init__(self, conv_nodes=[64], fc_nodes=[256, 128], learning_rate=0.001,
                 num_epochs=10, dropout_rate=0.5, batch_size=1,
                 learning_rate_decay=0.95, activation=rectify, validate_pct=0.1, momentum=0.9,
                 l1_penalty=1e-6, l2_penalty=1e-4, num_classes=21, f=2,
                 filter_size=(3, 3), num_channels=3, patch_size=(224, 224),
                 verbose=False):

        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.filter_size = filter_size
        self.patch_size = patch_size
        self.r = len(conv_nodes)*(filter_size[0]/2)
        self.extraction_step = (1, 1, 64, 64)
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

        # self.l2 = []
        self.build_net()
        self.l1 = regularize_layer_params(self.softmax, l1)*l1_penalty
        self.l2 = regularize_network_params(self.network, l2)*l2_penalty
        # self.l2 = regularize_layer_params(self.dense_layers[:-1], l2) * l2_penalty
        # self.l2_out = regularize_layer_params(self.softmax, l2) * 1e-4
        self.build_train_fn()

    def predict(self, x):

        mu = np.array([123.68, 116.779, 103.939], dtype=np.float32)

        x = np.float32(x) - mu

        x = np.expand_dims(np.rollaxis(x, 2, 0), axis=0)
        xs, ys = x.shape[2:]
        xp = 80 + (64 - xs % 64)/2 + 1
        yp = 80 + (64 - ys % 64)/2 + 1

        x = np.pad(x/255., ((0, 0), (0, 0), (xp, xp), (yp, yp)), mode='symmetric')

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

    def fit(self, x, y, idx_train=None, idx_test=None):

        validate_flag = idx_train is not None

        if validate_flag:

            self.build_validate_fn()

        else:

            idx_train = range(len(x))


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
        loss = aggregate(loss) + self.l1 + self.l2

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

    def run_epoch(self, x, y):

        train_err = 0
        train_batches = 0
        for batch in self.iterate_minibatches(x, y, self.batch_size, shuffle=True):
            inputs, targets = batch
            train_err += self.train_fn(inputs, targets)
            train_batches += 1

        if self.verbose:

            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    def run_epoch_validate(self, x_val, y_val):

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

    # def build_net(self):
        #
        # input_layer = InputLayer(shape=(self.batch_size, self.num_channels, None, None),
        #                          input_var=self.input_var)
        #
        # c00 = Conv2DDNNLayer(input_layer,
        #                      num_filters=64,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c01 = Conv2DDNNLayer(c00,
        #                      num_filters=64,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # p0 = MaxPool2DDNNLayer(c01,
        #                        pool_size=2)
        #
        # c10 = Conv2DDNNLayer(p0,
        #                      num_filters=128,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c11 = Conv2DDNNLayer(c10,
        #                      num_filters=128,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # p1 = MaxPool2DDNNLayer(c11,
        #                        pool_size=2)
        #
        # c20 = Conv2DDNNLayer(p1,
        #                      num_filters=256,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c21 = Conv2DDNNLayer(c20,
        #                      num_filters=256,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c22 = Conv2DDNNLayer(c21,
        #                      num_filters=256,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # p2 = MaxPool2DDNNLayer(c22,
        #                        pool_size=2)
        #
        # c30 = Conv2DDNNLayer(p2,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c31 = Conv2DDNNLayer(c30,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c32 = Conv2DDNNLayer(c31,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # p3 = MaxPool2DDNNLayer(c32,
        #                        pool_size=2)
        #
        # c40 = Conv2DDNNLayer(p3,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c41 = Conv2DDNNLayer(c40,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c42 = Conv2DDNNLayer(c41,
        #                      num_filters=512,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # p4 = MaxPool2DDNNLayer(c42,
        #                        pool_size=2)
        #
        # c50 = Conv2DDNNLayer(dropout(p4, p=self.dropout_rate),
        #                      num_filters=4096,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c51 = Conv2DDNNLayer(dropout(c50, p=self.dropout_rate),
        #                      num_filters=4096,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # self.l2 = regularize_layer_params([c50, c51], l2) * 5e-4
        #
        # c52 = Conv2DDNNLayer(c51,
        #                      num_filters=1000,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # feature_layer = concat((p2,
        #                         Upscale2DLayer(p3, scale_factor=2),
        #                         Upscale2DLayer(c52, scale_factor=4)))
        #
        # shape = get_output_shape(feature_layer)
        #
        # shuffle = DimshuffleLayer(feature_layer, pattern=(0, 2, 3, 1))
        #
        # reshape = ReshapeLayer(shuffle,
        #                        shape=(np.prod(np.array(shape)[2:]), shape[1]))
        #
        # self.softmax = DenseLayer(dropout(reshape, p=self.dropout_rate),
        #                           num_units=self.num_classes,
        #                           nonlinearity=softmax)
        #
        # self.network = ReshapeLayer(DimshuffleLayer(self.softmax, pattern=(1, 0)),
        #                             shape=(self.batch_size, self.num_classes) + shape[2:])

    def build_net(self):

        input_layer = InputLayer(shape=(self.batch_size, self.num_channels) + self.patch_size,
                                 input_var=self.input_var)

        c00 = Conv2DDNNLayer(input_layer,
                             num_filters=64,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c00.add_param(c00.W, c00.W.get_value().shape, trainable=False)
        c00.add_param(c00.b, c00.b.get_value().shape, trainable=False)

        c01 = Conv2DDNNLayer(c00,
                             num_filters=64,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c01.add_param(c01.W, c01.W.get_value().shape, trainable=False)
        c01.add_param(c01.b, c01.b.get_value().shape, trainable=False)

        p0 = MaxPool2DDNNLayer(c01,
                               pool_size=2)

        c10 = Conv2DDNNLayer(p0,
                             num_filters=128,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c10.add_param(c10.W, c10.W.get_value().shape, trainable=False)
        c10.add_param(c10.b, c10.b.get_value().shape, trainable=False)

        c11 = Conv2DDNNLayer(c10,
                             num_filters=128,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c11.add_param(c11.W, c11.W.get_value().shape, trainable=False)
        c11.add_param(c11.b, c11.b.get_value().shape, trainable=False)

        p1 = MaxPool2DDNNLayer(c11,
                               pool_size=2)

        c20 = Conv2DDNNLayer(p1,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c20.add_param(c20.W, c20.W.get_value().shape, trainable=False)
        c20.add_param(c20.b, c20.b.get_value().shape, trainable=False)

        c21 = Conv2DDNNLayer(c20,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c21.add_param(c21.W, c21.W.get_value().shape, trainable=False)
        c21.add_param(c21.b, c21.b.get_value().shape, trainable=False)

        c22 = Conv2DDNNLayer(c21,
                             num_filters=256,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c22.add_param(c22.W, c22.W.get_value().shape, trainable=False)
        c22.add_param(c22.b, c22.b.get_value().shape, trainable=False)

        p2 = MaxPool2DDNNLayer(c22,
                               pool_size=2)

        c30 = Conv2DDNNLayer(p2,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c30.add_param(c30.W, c30.W.get_value().shape, trainable=False)
        c30.add_param(c30.b, c30.b.get_value().shape, trainable=False)

        c31 = Conv2DDNNLayer(c30,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c31.add_param(c31.W, c31.W.get_value().shape, trainable=False)
        c31.add_param(c31.b, c31.b.get_value().shape, trainable=False)

        c32 = Conv2DDNNLayer(c31,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c32.add_param(c32.W, c32.W.get_value().shape, trainable=False)
        c32.add_param(c32.b, c32.b.get_value().shape, trainable=False)

        p3 = MaxPool2DDNNLayer(c32,
                               pool_size=2)

        c40 = Conv2DDNNLayer(p3,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c40.add_param(c40.W, c40.W.get_value().shape, trainable=False)
        c40.add_param(c40.b, c40.b.get_value().shape, trainable=False)

        c41 = Conv2DDNNLayer(c40,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c41.add_param(c41.W, c41.W.get_value().shape, trainable=False)
        c41.add_param(c41.b, c41.b.get_value().shape, trainable=False)

        c42 = Conv2DDNNLayer(c41,
                             num_filters=512,
                             filter_size=self.filter_size,
                             nonlinearity=self.activation,
                             pad='same',
                             W=GlorotUniform(gain='relu'))

        c42.add_param(c42.W, c42.W.get_value().shape, trainable=False)
        c42.add_param(c42.b, c42.b.get_value().shape, trainable=False)

        p4 = MaxPool2DDNNLayer(c42,
                               pool_size=2)

        # c50 = Conv2DDNNLayer(dropout(p4, p=self.dropout_rate),
        #                      num_filters=4096,
        #                      filter_size=(7, 7),
        #                      nonlinearity=self.activation,
        #                      W=GlorotUniform(gain='relu'))
        #
        # c50.add_param(c50.W, c50.W.get_value().shape, trainable=False)
        # c50.add_param(c50.b, c50.b.get_value().shape, trainable=False)
        #
        # c51 = Conv2DDNNLayer(dropout(c50, p=self.dropout_rate),
        #                      num_filters=4096,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c51.add_param(c51.W, c51.W.get_value().shape, trainable=False)
        # c51.add_param(c51.b, c51.b.get_value().shape, trainable=False)
        #
        # c52 = Conv2DDNNLayer(c51,
        #                      num_filters=1000,
        #                      filter_size=self.filter_size,
        #                      nonlinearity=self.activation,
        #                      pad='same',
        #                      W=GlorotUniform(gain='relu'))
        #
        # c52.add_param(c52.W, c52.W.get_value().shape, trainable=False)
        # c52.add_param(c52.b, c52.b.get_value().shape, trainable=False)

        # input_slice = SliceLayer(input_layer, indices=slice(80, 80 + 64), axis=2)
        # input_slice = SliceLayer(input_slice, indices=slice(80, 80 + 64), axis=3)

        c01_slice = SliceLayer(c01, indices=slice(80, 80 + 64), axis=2)
        c01_slice = SliceLayer(c01_slice, indices=slice(80, 80 + 64), axis=3)

        c11_slice = SliceLayer(c11, indices=slice(40, 40 + 32), axis=2)
        c11_slice = SliceLayer(c11_slice, indices=slice(40, 40 + 32), axis=3)

        c22_slice = SliceLayer(c22, indices=slice(20, 20 + 16), axis=2)
        c22_slice = SliceLayer(c22_slice, indices=slice(20, 20 + 16), axis=3)

        c32_slice = SliceLayer(c32, indices=slice(10, 10 + 8), axis=2)
        c32_slice = SliceLayer(c32_slice, indices=slice(10, 10 + 8), axis=3)

        c42_slice = SliceLayer(c42, indices=slice(5, 5 + 4), axis=2)
        c42_slice = SliceLayer(c42_slice, indices=slice(5, 5 + 4), axis=3)

        feature_layer = concat((c01_slice,
                                Upscale2DLayer(c11_slice, scale_factor=2),
                                Upscale2DLayer(c22_slice, scale_factor=4),
                                Upscale2DLayer(c32_slice, scale_factor=8),
                                Upscale2DLayer(c42_slice, scale_factor=16)))

        # ndens = len(self.fc_nodes)
        # dens = []
        # dens.append(Conv2DDNNLayer(dropout(feature_layer, p=self.dropout_rate),
        #                            num_filters=self.fc_nodes[0],
        #                            filter_size=(1, 1),
        #                            nonlinearity=self.activation,
        #                            W=GlorotUniform(gain='relu')))
        #
        # for i in range(1, ndens):
        #
        #     dens.append(Conv2DDNNLayer(dropout(dens[i-1], p=self.dropout_rate),
        #                                num_filters=self.fc_nodes[i],
        #                                filter_size=(1, 1),
        #                                nonlinearity=self.activation,
        #                                W=GlorotUniform(gain='relu')))
        #
        # self.dense_layers = dens
        #
        # shape = get_output_shape(dens[ndens-1])
        #
        # shuffle = DimshuffleLayer(self.dense_layers[ndens-1], pattern=(0, 2, 3, 1))

        shape = get_output_shape(feature_layer)
        shuffle = DimshuffleLayer(feature_layer, pattern=(0, 2, 3, 1))

        reshape = ReshapeLayer(shuffle,
                               shape=(np.prod(np.array(shape)[2:]), shape[1]))

        self.softmax = DenseLayer(dropout(reshape, p=0.),
                                  num_units=self.num_classes,
                                  nonlinearity=softmax)

        self.network = ReshapeLayer(DimshuffleLayer(self.softmax, pattern=(1, 0)),
                                    shape=(self.batch_size, self.num_classes) + shape[2:])

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):

        if shuffle:

            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):

            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            mu = np.array([123.68, 116.779, 103.939], dtype=np.float32)
            # mu = np.array([123.68/255., 116.779/255., 103.939/255.], dtype=np.float32)

            input = np.float32(inputs[excerpt[0]]) - mu

            input = np.expand_dims(np.rollaxis(input, 2, 0), axis=0)
            target = np.expand_dims(np.expand_dims(targets[excerpt[0]], axis=0), axis=0)

            xs, ys = input.shape[2:]

            xp = 80 + (64 - xs % 64)/2 + 1
            yp = 80 + (64 - ys % 64)/2 + 1

            input = np.pad(input, ((0, 0), (0, 0), (xp, xp), (yp, yp)), mode='symmetric')
            target = np.pad(target, ((0, 0), (0, 0), (xp, xp), (yp, yp)), mode='symmetric')

            if np.random.randint(0, 2) and shuffle:
                input = input[:, :, :, :-1]
                target = target[:, :, :, :-1]

            out = np.float32(self.extract_patches(input, self.extraction_step))
            out_targets = np.int32(self.extract_patches(target, self.extraction_step))
            valid = range(out.shape[0])

            if shuffle:
                # valid = np.random.choice(valid[0], int(valid[0].shape[0]*0.1), replace=False)
                np.random.shuffle(valid)

            out = out[valid]
            out_targets = out_targets[valid]

            inner_indices = range(out.shape[0])
            np.random.shuffle(inner_indices)

            for i in inner_indices:

                tars = out_targets[i:i+1, :, 80:-80, 80:-80]
                tars = tars.reshape((-1,))

                yield out[i:i+1], tars

    def iterate_minibatches_test(self, inputs, batchsize, shuffle=False):

        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

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

        n_h = i_h - p_h - 80 + 1
        n_w = i_w - p_w - 80 + 1

        # st_h = (i_h - p_h)/self.f
        # st_w = (i_w - p_w)/self.f

        for p, (i, j) in zip(patches,
                             product(range(80, n_h, p_h),
                                     range(80, n_w, p_w))):
            img[0, :, i:i + p_h, j:j + p_w] += p
            n[0, :, i:i + p_h, j:j + p_w] += 1

        return img/(n + 1e-9)
