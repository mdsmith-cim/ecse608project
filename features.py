import ConfigParser
import gc
import os
import socket
from ast import literal_eval
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy.misc as scm
import scipy.ndimage as ndi
from skimage import color
from skimage.feature import hog


# NOTE: We assume pixels_per_cell has the same size in both directions in multiple code locations
def calculateFeatures(trainData, trainLabels, testData, testLabels, featureType='hog', database='voc2011',
                      featureSettings={'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1),
                                       'scales': [1, 2, 4, 8]}):
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    hostname = socket.gethostname()
    dirOption = database + 'dir_' + hostname

    if featureType == 'hog':
        try:
            print "Attempting to load previously computed features from disk..."

            dbLocation = config.get('Databases', dirOption)
            trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
            testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))

            featureSettingsSaved1 = trainFile['featureSettings']
            featureSettingsSaved2 = testFile['featureSettings']

            for k,v in featureSettings.iteritems():
                if featureSettingsSaved1[k] != v or featureSettingsSaved2[k] != v:
                    raise Exception('Feature settings for saved data do not match request')

            hogTrain = trainFile['hogTrain']
            trainLabelsR = trainFile['trainLabels']

            hogTest = testFile['hogTest']
            testLabelsR = testFile['testLabels']

            print "Features loaded."

        except Exception as e:
            print e
            print "Unable to load data from disk; calculating features..."

            # Create data
            scales = featureSettings['scales']
            smallestScale = np.min(scales)

            calcHOG_partial_train = partial(calculateMultiscaleHOG, smallestScale=smallestScale,
                                            featureSettings=featureSettings, data=trainData)

            calcHOG_partial_test = partial(calculateMultiscaleHOG, smallestScale=smallestScale,
                                           featureSettings=featureSettings, data=testData)

            p = Pool()
            results_train = p.map_async(calcHOG_partial_train, scales)
            results_test = p.map_async(calcHOG_partial_test, scales)
            p.close()
            p.join()

            results_train = results_train.get()
            results_test = results_test.get()
            # We now have lists for training and testing of the data by scale, all resized to one uniform grid pattern

            # Get database location to save to
            dbLocation = config.get('Databases', dirOption)

            # Reshape label data
            pixels_per_cell = tuple(smallestScale * np.array(featureSettings['pixels_per_cell']))

            print "Resizing image class labels..."

            trainLabelsR = [scm.imresize(lab, 1. / pixels_per_cell[0], interp='nearest') for lab in trainLabels]
            testLabelsR = [scm.imresize(lab, 1. / pixels_per_cell[0], interp='nearest') for lab in testLabels]

            trainLabelsR = labelReshape(trainLabelsR)
            testLabelsR = labelReshape(testLabelsR)

            numScales = len(scales)

            for i in range(0, numScales):
                results_train[i] = hogReshape(results_train[i])
                results_test[i] = hogReshape(results_test[i])

            hogTrain = np.hstack(results_train)
            hogTest = np.hstack(results_test)

            # Save data to disk
            print "Saving data to disk..."

            np.savez_compressed(os.path.join(dbLocation, featureType + '_train.npz'), hogTrain=hogTrain,
                                trainLabels=trainLabelsR, featureSettings=featureSettings)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), hogTest=hogTest,
                                testLabels=testLabelsR, featureSettings=featureSettings)

            print "Data saved."

        return hogTrain, trainLabelsR, hogTest, testLabelsR
    else:
        raise NotImplementedError('Features other than HOG not implemented.')


def calculateMultiscaleHOG(scale, smallestScale, featureSettings, data):
    orientations = featureSettings['orientations']
    cells_per_block = featureSettings['cells_per_block']

    # note: we assume both directions are the same
    pixels_per_cell = tuple(scale * np.array(featureSettings['pixels_per_cell']))
    print "Calculating HOG features..."
    hogData = calculate_hog(data, orientations=orientations,
                            pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block)

    # hogTrain, hogTest now are lists of length # images of HOG features
    # We need to resize them to the smallest scale (i.e. the least pixels/grid square)
    print "Resizing HoG features..."
    if scale != smallestScale:
        pixels_per_cell_smallest = smallestScale * np.array(featureSettings['pixels_per_cell'])

        for i in range(0, len(hogData)):
            # print "Resizing feature %d/%d" % (i + 1, len(hogData))
            matrixSize = np.array(hogData[i].shape)[[0, 1, 4]]
            reshaped = np.reshape(hogData[i], matrixSize)

            factor = np.append(np.array(data[i].shape[0:2]) / pixels_per_cell_smallest,
                               orientations) / matrixSize.astype('float')

            hogData[i] = ndi.zoom(reshaped, factor, mode='nearest')

    else:
        hogData = [np.squeeze(x) for x in hogData]

    gc.collect()
    return hogData


def labelReshape(data):
    # Get how many rows we will need
    numObservations = 0
    for i in data:
        numObservations += np.prod(i.shape[0:2])

    # preallocate observation matrix
    observ = np.zeros((numObservations, 1))

    # Fill observation matrix
    idx = 0
    for i in data:
        num = np.prod(i.shape[0:2])
        observ[idx:idx + num] = np.reshape(i, (num, 1))
        idx += num
    return observ


def hogReshape(data):
    # Get how many rows we will need
    orientations = data[0].shape[2]
    numObservations = 0
    for i in data:
        numObservations += np.prod(i.shape[0:2])

    # preallocate observation matrix
    observ = np.zeros((numObservations, orientations))

    # Fill observation matrix
    idx = 0
    for i in data:
        num = np.prod(i.shape[0:2])
        observ[idx:idx + num] = np.reshape(i, (num, orientations))
        idx += num
    return observ


# HoG feature calculation
# input = skimage.io.collection
def calculate_hog(imgCol, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), verbose=False):
    if verbose:
        print "Computing HoG Features with settings:"
        print "Orientations: %i\nPixels/Cell: %s\nCells/Block: %s" % (
            orientations, (pixels_per_cell), (cells_per_block))

    fd = [hog(color.rgb2gray(img), orientations=orientations, pixels_per_cell=pixels_per_cell,
              cells_per_block=cells_per_block, feature_vector=False) for img in imgCol]
    if verbose:
        print "Computed %i HoG Features" % len(fd)
    return fd
