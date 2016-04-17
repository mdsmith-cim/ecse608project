import ConfigParser
import os
import socket
from ast import literal_eval

import numpy as np
import scipy.misc as scm
import scipy.ndimage as ndi
from skimage import color
from skimage.feature import hog
from multiprocessing import Pool
from functools import partial


def calculateFeatures(trainData, trainLabels, testData, testLabels, featureType='hog', database='voc2011',
                      featureSettings={'orientations': 8, 'pixels_per_cell': (2, 2), 'cells_per_block': (1, 1),
                                       'scales': [1, 2, 4, 8]}):
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    hostname = socket.gethostname()
    dirOption = database + 'dir_' + hostname

    if featureType == 'hog':
        try:
            print "Attempting to load previously computed features from disk..."
            # Below config line will throw an exception if the 'dataSaved' config entry does not exist
            # -> data was never saved
            config.getboolean(featureType, 'dataSaved')

            orientations = config.getint(featureType, 'orientations')
            pixels_per_cell = literal_eval(config.get(featureType, 'pixels_per_cell'))
            cells_per_block = literal_eval(config.get(featureType, 'cells_per_block'))
            scales = literal_eval(config.get(featureType, 'scales'))

            if orientations != featureSettings['orientations'] or pixels_per_cell != featureSettings[
                'pixels_per_cell'] or cells_per_block != featureSettings['cells_per_block'] or scales != \
                    featureSettings['scales']:
                raise Exception('Feature settings for saved data do not match request')

            dbLocation = config.get('Databases', dirOption)
            trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
            hogTrain = trainFile['hogTrain']
            trainLabelsR = trainFile['trainLabels']
            testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))
            hogTest = testFile['hogTest']
            testLabelsR = testFile['testLabels']

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
            results_train = p.map(calcHOG_partial_train, scales)
            results_test = p.map(calcHOG_partial_test, scales)
            # We now have lists for training and testing of the data by scale, all resized to one uniform grid pattern
            p.close()
            p.join()

            # Get database location to save to
            dbLocation = config.get('Databases', dirOption)

            # Reshape label data
            pixels_per_cell = tuple(smallestScale * np.array(featureSettings['pixels_per_cell']))

            print "Resizing image class labels..."
            trainLabelsR = [scm.imresize(lab, 1. / pixels_per_cell[0], interp='nearest') for lab in trainLabels]
            testLabelsR = [scm.imresize(lab, 1. / pixels_per_cell[0], interp='nearest') for lab in testLabels]

            trainLabelsR = labelReshape(trainLabelsR)
            testLabelsR = labelReshape(testLabelsR)

            bla = []
            for i in range(0,len(scales)):
                bla.append(hogReshape(results_train[i]))

            # Save data to disk
            print "Saving data to disk..."

            np.savez_compressed(os.path.join(dbLocation, featureType + '_train.npz'), hogTrain=hogTrain,
                                trainLabels=trainLabelsR)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), hogTest=hogTest,
                                testLabels=testLabelsR)

            # Note existence in config file
            if not (config.has_section(featureType)):
                config.add_section(featureType)

            config.set(featureType, 'dataSaved', 'True')
            config.set(featureType, 'orientations', str(orientations))
            config.set(featureType, 'pixels_per_cell', str(pixels_per_cell))
            config.set(featureType, 'cells_per_block', str(cells_per_block))

            with open('config.cfg', 'wb') as configfile:
                config.write(configfile)
            configfile.close()
            print "Data saved."

        return hogTrain, trainLabelsR, hogTest, testLabelsR
    else:
        raise NotImplementedError('Features other than HOG not implemented.')


def calculateMultiscaleHOG(scale, smallestScale, featureSettings, data):
    orientations = featureSettings['orientations']
    cells_per_block = featureSettings['cells_per_block']

    # note: we assume both directions are the same elsewhere in the code
    pixels_per_cell = tuple(scale * np.array(featureSettings['pixels_per_cell']))

    hogData = calculate_hog(data, orientations=orientations,
                            pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block)

    # hogTrain, hogTest now are lists of length # images of HOG features
    # We need to resize them to the smallest scale (i.e. the least pixels/grid square)
    print "Resizing HoG features..."
    if scale != smallestScale:
        factor = float(scale) / smallestScale
        hogData = [ndi.zoom(np.squeeze(x), (factor, factor, 1), mode='nearest') for x in
                   hogData]
    else:
        hogData = [np.squeeze(x) for x in hogData]

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
def calculate_hog(imgCol, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    print "Computing HoG Features with settings:"
    print "Orientations: %i\nPixels/Cell: %s\nCells/Block: %s" % (orientations, (pixels_per_cell), (cells_per_block))
    fd = [hog(color.rgb2gray(img), orientations=orientations, pixels_per_cell=pixels_per_cell,
              cells_per_block=cells_per_block, feature_vector=False) for img in imgCol]
    print "Computed %i HoG Features" % len(fd)
    return fd
