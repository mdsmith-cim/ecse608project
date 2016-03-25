import ConfigParser
import os
import socket

import numpy as np

import scipy.misc as scm
from skimage import color
from skimage.feature import hog

def calculateFeatures(trainData, trainLabels, testData, testLabels, featureType='hog', database='voc2011'):
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    hostname = socket.gethostname()
    dirOption = database + 'dir_' + hostname

    try:
        print "Attempting to load previously computed features from disk..."
        config.getboolean(featureType, 'dataSaved')
        dbLocation = config.get('Databases', dirOption)
        trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
        hogTrain = trainFile['hogTrain']
        trainLabelsR = trainFile['trainLabels']
        testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))
        hogTest = testFile['hogTest']
        testLabelsR = testFile['testLabels']


    except Exception as e:
        print "Unable to load data from disk; calculating features..."
        if (featureType == 'hog'):
            # Create data
            orientations = 8
            pixels_per_cell = (8, 8) # note: we assume both directions are the same elswhere in the code
            cells_per_block = (1, 1)
            hogTrain = hogReshape(calculate_hog(trainData, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block))
            hogTest = hogReshape(calculate_hog(testData, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                       cells_per_block=cells_per_block))

            # Get database location to save to
            dbLocation = config.get('Databases', dirOption)

            # Reshape label data
            print "Resizing image class labels..."
            trainLabelsR = [scm.imresize(lab, 1./pixels_per_cell[0], interp='nearest') for lab in trainLabels]
            testLabelsR = [scm.imresize(lab, 1./pixels_per_cell[0], interp='nearest') for lab in testLabels]

            trainLabelsR = labelReshape(trainLabelsR)
            testLabelsR = labelReshape(testLabelsR)

            # Save data to disk
            print "Saving data to disk..."

            np.savez_compressed(os.path.join(dbLocation, featureType + '_train.npz'), hogTrain=hogTrain,
                                trainLabels = trainLabelsR)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), hogTest= hogTest,
                                testLabels = testLabelsR)

            # Note existence in config file
            if not(config.has_section(featureType)):
                config.add_section(featureType)

            config.set(featureType, 'dataSaved', 'True')
            with open('config.cfg', 'wb') as configfile:
                config.write(configfile)
            configfile.close()
            print "Data saved."
        else:
            raise NotImplementedError('Features other than HOG not implemented.')

    return hogTrain, trainLabelsR, hogTest, testLabelsR


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
    orientations = data[0].shape[4]
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
    print "Orientations: %i\n Pixels/Cell: %s\n Cells/Block: %s" % (orientations, (pixels_per_cell), (cells_per_block))
    fd = [hog(color.rgb2gray(img), orientations=orientations, pixels_per_cell=pixels_per_cell,
              cells_per_block=cells_per_block, feature_vector=False) for img in imgCol]
    print "Computed %i HoG Features" % len(fd)
    return fd
