import ConfigParser
import os
import socket

import numpy as np

import calculate_hog as ch

import scipy.misc as scm


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
            pixels_per_cell = (8, 8)
            cells_per_block = (1, 1)
            hogTrain = ch.calculate_hog(trainData, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block)
            hogTest = ch.calculate_hog(testData, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                       cells_per_block=cells_per_block)

            # Get database location to save to
            dbLocation = config.get('Databases', dirOption)

            # Reshape label data
            print "Resizing image class labels..."
            trainLabelsR = [scm.imresize(lab, 1./8, interp='nearest').ravel() for lab in trainLabels]
            testLabelsR = [scm.imresize(lab, 1./8, interp='nearest').ravel() for lab in testLabels]

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
