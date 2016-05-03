import ConfigParser
import cPickle as pickle
import gc
import os
import socket
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy.misc as scm
import scipy.ndimage as ndi
from skimage import color
from skimage.feature import hog

# from sklearn.feature_extraction.image import PatchExtractor
# from sklearn.cluster import spectral_clustering
# from sklearn.feature_extraction import image
from skimage.segmentation import slic
from numpy.random import choice
from matplotlib import pyplot as plt
from scipy.stats import mode
import cv2

defaultSettings = {'hog': {'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1),
                           'scales': [1, 2, 4]}, 'images': {},
                   'segment': {'vec_length': 30000, 'n_segments': 20, 'sigma': 1, 'compactness': 10},
                   'segment2': {'step_size': 10, 'extended': True}}


# NOTE: We assume pixels_per_cell has the same size in both directions in multiple code locations
def calculateFeatures(trainData, trainLabels, testData, testLabels, featureType='hog', database='voc2011',
                      **settings):
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    hostname = socket.gethostname()
    dirOption = database + 'dir_' + hostname

    if featureType == 'hog':
        featureSettings = deepcopy(defaultSettings[featureType])
        featureSettings.update(**settings)
        try:
            print "Attempting to load previously computed features from disk..."

            dbLocation = config.get('Databases', dirOption)
            trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
            testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'rb')
            featureSettingsSaved = pickle.load(file)
            file.close()

            for k, v in featureSettings.iteritems():
                if featureSettingsSaved[k] != v:
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
                                trainLabels=trainLabelsR)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), hogTest=hogTest,
                                testLabels=testLabelsR)

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'wb')
            pickle.dump(featureSettings, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

            print "Data saved."

        return hogTrain, trainLabelsR, hogTest, testLabelsR
    elif featureType == 'images':
        raise NotImplementedError('Not implemented (yet) or probably ever')
    elif featureType == 'segment':

        featureSettings = deepcopy(defaultSettings[featureType])
        featureSettings.update(**settings)

        try:
            print "Attempting to load previously computed features from disk..."

            dbLocation = config.get('Databases', dirOption)
            trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
            testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'rb')
            featureSettingsSaved = pickle.load(file)
            file.close()

            for k, v in featureSettings.iteritems():
                if featureSettingsSaved[k] != v:
                    raise Exception('Feature settings for saved data do not match request')

            trft = trainFile['trft']
            trlb = trainFile['trlb']
            tra = trainFile['tra']

            tstft = testFile['tstft']
            tstlb = testFile['tstlb']
            tsta = testFile['tsta']

            print "Features loaded."

        except Exception as e:

            print e
            print "Unable to load data from disk; calculating features..."
            p = Pool()

            # tr = p.apply_async(segmentProcess, (trainData, trainLabels, featureSettings))
            # tst = p.apply_async(segmentProcess, (testData, testLabels, featureSettings))
            # p.close()
            # p.join()

            tr = segmentProcess(trainData, trainLabels, featureSettings)
            tst = segmentProcess(testData, testLabels, featureSettings)

            # tr = tr.get()
            # tst = tst.get()
            trft = tr[0]
            trlb = tr[1]
            tra = tr[2]
            tstft = tst[0]
            tstlb = tst[1]
            tsta = tst[2]

            print "Saving data to disk..."

            np.savez_compressed(os.path.join(dbLocation, featureType + '_train.npz'), trft=trft,
                                trlb=trlb, tra=tra)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), tstft=tstft,
                                tstlb=tstlb, tsta=tsta)

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'wb')
            pickle.dump(featureSettings, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

        return trft, trlb, tra, tstft, tstlb, tsta
    elif featureType == 'segment2':
        featureSettings = deepcopy(defaultSettings[featureType])
        featureSettings.update(**settings)

        try:
            print "Attempting to load previously computed features from disk..."

            dbLocation = config.get('Databases', dirOption)
            trainFile = np.load(os.path.join(dbLocation, featureType + '_train.npz'))
            testFile = np.load(os.path.join(dbLocation, featureType + '_test.npz'))

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'rb')
            featureSettingsSaved = pickle.load(file)
            file.close()

            for k, v in featureSettings.iteritems():
                if featureSettingsSaved[k] != v:
                    raise Exception('Feature settings for saved data do not match request')

            trft = trainFile['trft']
            trlb = trainFile['trlb']

            tstft = testFile['tstft']
            tstlb = testFile['tstlb']

            print "Features loaded."

        except Exception as e:

            print e
            print "Unable to load data from disk; calculating features..."
            p = Pool()

            tr = p.apply_async(segmentProcess2, (trainData, trainLabels, featureSettings))
            tst = p.apply_async(segmentProcess2, (testData, testLabels, featureSettings))
            p.close()
            p.join()

            # tr = segmentProcess2(trainData, trainLabels, featureSettings)
            # tst = segmentProcess2(testData, testLabels, featureSettings)

            tr = tr.get()
            tst = tst.get()
            trft = tr[0]
            trlb = tr[1]
            tstft = tst[0]
            tstlb = tst[1]

            print "Saving data to disk..."

            np.savez_compressed(os.path.join(dbLocation, featureType + '_train.npz'), trft=trft,
                                trlb=trlb)
            np.savez_compressed(os.path.join(dbLocation, featureType + '_test.npz'), tstft=tstft,
                                tstlb=tstlb)

            file = open(os.path.join(dbLocation, featureType + '_settings.dat'), 'wb')
            pickle.dump(featureSettings, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

        return trft, trlb, tstft, tstlb

    else:
        raise NotImplementedError('Not implemented.')


def segmentProcess(data, labels, featureSettings):
    feat = []
    labs = []

    vecLength = featureSettings['vec_length']
    n_segments = featureSettings['n_segments']
    sigma = featureSettings['sigma']
    compactness = featureSettings['compactness']

    assignments = []

    for imgIdx in range(0, len(data)):
        # print "Processing image %i/%i" % (imgIdx + 1, len(data))
        img = data[imgIdx]
        seg = slic(img, n_segments=n_segments, sigma=sigma, compactness=compactness)
        assignments.append(seg)
        for i in range(seg.min(), seg.max()):
            pix = img[seg == i]
            lab = labels[imgIdx][seg == i]
            try:
                reduced = choice(pix.ravel(), vecLength, replace=False)
            except ValueError:
                reduced = choice(pix.ravel(), vecLength, replace=True)
            commonLabel = mode(lab)[0]

            feat.append(reduced)
            labs.append(commonLabel)

    feat = np.vstack(feat)
    labs = np.vstack(labs)
    return feat, labs, assignments


def segmentProcess2(data, labels, featureSettings):
    feat = []
    labs = []

    step_size = featureSettings['step_size']

    extr = cv2.xfeatures2d.SURF_create(extended=featureSettings['extended'])

    for imgIdx in range(0, len(data)):
        # print "Processing image %i/%i" % (imgIdx + 1, len(data))
        img = data[imgIdx]
        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, grayImg.shape[0], step_size)
              for x in range(0, grayImg.shape[1], step_size)]
        kp2, des = extr.compute(grayImg, kp)

        feat.append(des)

        for i in kp:
            pt = i.pt
            size = i.size
            lab = labels[imgIdx][int(max(pt[1] - size / 2, 0)): int(min(grayImg.shape[0], pt[1] + size / 2)),
                  int(max(pt[0] - size / 2, 0)): int(min(pt[0] + size / 2, grayImg.shape[1]))]

            commonLabel = mode(lab, None)[0]

            labs.append(commonLabel)

    feat = np.vstack(feat)
    labs = np.vstack(labs)
    return feat, labs


# For segment2
def getLabeledImages2(images, computedLabels, **settings):
    featureSettings = deepcopy(defaultSettings['segment2'])
    featureSettings.update(**settings)

    step_size = featureSettings['step_size']

    result = []
    idx = 0

    for imgIdx in range(0, len(images)):
        # print "Processing image %i/%i" % (imgIdx + 1, len(images))
        img = images[imgIdx]

        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]

        tmpResult = np.zeros(img.shape[0:2])

        for i in kp:
            pt = i.pt
            size = i.size
            tmpResult[int(max(pt[1] - size / 2, 0)): int(min(img.shape[0], pt[1] + size / 2)),
            int(max(pt[0] - size / 2, 0)): int(min(pt[0] + size / 2, img.shape[1]))] = computedLabels[idx]

            idx += 1

        result.append(tmpResult)

    return result


def showLabels(images, labelMap, imageNumber):
    ext = [0, 1, 0, 1]
    plt.imshow(labelMap[imageNumber], extent=ext)
    plt.hold(True)
    plt.imshow(images[imageNumber], extent=ext, alpha=0.5)
    plt.show()


# This is for RGB w/ segment
def getLabeledImages(assignedLabels, mapping):
    result = []
    idx = 0
    for i in range(0, len(mapping)):
        mapp = mapping[i]

        tmpLabelAssigment = np.zeros(mapp.shape)
        for j in range(mapp.min(), mapp.max()):
            tmpLabelAssigment[mapp == j] = assignedLabels[idx]
            idx += 1

        result.append(tmpLabelAssigment)

    return result


def getLabeledImagesHOG(images, assignedLabels, **settings):
    featureSettings = deepcopy(defaultSettings['hog'])
    featureSettings.update(**settings)

    scales = featureSettings['scales']
    smallestScale = np.min(scales)

    pixels_per_cell = smallestScale * np.array(featureSettings['pixels_per_cell'])

    result = []


    totalNumObsv = 0
    beginIdx = 0
    for i in range(0, len(images)):
        # Figure out how many observations we got based on size
        img = images[i]
        imgShape = img.shape
        featureShape = np.array(imgShape) / pixels_per_cell
        numObsv = np.prod(imgShape)

        # for testing
        totalNumObsv += numObsv

        result.append(scm.imresize(np.reshape(assignedLabels[beginIdx : beginIdx + numObsv], featureShape), pixels_per_cell[0]))
        beginIdx += numObsv





# def imageProcess(data, labels):
#     # First: resize all images to the same size
#     # Get max size of images
#     maxSize = (0, 0, 0)
#     for img in data:
#         maxSize = np.maximum(maxSize, img.shape)
#
#     resizedImages = np.zeros(np.append(len(data), maxSize))
#     # Actually resize images now
#     print "Resizing images to %s" % str(maxSize)
#     for i in range(0, len(data)):
#         resizedImages[i] = scm.imresize(data[i], maxSize)
#
#     print "Resizing labels to match..."
#     maxSizeL = maxSize[0:2]
#     resizedLabels = np.zeros(np.append(len(labels), maxSizeL))
#     for i in range(0, len(labels)):
#         resizedLabels[i] = scm.imresize(data[i], maxSizeL)



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
