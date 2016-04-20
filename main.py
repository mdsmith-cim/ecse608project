import numpy as np

import features as ft
import load_data as ld
from basicClassifier.bscCls import BasicClassifier

classDefs = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
             8: 'cat', 9: 'chair',
             10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted plant',
             17: 'sheep', 18: 'sofa', 19: 'train', 20:
                 'tv/monitor'}


# import set_db_location as sdbl

def main():
    # Run this if you need to set the database path
    # sdbl.setDBLocation()

    # Main code
    trainData, trainLabels = ld.load_images('train')
    testData, testLabels = ld.load_images('test')
    # hogTrain, trainLabelsV, hogTest, testLabelsV = ft.calculateFeatures(trainData, trainLabels, testData, testLabels)
    segTrainFt, segTrainLab, segTestFt, segTestLab = ft.calculateFeatures(trainData,
                                                                          trainLabels,
                                                                          testData,
                                                                          testLabels,
                                                                          featureType='segment2')

    # gc.collect()
    cls = BasicClassifier('randomForest')

    # cls.trainModel(segTrainFt, segTrainLab)
    cls.loadFromDisk('rndForest_surf.dat')
    # gc.collect()

    # cls.saveToDisk('rndForest_surf.dat')
    # gc.collect()

    # print "Calculating score..."
    # cls.printScore(segTestFt, segTestLab)

    predictedLabelTest = cls.getPrediction(segTestFt)
    predictedLabelTrain = cls.getPrediction(segTrainFt)

    predictedLabelTrainMap = ft.getLabeledImages2(trainData, predictedLabelTrain)
    predictedLabelTestMap = ft.getLabeledImages2(testData, predictedLabelTest)

    print "TRAINING: SURF"
    printIUScore(predictedLabelTrainMap, trainLabels)
    print "TESTING: SURF"
    printIUScore(predictedLabelTestMap, testLabels)

    seg1TrainFt, seg1TrainLab, seg1TrainAnn, seg1TestFt, seg1TestLab, seg1TestAnn = ft.calculateFeatures(trainData,
                                                                                                         trainLabels,
                                                                                                         testData,
                                                                                                         testLabels,
                                                                                                         featureType='segment')

    cls = BasicClassifier('randomForest')

    # cls.trainModel(seg1TrainFt, seg1TrainLab)
    cls.loadFromDisk('rndForest_rgb.dat')
    # cls.saveToDisk('rndForest_rgb.dat')

    predictedLabelTest = cls.getPrediction(seg1TestFt)
    predictedLabelTrain = cls.getPrediction(seg1TrainFt)

    predictedLabelTrainMap = ft.getLabeledImages(predictedLabelTrain, seg1TrainAnn)
    predictedLabelTestMap = ft.getLabeledImages(predictedLabelTest, seg1TestAnn)

    print "TRAINING: RGB"
    printIUScore(predictedLabelTrainMap, trainLabels)
    print "TESTING: RGB"
    printIUScore(predictedLabelTestMap, testLabels)


def calculateAccuracyScore(predictedLabelMap, actualLabels):
    score = []
    for i in range(0, len(predictedLabelMap)):
        predLab = predictedLabelMap[i]
        actLab = actualLabels[i]
        totalPixels = float(np.prod(predLab.shape))
        score.append(np.sum(predLab == actLab) / totalPixels)

    avgScore = np.mean(score)
    print "Mean accuracy: %f" % avgScore


# predictedLabelMap = a list containing arrays with values set to the predicted label # (0-20)
# actuallabelMap = a list (same length as predictedLabelMap) containing arrays with values set to the
# ground truth label
def printIUScore(predictedLabelMap, actualLabelMap):
    intersect = np.zeros((21, 1))
    union = np.zeros((21, 1))

    for labIdx in range(0, len(predictedLabelMap)):
        for cl in range(0, 21):
            intersect[cl] += np.sum((actualLabelMap[labIdx] == cl) & (predictedLabelMap[labIdx] == cl))
            union[cl] += np.sum((actualLabelMap[labIdx] == cl) | (predictedLabelMap[labIdx] == cl))

    iuscoreByClass = intersect / union

    for cl in range(0, 21):
        scr = iuscoreByClass[cl]
        print "%s: %f" % (classDefs[cl], scr)
    print "Mean IU: %f" % np.mean(iuscoreByClass)


if __name__ == "__main__":
    main()
