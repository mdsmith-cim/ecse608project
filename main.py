import gc

import features as ft
import load_data as ld
from basicClassifier.bscCls import BasicClassifier
import numpy as np


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

    gc.collect()
    cls = BasicClassifier('randomForest')

    cls.trainModel(segTrainFt, segTrainLab)
    gc.collect()

    cls.saveToDisk('rndForest.dat')
    gc.collect()

    print "Calculating score..."
    cls.printScore(segTestFt, segTestLab)




def calculateAccuracyScore(predictedLabelMap, actualLabels):
    score = []
    for i in range(0, len(predictedLabelMap)):
        predLab = predictedLabelMap[i]
        actLab = actualLabels[i]
        totalPixels = float(np.prod(predLab.shape))
        score.append(np.sum(predLab == actLab) / totalPixels)

    avgScore = np.mean(score)
    print "Mean accuracy: %f" % avgScore

if __name__ == "__main__":
    main()
