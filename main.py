import features as ft
import load_data as ld
from basicClassifier.bscCls import BasicClassifier
import gc
#import set_db_location as sdbl

def main():
    # Run this if you need to set the database path
    # sdbl.setDBLocation()

    # Main code
    trainData, trainLabels = ld.load_images('train')
    testData, testLabels = ld.load_images('test')
    hogTrain, trainLabelsV, hogTest, testLabelsV = ft.calculateFeatures(trainData, trainLabels, testData, testLabels)

    gc.collect()
    cls = BasicClassifier('randomForest')
    cls.trainModel(hogTrain, trainLabelsV)
    gc.collect()

    cls.saveToDisk('rndForest.dat')
    gc.collect()

    print "Calculating score..."
    cls.printScore(hogTest, testLabelsV)


if __name__ == "__main__":
    main()
