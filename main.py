import features as ft
import load_data as ld
import removeBorders as rb

def main():
    # Run this if you need to set the database path
    # sdbl.setDBLocation()

    # Main code
    trainData, trainLabels = ld.load_images('train')
    testData, testLabels = ld.load_images('test')
    hogTrain, trainLabelsV, hogTest, testLabelsV = ft.calculateFeatures(trainData, trainLabels, testData, testLabels)

    trainLabelsV = rb.removeBorders(trainLabelsV)
    testLabelsV = rb.removeBorders(testLabelsV)

if __name__ == "__main__":
    main()
    # Things to do
    # - Multiple scales of HoG
