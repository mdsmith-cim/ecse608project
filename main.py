import features as ft
import load_data as ld


def main():
    # Run this if you need to set the database path
    # sdbl.setDBLocation()

    # Main code
    trainData = ld.load_data('train')
    testData = ld.load_data('test')
    hogTrain, hogTest = ft.calculateFeatures(trainData, testData)


if __name__ == "__main__":
    main()
    # Things to do
    # - Multiple scales of HoG

    # - Reshape ground truth into same shape
