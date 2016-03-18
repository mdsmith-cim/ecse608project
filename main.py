import load_data as ld
import features as ft

def main():

    trainData = ld.load_data('train')
    testData = ld.load_data('test')
    hogTrain, hogTest = ft.calculateFeatures(trainData, testData)


if __name__ == "__main__":
    main()
# Things to do
# - Multiple scales of HoG

# - Reshape ground truth into same shape