import numpy as np
from sklearn.svm import SVC

def runSVM(trainData, trainLabels, testData, testLabels):

    # TODO: parameters...
    clf = SVC(kernel='linear')
    print "Fitting SVM.."
    clf.fit(trainData, trainLabels.ravel())

    print "Predicting test data..."
    return clf, clf.predict(testData)

    #print "Accuracy: %f" % clf.score(testData, testLabels.ravel())
