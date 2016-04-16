from sklearn.linear_model import LogisticRegressionCV

def runLogistic(hogTrain, trainLabelsV, hogTest, testLabelsV):

    reg = LogisticRegressionCV(solver='lbfgs', multi_class='multinomial', n_jobs=4)

    reg.fit(hogTrain, trainLabelsV.ravel())


