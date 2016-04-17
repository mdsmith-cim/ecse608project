from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

import copy

classifierTypes = {'randomForest' : RandomForestClassifier,
                   'logisticRegression' : LogisticRegressionCV}

defaultValues = {'randomForest' : {'n_jobs': -1, 'n_estimators': 20, 'oob_score': True, 'verbose': 1, 'class_weight': 'balanced' },
                 'logisticRegression': {}}

class BasicClassifier:
    def __init__(self, classifierType, **kwargs):
        try:
            classType = classifierTypes[classifierType]
        except KeyError:
            raise NotImplementedError('Classifiers other than ' + str(classifierTypes.keys()) + ' not implemented.')

        parameters = copy.deepcopy(defaultValues[classifierType])
        parameters.update(kwargs)

        self.theClassifier = classType(**parameters)

    def trainModel(self, X, y):
        self.theClassifier.fit(X, y.ravel())
        print "Fit %i classes" % self.theClassifier.n_classes_
        print "Out of bag estimate score: %d" % self.theClassifier.oob_score_

    def getScore(self, testX, testy):
        print "Mean score on test data: %f" % self.theClassifier.score(testX, testy.ravel())

    def getClassifierObject(self):
        return self.theClassifier

    def getPrediction(self, dataX, datay):
        return self.theClassifier.predict(dataX, datay.ravel())

