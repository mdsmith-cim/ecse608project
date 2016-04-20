from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import ConfigParser
import socket
import cPickle as pickle
import os

import copy

classifierTypes = {'randomForest' : RandomForestClassifier,
                   'logisticRegression' : LogisticRegressionCV}

defaultValues = {'randomForest' : {'n_jobs': -1, 'n_estimators': 40, 'oob_score': True, 'verbose': 1, 'class_weight': 'balanced' },
                 'logisticRegression': {}}

class BasicClassifier:
    def __init__(self, classifierType, database='voc2011', **kwargs):
        try:
            classType = classifierTypes[classifierType]
        except KeyError:
            raise NotImplementedError('Classifiers other than ' + str(classifierTypes.keys()) + ' not implemented.')

        parameters = copy.deepcopy(defaultValues[classifierType])
        parameters.update(kwargs)

        self.theClassifier = classType(**parameters)

        config = ConfigParser.SafeConfigParser()
        config.read('config.cfg')

        self.dir = config.get('Databases', database + 'dir_' + socket.gethostname())

    def trainModel(self, X, y):
        print "Training model on %i observations" % X.shape[0]
        self.theClassifier.fit(X, y.ravel())
        print "Fit %i classes" % self.theClassifier.n_classes_
        print "Out of bag estimate score: %f" % self.theClassifier.oob_score_

    def printScore(self, testX, testy):
        print "Mean score on test data: %f" % self.theClassifier.score(testX, testy.ravel())

    def getClassifierObject(self):
        return self.theClassifier

    def getPrediction(self, dataX):
        return self.theClassifier.predict(dataX)

    def saveToDisk(self, filename):
        print "Saving to disk..."
        file = open(os.path.join(self.dir, filename), 'wb')
        pickle.dump(self.theClassifier, file, protocol=pickle.HIGHEST_PROTOCOL )
        file.close()

    def loadFromDisk(self, filename):
        print "Loading from disk..."
        file = open(os.path.join(self.dir, filename), 'rb')
        self.theClassifier = pickle.load(file)
        file.close()

