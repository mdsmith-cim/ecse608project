import calculate_hog as ch
import ConfigParser
import numpy as np
import os

def calculateFeatures(trainData, testData, featureType='hog', database='voc2011'):

    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    try:
        config.getboolean(featureType, 'dataSaved')
        dbLocation = config.get('Databases', database + 'dir')
        hogTrain = np.load(os.path.join(dbLocation,featureType + '_train.npy'))
        hogTest = np.load(os.path.join(dbLocation,featureType + '_test.npy'))


    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        if (featureType == 'hog'):
            # Create data
            orientations = 8
            pixels_per_cell = (8,8)
            cells_per_block = (1,1)
            hogTrain = ch.calculate_hog(trainData,orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
            hogTest = ch.calculate_hog(testData,orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

            # Get database location to save to
            dbLocation = config.get('Databases', database + 'dir')

            # Save data to disk
            print "Saving data to disk..."
            np.save(os.path.join(dbLocation,featureType + '_train.npy'), hogTrain)
            np.save(os.path.join(dbLocation,featureType + '_test.npy'), hogTest)

            # Note existence in config file
            config.add_section(featureType)

            config.set(featureType, 'dataSaved', 'True')
            with open('config.cfg', 'wb') as configfile:
                config.write(configfile)
            print "Data saved."
        else:
            raise NotImplementedError('Features other than HOG not implemented.')

    return hogTrain, hogTest
