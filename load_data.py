import ConfigParser
import os
import socket

import skimage.io as io


def load_data(imageset='train', database='voc2011'):
    optionname = database + 'dir_' + socket.gethostname()

    setToUse = 'train.txt'
    if imageset == 'test':
        setToUse = 'val.txt'

    # Load image names
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    db_path = config.get('Databases', optionname)

    imageListFile = os.path.join(db_path, 'ImageSets', 'Segmentation', setToUse)

    with open(imageListFile) as f:
        imageList = f.read().splitlines()

    imageList = [os.path.join(db_path, 'JPEGImages', s) + '.jpg' for s in imageList]

    collection = io.imread_collection(imageList, conserve_memory=False)

    print "Loaded %i images from file %s" % (len(collection), imageListFile)

    return collection
