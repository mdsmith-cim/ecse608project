import ConfigParser
import os
import socket
import scipy.misc as scm

import skimage.io as io

def load_images(imageset='train', database='voc2011'):
    optionname = database + 'dir_' + socket.gethostname()

    if imageset == 'train':
        setToUse = 'train.txt'
    elif imageset == 'test':
        setToUse = 'val.txt'
    else:
        raise Exception("Please specify training or testing data i.e. imageset='train' or imageset='test'")

    # Load image names
    config = ConfigParser.SafeConfigParser()
    config.read('config.cfg')

    try:
        db_path = config.get('Databases', optionname)
    except Exception as e:
        raise Exception("Unable to read path to database from config file.  Expected to find entry named " + optionname + " in [Databases] section.")

    imageListFile = os.path.join(db_path, 'ImageSets', 'Segmentation', setToUse)

    with open(imageListFile) as f:
        imageList = f.read().splitlines()

    imageListJPEG = [os.path.join(db_path, 'JPEGImages', s) + '.jpg' for s in imageList]

    collection = io.imread_collection(imageListJPEG, conserve_memory=False)

    print "Loaded %i images from file %s" % (len(collection), imageListFile)

    # Now we load the PNG class images

    classList = [os.path.join(db_path, 'SegmentationClass', s) + '.png' for s in imageList]

    # Use mode P to load labels

    labels = [scm.imread(i, mode='P') for i in classList]

    # Remove all 255's (segment borders)
    for i in labels:
        i[i == 255] = 0

    return collection, labels
