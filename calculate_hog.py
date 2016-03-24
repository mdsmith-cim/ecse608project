from numpy import array
from skimage import color
from skimage.feature import hog


# input = skimage.io.collection
def calculate_hog(imgCol, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    print "Computing HoG Features with settings:"
    print "Orientations: %i\n Pixels/Cell: %s\n Cells/Block: %s" % (orientations, (pixels_per_cell), (cells_per_block))
    fd = [hog(color.rgb2gray(img), orientations=orientations, pixels_per_cell=pixels_per_cell,
              cells_per_block=cells_per_block, feature_vector=True) for img in imgCol]
    print "Computed %i HoG Features" % len(fd)
    return array(fd)
