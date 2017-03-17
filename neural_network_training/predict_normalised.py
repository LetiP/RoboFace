# coding: utf-8

# from keras.models import Sequential
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
# from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib
from scipy import misc
import glob

# model = load_model('trained/pretrained_CelebA17Janabout90percentStillOverfit.h5')
# model = load_model('trained/pretrained_CelebA1.h5')
model = load_model('trained/pretrained_CelebA_normalised_augm.h5')
meanFace = np.load('mean_face_normalised.npy')

def mapAttributes(classes):
    with open('wanted_attributes_normalised.txt', 'r') as f:
        attributes = f.read()
    attributes = attributes.strip('\n').split(' ')

    result = []
    for i, cl in enumerate(classes):
        if cl == True:
            result.append(attributes[i])
    return result

files = glob.glob('../Face/leti/all_normalised/*.jpeg')
print "will predict on: ", len(files), 'files'
for i, file in enumerate(files):
    X_test = misc.imread(file).astype('float32')
    X_test -= meanFace
    # # for images from my webcam which have double size
    # X_test = misc.imresize(X_test, (218,178))
    X_test = np.expand_dims(X_test, axis=0)

    proba = model.predict_proba(X_test, batch_size=32, verbose=0)
    print file
    print proba
    print mapAttributes((proba > 0.6)[0])

# 2nd part: accuracy on each class

def loadFromHdf5(start, num, fileNum):
    global input_shape
    img_rows, img_cols = 128, 128
    X_train = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'images', start, start+num) #, normalizer=normalize_data)
    Y_train = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'labels', start, start+num)

    print("Done reading data from disk from {} to {}.".format(start, start+num))
    return X_train, np.array(Y_train)

accuracy = np.zeros((202599,13))
# accuracy = np.zeros((4096,13))

start, num = 0, 4096
while start + num < 202599: #202599:
    if start + num < 67533:
        fileNum = 1
        offset = 0
    elif start + num  > 67533 and start + num  < 2*67533:
        fileNum = 2
        offset = 67533
    else:
        fileNum = 3
        offset = 67533*2
    X_test, Y_test = loadFromHdf5(start-offset, num, fileNum)

    # subtracting the mean face
    X_test -= meanFace

    proba = model.predict_proba(X_test, batch_size=32, verbose=0)
    classes = proba > 0.6
    for i, c in enumerate(classes):
        accuracy[start + i] = (Y_test[i] == c)

    # print classes, 'CLASSES'
    # print Y_test
    start += num

on_classes = np.mean(accuracy, axis = 0)
print on_classes, 'accuracy on each class', on_classes.shape
print np.mean(on_classes), 'average accuracy'

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

# on_classes = np.array([0.81161309, 0.91842013,  0.79153895,  0.98008381,  0.95171743,  0.9510116,
#   0.89716139,  0.91579919,  0.89415545,  0.78403151,  0.70298471,  0.80345905,
#   0.91350895])

N = 13
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
rects1 = ax.bar(ind, on_classes, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('probabilities')
ax.set_title('Predicted probabilities for each class')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Black \n Hair', 'Blond \n Hair', 'Brown \n Hair', 'Eyeglasses', 'Gray \n Hair','Male', 'Mouth \n Slightly Open', 'No Beard', 'Smiling', 'Straight \n Hair', 'Wavy \n Hair', 'Earrings', 'Lipstick'))
ax.set_axisbelow(True)
ax.yaxis.grid(color='black', linestyle='dashed')

# def autolabel(rects):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')

# autolabel(rects1)
plt.savefig('accuracy_on_classes.pdf')
plt.show()