# coding: utf-8

# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
# import tensorflow as tf
import numpy as np
from util.progressbar import ProgressBar
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import Adadelta #, Adam, SGD
from keras.regularizers import l2, activity_l2
from keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from scipy import misc
from pims import ImageSequence
import h5py
import time

batch_size = 64 # 128
nb_classes = 13 # eigentlich no young
nb_epoch = 32 # 12
global input_shape

# input image dimensions
img_rows, img_cols = 128, 128
# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

def readLabels(fnWantedAttr, fnLabels):
    '''
    Read in the labels of interest from 'wanted_attributes.txt' and generate a numpy array for training and testing.
    '''
    with open(fnWantedAttr, 'r') as f:
        attributes = f.read()

    with open(fnLabels, 'r') as g:
        allLabels = g.readlines()

    print("Using following training labels: \n {}".format(attributes))
    
    labelNames = allLabels[1].strip('\n').split(' ')[:-2]
    attributes = attributes.strip('\n').split(' ')

    positions = []
    for i, names in enumerate(labelNames):
        positions.append(i) # +1 because the first pos is the image name

    wantedLabels = np.zeros((len(allLabels)-2, len(positions)))
    for i, line in enumerate(allLabels):
        if i in [0, 1]:
            continue
        else:
            labelsLong = line.strip('\n').split('  ')
            labels = []
            for lala in labelsLong:
                 labels.append(lala.strip('\n').split(' '))
            labels = [item for sublist in labels for item in sublist]
            for j, pos in enumerate(positions):
                if labelNames[pos] in attributes:
                    wantedLabels[i-2, j] = int(labels[pos+1])

    return wantedLabels[:, np.all(wantedLabels != 0, axis=0)]

def load_data(test_size=0.2, from_npy=True):
    if from_npy:
        X = np.load('/export/home/lparcala/dataFace/Xsaved.npy')
        y = np.load('/export/home/lparcala/dataFace/ysaved.npy')
    else:
        y = readLabels('wanted_attributes.txt', 'list_attr_celeba.txt')
        X = np.array(ImageSequence('../Face/img_align_celeba.zip'))
        X = X.astype('float32')
        
        # convert labels to boolean
        y[y<0] = 0
        y[y>0] = 1

        h5f = h5py.File('data.h5', 'w')
        h5f.create_dataset('images', data=X)
        h5f.create_dataset('labels', data=y)
        # np.save("/export/home/lparcala/dataFace/Xsaved.npy", X)
        np.save("/export/home/lparcala/dataFace/ysaved.npy", y)

    print("Done reading data from disk.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test

def loadFromHdf5(start, num, test, fileNum):
    global input_shape
    img_rows, img_cols = 128, 128
    X_train = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'images', start, start+num) #, normalizer=normalize_data)
    Y_train = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'labels', start, start+num)
    X_test = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'images', start+num, start+num+test) #, normalizer=normalize_data)
    Y_test = HDF5Matrix('/export/home/lparcala/dataFace/celebA{}.h5'.format(fileNum),
                    'labels', start+num, start+num+test)
    print("Done reading data from disk from {} to {}.".format(start, start+num+test))
    return X_train, Y_train, X_test, Y_test

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Convolution2D(32, 9, 9,
                        border_mode='valid',
                        input_shape=input_shape, subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(32, 7, 7))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dense(1024))
# model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid')) # SIGMOID, not softmax, because not mutually exclusive

# adadelta = Adadelta(clipnorm=1.)
adadelta = Adadelta(clipnorm=1.)

# def accuracyForEachClass(y_true, y_pred):
#     d = K.mean(K.abs(y_pred - y_true), axis=0)

#     print(K.gather(d,0), 'lala')
#     return {
#         'zero': d[0],
#         'one': d[1],
#         'two': d[2],
#         'three': d[3],
#         'four': d[4],
#         'five': d[5],
#         'six': d[6],
#         'seven': d[7],
#         'eight': d[8],
#         'nine': d[9],
#         'ten': d[10],
#         'eleven': d[11],
#         'twelve': d[12],
#         'thirteen': d[13],
#         'fourteen': d[14],
#         'fifteen':d[15]
#     }

# starting compilation
model.compile(loss='binary_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy']) #, accuracyForEachClass])

def size(model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])

print("The size of the model is {} parameters".format(size(model)))
meanFace = np.load('mean_face_normalised.npy')

x, acc, val_acc, loss, val_loss = [], [], [], [], []
first = []
part = 0
for e in range(nb_epoch):
    print("epoch %d" % e)
    # start, num, test = 0, 1024, 256
    start, num, test = 0, 1024, 512 
    while start + num + test < 202599:
        if start + num + test < 67533:
            fileNum = 1
            offset = 0
        elif start + num + test > 67533 and start + num + test < 2*67533:
            fileNum = 2
            offset = 67533
        else:
            fileNum = 3
            offset = 67533*2
        X_train, Y_train, X_test, Y_test = loadFromHdf5(start-offset, num, test, fileNum)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        # substracting the mean face
        X_train -= meanFace
        X_test -= meanFace

        augm = False
        # augm = True

        if augm == True:  
            datagen = ImageDataGenerator(
                                        featurewise_center=False,
                                        featurewise_std_normalization=False,
                                        rotation_range=10,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.3,
                                        zoom_range=0.2,
                                        channel_shift_range=0.2,
                                        horizontal_flip=True)
            datagen.fit(X_train)
            M = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=len(X_train), nb_epoch=1, verbose=1,
                       validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size), nb_val_samples=test)
        else:
            M = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, nb_epoch=1,
              verbose=1, shuffle='batch')
        start += num + test
        part += 1

        x.append(part)
        acc.append(M.history['acc'])
        val_acc.append(M.history['val_acc'])
        loss.append(M.history['loss'])
        val_loss.append(M.history['val_loss'])

        # plot
        if not part%5:
            fig = plt.figure(figsize=(18, 6))
            plt.subplot(121)
            plt.plot(x, acc)
            plt.plot(x, val_acc)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('{} iterations'.format(int(num/batch_size)))
            plt.legend(['train', 'test'], loc='lower right')

            plt.subplot(122)
            plt.plot(x, loss)
            plt.plot(x, val_loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('{} iterations'.format(int(num/batch_size)))
            plt.legend(['train', 'test'], loc='upper right')

            plt.savefig("Loss-accuracy.pdf")
            plt.close(fig)
            print("Saving figure ...")

    if augm:
        model.save('trained/pretrained_CelebA_normalised_augm.h5')
    else:
        model.save('trained/pretrained_CelebA_normalised_augm.h5')
    start = 0
    print("Skipping, ending {}".format(part))

model.save('trained/pretrained_CelebA_normalised{}.h5'.format(time.strftime("%m%d-%H")))
fig = plt.figure(figsize=(18, 6))
plt.subplot(121)
plt.plot(x, acc)
plt.plot(x, val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('{} iterations'.format(int(num/batch_size)))
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(122)
plt.plot(x, loss)
plt.plot(x, val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('{} iterations'.format(int(num/batch_size)))
plt.legend(['train', 'test'], loc='upper right')

plt.savefig("trained/Loss-accuracy-normalised{}.pdf".format(time.strftime("%m%d-%H")))
plt.close(fig)
print("Saving figure ...")
# 202598