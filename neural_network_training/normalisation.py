import os, sys
import numpy as np
import pandas as pd
import cv2
import math
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.transform import rotate

from scipy.misc import imresize, imrotate

IMAGE_SIZE = (128, 128)
IOD = 40.0

# -- read attribute and landmark files, and create a single data frame
# if not os.path.isfile('data/celebA.csv'):
#     data_dir = 'data/img_align_celeba'
#     attribute_fpath = 'data\list_attr_celeba.csv'
#     landmarks_fpath = 'data\list_landmarks_align_celeba.csv'
#     attributes = pd.read_csv(attribute_fpath)
#     landmarks = pd.read_csv(landmarks_fpath)
#     landmarks.drop('image_name', axis=1, inplace=True)
#     celebA = pd.concat([attributes, landmarks], axis=1)
#     celebA.to_csv('data\celebA.csv')
# else:
celebA = pd.read_csv('data/celebA.csv')


# -- All 40 attributes:
# 5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,
# Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,
# Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,
# Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,
# Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,
# Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,
# Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,
# Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young



# data = file.create_dataset('images',(celebA.shape[0], 128, 128, 3),
#                               maxshape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
#                               chunks=True # , dtype= ???
#                               )
# labels = file.create_dataset('labels',(celebA.shape[0], 40),
#                               maxshape=(None, 40),
#                               chunks=True)

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

def rotate_bound(image, angle, center):
    (cX, cY) = center
    (h, w) = image.shape[:2]
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


X = np.load('/export/home/lparcala/dataFace/Xsaved.npy')
out = np.zeros((X.shape[0]/3, 128, 128, 3))
for i in range(X.shape[0]/3):
    if i%1000==0:
        print("Done 1000 images")

    # # -- read images
    # fname = str(celebA['image_name'].iloc[i])
    # fpath = os.path.join('data','img_align_celeba', fname)
    # im = np.asarray(Image.open(fpath))

    # -- normalize faces using i.o.d
    left_eye = np.array([celebA['lefteye_y'].iloc[i], celebA['lefteye_x'].iloc[i]])
    right_eye = np.array([celebA['righteye_y'].iloc[i], celebA['righteye_x'].iloc[i]])
    scale = IOD / np.linalg.norm(left_eye - right_eye)
    left_eye = scale * left_eye
    right_eye = scale * right_eye
    im = resize(X[i], (int(scale*X[i].shape[0]), int(scale*X[i].shape[1])), mode='edge')
    # im = cv2.resize(X[i], (int(scale*X[i].shape[0]), int(scale*X[i].shape[1])), interpolation = cv2.INTER_CUBIC)

    diff = np.subtract(left_eye, right_eye)
    angle = math.atan2(diff[0], diff[1])
    im = rotate(im, -angle,center=(left_eye[0],left_eye[1]), preserve_range=True, mode='edge')
    # im = rotate_bound(im, -angle, (left_eye[0],left_eye[1]))

    iod = np.linalg.norm(left_eye - right_eye)
    im = im[ int(left_eye[0]-1.6*iod):int(left_eye[0]+2*iod), int(left_eye[1]-1.3*iod):int(right_eye[1]+1.3*iod),:]
    im = np.ascontiguousarray(im)
    # out[i-X.shape[0]/3] = resize(im, IMAGE_SIZE, mode='edge')
    try:
        out[i] = resize(im, IMAGE_SIZE, mode='edge')
        # out[i-X.shape[0]/3] = cv2.resize(im, IMAGE_SIZE, interpolation = cv2.INTER_CUBIC)
    except:
        # plt.imshow(X[i])
        # plt.show()
        print iod, im.shape
        print im, "image number", i, i
        out[i] = resize(X[i], IMAGE_SIZE, mode='edge')


# np.save("/export/home/lparcala/dataFace/out3.npy", out)
# print(out.shape)

print("writing out the hdf5 file", out.shape)
file = h5py.File('/export/home/lparcala/dataFace/celebA1.h5','w')
file.create_dataset('images', data=out)
# y = readLabels('wanted_attributes.txt', 'list_attr_celeba.txt')
y = readLabels('wanted_attributes_normalised.txt', 'list_attr_celeba.txt')
# convert labels to boolean
y[y<0] = 0
y[y>0] = 1
file.create_dataset('labels', data=y[0:X.shape[0]/3])

file.close()
