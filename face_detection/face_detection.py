import cv2
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
from scipy import misc
from scipy.misc import imresize
from skimage.transform import resize, rotate
import glob
import h5py
import math
from datetime import datetime
import face

IMAGE_SIZE = (128, 128)
IOD = 40.0

def imgCrop(image, cropBox, boxScale=1):
    off = 90    
    y = max(cropBox[1] - 3*off, 0)
    x = max(cropBox[0] - 2*off, 0)

    off = 50   
    y = max(cropBox[1] - 3*off, y)
    x = max(cropBox[0] - 2*off, x)

    off = 20
    y = max(cropBox[1] - 3*off, y)
    x = max(cropBox[0] - 2*off, x)


    cropped = image[y:cropBox[1]+cropBox[3]+90, x:cropBox[0]+cropBox[2]+30]
    dims = cropped.shape
    # print cropped.shape, cropBox
    # if dims[0] != IMAGE_SIZE[1] or dims[1] != IMAGE_SIZE[0]:
    #     return imresize(cropped, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # else:
    #     return cropped
    #misc.imsave("imgCrop.png", cropped)
    return cropped, x, y

def rotateBound(image, angle, center):
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

def normaliseImage(image, eyes, xcrop, ycrop):
    # -- normalize faces using i.o.d
    left_eye = eyes[0] + np.array([xcrop, ycrop, 0, 0])
    right_eye = eyes[1] + np.array([xcrop, ycrop, 0, 0])
    # print(left_eye, right_eye)
    # print(image.shape)
    # left_eye = np.array([celebA['lefteye_y'].iloc[i], celebA['lefteye_x'].iloc[i]])
    # right_eye = np.array([celebA['righteye_y'].iloc[i], celebA['righteye_x'].iloc[i]])
    scale = IOD / np.linalg.norm(left_eye - right_eye)
    left_eye = scale * left_eye
    right_eye = scale * right_eye
    im = resize(image, (int(scale*image.shape[0]), int(scale*image.shape[1])), mode='edge')
    # print(im.shape, scale)
    # print(left_eye, right_eye)

    diff = np.subtract(left_eye, right_eye)
    angle = math.atan2(diff[0], diff[1])
    #misc.imsave("previewBEFORErot.png", im)
    im = rotate(im, -angle,center=(left_eye[0],left_eye[1]), preserve_range=True, mode='edge')

    iod = np.linalg.norm(left_eye - right_eye)
    # print(im.shape, left_eye, right_eye)
    #misc.imsave("preview.png", im)
    xmin = int(left_eye[0]-1.6*iod)
    xmax = int(left_eye[0]+2*iod)
    ymin = int(left_eye[1]-1.3*iod)
    ymax = int(right_eye[1]+1.3*iod)
    xmin = max(0, xmin)
    xmax = min(im.shape[0], xmax)
    ymin = max(0, ymin)
    ymax = min(im.shape[1], ymax)
    im = im[xmin:xmax, ymin:ymax, :]

    #im = im[ int(left_eye[0]-1.6*iod):int(left_eye[0]+2*iod), int(left_eye[1]-1.3*iod):int(right_eye[1]+1.3*iod),:]
    # print(im.shape, scale, iod)
    # print(int(left_eye[1]-1.3*iod),int(right_eye[1]+1.3*iod))
    im = resize(im, IMAGE_SIZE, mode='edge')
    try:
        return im
        # return resize(im, IMAGE_SIZE, mode='edge')
    except:
        # plt.imshow(image)
        # plt.show()
        print(iod, im.shape)
        return None


def detectFace(image):
    # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # for each detected face, detect eyes and smile
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    unaltered_image = image.copy()
    eyes = None
    normalised_image = None
    for face in faces:
        (x,y,w,h) = face
        # show face bounding box on Webcam Preview
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # normalise image in order to predict on it
        # croppedImage = imgCrop(image, face, boxScale=1)
        # detect eyes for Inter Oculat Distance
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if len(eyes) == 2 and np.abs(eyes[0,1] - eyes[1,1]) < 10:
                offset1 = np.sqrt((eyes[0,2]**2+eyes[0,3]**2))*0.5
                offset2 = np.sqrt((eyes[1,2]**2+eyes[1,3]**2))*0.5
                real_eyes = eyes + np.array([[x+offset1,y+offset1,0,0],[x+offset2,y+offset2,0,0]])
                #real_eyes = eyes + np.array([[offset1,offset1,0,0],[offset2,offset2,0,0]])
                real_eyes = np.sort(real_eyes, axis = 0)
                cropped_image, xcrop, ycrop = imgCrop(unaltered_image, face)
                normalised_image = normaliseImage(cropped_image, real_eyes, -xcrop, -ycrop)
       

        
        # detect eyes
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # # detect smile
        # smile = smile_cascade.detectMultiScale(roi_gray)
        # for (sx,sy,sw,sh) in smile:
        #     cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

    return normalised_image, image

def mapAttributes(classes):
    with open('wanted_attributes_normalised.txt', 'r') as f:
        attributes = f.read()
    attributes = attributes.strip('\n').split(' ')

    result = []
    for i, cl in enumerate(classes):
        if cl == True:
            result.append(attributes[i])
    return result

if __name__ == "__main__":
    prediction = {0:'smile', 1:'talk', 2:'clap', 3:'wave'}
    # with h5py.File('trained/trained_webcam.h5',  "a") as f:
    #     try:
    #         del f['/optimizer_weights']
    #     except KeyError:
    #         print('Already deleted optimizer_weights due to incompatibility between keras versions. Nothing to be done here.')
    model = load_model('trained/pretrained_CelebA_normalised0203-05.h5')

    cv2.namedWindow("Webcam Preview")
    vc = cv2.VideoCapture(0) # 0 for built-in webcam, 1 for robot

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # cv2.imwrite('letiN/{}.jpg'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')), frame)
        normalised_image, frame = detectFace(frame)      

        # TODO cv2 format to normal jpeg like

        # NN stuff
        # for images from my webcam which have double size
        # X_test = misc.imresize(frame, (int(0.5*frame.shape[0]),int(0.5*frame.shape[1])))
        if normalised_image is not None:
            #normalised_image = cv2.cvtColor(normalised_image, cv2.COLOR_BGR2RGB)
            normalised_image = normalised_image[:,:,::-1]
            # subtract mean face
            meanFace = np.load('mean_face_normalised.npy')

            #misc.imsave("previewRGB.png", normalised_image)
            X_test = np.expand_dims(normalised_image, axis=0)
            X_test -= meanFace
            classes = model.predict_classes(X_test, batch_size=32, verbose=0)
            proba = model.predict_proba(X_test, batch_size=32, verbose=0)
            predicted_attributes = mapAttributes((proba > 0.2)[0])
            print( proba)
            print(predicted_attributes)
        # end NN stuff

        # postprocessing and reaction step

        cv2.imshow("Webcam Preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("Webcam Preview")

