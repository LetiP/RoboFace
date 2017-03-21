import cv2
import numpy as np
from keras.models import load_model
# from scipy import misc
from scipy.misc import imresize
from skimage.transform import resize, rotate
import h5py
import math
import face
import gTTS
from pygame import mixer, time

IMAGE_SIZE = (128, 128)
IOD = 40.0

def imgCrop(image, cropBox, boxScale=1):
    '''
    Crop an area around the detected face (by OpenCV) in order to feed it into the prediction algorithm (NN).
    '''
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

    return cropped, x, y

def rotateBound(image, angle, center):
    '''
    Rotates image. Used for image normalisation, so that the inter-ocular line is always horizontal for the NN.
    '''
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
    '''
    Normalize faces usinginter-ocular distance i.o.d
    '''
    # resite, such that i.o.d is always same
    left_eye = eyes[0] + np.array([xcrop, ycrop, 0, 0])
    right_eye = eyes[1] + np.array([xcrop, ycrop, 0, 0])
    scale = IOD / np.linalg.norm(left_eye - right_eye)
    left_eye = scale * left_eye
    right_eye = scale * right_eye
    im = resize(image, (int(scale*image.shape[0]), int(scale*image.shape[1])), mode='edge')

    # rotate to keep inter ocular line horizontal
    diff = np.subtract(left_eye, right_eye)
    angle = math.atan2(diff[0], diff[1])
    im = rotate(im, -angle,center=(left_eye[0],left_eye[1]), preserve_range=True, mode='edge')

    # new resizing for making the image compatible with the trained NN.
    iod = np.linalg.norm(left_eye - right_eye)
    xmin = int(left_eye[0]-1.6*iod)
    xmax = int(left_eye[0]+2*iod)
    ymin = int(left_eye[1]-1.3*iod)
    ymax = int(right_eye[1]+1.3*iod)
    xmin = max(0, xmin)
    xmax = min(im.shape[0], xmax)
    ymin = max(0, ymin)
    ymax = min(im.shape[1], ymax)
    im = im[xmin:xmax, ymin:ymax, :]
    im = resize(im, IMAGE_SIZE, mode='edge')
    
    return im

def detectFace(image):
    # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

    face_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_smile.xml')

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
        if len(eyes) == 2:
            left_eye = eyes[0][0:2]
            right_eye = eyes[1][0:2]
            x, y = np.mean(left_eye[0], right_eye[0]), np.mean(left_eye[1], right_eye[1])
            face.moveHead(x, y)
            # suggestion: skip this frame as prediction, so return None, image
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if len(eyes) == 2 and np.abs(eyes[0,1] - eyes[1,1]) < 10:
                offset1 = np.sqrt((eyes[0,2]**2+eyes[0,3]**2))*0.5
                offset2 = np.sqrt((eyes[1,2]**2+eyes[1,3]**2))*0.5
                real_eyes = eyes + np.array([[x+offset1,y+offset1,0,0],[x+offset2,y+offset2,0,0]])
                real_eyes = np.sort(real_eyes, axis = 0)
                cropped_image, xcrop, ycrop = imgCrop(unaltered_image, face)
                normalised_image = normaliseImage(cropped_image, real_eyes, -xcrop, -ycrop)

    return normalised_image, image

def mapAttributes(classes):
    '''
    Map the output probabilities to the correpsonding names, like 'smile', etc.
    '''
    with open('../face_detection/wanted_attributes_normalised.txt', 'r') as f:
        attributes = f.read()
    attributes = attributes.strip('\n').split(' ')

    result = []
    for i, cl in enumerate(classes):
        if cl == True:
            result.append(attributes[i])
    return result

def say(text):
    tts = gTTS(text=text, lang='en')
    tts.save("say.mp3")
    mixer.init()
    mixer.music.load('say.mp3')
    mixer.music.play()
    while mixer.music.get_busy():
        time.Clock().tick(10)

if __name__ == "__main__":
    roboFace = face.Face()
    roboface.neutral()
    # with h5py.File('trained/trained_webcam.h5',  "a") as f:
    #     try:
    #         del f['/optimizer_weights']
    #     except KeyError:
    #         print('Already deleted optimizer_weights due to incompatibility between keras versions. Nothing to be done here.')
    # load the trained neural network
    model = load_model('../face_detection/trained/pretrained_CelebA_normalised0203-05.h5')

    cv2.namedWindow("Webcam Preview")
    vc = cv2.VideoCapture(0) # 0 for built-in webcam, 1 for robot

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # cv2.imwrite('letiN/{}.jpg'.format(datetime.now().strftime('%Y-%m-%d_%H%M%S')), frame)
        normalised_image, frame = detectFace(frame)      

        # if a face is detected and the normalisation was successful, predict on it
        if normalised_image is not None:
            normalised_image = normalised_image[:,:,::-1]
            # subtract mean face
            meanFace = np.load('../face_detection/mean_face_normalised.npy')

            X_test = np.expand_dims(normalised_image, axis=0)
            X_test -= meanFace
            classes = model.predict_classes(X_test, batch_size=32, verbose=0)
            proba = model.predict_proba(X_test, batch_size=32, verbose=0)
            predicted_attributes = mapAttributes((proba > 0.4)[0])
            print( proba)
            print(predicted_attributes)
        # end NN stuff

        # postprocessing and reaction step
        if 'Smiling' in predicted_attributes:
            roboFace.happy()
        if 'Male' in predicted_attributes and 'No_Beard' in predicted_attributes and len(predicted_attributes) == 2:
            roboFace.unsure()
        if 'Male' not in predicted_attributes:
            say('You are a female, am I right?')

        roboFace.sad()
        roboFace.unsure()
        roboFace.angry()

        cv2.imshow("Webcam Preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("Webcam Preview")

  # Black_Hair Blond_Hair Brown_Hair Eyeglasses Gray_Hair Male
  # Mouth_Slightly_Open No_Beard Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Lipstick  
