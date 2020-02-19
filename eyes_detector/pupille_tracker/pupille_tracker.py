"""


"""


import cv2
import numpy as np
from scipy.spatial import distance as dist

from .pupil_movements.pupil_movements import face_movement
from .pupil_movements.pupil_movements import eyes_movements


#===================================================== Recuperate eyes informations.

def recuperate_eyes(landmarks, frame):
    """Recuperate DLIB eyes points"""

    #1) - convexhull = convex points of the contours.
    #2) - to numpy array
    eyes = (cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(36, 42)])),
            cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(42, 48)])))

    return eyes


def recuperate_extremums(eye_contours, frame):
    """Recuperate extremum for eyes movement and glob occular rayon"""

    #Extremums contours.
    x = tuple(eye_contours[eye_contours[:, :, 0].argmin()][0])  #left
    y = tuple(eye_contours[eye_contours[:, :, 1].argmin()][0])  #right
    w = tuple(eye_contours[eye_contours[:, :, 0].argmax()][0])  #top
    h = tuple(eye_contours[eye_contours[:, :, 1].argmax()][0])  #bottom
    
    #[cv2.circle(frame, (i), 1, (255, 0, 0), 1) for i in [x, w]]

    #Mean of the height of the eye.
    occular_glob = abs(int((y[1] - h[1]) / 2))

    return occular_glob, (x, y, w, h)


#===================================================== Get/Treat eye into a box from the frame.
def rectangle_eye_area(frame, eye, gray):
    """Recuperate contour of eyes in a box, make an egalizer,
    make a color and gray mask."""

    nb = 5
    #From the eyes contours, build a box.
    x, y, w, h = cv2.boundingRect(eye)

    #Region interest of the box from gray frame.
    cropGray = gray[y-nb : (y + h) + nb, x - nb : (x + w) + nb]
    #cv2.imshow("cropGray", cropGray)

    #Egalize the region on gray frame.
    cropEgalize = cv2.equalizeHist(cropGray)
    #cv2.imshow("cropMask", cropMask)

    #Recuperate Region on the frame.
    cropImg = frame[y-nb : (y+h)+nb, x-nb : (x+w)+nb]
    #cv2.imshow("cropImg", cropImg)

    return cropEgalize, cropImg


def eye_contour_masking(img, eye, gray):
    """Recuperate contour of eyes points, delimitate that
    recuperate color and gray mask."""

    nb = 5
    height, width = gray.shape[:2]

    black_frame = np.zeros((height, width), np.uint8)   #empty picture
    mask = np.full((height, width), 255, np.uint8)      #Mask
    cv2.fillPoly(mask, [eye], (0, 0, 255))              #

    #recuperate region interest.
    mask = cv2.bitwise_not(black_frame, gray.copy(), mask=mask)
    #cv2.imshow("mask", mask)

    (x, y, w, h) = cv2.boundingRect(eye)

    cropMask   = mask[y - nb  : (y+h) + nb, x - nb  : (x+w) + nb]   #Gray mask
    cropImg    = img[ y  - nb : (y+h) + nb, x - nb  : (x+w) + nb]   #Image mask
    crop_appli = img[ y  - 10 : (y+h) + 10, x - nb  : (x+w) + nb]   #for an extern appli

    return cropMask, cropImg, crop_appli


def superpose_contour_eye_rectangle(mask_eyes_gray, crop):
    """ - mask_eyes_gray's the interior of the eyes.
        - crop's the crop of the frame. (crop treat by an egalizer.)

        Superpose the mask (white part) on the crop for recuperate.
        the iris."""

    #cv2.imshow("mask_eyes_gray", mask_eyes_gray)
    #cv2.imshow("crop", crop)

    for i in range(mask_eyes_gray.shape[0]):
        for j in range(mask_eyes_gray.shape[1]):
            if mask_eyes_gray[i, j] > 200:
                crop[i, j] = 255

    #cv2.imshow("treat_crop", crop)
 
    return crop



#===================================================== Pupille/Threshold center part.
"""
1) - find threshold.
2) - treat threshold.
3) - find the center of threhsold.
4) - Mark center.
"""

def adjust_gamma(image, gamma):
    """We change gamma (light or shadow)"""

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def find_a_threshold(gaussian):

    #Run 0 - 200 value by 5 to 5.
    for thresh in range(0, 200, 5):

        #Test threshold and contours associates. 
        _, threshold = cv2.threshold(gaussian, thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #More than 1 contours stop.
        if len(contours) > 1:   
            break

    #Recuperate the last value - 50.
    _, threshold = cv2.threshold(gaussian, thresh - 50, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("threshold", threshold)

    return threshold, contours, thresh - 50


def threshold_treatment(threshold, gaussian, thresh):

    #Count black pixels.
    blackPx = cv2.countNonZero(threshold)
    kernel = np.ones((1,1), np.uint8)

    if blackPx > 10 or blackPx > 0: #No treatment.
        img_erosion = threshold

    #No eye thresold found, dilate it.
    elif blackPx == 0:

        _, threshold = cv2.threshold(gaussian, thresh + 10, 255,
                                     cv2.THRESH_BINARY_INV)
        img_erosion = cv2.dilate(threshold, kernel, iterations=1)

    return img_erosion

def center_threshold(mask_eyes_img, contours):

    #Variable out regulation.
    out = None, None, None

    #Find center of the threshold.
    #cv2.drawContours(mask_eyes_img, [contours[0]], -1, (0, 255, 0), 1)
    a = cv2.moments(contours[0])['m00']

    pupille_center = [(int(cv2.moments(contours[0])['m10']/a),
                      int(cv2.moments(contours[0])['m01']/a))
                      for cnt in contours if a > 0]

    #Value pupil regulation.
    if pupille_center != []:

        #Put red pixel into the center for eyes movements.
        x_center, y_center = pupille_center[0][0], pupille_center[0][1]
        mask_eyes_img[y_center, x_center] = 0, 0, 255

        #cv2.circle(mask_eyes_img, (x_center, y_center), rayon, (255, 255, 255), 1)
        out = x_center, y_center, mask_eyes_img

    return out


def find_center_pupille(crop, mask_eyes_img, rayon):
    """Gaussian filter, search the max solo contour on thresh,
    make an erod on 3 neighboors, find center of the contours."""

    #Out variable regulation.
    out = None, None, None

    #Ajust gamma of the crop.
    crop = adjust_gamma(crop, 3) #We add light shadow 0 - (0 + n) light where n is int.

    #Eliminate noise with gaussian blur.
    gaussian = cv2.GaussianBlur(crop, (9, 9), 50)

    #Find the maximum threshold value from one contour.
    threshold, contours, thresh = find_a_threshold(gaussian)

    #Treat threshold in the case nothing found.
    img_erosion = threshold_treatment(threshold, gaussian, thresh)

    #Contours of the trehsold.
    contours = cv2.findContours(img_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    #Contours Regulation.

    
    if len(contours) > 0:
        out = center_threshold(mask_eyes_img, contours)

    return out





#===================================================== Treatement for one eye.

def find_pupil_center(eye, frame, gray, rayon):
    """Recuperate egalized rectangle area or box area,
       recuperate contour eyes,
       Superpose egalized rectangle with contour eyes,
       find centers"""

    #Box egalized eyes areas
    gray_crop, color_crop = rectangle_eye_area(frame, eye, gray)

    #Contours of the broder of the eyes
    mask_eyes_gray, mask_eyes_img, crop_appli = eye_contour_masking(frame, eye, gray)

    #Superpose box and contours
    gray_crop = superpose_contour_eye_rectangle(mask_eyes_gray, gray_crop)

    #Define centers of pupils
    x_center, y_center, crop_eyes = find_center_pupille(gray_crop, mask_eyes_img, rayon)

    return ((x_center, y_center), crop_eyes, crop_appli)



def pupille_tracker(landmarks, frame, gray, head_box, blanck):

    #Recuperate DLIB points.
    eyes = recuperate_eyes(landmarks, frame)

    #Recuperate eye.
    right_eye, left_eye = eyes

    #Recuperate extremums points of eyes, and glob size.
    glob_right, extremum_right = recuperate_extremums(right_eye, frame)
    glob_left, extremum_left   = recuperate_extremums(left_eye, frame)

    #Right pupil.
    right_pupil = find_pupil_center(right_eye, frame, gray, glob_right)
    right_eye, crop_eyes_right, crop_appli_right = right_pupil

    #Left pupil
    left_pupil = find_pupil_center(left_eye, frame, gray, glob_left)
    left_eye, crop_eyes_left, crop_appli_left = left_pupil



    turning, down = face_movement(landmarks, frame, eyes, head_box)

    right_information = (frame, extremum_right, landmarks, head_box, eyes, glob_right, blanck, "right")
    eyes_movements(right_information)

    left_information = (frame, extremum_left, landmarks, head_box, eyes, glob_left, blanck, "left")
    eyes_movements(left_information)

    print("")



    return (right_eye, crop_eyes_right, crop_appli_right),\
           (left_eye, crop_eyes_left, crop_appli_left)


