"""
by any chance you fall here?

we resize the frame for speed and recover grayscale for face detection from dlib.
we define the eyes; by the points of dlib we recover

the convex points and transform them into points numpy in matrix for only
have the periphery of the eyes in a rectangle (inside eyes = white, exterior = white)

then we collect the points of the eyes to recover the rectangle
that surrounds the eyes we do a color equalization


we superpose the periphery and the rectangle,
we have an egalized picture Firstly and a focused eye secondly.
We apply all white pixels from the focused eye on the egalized picture.
We now only have pupil.

Now we search contours (only < 80% of the focused picture) and
take their centers !

force with you

"""


import cv2
import numpy as np

#===================================================== Recuperate eyes
def recuperate_eyes(landmarks):

    eyes = (cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(36, 42)])),
            cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(42, 48)])))
    return eyes


#===================================================== Get eye
def rectangle_eye_area(img, eye, gray):
    """Recuperate contour of eyes in a box, make an egalizer,
    make a color and gray mask."""

    nb = 5
    x, y, w, h = cv2.boundingRect(eye)
    cropMask = gray[y-nb : (y+h)+nb, x-nb : (x+w)+nb]
    cropMask = cv2.equalizeHist(cropMask)

    cropImg = img[y-nb : (y+h)+nb, x-nb : (x+w)+nb]

    return cropMask, cropImg


def eye_contour_masking(img, eye, gray):
    """Recuperate contour of eyes points, delimitate that
    recuperate color and gray mask."""

    nb = 5
    height, width = gray.shape[:2]

    black_frame = np.zeros((height, width), np.uint8)
    mask = np.full((height, width), 255, np.uint8)
    cv2.fillPoly(mask, [eye], (0, 0, 255))
    mask = cv2.bitwise_not(black_frame, gray.copy(), mask=mask)

    x, y, w, h = cv2.boundingRect(eye)

    cropMask = mask[y-nb : (y+h)+nb, x-nb : (x+w)+nb]
    cropImg = img[y-nb : (y+h)+nb, x-nb : (x+w)+nb]

    return cropMask, cropImg


def superpose_contour_eye_rectangle(mask_eyes_gray, crop):

    for i in range(mask_eyes_gray.shape[0]):
        for j in range(mask_eyes_gray.shape[1]):
            if mask_eyes_gray[i, j] > 200:
                crop[i, j] = 255

    return crop



#===================================================== Pupille center part

def find_center_pupille(crop, mask_eyes_img):
    """Gaussian filter, search the max solo contour on thresh,
    make an erod on 3 neighboors, find center of the contours."""

    out = None, None, None

    gaussian = cv2.GaussianBlur(crop, (9, 9), 0)

    for thresh in range(0, 200, 5):
        _, threshold = cv2.threshold(gaussian, thresh, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            break

    _, threshold = cv2.threshold(gaussian, thresh - 10, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(threshold, kernel, iterations=1) 

    contours = cv2.findContours(img_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours) > 0:

        a = cv2.moments(contours[0])['m00']
        pupille_center = [(int(cv2.moments(contours[0])['m10']/a),
                          int(cv2.moments(contours[0])['m01']/a)) for cnt in contours if a > 0]

        if pupille_center != []:
            x_center, y_center = pupille_center[0][0], pupille_center[0][1]
            #cv2.circle(mask_eyes_img, (x_center, y_center), 4, (0, 0, 255), 1)

            out = x_center, y_center, mask_eyes_img


    return out


#===================================================== Main

def find_pupil_center(eye, frame, gray):
    """Recuperate egalized rectangle area or box area,
       recuperate contour eyes,
       Superpose egalized rectangle with contour eyes,
       find centers"""

    #Box egalized eyes areas
    gray_crop, color_crop = rectangle_eye_area(frame, eye, gray)

    #Contours of the broder of the eyes
    mask_eyes_gray, mask_eyes_img = eye_contour_masking(frame, eye, gray)

    #Superpose box and contours
    gray_crop = superpose_contour_eye_rectangle(mask_eyes_gray, gray_crop)

    #Define centers of pupils
    x_center, y_center, crop_eyes = find_center_pupille(gray_crop, mask_eyes_img)

    return (x_center, y_center), crop_eyes



def pupille_tracker(landmarks, frame, gray):

    eyes = recuperate_eyes(landmarks)

    right_eye = eyes[0]
    left_eye = eyes[1]
 
    right_eye, crop_eyes_right = find_pupil_center(right_eye, frame, gray)
    left_eye, crop_eyes_left  = find_pupil_center(left_eye, frame, gray)

    return (right_eye, crop_eyes_right), (left_eye, crop_eyes_left)
