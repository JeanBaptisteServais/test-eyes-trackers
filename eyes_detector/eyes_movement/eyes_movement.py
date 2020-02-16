""" Here we take care about:
    the height and width of the box Who surround eyes (rectangle),

    we take care about center of pupil (from pupil tracker)
    and his position in percent by contribution of the frame.

    For have a detection we need the 2 eyes in the same time
    in the same position !

    Care blink and botttom movements must be differenciate.
"""
    



import cv2
import numpy as np




def recuperate_eyes(landmarks):
    """Recuperate eyes from dlib"""

    eyes = (cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                for n in range(36, 42)])),
            cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                for n in range(42, 48)])))

    return eyes


def eyes_box(eyes):
    """Place eyes convexhull from dlib in a box"""
    _, _, w, h = cv2.boundingRect(eyes)
    return w, h + 10


def top_movement(CONTENEUR, y_center, h, movements_dico, move):
    """If the last 2 movements is < of the actual height,
    and the center y of the pupil is > of 40% of the height of the box
    pupil is top
    """
    if len(CONTENEUR) >= 2 and h >= CONTENEUR[-2] + 2 and y_center >= int(0.40*h):
        movements_dico[move] = "haut"
    else:
        CONTENEUR.append(h)


def bot_movement(CONTENEUR, h, y_center, w, x_center, movements_dico, move1, move2):
    """If the current height of the box is < of the mean
    of all height - 1.5,
    and y center of pupil is < of 45% of the height box
    pupil is bottom.
    If pupil is bottom we drop x position because eyes movements bot
    are decreases
    """
    if h <= np.mean(CONTENEUR) - 1.5 and y_center <= int(0.45*h):
        movements_dico[move1] = "bas"

        if x_center >= int(0.80 * w):
           movements_dico[move2] = "bas gauche"
        elif x_center <= int(0.65 * w):
           movements_dico[move2] = "bas droite"



def x_movements(x_center, w, movements_dico, move):
    """x center pupil is > 80% of the width of the box = right
    x center pupil is < 72% of the width of the box = left"""

    if x_center >= int(0.85 * w):
        movements_dico[move] = "gauche"

    elif x_center <= int(0.72 * w):
        movements_dico[move] = "droite"



def main_movements(eyes, eye, CONTENEUR, movements_dico,
                   move1, move2, move3):

    #Eyes box
    w, h = eyes_box(eyes)

    #Center of pupil from pupil_tracker.py
    x_center, y_center = eye

    #Top movements
    top_movement(CONTENEUR, y_center, h, movements_dico, move1)

    #Bot movement
    bot_movement(CONTENEUR, h, y_center, w, x_center, movements_dico,
                 move2, move3)
    #X movements
    x_movements(x_center, w, movements_dico, move3)




def movements(movement, movements_dico):

    if movements_dico["right"] == "gauche" and movements_dico["left"] == "gauche":
        movement += "gauche "

    elif movements_dico["right"] == "droite" and movements_dico["left"] == "droite":
        movement += "droite "

    if movements_dico["right_top"] == "haut" and movements_dico["left_top"] == "haut":
        movement += "haut "

    elif movements_dico["right_bot"] == "bas" and movements_dico["left_bot"] == "bas":
        movement += "bas "

    if movements_dico["right"] == "bas gauche" and movements_dico["left"] == "bas gauche":
        movement += "bas gauche "

    elif movements_dico["right"] == "bas droite" and movements_dico["left"] == "bas droite":
        movement += "bas droite "

    return movement


LAST_H_RIGHT = []
LAST_H_LEFT = []

def eyes_movements(landmarks, frame, right_eye, left_eye):

    global LAST_H_RIGHT
    global LAST_H_LEFT

    movement = ""
    movements_dico = {"right": "", "left": "", "right_top": "",
                      "left_top": "", "right_bot": "", "left_bot": ""}

    eyes = recuperate_eyes(landmarks)


    if right_eye[0] is not None:
        main_movements(eyes[0], right_eye, LAST_H_RIGHT, movements_dico,
                   "right_top", "right_bot", "right")

    if left_eye[0] is not None:    
        main_movements(eyes[1], left_eye, LAST_H_LEFT, movements_dico,
                   "left_top", "left_bot", "left")


    movement = movements(movement, movements_dico)


    return movement



def eyes_contours(landmarks, frame, right_eye, left_eye):

    eyes = (cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(36, 42)])),
            cv2.convexHull(np.array([(landmarks.part(n).x, landmarks.part(n).y)
                    for n in range(42, 48)])))

    cv2.drawContours(frame, [eyes[0]], -1, (0, 0, 0), 1)
    cv2.drawContours(frame, [eyes[1]], -1, (0, 0, 0), 1)

    print(right_eye)
    print(left_eye)
    print("")


        



