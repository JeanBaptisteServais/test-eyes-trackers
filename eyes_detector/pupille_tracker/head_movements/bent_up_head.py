from numpy import min as np_min
from numpy import sin
from math import acos, pow
from scipy.spatial import distance as dist

DOWN_COUNTER = 0
UP_COUNTER = 0

def analyse_position(bent_up, head_box):

    global DOWN_COUNTER
    global UP_COUNTER
    out = ""
    w = head_box[2]
    #print(w, bent_up)

    if DOWN_COUNTER > 10 and bent_up >= 17:
        out = "position baissé"

    elif bent_up >= 17:
        out = "tete baissé"
        DOWN_COUNTER += 1

    elif UP_COUNTER > 10 and bent_up <= -4:
        out = "position levé"

    elif bent_up <= -4:
        out = "tete levé"
        UP_COUNTER += 1

    else:
        if UP_COUNTER > 10:
            out = "reprise position non levé"
        if DOWN_COUNTER > 10:
            out = "reprise position non enfouis"

        DOWN_COUNTER = 0
        UP_COUNTER = 0

    return out


def bent_up_head(right_eye, left_eye, nose, head_box):
    """Calculus distance beetween nose and eyes line"""

    global DOWN_COUNTER
    global UP_COUNTER

    d_eyes = dist.euclidean(right_eye, left_eye) 
    d1 = dist.euclidean(right_eye, nose) 
    d2 = dist.euclidean(left_eye, nose) 

    coeff = d1 + d2

    cosb = np_min( (pow(d2, 2) - pow(d1, 2) + pow(d_eyes, 2) ) / (2*d2*d_eyes) )
    bent_up = int(250*(d2*sin(acos(cosb))-coeff/3.5)/coeff)

    out = analyse_position(bent_up, head_box)

    return out
