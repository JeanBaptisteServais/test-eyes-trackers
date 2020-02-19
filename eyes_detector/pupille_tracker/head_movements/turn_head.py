from scipy.spatial import distance as dist
def turn_head(right_eye, left_eye, nose, head_box):
    """Calculus difference beetween left right distance"""

    d1 = dist.euclidean(right_eye, nose) 
    d2 = dist.euclidean(left_eye, nose)
    coeff = d1 + d2

    look_to = int(250*(d1-d2)/coeff)

    w = head_box[2]
    out = ""

    if look_to > int(0.25 * w):
        out = "gauche"

    elif look_to > int(0.145 * w):
        out = "legerement a gauche"


    if look_to < - int(0.25 * w):
        out = "droite"

    elif look_to < - int(0.145 * w):
        out = "legerement a droite"

    return out
