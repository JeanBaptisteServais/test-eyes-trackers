import cv2
import numpy as np
from dlib import get_frontal_face_detector, shape_predictor

def load_model_dlib(path_points_head):

    print("Search detector/predictor")

    try:
        detector = get_frontal_face_detector()
        predictor = shape_predictor(path_points_head)
        print("detector, predictor success")

    except:
        print("detector or predictor from dlib not find")


    return predictor, detector



def get_face_in_box(landmarks):
    """Head box detection"""

    points = [(landmarks.part(n).x, landmarks.part(n).y)
              for n in range(0, 68)]

    convexhull = cv2.convexHull(np.array(points))
    head_box = cv2.boundingRect(convexhull)

    return head_box


def recuperate_landmarks(gray_frame, predictor, detector):
    """Recuperate landmarks from dlib"""

    faces = detector(gray_frame)
    out = None, None

    if len(faces) > 0:
        landmarks = predictor(gray_frame, faces[0])
        out = faces, landmarks

    return out


def head_points(gray_frame, predictor, detector):

    out = None, None
    faces, landmarks = recuperate_landmarks(gray_frame, predictor, detector)
    if landmarks is not None:
        head_box = get_face_in_box(landmarks)
        out = landmarks, head_box

    return out








def eyes_points_for_head_analysis(landmarks):

    right_eye = landmarks.part(36).x, landmarks.part(36).y
    left_eye = landmarks.part(45).x, landmarks.part(45).y
    nose = landmarks.part(30).x, landmarks.part(30).y

    return right_eye, left_eye, nose
