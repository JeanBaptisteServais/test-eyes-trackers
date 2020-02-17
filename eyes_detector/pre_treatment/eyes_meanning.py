import time
import cv2
import os

import numpy as np

#Load dlib model
from recuperate_points.face_points import load_model_dlib

from recuperate_points.face_points import recuperate_landmarks
from recuperate_points.face_points import head_points
from recuperate_points.face_points import get_face_in_box
from recuperate_points.face_points import eyes_points_for_head_analysis

#Pupil part
from pupille_tracker.pupille_tracker import pupille_tracker

from paths import media_path, dlib_model, path_data, path_data_video, path_csv_file


NUMBER_FRAME = 500
EYE_MEAN = {}
HEAD_POSITION = {}

def number_of_file(path_data):

    data = os.listdir(path_data)
    data = sorted(data)
    number_file = len(data)

    print("Count file to treat : ", number_file)

    return data, number_file




def recuperate_eyes_mean_position(number_divide, video, path_csv_file):

    #NE PAS OUBLIER LA GROSSEUR DE LA TETE SI FOND CHANGE
    #OU DETECTION DE NOIR
    


    cap = cv2.VideoCapture(video)
    predictor, detector = load_model_dlib(dlib_model)

    No = 0
    nb_frame = 0
    global NUMBER_FRAME

    mean_rightX = 0
    mean_rightY = 0

    mean_leftX = 0
    mean_leftY = 0

    head_position = []

    while True:

        start_time_frame = time.time()

        _, frame = cap.read()

        #Video dimension traeatment.
        height, width = frame.shape[:2]
        width_resize =  int(width  / number_divide)
        height_resize = int(height / number_divide)
        frame = cv2.resize(frame, (width_resize, height_resize))

        #Gray threshold.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Recuperate landmarks and head box.
        landmarks, head_box = head_points(gray, predictor, detector) #face_points


        if landmarks is not None:

            #Recuperate pupil center, eyes constitution = (x, y), crop
            right_eye, left_eye = pupille_tracker(landmarks, frame, gray)

            print(right_eye[0], left_eye[0])

            mean_rightX += right_eye[0][0]
            mean_rightY += right_eye[0][1]

            mean_leftX  += left_eye[0][0]
            mean_leftY  += left_eye[0][1]

            head_position.append(head_box[2] * head_box[3])


        if nb_frame == NUMBER_FRAME:

            EYE_MEAN[No] = (mean_rightX/nb_frame, mean_rightY/nb_frame,
                            mean_leftX/nb/frame, mean_leftY/nb_frame)

            HEAD_POSITION[No] = 




            No += 1
            nb_frame = -1



        nb_frame += 1



        cv2.imshow("Frame", frame)

        #print(time.time() - start_time_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()




def meanning(number_divide):

    data, number_file = number_of_file(path_data)

    number_divide = 2.9999999999999973

    for video in data:
        video = path_data_video.format(video)
        recuperate_eyes_mean_position(number_divide, video, path_csv_file)









