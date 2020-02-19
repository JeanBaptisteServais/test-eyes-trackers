import time

start_time_import = time.time()

import cv2
import numpy as np

from recuperate_points.face_points import recuperate_landmarks
from recuperate_points.face_points import head_points
from recuperate_points.face_points import get_face_in_box
from recuperate_points.face_points import eyes_points_for_head_analysis


#Pupil part
from pupille_tracker.pupille_tracker import pupille_tracker

#Path to model, to media folder.
from paths import media_path, dlib_model
from paths import path_eyes_detector_stuff

#Blink part
from blinking.blinking_eyes import blinking_eyes
from blinking.blinking_eyes import blink_analysis
from blinking.blinking_eyes import final

#Load dlib model
from recuperate_points.face_points import load_model_dlib


print("Import time : ", time.time() - start_time_import)



def video_parameters(video_name, dlib_model):

    #Recuperate name of the video.
    video  = media_path.format(video_name)

    #Load DLIB.
    predictor, detector = load_model_dlib(dlib_model)


    cap = cv2.VideoCapture(video)

    #Recuperate video informations.
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_sec = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("coucou.avi", cv2.VideoWriter_fourcc('M','J','P','G'), int(frame_sec),
                          (frame_width, frame_height))

    start_time = time.time()


    return predictor, detector, cap, out, start_time

def displaying(alarm, frame):
    if alarm[0] is not "":
        cv2.putText(frame, alarm[0], (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    if alarm[1] is not "":
        cv2.putText(frame, alarm[1], (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,100,255), 2)
    if alarm[2] is not "":
        cv2.putText(frame, alarm[2], (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        start_time = time.time()


global_alarm = []
def video_capture_to_face(video_name):

    predictor, detector, cap, out, start_time = video_parameters(video_name, dlib_model)

    nb_frame = 0
    continuer = True
    while continuer:

        ret, frame = cap.read()

        if ret:

            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (int(width/2), int(height/2)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Recuperate landmarks and head box.
            landmarks, head_box = head_points(gray, predictor, detector) #face_points


            if landmarks is not None:

                #Recuperate blink algorythme
                blinking_frame, result = blinking_eyes(landmarks, head_box) #blinking_eyes

                timmer = time.time() - start_time
                alarm = blink_analysis(result, nb_frame, blinking_frame, timmer) #blinking_eyes

                displaying(alarm, frame)

                out.write(frame)
                nb_frame += 1

        else:
            continuer = False


    report = final()
    print(report)
    
    return report


video_capture_to_face("b.mp4")
