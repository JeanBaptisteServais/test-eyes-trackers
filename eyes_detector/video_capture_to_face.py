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

#Load dlib model
from recuperate_points.face_points import load_model_dlib


print("Import time : ", time.time() - start_time_import)




def make_parameters(video_name, dlib_model):

    #Recuperate name of the video.
    video  = media_path.format(video_name)

    #Load DLIB.
    predictor, detector = load_model_dlib(dlib_model)

    #Initialise video.
    cap = cv2.VideoCapture(video)

    #Recuperate video informations.
    frame_width  = int(cap.get(3))  #width.
    frame_height = int(cap.get(4))  #height.
    frame_sec = cap.get(cv2.CAP_PROP_FPS)   #Frame per second.

    #Empty video file.
    out = cv2.VideoWriter("coucou.avi", cv2.VideoWriter_fourcc('M','J','P','G'), int(frame_sec),
                          (frame_width, frame_height))

    return predictor, detector, cap, out




def put_picture(eyes, image_appli):

    #Image resize by contribution of mask region.
    image = cv2.resize(image_appli, (eyes[2].shape[1], eyes[2].shape[0]))

    #If background is blue del it.
    #else replace it by the picture animation.
    for i in range(eyes[2].shape[0]):
        for j in range(eyes[2].shape[1]):

            if image[i, j][0] > 200 and\
               image[i, j][1] > 200 and\
               image[i, j][2] < 60:
                pass
            else:
                eyes[2][i, j] = image[i, j]


def choice_picture():
    """Here choice picture from user asking"""
    pass


def video_capture_to_face(video_name, eyes_image, blink_image):

    #Create empty video file by contribution of original video parameters.
    predictor, detector, cap, out = make_parameters(video_name, dlib_model)

    #Download picture animation.
    eyes_image = cv2.imread(eyes_image)
    blink_image = cv2.imread(blink_image)


    while True:

        start_time_frame = time.time()

        _, frame = cap.read()

        #Treshold parameters.
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Recuperate landmarks and head box.
        landmarks, head_box = head_points(gray, predictor, detector) #face_points.


        if landmarks is not None:

            #Recuperate pupil center, eyes constitution = (x, y), crop.
            right_eye, left_eye = pupille_tracker(landmarks, frame, gray)

            #Recuperate blink algorythme.
            _, result = blinking_eyes(landmarks, head_box) #blinking_eyes.


            if result == "BLINK":   #Blink animation.
                if right_eye[0] is not None:
                    put_picture(right_eye, blink_image) #replace region by animation.

                if left_eye[0] is not None:
                    put_picture(left_eye, blink_image)

            elif result != "BLINK": #Eyes animations
                    
                if right_eye[0] is not None:
                    put_picture(right_eye, eyes_image)

                if left_eye[0] is not None:
                    put_picture(left_eye, eyes_image)

        #Savegarde video.
        out.write(frame)

        #Animations.
        #cv2.imshow("vcsqc", eyes_image)
        #cv2.imshow("Frame", frame)

        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break

    #cap.release()
    #cv2.destroyAllWindows()

blink_image = r"C:\Users\jeanbaptiste\Desktop\SYNERGO_SITE\Synergo\media\eyes_detector\fermture_gros_oeil.png"
eyes_image = r"C:\Users\jeanbaptiste\Desktop\SYNERGO_SITE\Synergo\media\eyes_detector\gros_oeil.png"
video_capture_to_face("aa.mp4", eyes_image, blink_image)
