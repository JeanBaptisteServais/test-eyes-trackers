#Import time
import time
start_time_import = time.time()

#Load Python libraies.
import os
import cv2
import numpy as np

#Load dlib model.
from recuperate_points.face_points import load_model_dlib

#Recuperate points from DLIB.
from recuperate_points.face_points import recuperate_landmarks
from recuperate_points.face_points import head_points
from recuperate_points.face_points import get_face_in_box
from recuperate_points.face_points import eyes_points_for_head_analysis

#Pupil tracker part.
from pupille_tracker.pupille_tracker import pupille_tracker

#Import paths.
from paths import media_path
from paths import dlib_model
from paths import path_data
from paths import path_data_video
from paths import path_csv_file
from paths import picture_app_3
from paths import data_save

from app_eye_traking.visit_site_web import retracage
from app_eye_traking.visit_site_web import timmer_treatment




print("Importation libraries took : ", time.time() - start_time_import)





TIMMER = []
POSITION_RIGHT = []
POSITION_LEFT = []


def recuperate_eyes_position(video, height, width):

    global TIMMER
    global POSITION_RIGHT
    global POSITION_LEFT


    cap = cv2.VideoCapture(video)
    predictor, detector = load_model_dlib(dlib_model)


    #Recuperate video informations.
    frame_width  = int(cap.get(3))  #width.
    frame_height = int(cap.get(4))  #height.
    frame_sec = cap.get(cv2.CAP_PROP_FPS)   #Frame per second.

    video_name = data_save + "/" + "video_jb" + ".avi"
    #writting = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 20,
    #                      (int(frame_width), int(frame_height)))


    BLANCK = np.zeros((height,width,3), np.uint8)

    eyes_time_functionality = time.time()
    while True:


        ret, frame = cap.read()

        if ret:

            #b, a = frame.shape[:2]
            #frame = cv2.resize(frame, (int(a /   0.8500000000000002), int(b /   0.8500000000000002)))#g 0.8500000000000002

            #Gray threshold.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Recuperate landmarks and head box.
            landmarks, head_box = head_points(gray, predictor, detector) #face_points

            if landmarks is not None:

                #Recuperate pupil center, eyes constitution = (x, y), crop
                informations = pupille_tracker(landmarks, frame, gray, head_box, BLANCK)
                right_pupil, left_pupil = informations

                POSITION_RIGHT.append(right_pupil)
                POSITION_LEFT.append(left_pupil)
                TIMMER.append((time.time() - eyes_time_functionality))

            #writting.write(frame)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                cap.release()
                cv2.destroyAllWindows()
            
                return "stop"


        else:

            IMG = cv2.imread(picture_app_3)

            position, box = retracage(POSITION_RIGHT, POSITION_LEFT, IMG)
            timmer_treatment(TIMMER, position, box, IMG)

            return "stop"







def run_data():
    """You can raise it !"""

    #Search video written from last operation and put it into a list.
    start_time_data_list = time.time()
    data = os.listdir(path_data)
    data = sorted(data)
    number_file = len(data)

    print("Count file to treat : ", number_file)
    print("running data took : ", time.time() - start_time_data_list)


    #From this list, run the video
    for video in data:
        video = path_data_video.format(video)
        video = path_data_video.format("g.mp4")

        print("\nin course: ", video)

        cap = cv2.VideoCapture(video)
        width_video  = int(cap.get(3))
        height_video = int(cap.get(4))


        #Recuperate eye position
        stop = recuperate_eyes_position(video, height_video, width_video)
        if stop == "stop":
            break


if __name__ == "__main__":
    run_data()


