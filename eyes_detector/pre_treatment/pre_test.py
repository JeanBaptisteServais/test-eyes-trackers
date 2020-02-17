import cv2
import dlib
import time
import numpy as np

#Recuperate points.
from recuperate_points.face_points import recuperate_landmarks
#Recuperate face in square.
from recuperate_points.face_points import get_face_in_box


def resizer(frame, predictor, detector):

    start_time_function = time.time()

    video_division = 0.35   #Increment this var for video size.
    height, width = frame.shape[:2] #Initial dimension of the video.

    find = False    #We loop until find a width face.
    ocontinue = True    #Loop condition.

    start_time_loop = time.time()
    while ocontinue:

        video_division += 0.05  #Incrementation.
        width_division  = int(width /  video_division)   #Divide width.
        height_division = int(height / video_division)   #Divide height.

        #Regulation of the hights dimensions.
        if width_division > 2000 or height_division > 2000:
            pass

        #Regulation of the smaller dimensions.
        elif width_division < 50 or height_division < 50:
            ocontinue = False

        elif width_division > 0 and height_division > 0:  #Ok dimension.

            #Resize the frame.
            frame = cv2.resize(frame, (width_division, height_division))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Gray threshold.

            #Recuperate landmarks.
            faces, landmarks = recuperate_landmarks(gray, predictor, detector)

            #Regulation of no landmarks.
            if landmarks is not None:
                head = get_face_in_box(landmarks)   #Box the face.
                x, y, w, h = head   #Recuperate the square box face.
                #print(w, y, video_division) #Display dimension

                if w <= 93: #We exceed 93 width pixels.
                    find = True #Condition for stop loop from main.
                    ocontinue = False #Condition for stop the current loop.

            #Higtly recommande if first time watch.
            #cv2.imshow("frameframe", frame)
            #cv2.waitKey(0)

        print("Loop time : ", time.time() - start_time_loop)

    print("\nFunction RESIZER time : ", time.time() - start_time_function)
    return video_division, find



def search_video_size(video, predictor, detector, dlib_model):

    start_time_function = time.time()
    
    print("\nSearch a width head to 93 pixels")

    #Initialise video.
    cap = cv2.VideoCapture(video)

    #Loop condition.
    search_video_size = True
    while search_video_size:

        _, frame = cap.read() #Window
        #Search dimensions.
        video_size, find = resizer(frame,  predictor, detector)

        if find is True:    #We found a dimensions.
            search_video_size = False

    print("Find 93 head with parameter to apply : ", video_size)
    print("Function SEARCH VIDEO SIZE time : ", time.time() - start_time_function)
    return video_size
