import cv2

def draw(head_box, frame, right_eye, left_eye):

    #Draw
    x, y, w, h = head_box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, 'Head', ((x+w) - 30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)


    if right_eye[0][0] is not None:
        cv2.circle(right_eye[1], (right_eye[0]), 4, (0, 0, 255), 1)
    if left_eye[0][0] is not None:
        cv2.circle(left_eye[1], (left_eye[0]), 4, (0, 0, 255), 1)
