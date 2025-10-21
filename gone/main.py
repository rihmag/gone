import cv2
import numpy as np
import mediapipe as mp
from fire import Fire
from punch import Punch
from collections import deque as dq
import math as math 
from rotation import Rotation
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
holistic = mp.solutions.holistic

# Open webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video file punching_video.mp4")
    exit()

print("Fire Pose Detector Started!")
print("Position yourself with:")
print("- Both hands open with fingers spread")
print("- Both arms straight up above shoulders")
print("Press 'q' to quit")

with holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5) as h:
    # Initialize separate hands and pose for Fire class
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    fire_object = Fire(mp_hands, mp_pose, mp_drawing, np, cv2)
    punch_object = Punch(mp_drawing,np,cv2,dq)
    rotation_object  = Rotation(pose ,mp_drawing,cv2,math)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        hand_result = hands.process(rgb)
        
        # Draw pose landmarks
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        # Detect fire pose
        status_text, fire_detected = fire_object.Detect_Fire(frame, pose_result, hand_result)
        draw_frame = punch_object.punch_execute(frame,pose_result)
        rotate_frame  = rotation_object.execute(frame,pose_result)
        
        # Display frame
        cv2.imshow('Fire Pose Detector', rotate_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    hands.close()
    pose.close()

cap.release()
cv2.destroyAllWindows()