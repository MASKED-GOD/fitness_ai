import cv2
import mediapipe as mp
import numpy as np 

left_counter = 0
left_stage = None
right_counter = 0
right_stage = None

def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Left Arm
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if left_angle > 160:
                left_stage = "down"
            if left_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1
                print("Left Reps:", left_counter)

            # Right Arm
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if right_angle > 160:
                right_stage = "down"
            if right_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1
                print("Right Reps:", right_counter)

        except:
            pass

        # Draw Left Arm
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Draw Right Arm
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                   mp_drawing.DrawingSpec(color=(66, 117, 245), thickness=2, circle_radius=2), 
                                   mp_drawing.DrawingSpec(color=(230, 66, 245), thickness=2, circle_radius=2))

        # Display Counters
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'Left Reps', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(left_counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(image, (480, 0), (640, 73), (66, 117, 245), -1)
        cv2.putText(image, 'Right Reps', (495, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(right_counter),
                    (490, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()