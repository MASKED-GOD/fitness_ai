import cv2
import mediapipe as mp
import numpy as np 
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- CURL COUNTER (unchanged) ---
def run_curl_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    left_stage = None
    right_stage = None
    set_num = 1
    left_counter = 0
    right_counter = 0
    left_set_counter = 0
    right_set_counter = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
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
                    left_set_counter += 1
                    print(f"{name} - Set {set_num} - Left Reps: {left_set_counter}")
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
                    right_set_counter += 1
                    print(f"{name} - Set {set_num} - Right Reps: {right_set_counter}")
            except:
                pass
            # Draw Landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            # Display Counters and Set Info
            cv2.rectangle(image, (0, 0), (225, 100), (245, 117, 16), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Left Reps', (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_set_counter),
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (480, 0), (640, 100), (66, 117, 245), -1)
            cv2.putText(image, 'Right Reps', (495, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_set_counter),
                        (490, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # Show break timer if set is complete
            if left_set_counter >= reps_per_set and right_set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (150, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                left_set_counter = 0
                right_set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # After all sets
        cap.release()
        cv2.destroyAllWindows()
        # Show completion message
        done_img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', done_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

# --- SQUAT COUNTER ---
def run_squat_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    set_num = 1
    squat_counter = 0
    set_counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                # Use right leg for squats
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
                    squat_counter += 1
                    set_counter += 1
                    print(f"{name} - Set {set_num} - Squat Reps: {set_counter}")
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Overlay
            cv2.rectangle(image, (0, 0), (225, 100), (0, 200, 0), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Squat Reps', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        done_img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', done_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

# --- SIT-UP COUNTER ---
def run_situp_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    set_num = 1
    situp_counter = 0
    set_counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                # Use left shoulder and left hip for sit-ups
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                angle = calculate_angle(shoulder, hip, knee)
                if angle > 150:
                    stage = "down"
                if angle < 100 and stage == 'down':
                    stage = "up"
                    situp_counter += 1
                    set_counter += 1
                    print(f"{name} - Set {set_num} - Sit-up Reps: {set_counter}")
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0, 0), (225, 100), (200, 100, 0), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Sit-up Reps', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        done_img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', done_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

# --- SHOULDER PRESS COUNTER ---
def run_shoulder_press_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    set_num = 1
    press_counter = 0
    set_counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                # Use right arm for shoulder press
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                if angle < 90:
                    stage = "down"
                if angle > 160 and stage == 'down':
                    stage = "up"
                    press_counter += 1
                    set_counter += 1
                    print(f"{name} - Set {set_num} - Shoulder Press Reps: {set_counter}")
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0, 0), (300, 100), (100, 0, 200), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Shoulder Press Reps', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        done_img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', done_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def run_jump_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    set_num = 1
    jump_counter = 0
    set_counter = 0
    stage = None
    last_ankle_y = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                if last_ankle_y is not None:
                    # Detect jump: ankles go up (y decreases), then down (y increases)
                    if stage == 'down' and avg_ankle_y < last_ankle_y - 0.05:
                        stage = 'up'
                    if stage == 'up' and avg_ankle_y > last_ankle_y + 0.05:
                        jump_counter += 1
                        set_counter += 1
                        print(f"{name} - Set {set_num} - Jump Reps: {set_counter}")
                        stage = 'down'
                else:
                    stage = 'down'
                last_ankle_y = avg_ankle_y
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0, 0), (225, 100), (0, 150, 255), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Jump Reps', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        done_img = np.zeros((400, 700, 3), dtype=np.uint8)
        cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', done_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def run_pushup_counter(name, sets=2, reps_per_set=13, break_duration=20):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    set_num = 1
    pushup_counter = 0
    set_counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and set_num <= sets:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                # Both arms: average the angle
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                avg_angle = (l_angle + r_angle) / 2
                if avg_angle < 80:
                    stage = "down"
                if avg_angle > 160 and stage == 'down':
                    stage = "up"
                    pushup_counter += 1
                    set_counter += 1
                    print(f"{name} - Set {set_num} - Push-up Reps: {set_counter}")
            except:
                pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0, 0), (300, 100), (0, 100, 200), -1)
            cv2.putText(image, f'Set: {set_num}/{sets}', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Push-up Reps', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(set_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            if set_counter >= reps_per_set:
                cv2.putText(image, 'Set Complete! Break Time...', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
                cv2.waitKey(1000)
                for t in range(break_duration, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (100, 150), (540, 330), (0, 0, 0), -1)
                    cv2.putText(frame, f'Break: {t} sec', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Next: Set {set_num+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Mediapipe Feed', frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                set_num += 1
                set_counter = 0
            else:
                cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    done_img = np.zeros((400, 700, 3), dtype=np.uint8)
    cv2.putText(done_img, f'Workout Complete!', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
    cv2.putText(done_img, f'Great job, {name}!', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
    cv2.imshow('Mediapipe Feed', done_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()