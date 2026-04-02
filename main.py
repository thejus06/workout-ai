import cv2
import mediapipe as mp
import time
import subprocess
from collections import deque
from utils import calculate_angle

# ---------------- VOICE SETUP ----------------
def speak(text):
    subprocess.Popen(
        ['powershell', '-Command',
         f'Add-Type -AssemblyName System.Speech; '
         f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
         f'$s.Rate = 2; '
         f'$s.Speak("{text}")'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- STATE ----------------
counter = 0
stage = "UP"

last_voice_time = 0
voice_cooldown = 1.5

UP_ANGLE   = 160
DOWN_ANGLE = 80

alpha = 0.7
prev_angle = None

DIRECTION_BUFFER = 5
shoulder_y_buffer = deque(maxlen=DIRECTION_BUFFER)

min_angle_reached = False
last_rep_time = 0
cooldown = 0.5

calibration_frames = []
CALIBRATE_N = 30
depth_threshold = None
lowest_y = None

# ---------------- HELPERS ----------------
VISIBILITY_THRESHOLD = 0.6

def landmarks_visible(lm, indices):
    return all(lm[i].visibility > VISIBILITY_THRESHOLD for i in indices)

def get_direction(buf):
    if len(buf) < DIRECTION_BUFFER:
        return 'neutral'
    delta = buf[-1] - buf[0]
    if delta > 0.008:
        return 'down'
    elif delta < -0.008:
        return 'up'
    return 'neutral'

def detect_view(lm):
    ls, rs = lm[11], lm[12]
    x_spread = abs(ls.x - rs.x)
    z_spread = abs(ls.z - rs.z)
    if x_spread > 0.18:
        return "FRONT"
    elif z_spread > 0.08 or x_spread < 0.10:
        return "SIDE"
    return "FRONT"

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        view = detect_view(lm)

        # ---- Calibration phase ----
        if depth_threshold is None:
            if landmarks_visible(lm, [11, 12]):
                calibration_frames.append((lm[11].y + lm[12].y) / 2)
            if len(calibration_frames) >= CALIBRATE_N:
                shoulder_range = max(calibration_frames) - min(calibration_frames)
                depth_threshold = max(0.08, shoulder_range * 0.4)
                lowest_y = min(calibration_frames)
                cv2.putText(frame, "Calibrated!", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Calibrating... {len(calibration_frames)}/{CALIBRATE_N}",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # ---- SIDE VIEW ----
        if view == "SIDE":
            required = [11, 13, 15, 23, 27]
            if not landmarks_visible(lm, required):
                cv2.putText(frame, "Landmarks not clear", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            else:
                shoulder = [lm[11].x, lm[11].y]
                elbow    = [lm[13].x, lm[13].y]
                wrist    = [lm[15].x, lm[15].y]
                hip      = [lm[23].x, lm[23].y]
                ankle    = [lm[27].x, lm[27].y]

                arm_angle  = calculate_angle(shoulder, elbow, wrist)
                body_angle = calculate_angle(shoulder, hip, ankle)

                if prev_angle is None:
                    prev_angle = arm_angle
                angle = alpha * arm_angle + (1 - alpha) * prev_angle
                prev_angle = angle

                is_body_straight = body_angle > 155

                if not is_body_straight:
                    if time.time() - last_voice_time > voice_cooldown:
                        speak("Keep your body straight")
                        last_voice_time = time.time()

                shoulder_y = lm[11].y
                shoulder_y_buffer.append(shoulder_y)
                direction = get_direction(shoulder_y_buffer)

                if lowest_y is None:
                    lowest_y = shoulder_y
                if shoulder_y > lowest_y:
                    lowest_y = shoulder_y

                if angle < DOWN_ANGLE and direction == 'down' and is_body_straight:
                    min_angle_reached = True
                    if stage != "DOWN":
                        stage = "DOWN"
                        if time.time() - last_voice_time > voice_cooldown:
                            speak("Down")
                            last_voice_time = time.time()

                if (angle > UP_ANGLE and min_angle_reached
                        and direction == 'up' and is_body_straight
                        and depth_threshold is not None):
                    depth = lowest_y - shoulder_y
                    if depth > depth_threshold:
                        current_time = time.time()
                        if current_time - last_rep_time > cooldown:
                            counter += 1
                            last_rep_time = current_time
                            min_angle_reached = False
                            lowest_y = shoulder_y
                            if stage != "UP":
                                stage = "UP"
                                if time.time() - last_voice_time > voice_cooldown:
                                    speak("Good rep")
                                    last_voice_time = time.time()

        # ---- FRONT VIEW ----
        else:
            required = [11, 12, 13, 14, 15, 16]
            if not landmarks_visible(lm, required):
                cv2.putText(frame, "Landmarks not clear", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            else:
                l_angle = calculate_angle([lm[11].x, lm[11].y],
                                          [lm[13].x, lm[13].y],
                                          [lm[15].x, lm[15].y])
                r_angle = calculate_angle([lm[12].x, lm[12].y],
                                          [lm[14].x, lm[14].y],
                                          [lm[16].x, lm[16].y])
                avg_angle = (l_angle + r_angle) / 2

                if prev_angle is None:
                    prev_angle = avg_angle
                angle = alpha * avg_angle + (1 - alpha) * prev_angle
                prev_angle = angle

                shoulder_y = (lm[11].y + lm[12].y) / 2
                shoulder_y_buffer.append(shoulder_y)
                direction = get_direction(shoulder_y_buffer)

                if angle < DOWN_ANGLE and direction == 'down':
                    min_angle_reached = True
                    if stage != "DOWN":
                        stage = "DOWN"
                        if time.time() - last_voice_time > voice_cooldown:
                            speak("Down")
                            last_voice_time = time.time()

                if (angle > UP_ANGLE and min_angle_reached and direction == 'up'):
                    current_time = time.time()
                    if current_time - last_rep_time > cooldown:
                        counter += 1
                        last_rep_time = current_time
                        min_angle_reached = False
                        if stage != "UP":
                            stage = "UP"
                            if time.time() - last_voice_time > voice_cooldown:
                                speak("Good rep")
                                last_voice_time = time.time()

        # ---- UI ----
        cv2.rectangle(frame, (0, 0), (320, 145), (0, 0, 0), -1)

        cv2.putText(frame, f"REPS: {counter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.putText(frame, f"STAGE: {stage}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"VIEW: {view}", (180, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if depth_threshold is not None:
            cv2.putText(frame, f"THR: {depth_threshold:.2f}", (180, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Workout AI", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()