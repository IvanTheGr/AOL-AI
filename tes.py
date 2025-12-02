import streamlit as st  # type: ignore
import cv2   # type: ignore
import numpy as np # type: ignore
import os
import time
from datetime import datetime
from PIL import Image # type: ignore
from pathlib import Path
from tensorflow.keras.models import load_model   # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array   # type: ignore
from collections import deque  

# simulated credentials
USER_CREDENTIALS = {"admin": "1234"}

# create session folder for saving suspicious movement pictures and videos
DOCUMENTS_PATH = Path(r"C:/Users/Public/Documents/Python/cheating_detection")
session_folder_name = f"cheating_detection/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SESSION_FOLDER = DOCUMENTS_PATH / session_folder_name
SESSION_FOLDER.mkdir(parents=True, exist_ok=True)
st.session_state['SESSION_FOLDER'] = str(SESSION_FOLDER)

if 'violation_count' not in st.session_state:
    st.session_state['violation_count'] = 0
if 'last_expression' not in st.session_state:
    st.session_state['last_expression'] = ""

MAX_VIOLATIONS = 3

# buffer and frame rate for suspicious video clips
FRAME_RATE = 20        # adjust if your webcam has a different FPS
CLIP_SECONDS = 5       # length of suspicious clip in seconds
BUFFER_SIZE = FRAME_RATE * CLIP_SECONDS

# load facial expression model
model_path = os.path.join(os.getcwd(), "emotion_model.h5")
if os.path.exists(model_path):
    emotion_model = load_model(model_path, compile=False)
else:
    emotion_model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- gaze model (TensorFlow) ---
gaze_model = None
use_gaze_model = False
gaze_model_path = os.path.join(os.getcwd(), "gaze_model.h5")  # You must provide this model
if os.path.exists(gaze_model_path):
    try:
        gaze_model = load_model(gaze_model_path, compile=False)
        use_gaze_model = True
    except Exception as e:
        print(f"Failed to load gaze model: {e}")
        use_gaze_model = False

# OpenCV Haar cascades for face / eyes fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# utility functions
def detect_expression(face_frame):
    if emotion_model is None:
        return "Neutral"
    face_gray = cv2.cvtColor(face_frame, cv2.COLOR_RGB2GRAY)
    
    face_resized = cv2.resize(face_gray, (64, 64))
    face_resized = face_resized.astype("float") / 255.0
    face_resized = img_to_array(face_resized)
    face_resized = np.expand_dims(face_resized, axis=0)
    preds = emotion_model.predict(face_resized, verbose=0)[0]
    emotion = emotion_labels[np.argmax(preds)]
    return emotion

def save_suspicious_image(frame, reason):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = SESSION_FOLDER / f"{reason}_{timestamp}.jpg"
    try:
        cv2.imwrite(str(filename), frame)
    except Exception as e:
        print(f"Failed to save image: {e}")
    return filename

def clear_suspicious_images():
    for file in SESSION_FOLDER.glob("*.jpg"):
        file.unlink()

def start_video_recording(path, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(path), fourcc, 20.0, frame_size)

def save_suspicious_clip(buffer, output_path, frame_size):
    if len(buffer) < 2:
        return  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FRAME_RATE, frame_size)
    for f in buffer:
        if f.shape[1] != frame_size[0] or f.shape[0] != frame_size[1]:
            f = cv2.resize(f, frame_size)
        out.write(f)
    out.release()

def calculate_eye_aspect_ratio(eye):
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# gaze detection function
def detect_gaze(face_bbox, frame):
    x, y, w, h = face_bbox
    img_h, img_w = frame.shape[:2]

    # crop face region
    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    if face_rgb.size == 0:
        return "no_face"

    # if gaze model is available, use this function
    if use_gaze_model and gaze_model is not None:
        try:
            
            inp = cv2.resize(face_rgb, (64, 64))
            inp = inp.astype("float") / 255.0
            inp = img_to_array(inp)
            inp = np.expand_dims(inp, axis=0)
            preds = gaze_model.predict(inp, verbose=0)[0]
            
            idx = int(np.argmax(preds))
            mapping = ['center', 'left', 'right', 'up', 'down']
            if idx < len(mapping):
                return mapping[idx]
            else:
                return "center"
        except Exception as e:
           
            print(f"Gaze model prediction error: {e}")

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
    if len(eyes) == 0:
        
        return "blink"

    # pick two largest eyes (if available)
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centers_x = []
    centers_y = []
    for (ex, ey, ew, eh) in eyes:
        centers_x.append(ex + ew // 2)
        centers_y.append(ey + eh // 2)

    avg_x = np.mean(centers_x)
    avg_y = np.mean(centers_y)

    # initialize references
    if 'eye_x_reference' not in st.session_state:
        st.session_state['eye_x_reference'] = avg_x + x
    if 'eye_y_reference' not in st.session_state:
        st.session_state['eye_y_reference'] = avg_y + y

    dx = (avg_x + x) - st.session_state['eye_x_reference']
    dy = (avg_y + y) - st.session_state['eye_y_reference']

    horizontal_thresh = max(8, int(w * 0.06))
    vertical_thresh = max(8, int(h * 0.06))

    if dx < -horizontal_thresh:
        return "left"
    if dx > horizontal_thresh:
        return "right"
    if dy < -vertical_thresh:
        return "up"
    if dy > vertical_thresh:
        return "down"
    return "center"

def log_violation(log_messages, reason, frame):
    st.session_state['violation_count'] += 1
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_messages.append(f"{timestamp} - ⚠️ {reason}. Stay focused! Violation {st.session_state['violation_count']}/{MAX_VIOLATIONS}.")
    save_suspicious_image(frame, reason)
    if st.session_state['violation_count'] >= MAX_VIOLATIONS:
        log_messages.append(f"{timestamp} - ❌ Account access disabled due to violations.")
        st.error("Access denied. You can no longer continue the exam.")
        st.stop()

def login():
    st.sidebar.header("Login Required")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.sidebar.success("Login Successful!")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

def main_dashboard():
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    if st.session_state['webcam_running']:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():  
            st.error("Unable to access webcam.")
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("Unable to read frame from webcam. Please check your camera connection.")
            cap.release()
            return
        frame_height, frame_width = frame.shape[:2]

        log_messages = []
        violation_start_time = None
        video_filename = SESSION_FOLDER / f"exam_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_writer = start_video_recording(video_filename, (frame_width, frame_height))
        recording_exam = True

        frame_buffer = deque(maxlen=BUFFER_SIZE)  

        def log(message):
            log_messages.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

        gaze_placeholder = st.markdown("**Current Gaze Direction:** N/A")
        frame_placeholder = st.empty()
        logs_placeholder = st.empty()
        stop_button = st.button("Stop Webcam")

        while cap.isOpened() and st.session_state['webcam_running']:
            ret, frame = cap.read()
            if not ret:
                st.warning("No frame detected!")
                break

            frame_buffer.append(frame.copy())  

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            # face detection using Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            face_detected = len(faces) > 0
            current_gaze = "N/A"
            suspicious_activity_detected = False
            suspicious_reason = ""

            if face_detected:
                (x, y, w, h) = faces[0]
                gaze = detect_gaze((x, y, w, h), frame)
                current_gaze = gaze
                if gaze in ["left", "right", "up", "down"]:
                    suspicious_activity_detected = True
                    suspicious_reason = f"gaze_{gaze}"

                # crop face region for expression detection similar to previous coords approach
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                face_crop = rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    expression = detect_expression(face_crop)
                    if expression != st.session_state['last_expression']:
                        st.session_state['last_expression'] = expression
                        log(f"🧠 Expression Detected: {expression}")
                        if expression in ['Angry', 'Sad', 'Fear', 'Surprise', 'Disgust']:
                            suspicious_activity_detected = True
                            suspicious_reason = f"expression_{expression.lower()}"
            else:
                suspicious_activity_detected = True
                suspicious_reason = "no_face"

            if suspicious_activity_detected:
                if violation_start_time is None:
                    violation_start_time = time.time()
                elif time.time() - violation_start_time >= 10:
                    # save video clip of suspicious behavior
                    clip_filename = SESSION_FOLDER / f"clip_{suspicious_reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    save_suspicious_clip(frame_buffer, clip_filename, (frame_width, frame_height))
                    if recording_exam:
                        video_writer.release()
                        recording_exam = False
                        log(f"❌ Suspicious activity detected, exam recording stopped.")
                        log(f"📼 Suspicious recording saved: {video_filename}")
                        log(f"📹 Suspicious clip saved: {clip_filename}")
                    log_violation(log_messages, suspicious_reason, frame)
                    violation_start_time = None
            else:
                if recording_exam:
                    video_writer.write(frame)
                violation_start_time = None

            gaze_placeholder.markdown(f"**Current Gaze Direction:** {current_gaze}")
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            logs_placeholder.markdown(
                f"""<div id=\"log\" style=\"height: 300px; background-color:#000000; color:white; \
                    overflow-y: scroll; padding:10px; border:1px solid #333;\">
                    {"<br>".join(log_messages[-15:])}
                </div>""",
                unsafe_allow_html=True
            )

            if stop_button:
                if recording_exam:
                    video_writer.release()
                    log(f"📼 Final recording saved: {video_filename}")
                clear_suspicious_images()
                st.session_state['webcam_running'] = False
                break

        cap.release()
        if recording_exam:
            video_writer.release()
            log(f"📼 Final recording saved: {video_filename}")
        cv2.destroyAllWindows()
        st.success("Webcam stopped successfully.")
        logs_placeholder.markdown(
            f"""<div id=\"log\" style=\"height: 300px; background-color:#000000; color:white; \
                overflow-y: scroll; padding:10px; border:1px solid #333;\">
                {"<br>".join(log_messages[-15:])}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        if st.button("Start Webcam"):
            st.session_state['webcam_running'] = True
            st.rerun()


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.title("Cheating Detection System")

if not st.session_state['logged_in']:
    login()
else:
    main_dashboard()

