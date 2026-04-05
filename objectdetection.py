# 📦 Import libraries
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from gtts import gTTS
import pytesseract
from paddleocr import PaddleOCR
import os
import time

# 📌 Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🧠 Load YOLO model
def load_yolo_model(model_path='yolov8m.pt'):
    return YOLO(model_path)

# 🎯 Object detection
def detect_objects(model, img, conf_threshold=0.5, nms_threshold=0.4):
    results = model(img, conf=conf_threshold, iou=nms_threshold)
    return results

# ✍️ Annotate frame with detected objects
def annotate_objects(img, results, conf_threshold=0.5):
    detected_objects = []

    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                detected_objects.append(label)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label}: {conf:.2f}', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return detected_objects

# 😊 Emotion detection (only once every few seconds)
def detect_emotions(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return None

# ✍️ Text detection (only once every few seconds)
def detect_text(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text1 = pytesseract.image_to_string(gray)

        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(gray, cls=True)
        texts2 = []
        for line in result[0]:
            texts2.append(line[1][0])

        combined_text = text1 + " ".join(texts2)
        return combined_text.strip() if combined_text.strip() else None
    except Exception as e:
        print(f"Text detection error: {e}")
        return None

# 🔈 Play audio feedback
def play_audio_feedback(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # For Windows

# 🎥 Real-time webcam detection
def real_time_detection(model_path='yolov8m.pt', conf_threshold=0.5, nms_threshold=0.4):
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    last_detection_time = time.time()

    detected_emotion = None
    detected_text = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection every frame
        results = detect_objects(model, frame, conf_threshold, nms_threshold)
        detected_objects = annotate_objects(frame, results, conf_threshold)

        # Emotion and text detection every 5 seconds (for speed)
        if time.time() - last_detection_time > 5:
            detected_emotion = detect_emotions(frame)
            detected_text = detect_text(frame)
            last_detection_time = time.time()

        if detected_emotion:
            cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            detected_objects.append(f"Emotion: {detected_emotion}")

        if detected_text:
            cv2.putText(frame, f'Text: {detected_text[:30]}...', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            detected_objects.append(f"Text detected")

        cv2.imshow('Real-time Detection', frame)

        # Give audio feedback if there are detections
        if detected_objects:
            feedback = "Detected: " + ", ".join(detected_objects)
            print(feedback)
            play_audio_feedback(feedback)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🚀 Start
real_time_detection()

