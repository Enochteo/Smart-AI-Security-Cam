from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import json
from ultralytics import YOLO
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from time import time
last_email_time = 0
EMAIL_COOLDOWN = 30  # seconds


# Load environment variables
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

# Initialize Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)
app.secret_key = os.urandom(24)

# Global state
paused = False
prev_frame = None  # ✅ Now global and persistent
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Settings load/save
def load_settings():
    with open("settings.json", "r") as f:
        return json.load(f)

def save_settings(settings):
    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)

# Email notification
def send_email_alert(snapshot_path):
    msg = EmailMessage()
    msg["Subject"] = "Motion Detected!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("Motion was detected. See the attached image.")

    with open(snapshot_path, "rb") as f:
        file_data = f.read()
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=os.path.basename(snapshot_path))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("✅ Email sent!")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# Video frame generator
def generate_frames():
    global paused, prev_frame, last_email_time
    prev_frame = None
    while True:
        settings = load_settings()
        if paused:
            continue

        success, frame = camera.read()
        if not success or frame is None:
            continue

        frame = cv2.resize(frame, (480, 360))

        if settings["detection_mode"] == "motion":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)  # More sensitive

            if prev_frame is None:
                prev_frame = gray
            else:
                frame_delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("Contours found:", len(contours))

                motion_detected = False
                for contour in contours:
                    area = cv2.contourArea(contour)
                    print("Contour area:", area)

                    if area < int(settings["sensitivity"]) * 100:
                        continue
                    cv2.waitKey(1)
                    motion_detected = True
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Motion Detected", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    if time() - last_email_time > EMAIL_COOLDOWN:
                        send_email_alert(filename)
                        last_email_time = time()

                    send_email_alert(filename)

                if not motion_detected:
                    cv2.putText(frame, "Monitoring...", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                prev_frame = gray

        elif settings["detection_mode"] == "ai":
            results = model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Always show sensitivity level
        cv2.putText(frame, f"Sensitivity: {settings['sensitivity']}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route("/")
def index():
    settings = load_settings()
    return render_template("index.html", paused=paused, settings=settings)

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    global settings
    if request.method == "POST":
        settings = load_settings()
        settings["detection_mode"] = request.form.get("detection_mode")
        settings["sensitivity"] = int(request.form.get("sensitivity"))
        save_settings(settings)
        flash("Settings updated successfully!", "success")
        return redirect(url_for("settings_page"))

    settings = load_settings()
    return render_template("settings.html", detection_mode=settings["detection_mode"], settings=settings)

@app.route("/toggle_feed", methods=["POST"])
def toggle_feed():
    global paused
    paused = not paused
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(port=5001, debug=True)
