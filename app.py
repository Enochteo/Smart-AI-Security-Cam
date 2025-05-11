from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import json
from ultralytics import YOLO
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()
# Load environment variables
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
# Function to send email
TO_EMAIL = os.getenv("TO_EMAIL")
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")


app = Flask(__name__)
camera = cv2.VideoCapture(0)
app.secret_key = os.urandom(24)

detection_mode = "motion"
paused = False

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def load_settings():
    with open("settings.json", "r") as f:
        settings = json.load(f)
    return settings

def save_settings(settings):
    with open("settings.json", "w") as f:
        json.dump(settings, f, indent = 4)
settings = load_settings()

def generate_frames():
    global paused
    prev_frame = None
    while True:
        if paused:
            continue
        success, frame = camera.read()
        if not success or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))

        if settings["detection_mode"] == "motion":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            frame_delta = cv2.absdiff(prev_frame, gray)
            prev_frame = gray

            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")
                cv2.imwrite(filename, frame)
                send_email_alert(filename)


        elif settings["detection_mode"] == "ai":
            results = model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Always encode the final frame once, regardless of mode
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
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
            print("Email sent!")
    except Exception as e:
        print(f"Email failed: {e}")


@app.route("/")
def index():
    return render_template("index.html", paused=paused)

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    global settings
    if request.method == "POST":
        settings["detection_mode"] = request.form.get("detection_mode")
        save_settings(settings)
        flash("Settings updated successfully!", "success")
        return redirect(url_for("settings_page"))

    return render_template("settings.html", detection_mode=settings["detection_mode"])

@app.route("/toggle_feed", methods=["POST"])
def toggle_feed():
    global paused
    paused = not paused
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(port=5001, debug=True)