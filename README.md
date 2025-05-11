# 🔐 Smart AI Security Camera

An intelligent, real-time home security system built with Flask, OpenCV, and YOLOv8. Detects motion or AI-based objects and streams live video via a web interface. Supports email alerts and snapshot saving.

---

##  Features

-  Motion Detection using OpenCV
-  Object Detection using YOLOv8 (Ultralytics)
-  Automatic Snapshots on Detection
-  Email Alerts with Image Attachment
-  Toggle Live Feed On/Off
-  Bootstrap-based Responsive Web Interface

---

## Demo

![screenshot](static/demo_ui.jpg)  
To be added later... Calmzz

---

##  Tech Stack

- Python 3.10  
- Flask  
- OpenCV  
- Ultralytics YOLOv8  
- Bootstrap 5  
- Gmail SMTP for Email Alerts  

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-security-cam-ai.git
cd smart-security-cam-ai
```

### 2. Create a Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
EMAIL_ADDRESS=youremail@gmail.com
EMAIL_PASSWORD=yourapppassword
TO_EMAIL=receiver@gmail.com
```

### 5. Run the App
```bash
python app.py
```

##  Folder Structure
smart-security-cam-ai/
│
├── app.py
├── settings.json
├── snapshots/
├── templates/
│   ├── index.html
│   └── settings.html
├── static/
│   └── demo_ui.jpg
├── .env
├── requirements.txt
└── README.md



