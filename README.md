# Facial Attendance System (Drone & DeepFace AI Integrated)

An automated, web-based **Facial Attendance System** built with **Python**, **Flask**, **SQLite**, **DeepFace AI**, **OpenCV**, and **Drone Camera Stream Integration**.

---

## 📌 Technology Stack & Concept

* **Backend Framework:** Python 3 & Flask
* **Database:** SQLite3 (stores student details and attendance logs)
* **Facial Recognition & Deep Learning:**
  * **DeepFace (VGG-Face model)**: Generates high-dimensional facial embeddings for exact identity matching using cosine similarity.
  * **OpenCV (`cv2`)**: Used for video stream decoding, frame capture, and Haar Cascade face detection.
* **Drone Camera Integration:**
  * Supports direct video feeds from Drone camera streams (RTSP, UDP, HTTP MJPEG stream, or video device index).
* **Frontend UI:** HTML5, CSS3, JavaScript (Fetch API, Bootstrap 5, live stream player with camera source switcher).

---

## 🚀 Key Features

1. **Drone Camera Stream & Local Webcam Support**
   * Seamlessly stream video directly from a **Drone Camera** (e.g. RTSP `rtsp://<drone-ip>:554/live`, HTTP/UDP) or switch to a **Local Webcam**.
   * Configure Drone stream URLs dynamically via the web UI settings modal or environment variables.

2. **DeepFace AI Facial Recognition**
   * Deep learning facial feature extraction and distance matching using DeepFace.
   * High accuracy verification without manual entry.

3. **Student Registration & Attendance Logs**
   * Capture student face snapshots via Drone Stream or Webcam and associate with Name & Registration Number.
   * Automatic daily attendance logging with duplicate detection and timestamp records.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
* Python 3.8+
* `pip` and `venv` (Python Virtual Environment)

### 2. Setting Up Virtual Environment (`venv`)

It is strongly recommended to use a Python virtual environment (`venv`) to run the system:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate

# On Windows (Command Prompt):
# venv\Scripts\activate.bat

# On Windows (PowerShell):
# venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚁 Drone Camera Setup & Configuration

You can configure the Drone Camera Stream source in two ways:

1. **Via Environment Variable:**
   Set `DRONE_STREAM_URL` before running the application:
   ```bash
   export DRONE_STREAM_URL="rtsp://192.168.1.1:554/live"
   python3 app.py
   ```
2. **Via Web Interface:**
   * Open the web app in your browser.
   * Click **"Drone Stream Settings"** in the top navigation bar.
   * Enter your Drone's RTSP stream URL (e.g., `rtsp://192.168.1.100:554/stream`), HTTP MJPEG URL, or camera index (`0`), then click **"Save & Reconnect"**.

---

## 🏃 Usage & How to Run

### 1. Start the Server
With `venv` activated, run the Flask application:
```bash
python3 app.py
```
By default, the server starts on `http://127.0.0.1:5000`.

### 2. Access the Application
Open your web browser and navigate to `http://127.0.0.1:5000`.

### 3. Workflow Guide
* **Select Camera Source:**
  * Toggle between **Drone Camera Stream** or **Local Webcam** using the active camera toolbar.
* **Registering a Student:**
  1. Go to the **Student Registration** tab.
  2. Enter **Full Name** and **Registration Number**.
  3. Align the student's face in the live Drone/Webcam feed and click **"Capture & Register"**.
* **Marking Attendance:**
  1. Go to the **Mark Attendance** tab.
  2. Position the student's face in front of the Drone/Webcam view and click **"Scan Face & Mark Attendance"**.
  3. DeepFace will recognize the student and record attendance.
* **Viewing Logs:**
  1. Open the **Attendance Logs & Students** tab to inspect student records and timestamped attendance logs.

---

## 🧪 Running Unit Tests

Run unit tests using Python's built-in `unittest`:
```bash
python3 -m unittest test_app.py
```
