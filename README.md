# Facial Attendance System

An automated, web-based **Facial Attendance System** built with **Python**, **Flask**, **SQLite**, and **OpenCV** (using Haar Cascade classifier for face detection and Local Binary Patterns Histograms - LBPH for face recognition).

---

## 📌 Technology Stack & Concept

* **Backend Web Framework:** Python 3 & Flask
* **Database:** SQLite3 (stores student details and attendance logs)
* **Computer Vision & Face Recognition:**
  * **OpenCV (`cv2`)**: Used for real-time video frame decoding and processing.
  * **Haar Cascade Classifier**: Used for detecting human faces in images.
  * **LBPH (Local Binary Patterns Histograms) Face Recognizer**: Used for training face model features and recognizing individual faces based on texture histogram analysis.
* **Frontend UI:** HTML5, CSS3, JavaScript (Fetch API & MediaDevices API for webcam stream), Bootstrap 5.

---

## 🚀 Key Features

1. **Facial Attendance Marking (No Manual Input Required)**
   * When a student stands in front of the camera and clicks **"Scan Face & Mark Attendance"**, the system uses facial recognition alone to identify the student.
   * If recognized, attendance is logged in the database for the day with a timestamp.
   * If the student's face is not in the system, it alerts that **"Student is not registered."**

2. **Student Registration**
   * Students provide their **Full Name** and **Registration Number** alongside a snapshot captured via webcam.
   * The facial image is stored and automatically trains/updates the LBPH facial recognition model.

3. **Attendance Logs & Student Records**
   * Tabbed interface displaying all registered students and real-time attendance logs.

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
* Python 3.8+
* `pip`

### 2. Install Dependencies
```bash
pip install opencv-python opencv-contrib-python numpy flask
```

---

## 🏃 Usage & How to Run

### 1. Start the Server
Run the Flask application:
```bash
python3 app.py
```
By default, the server will start on `http://127.0.0.1:5000`.

### 2. Access the Application
Open your web browser and navigate to `http://127.0.0.1:5000`.

### 3. Workflow Guide
* **Registering a Student:**
  1. Click on the **Student Registration** tab.
  2. Enter the student's **Full Name** and **Registration Number**.
  3. Align face in front of the webcam and click **"Capture & Register"**.
* **Marking Attendance:**
  1. Go to the **Mark Attendance** tab.
  2. Position face in front of the webcam and click **"Scan Face & Mark Attendance"**.
  3. The system will recognize the student and display their attendance status. If unrecognized, it displays **"Student is not registered"**.
* **Viewing Logs:**
  1. Go to the **Attendance Logs & Students** tab to view the list of registered students and attendance records.

---

## 🧪 Running Unit Tests

Run the test suite using Python's built-in `unittest`:
```bash
python3 -m unittest test_app.py
```
