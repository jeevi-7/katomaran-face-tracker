# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-powered face-tracking system that detects faces in video streams, recognizes registered visitors, automatically registers new visitors, and records visitor entry and exit events.

**Hackathon:** Katomaran

**Project Type:** Computer Vision, Face Recognition, Visitor Analytics

## Demo Video

Watch the project demonstration:

**Demo:** [Watch the Intelligent Face Tracker demonstration](https://youtu.be/OYaION_syKg)


## 1. Project Overview

The Intelligent Face Tracker is designed to automate visitor identification and activity logging using computer vision and face recognition.

The application processes video frames, detects faces, generates face embeddings, and compares them with previously registered visitors. When a matching visitor is identified, the system reuses the existing visitor ID. When an unknown visitor is detected, the application can register a new visitor.

Visitor events are stored in a MySQL database, while application logs and cropped face images support monitoring and troubleshooting.

## 2. Objectives

- Detect faces in video frames.
- Generate face embeddings for recognition.
- Automatically register new visitors.
- Identify previously registered visitors using unique IDs.
- Track faces across video frames.
- Record visitor entry and exit events.
- Store visitor metadata and event history in a database.
- Provide a dashboard for reviewing visitor information and analytics.

## 3. Features

### Face Detection
Detects faces in video frames using the configured face-detection model.

### Face Recognition
Uses InsightFace to generate face embeddings and compare detected faces with registered visitor embeddings.

### Automatic Registration
Assigns a new visitor ID when a detected face does not match an existing registered visitor according to the configured recognition threshold.

### Face Tracking
Associates face detections across frames to help maintain visitor identities during video processing.

### Entry and Exit Logging
Records visitor entry and exit events with timestamps and image paths.

### Image Storage
Saves cropped face images in date-organized output folders.

### Database Integration
Uses MySQL to store registered visitors, face embeddings, and event records.

### Dashboard and Analytics
Provides Streamlit interfaces for reviewing available visitor information and analytics.

### Application Logging
Maintains an event log to help inspect recognition, tracking, registration, and visitor event processing.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Video processing | OpenCV |
| Face detection | YOLO face-detection model |
| Face recognition | InsightFace |
| Embedding comparison | NumPy and cosine similarity |
| Database | MySQL |
| Dashboard | Streamlit |
| Data configuration | JSON |
| Event logging | Python logging |
| Development environment | Visual Studio Code |

## 5. System Architecture

```text
                 Input Video
                      |
                      v
              Video Frame Processing
                      |
                      v
                Face Detection
                      |
                      v
             Face Embedding Generation
                      |
                      v
             Compare with Registered
                  Visitors
                      |
              +-------+-------+
              |               |
              v               v
         Known Visitor    New Visitor
              |               |
              v               v
        Reuse Visitor ID   Register ID
              |               |
              +-------+-------+
                      |
                      v
                Face Tracking
                      |
                      v
              Entry / Exit Events
                      |
            +---------+----------+
            |         |          |
            v         v          v
          MySQL    Face Images   Event Logs
            |
            v
       Streamlit Dashboard
       and Visitor Analytics
```

## 6. Project Structure

The main files in the repository include:

```text
katomaran-face-tracker/
├── recognize_faces.py
├── tracker.py
├── database.py
├── detect_faces.py
├── dashboard.py
├── visitor_analytics.py
├── visitor_report.py
├── export_report.py
├── test_database.py
├── test_tracker.py
├── requirements.txt
├── README.md
├── .gitignore
├── config.json
├── models/
├── logs/
└── output_images/
```

Some files and folders, including local configuration, model weights, logs, generated images, and sample videos, may be excluded from GitHub by `.gitignore`. The structure above describes the application's main components; excluded files may exist only in the local development environment.

## 7. Prerequisites

Before running the application, install or configure:

- Python compatible with the project's dependencies.
- MySQL Server.
- Git.
- Visual Studio Code or another Python editor.
- The required face-detection model weights.
- A sample video for local testing.

The application was developed and tested in a Windows environment. Hardware requirements depend on video resolution, frame-processing settings, and model configuration.

## 8. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/jeevi-7/katomaran-face-tracker.git
cd katomaran-face-tracker
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment on Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The execution-policy command applies only to the current PowerShell session.

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the dependency versions recorded in `requirements.txt` and verify compatibility with your Python version.

### Step 5: Configure MySQL

Create a MySQL database named:

```sql
CREATE DATABASE katomaran_db;
```

Configure the database connection in the application according to `database.py` and any required local configuration.

**Security:** Never commit database passwords, API keys, or other secrets to GitHub.

### Step 6: Set the database password

In PowerShell, set the password as an environment variable:

```powershell
$env:KATOMARAN_DB_PASSWORD = Read-Host "Enter your MySQL root password"
```

Enter the password when prompted. This variable is available only in the current PowerShell session.

### Step 7: Prepare local assets

Make sure the following are available locally if required by your configuration:

- The face-detection model weights.
- The sample video.
- The required configuration file.
- A working MySQL database.

These files may not be included in the public repository because of `.gitignore` rules.

## 9. Configuration

The application uses a local `config.json` file for configurable settings.

Example configuration:

```json
{
  "video_source": "sample.mp4",
  "detection_skip_frames": 5,
  "exit_timeout_seconds": 3,
  "recognition_threshold": 0.45,
  "tracker_max_missing_frames": 15,
  "image_output_dir": "output_images",
  "log_dir": "logs",
  "database_name": "katomaran_db"
}
```

The values above are an example based on the development configuration. Adjust the settings to match your local environment and the parameters supported by the application.

- `video_source`: Video input path.
- `detection_skip_frames`: Face-detection frame-skipping interval.
- `exit_timeout_seconds`: Configured exit timeout.
- `recognition_threshold`: Face-recognition similarity threshold.
- `tracker_max_missing_frames`: Maximum missing frames used by the tracker.
- `image_output_dir`: Directory for saved face images.
- `log_dir`: Directory for application logs.
- `database_name`: MySQL database name.

Keep local configuration files containing environment-specific values out of version control when appropriate.

## 10. Running the Application

Run these commands from the project root after activating the virtual environment and configuring the database password.

### Main face-tracking application

```powershell
python recognize_faces.py
```

### Main dashboard

Open a second terminal and run:

```powershell
python -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit, typically:

```text
http://localhost:8501
```

### Visitor analytics

Run in another terminal if you want to open the analytics interface separately:

```powershell
python -m streamlit run visitor_analytics.py
```

Streamlit may select a different port if another application is already using the default port.

### Database test

```powershell
python test_database.py
```

Run this test only after configuring the database connection and ensuring that the required tables exist.

## 11. Database Design

The application uses MySQL to store visitor and event information.

### Visitors table

Stores registered visitor identities and face embeddings.

Typical fields include:

- `visitor_id`: Unique visitor identifier.
- `embedding`: Stored face embedding.
- `created_at`: Visitor registration timestamp.

### Events table

Stores visitor activity.

Typical fields include:

- `event_id`: Event identifier.
- `visitor_id`: Associated visitor.
- `event_type`: Entry or exit.
- `event_time`: Event timestamp.
- `image_path`: Path to the associated face image.

The event table references registered visitor IDs through a foreign key.

## 12. Logs and Output Images

The application uses a local log directory and image output directory.

Typical local paths:

```text
logs/events.log
output_images/
```

The log file can be inspected in PowerShell:

```powershell
Get-Content .\logs\events.log -Tail 15
```

Open the output folder in Windows Explorer:

```powershell
explorer .\output_images
```

The actual records and generated images depend on the video processed and the application's runtime behaviour.

## 13. Testing

The project includes test scripts for selected components.

### Database testing

```powershell
python test_database.py
```

### Tracker testing

```powershell
python test_tracker.py
```

### Python syntax check

```powershell
python -m py_compile recognize_faces.py
```

A syntax check verifies Python syntax only. It does not guarantee that the database, face recognition, tracking, or dashboard features are all functioning correctly.

## 14. Limitations and Assumptions

- Recognition accuracy depends on image quality, lighting, face angle, and the configured similarity threshold.
- Multiple similar-looking faces may require additional validation.
- Visitor IDs and event counts depend on the tracking and re-identification logic.
- Camera availability and video quality affect detection performance.
- The current development setup uses a local video source; an RTSP camera requires appropriate configuration and testing.
- Performance depends on the hardware, model size, and video-processing settings.
- Face images and embeddings are sensitive biometric data and should be handled securely with appropriate access controls and consent.

## 15. Future Improvements

- Improve recognition accuracy under different lighting conditions.
- Add support for configurable RTSP camera streams.
- Improve tracking and entry/exit event deduplication.
- Add more detailed visitor analytics and report exports.
- Improve database configuration and deployment.
- Add secure authentication and access controls to the dashboard.
- Evaluate performance on different hardware configurations.

## 16. Project Demo

**Video demonstration:** [Watch the project demo](PASTE_YOUR_DEMO_VIDEO_LINK_HERE)

The demonstration covers the face-tracking application, visitor recognition, database records, saved images, logs, and dashboard.

Replace the placeholder with the actual publicly accessible or unlisted video link before submission.

## 17. Author and Repository

**GitHub Repository:** https://github.com/jeevi-7/katomaran-face-tracker

This project was developed as a hackathon project to demonstrate an integrated face-tracking and visitor-management workflow.

---

This project is a part of a hackathon run by https://katomaran.com
