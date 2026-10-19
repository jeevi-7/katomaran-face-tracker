\# Katomaran Face Tracker



AI-powered visitor face recognition and tracking system built with Python, InsightFace, OpenCV, and MySQL.



\## Project Overview



Katomaran Face Tracker detects faces in video, generates face embeddings, identifies registered visitors, tracks them across frames, and records visitor entry and exit events.



The system also provides a Streamlit dashboard for viewing visitor statistics and activity reports.



\## Features



\- Face detection using InsightFace and OpenCV

\- Face recognition using facial embeddings

\- Visitor tracking across video frames

\- Automatic visitor registration in MySQL

\- Entry and exit event logging

\- Cropped face image storage

\- Visitor activity reports

\- CSV report export

\- Streamlit dashboard and visitor analytics



\## Technology Stack



| Component | Technology |

|---|---|

| Programming language | Python |

| Face detection and recognition | InsightFace |

| Video processing | OpenCV |

| Numerical operations | NumPy |

| Database | MySQL |

| Dashboard | Streamlit |

| Reports | CSV and Pandas |



\## Project Structure



```text

katomaran-face-tracker/

├── .venv/

├── models/

│   └── yolov8n-face.pt

├── logs/

├── output\_images/

├── config.json

├── database.py

├── detect\_faces.py

├── tracker.py

├── recognize\_faces.py

├── recognize\_faces\_backup.py

├── test\_database.py

├── test\_tracker.py

├── visitor\_report.py

├── export\_report.py

├── dashboard.py

├── visitor\_analytics.py

├── sample.mp4

└── recognition\_tracking\_output.mp4

```



Generated files and folders may vary depending on testing.



\## Requirements



\- Windows 10 or Windows 11

\- Python installation compatible with the project's dependencies

\- MySQL Server

\- VS Code or another Python editor



\## Setup



\### 1. Open the project directory



```powershell

cd G:\\katomaran-face-tracker

```



\### 2. Activate the virtual environment



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script activation, use:



```powershell

.\\.venv\\Scripts\\python.exe --version

```



\### 3. Configure the database password



Set the MySQL root password in the current PowerShell session:



```powershell

$env:KATOMARAN\_DB\_PASSWORD = Read-Host "Enter MySQL password"

```



\### 4. Check the database connection



```powershell

python .\\database.py

```



The database `katomaran\_db` and its required tables must already be configured.



\## Run Face Recognition



```powershell

python .\\recognize\_faces.py

```



The configured video is processed, visitor events are stored in MySQL, and the processed video is saved as `recognition\_tracking\_output.mp4`.



Press `Q` in the video window to stop processing.



\## Run the Main Dashboard



```powershell

python -m streamlit run .\\dashboard.py

```



The dashboard displays visitor counts, entry and exit statistics, visitor activity, filters, and CSV downloads.



\## Run Visitor Analytics



```powershell

python -m streamlit run .\\visitor\_analytics.py

```



This page provides event statistics, activity summaries by visitor, and the latest recorded event.



\## Generate Reports



Display visitor events:



```powershell

python .\\visitor\_report.py

```



Export visitor events to CSV:



```powershell

python .\\export\_report.py

```



\## Output



The system generates:



\- Visitor records and face embeddings in MySQL

\- Entry and exit event records

\- Cropped face images under `output\_images/`

\- Application logs under `logs/`

\- A processed video file

\- A CSV visitor report



\## Security Notes



\- Do not commit database passwords, API keys, or other secrets.

\- Keep `.venv/`, generated videos, and unnecessary test images out of version control.

\- Use appropriate consent and access controls when processing facial data.



\## Future Improvements



\- Improve tracking reliability and reduce false entry/exit events

\- Optimize video processing performance

\- Add date-range filters and richer analytics

\- Improve dashboard usability

\- Add deployment documentation and automated tests



\## Project Status



The application has been tested locally with video input, MySQL event logging, image storage, CSV export, and Streamlit dashboards.



