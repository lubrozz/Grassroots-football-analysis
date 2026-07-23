# Grassroots Football Analysis

A master's thesis project focused on developing an AI-assisted football analysis platform for grassroots clubs. The system processes match recordings, detects and tracks players, and generates analytics to support coaches.

## Project Structure
```
backend/
├── src/
│   ├── analytics/
│   ├── api/
│   ├── calibration/
│   ├── config/
│   ├── core/
│   ├── detection/
│   ├── tracking/
│   ├── utils/
│   └── video/
├── data/
│   ├── input/
│   ├── output/
│   └── sample/
├── models/
├── tests/
├── main.py
└── requirements.txt
```

## Requirements
* Python 3.12 (recommended)
* Git

## Clone the Repository
```
git clone <repository-url>
cd Grassroots-football-analysis/backend
```

## Create a Virtual Environment
### Windows
```
python -m venv .venv
```

Activate it:

### PowerShell
```
.venv\Scripts\Activate.ps1
```

###  Command Prompt
```
.venv\Scripts\activate.bat
```

## Install Dependencies
```
pip install -r requirements.txt
```

If requirements.txt has not yet been generated, install the required packages manually and then save them:
```
pip install opencv-python numpy
pip freeze > requirements.txt
```

## Running the Application

From the backend directory:
```
python main.py
```
The current implementation:

* Loads a video from data/input/
* Reads the video's metadata
* Prints metadata to the terminal
* Reads every frame to verify the video pipeline

## Input Videos

Place test videos inside:

`backend/data/input/`

Example:
```
backend/data/input/
└── veo-tracker-highlights-test.mp4
```

Update the path in main.py if using a different filename.

## Current Features
* Video loading
* Metadata extraction
* Frame iteration
* Project architecture
* Type-safe VideoMetadata dataclass

## Planned Features
* Video player
* Video writer
* Player detection (YOLO)
* Player tracking
* Pitch calibration
* Football analytics
* REST API
* React frontend

## Development Workflow

Development follows a feature-branch workflow.

Example:
```
development
│
├── feature/video-loader
├── feature/video-player
├── feature/yolo-detection
└── feature/player-tracking
```

Each feature is developed in its own branch and merged into development after testing.

## License

This project is developed as part of a Master's thesis and is currently intended for research and educational purposes.