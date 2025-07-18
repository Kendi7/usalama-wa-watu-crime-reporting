# Nairobi County Crime Reporting System (AI-Powered)

A simple Streamlit web app for reporting crimes in Nairobi County, powered by AI for sentiment analysis and object detection.

## Features
- Submit crime reports with text and optional images
- AI sentiment analysis of report text (coming soon)
- AI object detection in uploaded images (coming soon)
- Map and table view of all reports

## Setup Instructions

1. **Clone the repository**
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Go to `http://localhost:8501` in your browser
- Submit a crime report or view existing reports

## To Do
- Integrate AI sentiment analysis
- Integrate AI object detection
- Add map visualization 