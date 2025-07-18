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

## Descriptions
1  Star (Very Negative)
"A group of armed men broke into my house last night and threatened my family. We are traumatized and scared for our lives."
"There was a violent robbery at the shop near my home. The attackers beat up the cashier and stole everything."
"Someone snatched my phone in town. I tried to shout for help but nobody responded."
"My car was vandalized in the parking lot. This is the second time it has happened this month."

2 Stars (Neutral)
"I saw a suspicious person loitering around the estate. Not sure if anything happened, but it looked odd."
"There was a minor argument between two people on the street, but it didn’t escalate."

5 Stars (Very Positive)
"A lost child was found near the bus stop and safely reunited with their parents."
"I witnessed a traffic accident, but thankfully no one was seriously injured and help arrived quickly."
"A community group organized a clean-up and safety patrol in our neighborhood, making everyone feel safer."
"I saw police officers helping an elderly woman cross the busy road. It was heartwarming to see such kindness."
