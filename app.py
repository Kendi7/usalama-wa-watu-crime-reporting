import streamlit as st
import pandas as pd
from PIL import Image
import os
from transformers import pipeline
import torch
from streamlit_folium import st_folium
import folium
from ultralytics import YOLO

st.set_page_config(page_title="Nairobi Crime Reporting AI App", layout="wide")

st.title("Nairobi County Crime Reporting System (AI-Powered)")

# Initialize session state for reports
def init_reports():
    if 'reports' not in st.session_state:
        st.session_state['reports'] = []

init_reports()

# Load sentiment analysis pipeline once
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = get_sentiment_pipeline()

# Load YOLOv8 model once
@st.cache_resource
def get_yolo_model():
    return YOLO('yolov8n.pt')

yolo_model = get_yolo_model()

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Report Crime", "View Reports"])

if page == "Report Crime":
    st.header("Report a Crime")
    with st.form("crime_form"):
        description = st.text_area("Describe the incident", help="What happened? Where? When? Any details?")
        image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
        location = st.selectbox("Select location (sub-county)", [
            "Westlands", "Kasarani", "Lang'ata", "Embakasi", "Starehe", "Dagoretti", "Kamukunji", "Makadara", "Mathare", "Kibra", "Roysambu", "Ruaraka", "Other"
        ])
        contact = st.text_input("Contact info (optional)")
        submitted = st.form_submit_button("Submit Report")

    if submitted:
        # Run sentiment analysis
        if description.strip():
            sentiment_result = sentiment_pipeline(description)[0]
            label = sentiment_result['label']
            # Map star rating to sentiment
            star_to_sentiment = {
                '1 star': 'Very Negative',
                '2 stars': 'Negative',
                '3 stars': 'Neutral',
                '4 stars': 'Positive',
                '5 stars': 'Very Positive',
            }
            sentiment_label = star_to_sentiment.get(label, label)
            sentiment = f"{sentiment_label} (score: {sentiment_result['score']:.2f})"
            # Map sentiment to urgency and emoji
            sentiment_to_urgency = {
                'Very Negative': ('High', '🚨'),
                'Negative': ('Medium-High', '⚠️'),
                'Neutral': ('Medium/Low', '🟡'),
                'Positive': ('Low', '✅'),
                'Very Positive': ('Very Low', '🎉'),
            }
            urgency, urgency_emoji = sentiment_to_urgency.get(sentiment_label, ('Medium/Low', '🟡'))
        else:
            sentiment = "No description provided"
            urgency = "Unknown"
            urgency_emoji = "❓"
        # Run object detection if image is uploaded
        detected_objects = "No image uploaded"
        img_path = None
        if image:
            img_path = os.path.join("uploads", image.name)
            os.makedirs("uploads", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(image.read())
            # Object detection using ultralytics YOLO
            results = yolo_model(img_path)
            labels = set()
            for r in results:
                if hasattr(r, 'names') and hasattr(r, 'boxes'):
                    for c in r.boxes.cls:
                        labels.add(r.names[int(c)])
            detected_objects = ', '.join(labels) if labels else 'No objects detected'
        report = {
            "description": description,
            "image": img_path,
            "location": location,
            "contact": contact,
            "sentiment": sentiment,
            "urgency": f"{urgency} {urgency_emoji}",
            "objects": detected_objects
        }
        st.session_state['reports'].append(report)
        st.success("Report submitted!")
        st.markdown(f"**Sentiment Analysis:** {sentiment}")
        st.markdown(f"**Urgency Level:** {urgency} {urgency_emoji}")
        st.markdown(f"**Detected Objects:** {detected_objects}")
        st.write(report)

elif page == "View Reports":
    st.header("Crime Reports Map & Table")
    reports = st.session_state.get('reports', [])
    if not reports:
        st.info("No reports yet.")
    else:
        df = pd.DataFrame(reports)
        st.dataframe(df)
        # Map visualization
        # Nairobi center coordinates
        nairobi_center = [-1.286389, 36.817223]
        m = folium.Map(location=nairobi_center, zoom_start=11)
        # Sub-county coordinates (approximate)
        sub_county_coords = {
            "Westlands": [-1.2647, 36.8121],
            "Kasarani": [-1.2195, 36.8961],
            "Lang'ata": [-1.3621, 36.7517],
            "Embakasi": [-1.3341, 36.8947],
            "Starehe": [-1.2833, 36.8500],
            "Dagoretti": [-1.3081, 36.7381],
            "Kamukunji": [-1.2833, 36.8500],
            "Makadara": [-1.3000, 36.8667],
            "Mathare": [-1.2667, 36.8667],
            "Kibra": [-1.3171, 36.7924],
            "Roysambu": [-1.2100, 36.9000],
            "Ruaraka": [-1.2500, 36.8833],
            "Other": nairobi_center
        }
        for idx, row in df.iterrows():
            coords = sub_county_coords.get(row['location'], nairobi_center)
            # Color by sentiment
            sentiment = row.get('sentiment', '').lower()
            if 'negative' in sentiment or 'urgent' in sentiment:
                color = 'red'
            elif 'neutral' in sentiment:
                color = 'orange'
            elif 'positive' in sentiment:
                color = 'green'
            else:
                color = 'blue'
            popup = folium.Popup(f"<b>Description:</b> {row['description']}<br>"
                                 f"<b>Sentiment:</b> {row['sentiment']}<br>"
                                 f"<b>Urgency:</b> {row.get('urgency', '')}<br>"
                                 f"<b>Objects:</b> {row['objects']}<br>"
                                 f"<b>Contact:</b> {row['contact']}", max_width=300)
            folium.Marker(
                location=coords,
                popup=popup,
                icon=folium.Icon(color=color)
            ).add_to(m)
        st_folium(m, width=700, height=500) 