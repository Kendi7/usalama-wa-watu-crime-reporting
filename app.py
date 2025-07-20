import streamlit as st
import pandas as pd
from PIL import Image
import os
from transformers import pipeline
import torch
from streamlit_folium import st_folium
import folium
from ultralytics import YOLO
import base64
import requests
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import sqlite3
import time

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

# Add local image captioning model loader and function
@st.cache_resource
def get_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

def get_local_image_caption(image_path):
    model, feature_extractor, tokenizer = get_captioning_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16)  # greedy decoding only
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def map_caption_to_sentiment(caption):
    negative_keywords = [
        'violence', 'injury', 'attack', 'gun', 'knife', 'blood', 'fire', 'explosion', 'fight', 'crying', 'sad', 'danger', 'accident', 'protest', 'riot', 'dead', 'death', 'hurt', 'wound'
    ]
    positive_keywords = [
        'smile', 'happy', 'safe', 'peace', 'joy', 'celebration', 'calm', 'help', 'rescue', 'hug', 'love'
    ]
    caption_lower = caption.lower()
    if any(word in caption_lower for word in negative_keywords):
        return "Very Negative"
    elif any(word in caption_lower for word in positive_keywords):
        return "Very Positive"
    else:
        return "Neutral"

def map_image_urgency(image_sentiment):
    sentiment_to_urgency = {
        'Very Negative': ('High', '🚨'),
        'Very Positive': ('Very Low', '🎉'),
        'Neutral': ('Medium/Low', '🟡'),
    }
    return sentiment_to_urgency.get(image_sentiment, ('Medium/Low', '🟡'))

def combine_sentiment_and_urgency(text_sentiment, text_urgency, text_urgency_emoji, image_sentiment, image_urgency, image_urgency_emoji):
    # Priority: High > Medium/Low > Very Low
    urgency_order = {'High': 3, 'Medium/Low': 2, 'Very Low': 1}
    # Pick the higher urgency
    if urgency_order.get(text_urgency, 2) >= urgency_order.get(image_urgency, 2):
        combined_urgency = text_urgency
        combined_urgency_emoji = text_urgency_emoji
    else:
        combined_urgency = image_urgency
        combined_urgency_emoji = image_urgency_emoji
    # Sentiment: Very Negative > Negative > Neutral > Positive > Very Positive
    sentiment_order = {
        'Very Negative': 1, 'Negative': 2, 'Neutral': 3, 'Positive': 4, 'Very Positive': 5
    }
    # Use the more negative sentiment
    if sentiment_order.get(text_sentiment.split()[0], 3) <= sentiment_order.get(image_sentiment.split()[0], 3):
        combined_sentiment = text_sentiment
    else:
        combined_sentiment = image_sentiment
    return combined_sentiment, combined_urgency, combined_urgency_emoji

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        image TEXT,
        location TEXT,
        contact TEXT,
        sentiment TEXT,
        urgency TEXT,
        objects TEXT,
        image_sentiment TEXT,
        image_caption TEXT,
        image_urgency TEXT
    )''')
    conn.commit()
    conn.close()
init_db()

def save_report_to_db(report):
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''INSERT INTO reports (description, image, location, contact, sentiment, urgency, objects, image_sentiment, image_caption, image_urgency)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (report['description'], report['image'], report['location'], report['contact'], report['sentiment'], report['urgency'], report['objects'], report.get('image_sentiment'), report.get('image_caption'), report.get('image_urgency')))
    conn.commit()
    conn.close()

def load_reports_from_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('SELECT description, image, location, contact, sentiment, urgency, objects, image_sentiment, image_caption, image_urgency FROM reports')
    rows = c.fetchall()
    conn.close()
    keys = ['description', 'image', 'location', 'contact', 'sentiment', 'urgency', 'objects', 'image_sentiment', 'image_caption', 'image_urgency']
    return [dict(zip(keys, row)) for row in rows]

# --- Auth ---
def login():
    if st.session_state.get('logged_in', False):
        if st.button('Logout', key='logout_button'):
            st.session_state['logged_in'] = False
            st.rerun()
        st.success('Login successful!')
        return True
    st.subheader('Login')
    username = st.text_input('Username', key='login_username')
    password = st.text_input('Password', type='password', key='login_password')
    if st.button('Login', key='login_button'):
        if username == 'admin' and password == 'admin123':
            st.session_state['logged_in'] = True
            st.success('Login successful!')
            st.rerun()
        else:
            st.error('Invalid credentials')
    return st.session_state.get('logged_in', False)

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Report Crime", "View Reports"])

if page == "Report Crime":
    # Show toast if flag is set
    if st.session_state.get('show_report_toast', False):
        st.toast("Report submitted!", icon="✅")
        st.session_state['show_report_toast'] = False
    # Clear form fields if reset flag is set
    if st.session_state.get('reset_crime_form', False):
        st.session_state['crime_form_description'] = ''
        st.session_state['crime_form_location'] = None
        st.session_state['crime_form_contact'] = ''
        st.session_state['reset_crime_form'] = False
    st.header("Report a Crime")
    with st.form("crime_form"):
        description = st.text_area("Describe the incident", help="What happened? Where? When? Any details?", key="crime_form_description")
        image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"], key="crime_form_image")
        location = st.selectbox("Select location (sub-county)", [
            "Westlands", "Kasarani", "Lang'ata", "Embakasi", "Starehe", "Dagoretti", "Kamukunji", "Makadara", "Mathare", "Kibra", "Roysambu", "Ruaraka", "Other"
        ], key="crime_form_location")
        contact = st.text_input("Contact info (optional)", key="crime_form_contact")
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
            # --- Custom logic for negative keywords ---
            negative_keywords = [
                'attack', 'kill', 'robbery', 'assault', 'theft', 'violence', 'murder', 'rape', 'shooting', 'stab',
                'injury', 'danger', 'threat', 'gun', 'knife', 'explosion', 'bomb', 'terror', 'crime', 'goon', 'gang',
                'thief', 'hijack', 'kidnap', 'abduct', 'arson', 'riot', 'fight', 'abuse', 'molest', 'harass', 'rape'
            ]
            if (sentiment_label in ['Positive', 'Very Positive'] and any(word in description.lower() for word in negative_keywords)):
                sentiment_label = 'Very Negative'
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
        image_sentiment = "No image uploaded"
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
            # Local image captioning (as sentiment proxy)
            caption = get_local_image_caption(img_path)
            image_sentiment = map_caption_to_sentiment(caption)
            image_urgency, image_urgency_emoji = map_image_urgency(image_sentiment)

        if image and description.strip():
            # Combine sentiment and urgency
            combined_sentiment, combined_urgency, combined_urgency_emoji = combine_sentiment_and_urgency(
                sentiment_label, urgency, urgency_emoji, image_sentiment, image_urgency, image_urgency_emoji
            )
            report = {
                "description": description,
                "image": img_path,
                "location": location,
                "contact": contact,
                "sentiment": combined_sentiment,
                "urgency": f"{combined_urgency} {combined_urgency_emoji}",
                "objects": detected_objects,
                "image_sentiment": image_sentiment,
                "image_caption": caption,
                "image_urgency": f"{image_urgency} {image_urgency_emoji}",
            }
            st.session_state['reports'].append(report)
            save_report_to_db(report)
            st.session_state['reset_crime_form'] = True
            st.session_state['show_report_toast'] = True
            st.rerun()
        else:
            report = {
                "description": description,
                "image": img_path,
                "location": location,
                "contact": contact,
                "sentiment": sentiment,
                "urgency": f"{urgency} {urgency_emoji}",
                "objects": detected_objects,
                "image_sentiment": image_sentiment,
                "image_caption": caption if image else None,
                "image_urgency": f"{image_urgency} {image_urgency_emoji}" if image else None
            }
            st.session_state['reports'].append(report)
            save_report_to_db(report)
            st.session_state['reset_crime_form'] = True
            st.session_state['show_report_toast'] = True
            st.rerun()

elif page == "View Reports":
    if not st.session_state.get('logged_in', False):
        if not login():
            st.stop()
    st.header("Crime Reports")
    reports = load_reports_from_db()
    if not reports:
        st.info("No reports yet.")
    else:
        df = pd.DataFrame(reports)
        # --- Filter Controls ---
        st.subheader("Filter Reports")
        col1, col2, col3 = st.columns(3)
        with col1:
            location_options = df['location'].unique().tolist()
            selected_locations = st.multiselect("Location", location_options)
        with col2:
            urgency_options = df['urgency'].unique().tolist()
            selected_urgencies = st.multiselect("Urgency", urgency_options)
        with col3:
            sentiment_options = df['sentiment'].unique().tolist()
            selected_sentiments = st.multiselect("Sentiment", sentiment_options)
        # If nothing is selected in any filter, show all
        if not selected_locations:
            selected_locations = location_options
        if not selected_urgencies:
            selected_urgencies = urgency_options
        if not selected_sentiments:
            selected_sentiments = sentiment_options
        # Apply filters
        filtered_df = df[
            df['location'].isin(selected_locations) &
            df['urgency'].isin(selected_urgencies) &
            df['sentiment'].isin(selected_sentiments)
        ]
        st.markdown("---")
        # --- Expander Layout ---
        for idx, row in filtered_df.iterrows():
            with st.expander(f"{row['location']} | {row['sentiment']} | {row['urgency']} | {row['description'][:30]}..."):
                card_cols = st.columns([1, 2])
                # Image section
                with card_cols[0]:
                    if row['image'] and os.path.exists(row['image']):
                        st.image(row['image'], caption="Uploaded Image", use_container_width=True)
                    else:
                        st.write("No image uploaded.")
                # Info section
                with card_cols[1]:
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Sentiment:** {row['sentiment']}")
                    st.markdown(f"**Urgency:** {row['urgency']}")
                    st.markdown(f"**Image Sentiment:** {row.get('image_sentiment', 'N/A')}")
                    st.markdown(f"**Image Urgency:** {row.get('image_urgency', 'N/A')}")
                    st.markdown(f"**Description:** {row['description']}")
                    st.markdown(f"**Image Caption:** {row.get('image_caption', 'N/A')}")
                    st.markdown(f"**Detected Objects:** {row['objects']}")
                    st.markdown(f"**Contact:** {row['contact']}")
            st.markdown("---")
        # --- Map Visualization (optional, keep at bottom) ---
        st.subheader("Map of Reports")
        nairobi_center = [-1.286389, 36.817223]
        m = folium.Map(location=nairobi_center, zoom_start=11)
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
        for idx, row in filtered_df.iterrows():
            coords = sub_county_coords.get(row['location'], nairobi_center)
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