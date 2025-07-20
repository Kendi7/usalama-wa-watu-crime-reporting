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
            image_sentiment = get_local_image_caption(img_path)
        report = {
            "description": description,
            "image": img_path,
            "location": location,
            "contact": contact,
            "sentiment": sentiment,
            "urgency": f"{urgency} {urgency_emoji}",
            "objects": detected_objects,
            "image_sentiment": image_sentiment
        }
        st.session_state['reports'].append(report)
        st.success("Report submitted!")
        st.markdown(f"**Sentiment Analysis:** {sentiment}")
        st.markdown(f"**Image Sentiment:** {image_sentiment}")
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
        # Filtering and search widgets
        st.subheader("Filter & Search Reports")
        search_text = st.text_input("Search (description, contact, location, objects)")
        location_options = df['location'].unique().tolist()
        urgency_options = df['urgency'].unique().tolist()
        # No default selection
        selected_locations = st.multiselect("Filter by Location", location_options)
        selected_urgencies = st.multiselect("Filter by Urgency", urgency_options)
        # If nothing is selected, show all
        if not selected_locations:
            selected_locations = location_options
        if not selected_urgencies:
            selected_urgencies = urgency_options

        # Apply filters
        filtered_df = df[
            df['location'].isin(selected_locations) &
            df['urgency'].isin(selected_urgencies)
        ]
        if search_text:
            mask = (
                filtered_df['description'].str.contains(search_text, case=False, na=False) |
                filtered_df['contact'].str.contains(search_text, case=False, na=False) |
                filtered_df['location'].str.contains(search_text, case=False, na=False) |
                filtered_df['objects'].str.contains(search_text, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        st.dataframe(filtered_df)
        # Show images as thumbnails in a custom table
        st.subheader("Reports with Images")
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Report {idx+1} - {row['location']}"):
                # Show image if available
                if row['image'] and os.path.exists(row['image']):
                    st.image(row['image'], caption="Uploaded Image", width=400)
                else:
                    st.write("No image uploaded.")
                # Show other details below the image
                details = f"""
                **Description:** {row['description']}  \n                **Location:** {row['location']}  \n                **Contact:** {row['contact']}  \n                **Sentiment:** {row['sentiment']}  \n                **Image Sentiment:** {row.get('image_sentiment', 'N/A')}  \n                **Urgency:** {row.get('urgency', '')}  \n                **Objects:** {row['objects']}  \n                """
                st.markdown(details)
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