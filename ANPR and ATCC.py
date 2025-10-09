"""
ANPR & ATCC System - Professional Edition v5.0
Enhanced dashboard for vehicle detection and license plate recognition

Features:
- Black theme with professional design
- Real-time vehicle detection
- Enhanced OCR for license plates
- Analytics dashboard
- Export functionality

Run: streamlit run anpr_app.py
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64
from datetime import datetime
import io

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="ANPR & ATCC System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING - BLACK THEME
# ===========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* BLACK BACKGROUND */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%) !important;
        padding: 2rem !important;
    }
    
    .stApp {
        background: #000000 !important;
    }
    
    .block-container {
        background: #000000 !important;
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
    
    .main * {
        color: #ffffff !important;
    }
    
    .element-container {
        color: #ffffff !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .app-header h1 {
        color: white !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
        margin: 10px 0 0 0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #FFD700 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Input fields styling */
    input, select, textarea {
        background-color: rgba(20, 20, 20, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        padding: 10px !important;
        font-weight: 500 !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2) !important;
    }
    
    /* Select boxes */
    [data-baseweb="select"] {
        background-color: rgba(20, 20, 20, 0.9) !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: rgba(20, 20, 20, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(102, 126, 234, 0.15) !important;
        border: 3px dashed #667eea !important;
        border-radius: 15px !important;
        padding: 20px !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        border-color: #FFD700 !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 35px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
    
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%) !important;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%) !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 35px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4) !important;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(72, 187, 120, 0.6) !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        border-left: 5px solid #667eea !important;
        padding: 15px !important;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: rgba(72, 187, 120, 0.2) !important;
        border-left-color: #48bb78 !important;
    }
    
    .stError {
        background: rgba(245, 101, 101, 0.2) !important;
        border-left-color: #f56565 !important;
    }
    
    .stInfo {
        background: rgba(66, 153, 225, 0.2) !important;
        border-left-color: #4299e1 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(10, 10, 10, 0.8) !important;
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border-color: transparent !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 10px !important;
    }
    
    .stRadio [role="radiogroup"] label {
        background: rgba(102, 126, 234, 0.1) !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease;
        font-weight: 600 !important;
    }
    
    .stRadio [role="radiogroup"] label:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: #667eea !important;
        transform: translateY(-2px);
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Slider styling */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 15px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .dataframe tbody tr {
        background: rgba(10, 10, 10, 0.9) !important;
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        transform: scale(1.01);
    }
    
    .dataframe tbody td {
        color: #ffffff !important;
        padding: 12px !important;
        border-bottom: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.3) !important;
        margin: 20px 0 !important;
    }
    
    /* Make headings visible */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Paragraph text */
    p {
        color: rgba(255, 255, 255, 0.9) !important;
        line-height: 1.6 !important;
    }
    
    /* Image captions */
    .caption {
        color: rgba(255, 255, 255, 0.8) !important;
        font-style: italic;
    }
    
    /* Plate detection box */
    .plate-box {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.2) 0%, rgba(255, 165, 0, 0.2) 100%);
        border: 3px solid #FFD700;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
    }
    
    .plate-text {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #FFD700 !important;
        letter-spacing: 2px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Number input */
    [data-testid="stNumberInput"] input {
        background-color: rgba(20, 20, 20, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
    }
    
    /* Text input */
    [data-testid="stTextInput"] input {
        background-color: rgba(20, 20, 20, 0.9) !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
    }
    
    /* Column styling */
    [data-testid="column"] {
        background: rgba(10, 10, 10, 0.5);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Remove white padding around columns */
    .row-widget {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# MOTION TRACKER CLASS
# ===========================
class AdvancedMotionTracker:
    """Enhanced centroid-based tracker with unique ID assignment"""
    def __init__(self, max_displacement=75, life_threshold=40):
        self._next_id = 1
        self._centroids = {}
        self._lost_counter = {}
        self._history = {}
        self.max_displacement = max_displacement
        self.life_threshold = life_threshold

    def _register(self, centroid):
        self._centroids[self._next_id] = centroid
        self._lost_counter[self._next_id] = 0
        self._history[self._next_id] = [centroid]
        self._next_id += 1
        return self._next_id - 1

    def _deregister(self, obj_id):
        del self._centroids[obj_id]
        del self._lost_counter[obj_id]
        if obj_id in self._history:
            del self._history[obj_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self._lost_counter.keys()):
                self._lost_counter[obj_id] += 1
                if self._lost_counter[obj_id] > self.life_threshold:
                    self._deregister(obj_id)
            return dict(self._centroids)

        input_centroids = []
        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))

        if not self._centroids:
            for c in input_centroids:
                self._register(c)
            return dict(self._centroids)

        ids = list(self._centroids.keys())
        existing = list(self._centroids.values())
        D = np.zeros((len(existing), len(input_centroids)), dtype=float)
        
        for i, e in enumerate(existing):
            for j, c in enumerate(input_centroids):
                D[i, j] = np.hypot(e[0] - c[0], e[1] - c[1])

        assigned_existing = set()
        assigned_input = set()
        rows, cols = np.unravel_index(np.argsort(D, axis=None), D.shape)
        
        for r, c in zip(rows, cols):
            if r in assigned_existing or c in assigned_input:
                continue
            if D[r, c] > self.max_displacement:
                continue
            
            obj_id = ids[r]
            self._centroids[obj_id] = input_centroids[c]
            self._lost_counter[obj_id] = 0
            self._history[obj_id].append(input_centroids[c])
            assigned_existing.add(r)
            assigned_input.add(c)

        for idx, c in enumerate(input_centroids):
            if idx not in assigned_input:
                self._register(c)

        for idx, obj_id in enumerate(ids):
            if idx not in assigned_existing:
                self._lost_counter[obj_id] += 1
                if self._lost_counter[obj_id] > self.life_threshold:
                    self._deregister(obj_id)

        return dict(self._centroids)

# ===========================
# UTILITY FUNCTIONS
# ===========================
@st.cache_resource
def load_yolo_model(model_path=None):
    """Load YOLO model with caching"""
    try:
        if model_path:
            return YOLO(model_path)
        try:
            return YOLO('yolov8n.pt')
        except:
            return YOLO('yolo11n.pt')
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

def enhance_plate_image(crop):
    """Advanced plate image preprocessing"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if max(h, w) < 200:
        scale = max(3, int(300 / max(h, w)))
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return [thresh1, thresh2, thresh3, gray]

def extract_plate_text(plate_img):
    """Extract text from plate using multiple OCR attempts"""
    if plate_img is None or plate_img.size == 0:
        return ""
    
    try:
        processed_images = enhance_plate_image(plate_img)
        configs = [
            "--psm 7 --oem 3 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 8 --oem 3 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 6 --oem 3 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 11 --oem 3 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ]
        
        best_text = ""
        max_confidence = 0
        
        for img in processed_images:
            for config in configs:
                try:
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    text_parts = []
                    confidences = []
                    
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 30:
                            text = data['text'][i]
                            if text.strip():
                                text_parts.append(text)
                                confidences.append(int(conf))
                    
                    if text_parts:
                        combined_text = "".join(text_parts)
                        avg_conf = sum(confidences) / len(confidences)
                        
                        if avg_conf > max_confidence and len(combined_text) >= 4:
                            best_text = combined_text
                            max_confidence = avg_conf
                except:
                    continue
        
        best_text = "".join([c for c in best_text.upper() if c.isalnum()])
        return best_text if len(best_text) >= 4 else ""
        
    except Exception as e:
        return ""

def display_plate_detections(detections, frame):
    """Display detected license plates with extracted text"""
    plates_found = []
    
    for det in detections:
        if det.get("plate_text") and len(det["plate_text"]) >= 4:
            x1, y1, x2, y2 = det["box"]
            try:
                plate_crop = frame[y1:y2, x1:x2]
                plates_found.append({
                    "text": det["plate_text"],
                    "image": plate_crop,
                    "confidence": det["conf"],
                    "vehicle_id": det.get("id", "N/A")
                })
            except:
                pass
    
    return plates_found

def draw_detections(frame, detections, class_names):
    """Draw beautiful detection boxes and labels"""
    output = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls = det["cls"]
        conf = det["conf"]
        
        if "car" in class_names.get(cls, "").lower():
            color = (102, 255, 102)
        elif "truck" in class_names.get(cls, "").lower():
            color = (255, 102, 102)
        elif "bus" in class_names.get(cls, "").lower():
            color = (255, 204, 102)
        else:
            color = (102, 126, 234)
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 4)
        
        label_text = f"{class_names.get(cls, str(cls))} {conf:.2f}"
        if det.get("id"):
            label_text += f" | ID:{det['id']}"
        
        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - label_h - 15), (x1 + label_w + 15, y1), color, -1)
        cv2.putText(output, label_text, (x1 + 7, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        if det.get("plate_text"):
            plate_text = det["plate_text"]
            text_width = len(plate_text) * 18 + 20
            cv2.rectangle(output, (x1, y2 + 5), (x1 + text_width, y2 + 40), (0, 215, 255), -1)
            cv2.rectangle(output, (x1, y2 + 5), (x1 + text_width, y2 + 40), (255, 255, 255), 2)
            cv2.putText(output, plate_text, (x1 + 10, y2 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3)
    
    return output

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if "tracker" not in st.session_state:
    st.session_state["tracker"] = AdvancedMotionTracker(max_displacement=80, life_threshold=40)

if "detection_logs" not in st.session_state:
    st.session_state["detection_logs"] = []

if "unique_vehicles" not in st.session_state:
    st.session_state["unique_vehicles"] = set()

if "class_distribution" not in st.session_state:
    st.session_state["class_distribution"] = {}

if "latest_plates" not in st.session_state:
    st.session_state["latest_plates"] = []

# ===========================
# APP HEADER
# ===========================
st.markdown("""
<div class="app-header">
    <h1>🚗 ANPR & ATCC System</h1>
    <p>🔍 Automatic Number Plate Recognition & Traffic Control Center</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# SIDEBAR CONFIGURATION
# ===========================
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")
    st.markdown("---")
    
    st.markdown("### 🤖 Model Settings")
    uploaded_model = st.file_uploader("Upload Custom YOLO Model (.pt)", type=["pt"], help="Upload your custom trained model")
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.25, 0.05, help="Minimum confidence for detections")
    inference_size = st.selectbox("📐 Inference Size", [416, 640, 896, 1280], index=1, help="Larger size = better accuracy but slower")
    
    st.markdown("---")
    
    st.markdown("### 🔍 OCR Settings")
    enable_ocr = st.checkbox("✅ Enable License Plate OCR", value=True, help="Extract text from license plates")
    show_ocr_debug = st.checkbox("🐛 Show OCR Debug Info", value=False, help="Display OCR preprocessing steps")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Tracker Settings")
    max_movement = st.slider("🏃 Max Movement (pixels)", 20, 200, 80, 10, help="Maximum distance a vehicle can move between frames")
    tracker_life = st.slider("⏱️ Tracker Life (frames)", 10, 100, 40, 5, help="How long to keep tracking lost vehicles")
    
    st.markdown("---")
    
    st.markdown("### 💾 Export Settings")
    auto_save = st.checkbox("💾 Auto-save Logs", value=True)
    log_filename = st.text_input("📁 Log Filename", value="anpr_logs.csv")
    
    st.markdown("---")
    st.markdown("### 📊 Live Statistics")
    st.metric("🚗 Total Detections", len(st.session_state["detection_logs"]))
    st.metric("🆔 Unique Vehicles", len(st.session_state["unique_vehicles"]))
    plates_detected = sum(1 for log in st.session_state["detection_logs"] if log.get("plate_text", ""))
    st.metric("🔤 Plates Read", plates_detected)
    
    if st.session_state["latest_plates"]:
        st.markdown("---")
        st.markdown("### 🔤 Recent Plates")
        for plate_info in st.session_state["latest_plates"][-5:]:
            st.markdown(f"""
            <div class="plate-box">
                <div class="plate-text">{plate_info['text']}</div>
                <small>ID: {plate_info.get('vehicle_id', 'N/A')} | Conf: {plate_info.get('confidence', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

# ===========================
# LOAD MODEL
# ===========================
if uploaded_model:
    temp_path = Path("temp_model.pt")
    temp_path.write_bytes(uploaded_model.read())
    model = load_yolo_model(str(temp_path))
else:
    model = load_yolo_model()

if "tracker" in st.session_state:
    st.session_state["tracker"].max_displacement = max_movement
    st.session_state["tracker"].life_threshold = tracker_life

# ===========================
# MAIN CONTENT TABS - ONLY 3 TABS NOW
# ===========================
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📊 Analytics Dashboard", "💾 Export Data"])

# ===========================
# FRAME PROCESSING FUNCTION
# ===========================
def process_detection_frame(frame):
    """Process frame and return detections with annotations"""
    detections = []
    
    if model is None:
        return detections, frame
    
    try:
        results = model.predict(source=frame, conf=confidence_threshold, imgsz=inference_size, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        for i in range(len(boxes)):
            xy = boxes.xyxy[i]
            x1, y1, x2, y2 = map(int, xy)
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])
            
            plate_text = ""
            if enable_ocr and (y2 - y1) > 15 and (x2 - x1) > 15:
                try:
                    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if crop.size > 0:
                        plate_text = extract_plate_text(crop)
                        
                        if show_ocr_debug and plate_text:
                            st.sidebar.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 
                                           caption=f"✅ Detected: {plate_text}", 
                                           width=150)
                except Exception as e:
                    if show_ocr_debug:
                        st.sidebar.error(f"❌ OCR Error: {str(e)}")
            
            detections.append({
                "box": [x1, y1, x2, y2],
                "cls": cls,
                "conf": conf,
                "plate_text": plate_text
            })
        
        boxes_list = [d["box"] for d in detections]
        tracker_output = st.session_state["tracker"].update(boxes_list)
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            min_dist = float("inf")
            assigned_id = None
            
            for obj_id, (tx, ty) in tracker_output.items():
                dist = np.hypot(cx - tx, cy - ty)
                if dist < min_dist:
                    min_dist = dist
                    assigned_id = obj_id
            
            if assigned_id and min_dist < max_movement:
                det["id"] = assigned_id
                st.session_state["unique_vehicles"].add(assigned_id)
                
                class_name = model.names.get(det["cls"], str(det["cls"]))
                if class_name not in st.session_state["class_distribution"]:
                    st.session_state["class_distribution"][class_name] = set()
                st.session_state["class_distribution"][class_name].add(assigned_id)
                
                if det["plate_text"]:
                    plate_info = {
                        "text": det["plate_text"],
                        "vehicle_id": assigned_id,
                        "confidence": det["conf"],
                        "timestamp": datetime.now()
                    }
                    if not any(p["text"] == plate_info["text"] and p["vehicle_id"] == plate_info["vehicle_id"] 
                             for p in st.session_state["latest_plates"][-10:]):
                        st.session_state["latest_plates"].append(plate_info)
        
        annotated_frame = draw_detections(frame, detections, model.names if model else {})
        
        return detections, annotated_frame
    
    except Exception as e:
        st.error(f"❌ Detection error: {str(e)}")
        return detections, frame

# ===========================
# TAB 1: LIVE DETECTION
# ===========================
with tab1:
    st.markdown("## 🎥 Real-Time Vehicle & License Plate Detection")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        detection_mode = st.radio(
            "🔹 Select Input Source:",
            ["📷 Upload Image", "🎬 Upload Video", "📹 Webcam"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if detection_mode == "📷 Upload Image":
            uploaded_image = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png", "bmp"], 
                                             help="Upload a clear image with visible vehicles and license plates")
            
            if uploaded_image:
                if st.button("🚀 START DETECTION", use_container_width=True):
                    image_bytes = uploaded_image.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        with st.spinner("🔍 Detecting vehicles and reading license plates..."):
                            detections, annotated = process_detection_frame(frame)
                            
                            timestamp = datetime.now()
                            for det in detections:
                                st.session_state["detection_logs"].append({
                                    "timestamp": timestamp,
                                    "source": "image",
                                    "class": det["cls"],
                                    "confidence": det["conf"],
                                    "vehicle_id": det.get("id"),
                                    "plate_text": det.get("plate_text", "")
                                })
                            
                            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                   caption="✅ Detection Results", 
                                   use_container_width=True)
                            st.success(f"✅ Successfully detected {len(detections)} objects!")
                            
                            plates = display_plate_detections(detections, frame)
                            if plates:
                                st.markdown("---")
                                st.markdown("### 🔍 Detected License Plates")
                                
                                cols_per_row = 3
                                for idx in range(0, len(plates), cols_per_row):
                                    cols = st.columns(cols_per_row)
                                    for col_idx, col in enumerate(cols):
                                        plate_idx = idx + col_idx
                                        if plate_idx < len(plates):
                                            plate_data = plates[plate_idx]
                                            with col:
                                                st.markdown(f"""
                                                <div class="plate-box">
                                                    <div class="plate-text">{plate_data['text']}</div>
                                                    <p>Confidence: {plate_data['confidence']:.2f}</p>
                                                    <p>Vehicle ID: {plate_data['vehicle_id']}</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                st.image(cv2.cvtColor(plate_data["image"], cv2.COLOR_BGR2RGB), 
                                                        use_container_width=True)
                            else:
                                st.info("ℹ️ No license plate text detected. Ensure good lighting and clear plate visibility.")
                    else:
                        st.error("❌ Failed to decode image. Please try another file.")
        
        elif detection_mode == "🎬 Upload Video":
            uploaded_video = st.file_uploader("📤 Choose a video file", type=["mp4", "avi", "mov", "mkv"],
                                             help="Upload a video with vehicles for batch processing")
            
            if uploaded_video:
                if st.button("🚀 PROCESS VIDEO", use_container_width=True):
                    temp_video = Path("temp_video.mp4")
                    temp_video.write_bytes(uploaded_video.read())
                    
                    cap = cv2.VideoCapture(str(temp_video))
                    if not cap.isOpened():
                        st.error("❌ Could not open video file")
                    else:
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        plates_display = st.empty()
                        
                        frame_count = 0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        detected_plates_video = []
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            
                            if frame_count % 3 == 0:
                                detections, annotated = process_detection_frame(frame)
                                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                                      use_container_width=True)
                                
                                for det in detections:
                                    st.session_state["detection_logs"].append({
                                        "timestamp": datetime.now(),
                                        "source": "video",
                                        "class": det["cls"],
                                        "confidence": det["conf"],
                                        "vehicle_id": det.get("id"),
                                        "plate_text": det.get("plate_text", "")
                                    })
                                    
                                    if det.get("plate_text"):
                                        detected_plates_video.append(det["plate_text"])
                                
                                if detected_plates_video:
                                    unique_plates = list(set(detected_plates_video[-10:]))
                                    plates_display.success(f"🔤 Plates detected: {', '.join(unique_plates)}")
                            
                            progress_bar.progress(min(frame_count / total_frames, 1.0))
                            status_text.info(f"🎬 Processing frame {frame_count}/{total_frames}")
                        
                        cap.release()
                        st.success(f"✅ Video processing complete! Processed {frame_count} frames")
                        
                        if detected_plates_video:
                            st.markdown("### 📋 All Detected Plates in Video")
                            unique_plates = list(set(detected_plates_video))
                            st.markdown(f"**Total Unique Plates:** {len(unique_plates)}")
                            st.write(", ".join(unique_plates))
        
        elif detection_mode == "📹 Webcam":
            st.info("📹 Click 'Start Webcam' to begin real-time detection")
            
            webcam_col1, webcam_col2 = st.columns(2)
            
            with webcam_col1:
                start_webcam = st.button("▶️ START WEBCAM", use_container_width=True)
            with webcam_col2:
                stop_webcam = st.button("⏹️ STOP WEBCAM", use_container_width=True)
            
            if start_webcam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not access webcam. Please check permissions and connections.")
                else:
                    webcam_placeholder = st.empty()
                    webcam_plates = st.empty()
                    
                    st.success("✅ Webcam active! Press 'Stop Webcam' to end session.")
                    
                    frame_counter = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_counter += 1
                        
                        if frame_counter % 2 == 0:
                            detections, annotated = process_detection_frame(frame)
                            webcam_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                                    use_container_width=True)
                            
                            for det in detections:
                                st.session_state["detection_logs"].append({
                                    "timestamp": datetime.now(),
                                    "source": "webcam",
                                    "class": det["cls"],
                                    "confidence": det["conf"],
                                    "vehicle_id": det.get("id"),
                                    "plate_text": det.get("plate_text", "")
                                })
                            
                            if st.session_state["latest_plates"]:
                                recent_plates_text = [p["text"] for p in st.session_state["latest_plates"][-5:]]
                                if recent_plates_text:
                                    webcam_plates.success(f"🔤 Recent plates: {', '.join(recent_plates_text)}")
                        
                        if stop_webcam:
                            break
                        
                        time.sleep(0.03)
                    
                    cap.release()
                    st.success("✅ Webcam session ended successfully")
    
    with col2:
        st.markdown("### 📊 Live Statistics")
        st.metric("🚗 Active Detections", len(st.session_state["detection_logs"]))
        st.metric("🆔 Unique Vehicles", len(st.session_state["unique_vehicles"]))
        plates_count = sum(1 for log in st.session_state["detection_logs"] if log.get("plate_text", ""))
        st.metric("🔤 Plates Read", plates_count)
        
        if st.session_state["class_distribution"]:
            st.markdown("---")
            st.markdown("### 🚗 Vehicle Types")
            for vehicle_type, ids in st.session_state["class_distribution"].items():
                st.metric(vehicle_type.capitalize(), len(ids))

# ===========================
# TAB 2: ANALYTICS DASHBOARD
# ===========================
with tab2:
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("---")
    
    if len(st.session_state["detection_logs"]) == 0:
        st.info("👋 No data yet! Start detecting vehicles in the Live Detection tab.")
    else:
        df_logs = pd.DataFrame(st.session_state["detection_logs"])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: white; text-align: center;'>Total Detections</h3>
                <h1 style='color: #667eea; text-align: center;'>{}</h1>
            </div>
            """.format(len(df_logs)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: white; text-align: center;'>Unique Vehicles</h3>
                <h1 style='color: #764ba2; text-align: center;'>{}</h1>
            </div>
            """.format(len(st.session_state["unique_vehicles"])), unsafe_allow_html=True)
        
        with col3:
            plates_detected = df_logs["plate_text"].astype(str).str.len() >= 4
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: white; text-align: center;'>Plates Read</h3>
                <h1 style='color: #48bb78; text-align: center;'>{}</h1>
            </div>
            """.format(plates_detected.sum()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: white; text-align: center;'>Vehicle Types</h3>
                <h1 style='color: #ed8936; text-align: center;'>{}</h1>
            </div>
            """.format(len(st.session_state["class_distribution"])), unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚗 Vehicle Distribution")
            class_counts = {k: len(v) for k, v in st.session_state["class_distribution"].items()}
            if class_counts:
                fig = px.pie(
                    values=list(class_counts.values()),
                    names=list(class_counts.keys()),
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                    hole=0.4
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Detection Timeline")
            df_time = df_logs.copy()
            df_time['hour'] = pd.to_datetime(df_time['timestamp']).dt.hour
            timeline = df_time.groupby('hour').size().reset_index(name='count')
            
            fig = px.bar(
                timeline,
                x='hour',
                y='count',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Hour of Day",
                yaxis_title="Number of Detections"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔤 License Plate Analytics")
        
        df_plates = df_logs[df_logs["plate_text"].astype(str).str.len() >= 4].copy()
        
        if not df_plates.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Most Frequent Plates")
                top_plates = df_plates["plate_text"].value_counts().head(10)
                fig = px.bar(
                    x=top_plates.index,
                    y=top_plates.values,
                    labels={'x': 'License Plate', 'y': 'Count'},
                    color_discrete_sequence=['#FFD700']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="License Plate Number",
                    yaxis_title="Detection Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🕐 Recent Unique Plates")
                unique_plates = df_plates.drop_duplicates(subset=['plate_text'], keep='last')
                display_df = unique_plates[['timestamp', 'plate_text', 'confidence', 'vehicle_id']].tail(10).copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
                st.dataframe(display_df, use_container_width=True, height=350)
        else:
            st.info("ℹ️ No license plates with text detected yet. Enable OCR and upload images with visible plates.")
        
        st.markdown("---")
        
        st.markdown("### 📋 Recent Detection Logs")
        df_display = df_logs.tail(100).copy()
        df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_display, use_container_width=True, height=400)

# ===========================
# TAB 3: EXPORT DATA
# ===========================
with tab3:
    st.markdown("## 💾 Export Detection Data")
    st.markdown("---")
    
    if len(st.session_state["detection_logs"]) == 0:
        st.info("👋 No data to export yet! Start detecting vehicles first.")
    else:
        df_export = pd.DataFrame(st.session_state["detection_logs"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Data Preview")
            st.dataframe(df_export.head(100), use_container_width=True, height=400)
            
            st.markdown("### 📈 Export Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("📝 Total Records", len(df_export))
            with stats_col2:
                date_range = f"{df_export['timestamp'].min().date()} to {df_export['timestamp'].max().date()}"
                st.metric("📅 Date Range", date_range if len(date_range) < 30 else "Multiple days")
            with stats_col3:
                plates_count = (df_export['plate_text'].astype(str).str.len() >= 4).sum()
                st.metric("🔤 Plates Captured", plates_count)
        
        with col2:
            st.markdown("### ⚙️ Export Options")
            
            export_format = st.selectbox(
                "📁 Export Format",
                ["CSV", "JSON", "Excel"]
            )
            
            include_all = st.checkbox("✅ Include All Fields", value=True)
            
            if not include_all:
                fields_to_export = st.multiselect(
                    "Select Fields to Export",
                    df_export.columns.tolist(),
                    default=df_export.columns.tolist()
                )
            else:
                fields_to_export = df_export.columns.tolist()
            
            custom_filename = st.text_input(
                "📄 Custom Filename",
                value=f"anpr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            st.markdown("---")
            
            if export_format == "CSV":
                csv_data = df_export[fields_to_export].to_csv(index=False)
                st.download_button(
                    label="📥 DOWNLOAD CSV",
                    data=csv_data,
                    file_name=f"{custom_filename}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "JSON":
                json_data = df_export[fields_to_export].to_json(orient='records', date_format='iso', indent=2)
                st.download_button(
                    label="📥 DOWNLOAD JSON",
                    data=json_data,
                    file_name=f"{custom_filename}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_export[fields_to_export].to_excel(writer, index=False, sheet_name='Detections')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 DOWNLOAD EXCEL",
                    data=excel_data,
                    file_name=f"{custom_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            if auto_save:
                save_path = Path(log_filename)
                try:
                    df_export.to_csv(save_path, index=False)
                    st.success(f"✅ Auto-saved to: {save_path.name}")
                except:
                    st.warning("⚠️ Auto-save failed")
            
            st.markdown("---")
            
            if st.button("🗑️ CLEAR ALL DATA", use_container_width=True, type="secondary"):
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ YES, CLEAR", use_container_width=True):
                        st.session_state["detection_logs"] = []
                        st.session_state["unique_vehicles"] = set()
                        st.session_state["class_distribution"] = {}
                        st.session_state["latest_plates"] = []
                        st.session_state["tracker"] = AdvancedMotionTracker()
                        st.success("✅ All data cleared!")
                        st.rerun()
                with confirm_col2:
                    st.button("❌ CANCEL", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📄 Generate Report")
        
        report_cols = st.columns(3)
        
        with report_cols[0]:
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Detailed Analysis", "License Plate Report"]
            )
        
        with report_cols[1]:
            report_format_choice = st.selectbox(
                "Report Format",
                ["Markdown", "HTML", "Plain Text"]
            )
        
        with report_cols[2]:
            if st.button("📝 GENERATE REPORT", use_container_width=True):
                with st.spinner("📝 Generating report..."):
                    time.sleep(1)
                    
                    plates_df = df_export[df_export['plate_text'].astype(str).str.len() >= 4]
                    
                    report_content = f"""
# 🚗 ANPR & ATCC System Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_type}

---

## 📊 Executive Summary

### Key Metrics
- **Total Detections:** {len(df_export)}
- **Unique Vehicles:** {len(st.session_state['unique_vehicles'])}
- **License Plates Captured:** {len(plates_df)}
- **OCR Success Rate:** {(len(plates_df)/len(df_export)*100):.2f}%

---

## 🚗 Vehicle Classification

"""
                    for vehicle_type, ids in st.session_state['class_distribution'].items():
                        report_content += f"- **{vehicle_type.capitalize()}:** {len(ids)} unique vehicles\n"
                    
                    report_content += f"""

---

## 🔤 Top 10 License Plates

"""
                    if not plates_df.empty:
                        top_plates = plates_df['plate_text'].value_counts().head(10)
                        for idx, (plate, count) in enumerate(top_plates.items(), 1):
                            report_content += f"{idx}. **{plate}** - Detected {count} time(s)\n"
                    else:
                        report_content += "No plates detected.\n"
                    
                    report_content += f"""

---

*Report generated by ANPR & ATCC System*
"""
                    
                    st.markdown(report_content)
                    
                    if report_format_choice == "Markdown":
                        st.download_button(
                            label="📥 DOWNLOAD REPORT",
                            data=report_content,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    elif report_format_choice == "HTML":
                        html_content = f"<html><body style='background:#0a0a0a;color:#fff;padding:40px;'><pre>{report_content}</pre></body></html>"
                        st.download_button(
                            label="📥 DOWNLOAD REPORT",
                            data=html_content,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    else:
                        st.download_button(
                            label="📥 DOWNLOAD REPORT",
                            data=report_content,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
