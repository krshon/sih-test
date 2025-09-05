import streamlit as st
import tempfile
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="🌱 Eco-Points Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Main background with dark green gradient */
    .main .block-container {
        background: linear-gradient(135deg, #0a0f0a, #0d1f0d, #1a331a);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        color: #e5e7eb;
    }
    
    /* Custom header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #065f46, #047857, #059669);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(5, 150, 105, 0.5);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header p {
        color: #d1fae5;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 400;
    }
    
    /* Upload area styling */
    .upload-section {
        background: #111827;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #10b981;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }
    
    .upload-section:hover {
        border-color: #34d399;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    
    /* Stats cards */
    .stats-card {
        background: #1f2937;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .points-display {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .points-label {
        color: #9ca3af;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    /* Activity cards */
    .activity-card {
        background: linear-gradient(135deg, #064e3b, #065f46);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.6);
    }
    
    .activity-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #6ee7b7;
        margin: 0 0 0.5rem 0;
    }
    
    .activity-points {
        color: #34d399;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Detection results styling */
    .detection-results {
        background: #1f2937;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
    }
    
    /* Tips section */
    .tips-section {
        background: linear-gradient(135deg, #1e293b, #111827);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border-left: 4px solid #facc15;
    }
    
    .tips-title {
        color: #facc15;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    .tip-item {
        color: #fef9c3;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .tip-item::before {
        content: "🌱";
        position: absolute;
        left: 0;
        top: 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.5);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #34d399, #10b981);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# Load YOLOv8 model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# Enhanced eco points mapping with descriptions
ECO_POINTS = {
    "bicycle": {"points": 5, "description": "Eco-friendly transportation", "icon": "🚲"},
    "potted plant": {"points": 5, "description": "Indoor air purification", "icon": "🪴"},
    "tree": {"points": 8, "description": "Carbon absorption champion", "icon": "🌳"},
    "person and bicycle": {"points": 12, "description": "Active green commuting", "icon": "🚴"},
    "person and tree": {"points": 15, "description": "Tree planting/care activity", "icon": "🌱"},
    "person and plant": {"points": 10, "description": "Gardening & plant care", "icon": "👨‍🌾"},
    "boat": {"points": 3, "description": "Water transportation", "icon": "⛵"}
}

def calculate_points(image_path):
    results = model(image_path)
    detected_labels = [results[0].names[int(box.cls)] for box in results[0].boxes]
    detected_set = set(detected_labels)
    score = 0
    activities_found = []

    # Check compound activities first
    compound_activities = [
        ("person", "bicycle", "person and bicycle"),
        ("person", "tree", "person and tree"),
        ("person", "potted plant", "person and plant")
    ]
    
    used_in_compound = set()
    
    for item1, item2, compound in compound_activities:
        if item1 in detected_set and item2 in detected_set:
            score += ECO_POINTS[compound]["points"]
            activities_found.append(compound)
            used_in_compound.update([item1, item2])

    # Add individual object points (avoid double-counting)
    for label in detected_labels:
        if label in ECO_POINTS and label not in used_in_compound:
            score += ECO_POINTS[label]["points"]
            if label not in [act for act in activities_found]:
                activities_found.append(label)

    return score, activities_found, results[0]

def annotate_image(image_path, yolo_result):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = ["#22c55e", "#16a34a", "#15803d", "#f59e0b", "#ef4444", "#8b5cf6"]
    
    for i, box in enumerate(yolo_result.boxes):
        cls = int(box.cls)
        label = yolo_result.names[cls]
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw label background
        label_text = f"{label} ({confidence:.2f})"
        bbox = draw.textbbox((x1, y1-30), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        
        # Draw label text
        draw.text((x1, y1-30), label_text, fill="white", font=font)
    
    return image

def create_points_chart(activities):
    if not activities:
        return None
        
    labels = []
    values = []
    icons = []
    
    for activity in activities:
        if activity in ECO_POINTS:
            labels.append(activity.replace("_", " ").title())
            values.append(ECO_POINTS[activity]["points"])
            icons.append(ECO_POINTS[activity]["icon"])
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f"{icon} {val} pts" for icon, val in zip(icons, values)],
            textposition='auto',
            marker_color=['#22c55e', '#16a34a', '#15803d', '#f59e0b', '#ef4444'][:len(values)]
        )
    ])
    
    fig.update_layout(
        title="🎯 Points Breakdown by Activity",
        xaxis_title="Eco Activities Detected",
        yaxis_title="Points Earned",
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12)
    )
    
    return fig

# Main App Layout
def main():
    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Eco-Points Detector</h1>
        <p>Earn points for eco-friendly activities! Upload photos of green actions and see your environmental impact.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🎯 How to Earn Points")
        
        for activity, data in ECO_POINTS.items():
            if "person and" not in activity:  # Show individual activities
                st.markdown(f"""
                **{data['icon']} {activity.replace('_', ' ').title()}**  
                *{data['points']} points* - {data['description']}
                """)
        
        st.markdown("### 🏆 Bonus Activities")
        for activity, data in ECO_POINTS.items():
            if "person and" in activity:  # Show compound activities
                st.markdown(f"""
                **{data['icon']} {activity.replace('_', ' ').title()}**  
                *{data['points']} points* - {data['description']}
                """)
        
        st.markdown("---")
        st.markdown("### 📊 Your Stats")
        if 'total_sessions' not in st.session_state:
            st.session_state.total_sessions = 0
        if 'total_points' not in st.session_state:
            st.session_state.total_points = 0
            
        st.metric("Total Sessions", st.session_state.total_sessions)
        st.metric("Total Points", st.session_state.total_points)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Your Eco Activity Photo</h3>
            <p>Take a photo of yourself biking, planting trees, gardening, or any eco-friendly activity!</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your eco-friendly activity for best results!"
        )
        
        if uploaded_file is not None:
            # Process the image
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = os.path.join(tmpdir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Calculate points and detect activities
                with st.spinner("🔍 Analyzing your eco-activity..."):
                    points, activities, yolo_result = calculate_points(temp_path)
                    annotated_img = annotate_image(temp_path, yolo_result)
                
                # Update session stats
                st.session_state.total_sessions += 1
                st.session_state.total_points += points
                
                # Display results
                if points > 0:
                    st.markdown(f"""
                    <div class="success-message">
                        🎉 Congratulations! You earned <strong>{points} Eco-Points</strong>! 🌟
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("🤔 No eco-activities detected. Try uploading a photo with bicycles, plants, trees, or people doing eco-friendly activities!")
                
                # Show annotated image
                st.image(annotated_img, caption="🔍 Detected Objects & Activities", use_column_width=True)
                
                # Show activities breakdown
                if activities:
                    st.markdown('<div class="detection-results">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Activities Detected")
                    
                    for activity in activities:
                        if activity in ECO_POINTS:
                            data = ECO_POINTS[activity]
                            st.markdown(f"""
                            <div class="activity-card">
                                <div class="activity-name">{data['icon']} {activity.replace('_', ' ').title()}</div>
                                <div class="activity-points">+{data['points']} points - {data['description']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show points chart
                    chart = create_points_chart(activities)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        # Points display
        st.markdown(f"""
        <div class="stats-card">
            <div class="points-display">{points if 'points' in locals() else 0}</div>
            <div class="points-label">Points This Session</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips section
        st.markdown("""
        <div class="tips-section">
            <div class="tips-title">💡 Pro Tips</div>
            <div class="tip-item">Take clear, well-lit photos</div>
            <div class="tip-item">Include yourself in the activity</div>
            <div class="tip-item">Show the full object/activity</div>
            <div class="tip-item">Try different eco-friendly actions</div>
            <div class="tip-item">Plant care = bonus points!</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Environmental impact
        if 'points' in locals() and points > 0:
            st.markdown("### 🌍 Your Environmental Impact")
            co2_saved = points * 0.5  # Rough estimation
            trees_equivalent = points / 10
            
            st.metric("CO₂ Saved (kg)", f"{co2_saved:.1f}")
            st.metric("Tree Equivalent", f"{trees_equivalent:.1f}")

if __name__ == "__main__":
    main()
