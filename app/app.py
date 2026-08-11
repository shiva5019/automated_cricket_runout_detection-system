import streamlit as st
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(
    page_title="Cricket Run-Out Detection",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    .block-container { padding-top: 0rem; padding-bottom: 0rem; }

    .hero {
        background: linear-gradient(135deg, #0a0a1a, #0d1b2a, #0a1628);
        border-radius: 20px;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #1a3a5c;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-title span {
        color: #00d4ff;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-top: 10px;
        margin-bottom: 30px;
    }
    .badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
    }
    .badge-green {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        color: #00ff88;
    }
    .stat-card {
        background: linear-gradient(135deg, #0d1b2a, #0a1628);
        border: 1px solid #1a3a5c;
        border-radius: 16px;
        padding: 25px 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .stat-label {
        color: #a0aec0;
        font-size: 0.9rem;
        margin: 5px 0 0 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #0d1b2a, #111827);
        border: 1px solid #1a3a5c;
        border-radius: 16px;
        padding: 25px;
        height: 100%;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
        margin: 0 0 8px 0;
    }
    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .nav-container {
        background: #0d1b2a;
        border: 1px solid #1a3a5c;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 25px;
    }
    .step-card {
        background: #0d1b2a;
        border: 1px solid #1a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .step-num {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #00d4ff, #0080ff);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        margin: 0 auto 12px auto;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH  = Path(__file__).parent / "models" / "best_model_resnet.keras"
RESULTS_DIR = Path(__file__).parent / "results"
CLASS_NAMES = ["not_out", "out"]
IMG_SIZE    = (224, 224)

@st.cache_resource
def load_resnet_model():
    return load_model(MODEL_PATH)

model = load_resnet_model()

def predict_image(img):
    img       = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob      = model.predict(img_array, verbose=0)[0][0]
    label     = CLASS_NAMES[int(prob > 0.5)]
    confidence = prob if prob > 0.5 else 1 - prob
    return label, float(confidence)

def predict_frame(frame):
    img       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img       = cv2.resize(img, IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob      = model.predict(img_array, verbose=0)[0][0]
    label     = CLASS_NAMES[int(prob > 0.5)]
    confidence = prob if prob > 0.5 else 1 - prob
    return label, float(confidence)

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown('<div class="nav-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
with col2:
    if st.button("🖼️ Image Prediction", use_container_width=True):
        st.session_state.page = "Image"
with col3:
    if st.button("🎥 Video Detection", use_container_width=True):
        st.session_state.page = "Video"
with col4:
    if st.button("📊 Project Summary", use_container_width=True):
        st.session_state.page = "Summary"
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">🏏 Cricket <span>Run-Out</span> Detection</p>
        <p class="hero-subtitle">AI-Powered Automated Run-Out Decision System for Cricket Matches</p>
        <div>
            <span class="badge">🤖 Deep Learning</span>
            <span class="badge badge-green">✓ ResNet50V2</span>
            <span class="badge">📊 95% Accuracy</span>
            <span class="badge badge-green">🎥 Video Analysis</span>
            <span class="badge">🖼️ Image Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (col1, "1800",       "Training Images",  "#00d4ff"),
        (col2, "ResNet50V2", "Final Model",      "#00ff88"),
        (col3, "95.00%",     "Test Accuracy",    "#00d4ff"),
        (col4, "100%",       "Video Accuracy",   "#00ff88"),
    ]
    for col, value, label, color in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-value" style="color:{color};">{value}</p>
                <p class="stat-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<h3 style="text-align:center;color:white;">⚡ Features</h3>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        (col1, "🖼️", "Image Prediction", "Upload any cricket image and get instant run-out or not-out prediction with a confidence score.", "#00d4ff"),
        (col2, "🎥", "Video Detection",   "Upload a cricket video and get frame by frame analysis with prediction timeline and sample frames.", "#00ff88"),
        (col3, "📊", "Project Summary",   "View full model comparison, training curves, confusion matrices and complete project results.", "#00d4ff"),
    ]
    for col, icon, title, desc, color in features:
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <p class="feature-title" style="color:{color};">{title}</p>
                <p class="feature-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<h3 style="text-align:center;color:white;">🔄 How It Works</h3>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    steps = [
        (col1, "1", "Upload",   "Upload a cricket image or video clip"),
        (col2, "2", "Analyse",  "ResNet50V2 processes each frame"),
        (col3, "3", "Predict",  "Model predicts OUT or NOT OUT"),
        (col4, "4", "Decision", "Final decision with confidence score"),
    ]
    for col, num, title, desc in steps:
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <p style="color:white;font-weight:700;margin:0 0 5px 0;">{title}</p>
                <p style="color:#718096;font-size:0.85rem;margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="color:#4a5568;text-align:center;font-size:0.85rem;">Built by J. Shiva &nbsp;|&nbsp; Cricket Run-Out Detection &nbsp;|&nbsp; Deep Learning Project</p>', unsafe_allow_html=True)

elif st.session_state.page == "Image":
    st.markdown("## 🖼️ Image Prediction")
    st.markdown('<p style="color:#a0aec0;">Upload a cricket image to detect whether the batsman is out or not out.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        img  = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analysing image..."):
                label, confidence = predict_image(img)

            st.markdown("<br>", unsafe_allow_html=True)

            if label == "out":
                st.markdown("""
                <div style="background:linear-gradient(135deg,#7b0000,#1a0000);
                            border:2px solid #ff4444;border-radius:16px;
                            padding:30px;text-align:center;">
                <h1 style="color:#ff4444;font-size:3rem;">🔴 OUT</h1>
                <h3 style="color:#ffaaaa;">Batsman is Run Out</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:linear-gradient(135deg,#003d1a,#001a0d);
                            border:2px solid #00ff88;border-radius:16px;
                            padding:30px;text-align:center;">
                <h1 style="color:#00ff88;font-size:3rem;">🟢 NOT OUT</h1>
                <h3 style="color:#aaffcc;">Batsman is Safe</h3>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="background:#1a1a2e;border-radius:12px;padding:20px;border:1px solid #16213e;text-align:center;"><p style="color:#a0aec0;margin:0;">Confidence Score</p><h2 style="color:#00d4ff;margin:5px 0;">{confidence:.2%}</h2></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(confidence))
    else:
        st.markdown("""
        <div style="background:#1a1a2e;border-radius:12px;padding:40px;
                    text-align:center;border:2px dashed #2d3748;">
        <h3 style="color:#4a5568;">📁 Upload a Cricket Image</h3>
        <p style="color:#4a5568;">Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "Video":
    st.markdown("## 🎥 Video Detection")
    st.markdown('<p style="color:#a0aec0;">Upload a cricket video to detect run-out decisions frame by frame.</p>', unsafe_allow_html=True)

    uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.markdown(f'<p style="color:#00d4ff;">Video loaded — {total} frames at {fps:.0f} fps</p>', unsafe_allow_html=True)

        labels      = []
        confs       = []
        frames_data = []
        frame_num   = 0
        progress    = st.progress(0)
        status      = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            label, confidence = predict_frame(frame)
            labels.append(label)
            confs.append(confidence)
            frames_data.append((frame_num, label, confidence, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_num += 1
            progress.progress(frame_num / total)
            status.text(f"Processing frame {frame_num} / {total}")

        cap.release()
        progress.empty()
        status.empty()

        out_count     = labels.count("out")
        not_out_count = labels.count("not_out")
        decision      = "out" if out_count > not_out_count else "not_out"
        avg_conf      = np.mean(confs)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">Total Frames</p><h2 style="color:#00d4ff;margin:0;">{total}</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">Out Frames</p><h2 style="color:#ff4444;margin:0;">{out_count}</h2></div>', unsafe_allow_html=True)
        col3.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">Not Out Frames</p><h2 style="color:#00ff88;margin:0;">{not_out_count}</h2></div>', unsafe_allow_html=True)
        col4.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">Avg Confidence</p><h2 style="color:#00d4ff;margin:0;">{avg_conf:.1%}</h2></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if decision == "out":
            st.markdown("""
            <div style="background:linear-gradient(135deg,#7b0000,#1a0000);
                        border:2px solid #ff4444;border-radius:16px;
                        padding:25px;text-align:center;">
            <h1 style="color:#ff4444;">🔴 FINAL DECISION : OUT</h1>
            <h3 style="color:#ffaaaa;">❌ Batsman is Run Out</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#003d1a,#001a0d);
                        border:2px solid #00ff88;border-radius:16px;
                        padding:25px;text-align:center;">
            <h1 style="color:#00ff88;">🟢 FINAL DECISION : NOT OUT</h1>
            <h3 style="color:#aaffcc;">✅ Batsman is Safe</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎞️ Sample Frames — Out Predictions")
        out_frames = [(n, l, c, f) for n, l, c, f in frames_data if l == "out"]
        if out_frames:
            step    = max(1, len(out_frames) // 6)
            samples = out_frames[::step][:6]
            cols    = st.columns(len(samples))
            for col, (n, l, c, f) in zip(cols, samples):
                col.image(f, use_container_width=True)
                col.markdown(f'<p style="color:#ff4444;text-align:center;font-size:12px;">F{n} | {c:.0%}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4a5568;">No out frames detected.</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎞️ Sample Frames — Not Out Predictions")
        not_out_frames = [(n, l, c, f) for n, l, c, f in frames_data if l == "not_out"]
        if not_out_frames:
            step    = max(1, len(not_out_frames) // 6)
            samples = not_out_frames[::step][:6]
            cols    = st.columns(len(samples))
            for col, (n, l, c, f) in zip(cols, samples):
                col.image(f, use_container_width=True)
                col.markdown(f'<p style="color:#00ff88;text-align:center;font-size:12px;">F{n} | {c:.0%}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4a5568;">No not_out frames detected.</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Frame by Frame Prediction Timeline")
        fig, ax = plt.subplots(figsize=(12, 4))
        timestamps = [f / fps for f in range(len(labels))]
        colors     = ["#ff4444" if l == "out" else "#00ff88" for l in labels]
        ax.bar(timestamps, confs, color=colors, width=1/fps, alpha=0.9)
        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel("Confidence", color="white")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#2d3748")
        ax.spines["left"].set_color("#2d3748")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

    else:
        st.markdown("""
        <div style="background:#1a1a2e;border-radius:12px;padding:40px;
                    text-align:center;border:2px dashed #2d3748;">
        <h3 style="color:#4a5568;">📁 Upload a Cricket Video</h3>
        <p style="color:#4a5568;">Supported formats: MP4, AVI, MOV</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "Summary":
    st.markdown("## 📊 Project Summary")
    st.markdown('<p style="color:#a0aec0;">Full comparison of MobileNetV2 and ResNet50V2 with training results.</p>', unsafe_allow_html=True)

    st.markdown("### 📁 Dataset")
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, value, color in [
        (col1, "Raw Images",  "225",  "#00d4ff"),
        (col2, "Augmented",   "1800", "#00ff88"),
        (col3, "Train",       "1260", "#00d4ff"),
        (col4, "Val",         "270",  "#00ff88"),
        (col5, "Test",        "270",  "#00d4ff"),
    ]:
        col.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">{label}</p><h2 style="color:{color};margin:0;">{value}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🖼️ Sample Dataset Images")
    st.image(str(RESULTS_DIR / "sample_images.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#1a1a2e;border-radius:12px;padding:20px;border:1px solid #16213e;">
        <h3 style="color:#a0aec0;">MobileNetV2</h3>
        <table style="width:100%;color:white;">
        <tr><td style="color:#a0aec0;padding:5px;">Test Accuracy</td><td style="color:#00d4ff;padding:5px;">93.00%</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Macro F1</td><td style="color:#00d4ff;padding:5px;">0.93</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Total Errors</td><td style="color:#ff4444;padding:5px;">20 / 270</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Best Epoch</td><td style="color:#00d4ff;padding:5px;">20</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#003d1a;border-radius:12px;padding:20px;border:2px solid #00ff88;">
        <h3 style="color:#00ff88;">ResNet50V2 ✓ Winner</h3>
        <table style="width:100%;color:white;">
        <tr><td style="color:#a0aec0;padding:5px;">Test Accuracy</td><td style="color:#00ff88;padding:5px;">95.00%</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Macro F1</td><td style="color:#00ff88;padding:5px;">0.95</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Total Errors</td><td style="color:#00ff88;padding:5px;">14 / 270</td></tr>
        <tr><td style="color:#a0aec0;padding:5px;">Best Epoch</td><td style="color:#00ff88;padding:5px;">18</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Training Curves")
    st.image(str(RESULTS_DIR / "training_curves_comparison.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🟦 Confusion Matrices")
    st.image(str(RESULTS_DIR / "confusion_matrix_comparison.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Batch Predictions on Test Images")
    st.image(str(RESULTS_DIR / "batch_predictions.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎥 Video Detection Timeline")
    st.image(str(RESULTS_DIR / "prediction_timeline.png"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Final Results")
    col1, col2, col3, col4 = st.columns(4)
    for col, label, value, color in [
        (col1, "Final Model",    "ResNet50V2", "#00ff88"),
        (col2, "Test Accuracy",  "95.00%",     "#00d4ff"),
        (col3, "Macro F1",       "0.95",       "#00ff88"),
        (col4, "Video Accuracy", "100%",       "#00d4ff"),
    ]:
        col.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:15px;text-align:center;border:1px solid #16213e;"><p style="color:#a0aec0;margin:0;">{label}</p><h2 style="color:{color};margin:0;">{value}</h2></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="color:#4a5568;text-align:center;font-size:0.85rem;">Built by J. Shiva &nbsp;|&nbsp; Cricket Run-Out Detection &nbsp;|&nbsp; Deep Learning Project</p>', unsafe_allow_html=True)