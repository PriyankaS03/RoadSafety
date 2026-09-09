"""
pages/live_detection.py – Real-time webcam + YOLO detection
"""
import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path
from utils.detection import (
    mock_frame_detection,
    real_frame_detection,
    classify_pothole_severity,
    get_random_tn_location,
    YOLO_AVAILABLE
)
from utils.db import add_report

# Try to get real detector
def get_detections(frame, use_real_yolo=True):
    """Get detections using real YOLO or mock"""
    if use_real_yolo and YOLO_AVAILABLE:
        try:
            return real_frame_detection(frame)
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return mock_frame_detection(frame)
    else:
        return mock_frame_detection(frame)

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for d in detections:
        x, y, w, h = d["bbox"]
        color = d["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{d['label']} ({d['confidence']:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame

def show_live_detection():
    st.markdown("""
    <div class='section-header'>📹 Live Detection Engine</div>
    <div class='section-sub'>Real-time AI hazard detection via webcam or demo mode</div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 1])

    with col_r:
        st.markdown("""
        <div style='background:white; border-radius:16px; padding:20px;
             box-shadow:0 2px 12px rgba(0,0,0,0.07); border:1px solid #e8edf2;'>
        <b style='font-size:0.9rem; color:#0f172a'>🎛️ Detection Controls</b>
        """, unsafe_allow_html=True)

        mode = st.radio("Input Mode", ["🎭 Demo Mode", "📷 Webcam"], index=0)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.65, 0.05)

        # YOLO toggle
        use_real_yolo = st.toggle("🤖 Use Real YOLO Model", value=YOLO_AVAILABLE)
        if use_real_yolo and not YOLO_AVAILABLE:
            st.warning("YOLO not available - using mock detection")

        auto_report = st.checkbox("Auto-report to DB", value=True)
        scan_interval = st.slider("Scan every N frames", 5, 30, 10)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#64748b'>
            <b>Detection Classes:</b><br>
            🔴 Accident (Red box)<br>
            🟠 Pothole (Orange box)<br>
            🟢 Good Road (Green)<br>
            🟡 Crack (Yellow)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Alert log
        if "detection_log" not in st.session_state:
            st.session_state.detection_log = []

        st.markdown("### 📋 Recent Detections")
        for log in st.session_state.detection_log[-6:]:
            badge_clr = "#ef4444" if "Accident" in log else "#f59e0b"
            st.markdown(f"""
            <div style='background:{badge_clr}11; border-left:3px solid {badge_clr};
                 border-radius:8px; padding:8px 10px; margin:4px 0; font-size:0.78rem; color:#334155'>
                {log}
            </div>""", unsafe_allow_html=True)

    with col_l:
        if "📷 Webcam" in mode:
            st.info("📷 Click **Start Webcam** then allow camera access in your browser.")
            run_webcam = st.toggle("▶️ Start Webcam Detection", value=False)

            if run_webcam:
                frame_window = st.image([], use_container_width=True)
                alert_box    = st.empty()
                cap = cv2.VideoCapture(0)
                frame_count = 0

                try:
                    while run_webcam:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Cannot access webcam. Try Demo Mode.")
                            break

                        frame_count += 1
                        detections = []
                        if frame_count % scan_interval == 0:
                            detections = get_detections(frame, use_real_yolo)

                            for d in detections:
                                if d["confidence"] >= confidence_threshold:
                                    loc = get_random_tn_location()
                                    log_entry = f"⏱ {d['label']} | {d['confidence']:.0%} | {loc['name']}"
                                    if log_entry not in st.session_state.detection_log:
                                        st.session_state.detection_log.append(log_entry)

                                    if auto_report:
                                        sev = d["severity"]
                                        damage_rate = d.get("damage_rate", 0)
                                        add_report(
                                            "accident" if d["label"] == "Accident" else "pothole",
                                            sev, loc["name"], loc["lat"], loc["lon"],
                                            damage_rate=damage_rate
                                        )
                                    if d["label"] == "Accident":
                                        alert_box.error(f"🚨 ACCIDENT DETECTED! ({d['confidence']:.0%}) at {loc['name']}")
                                    else:
                                        alert_box.warning(f"🕳️ POTHOLE DETECTED! ({d['confidence']:.0%}) – {sev.upper()} severity")

                        annotated = draw_detections(frame.copy(), detections)
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame_rgb, use_container_width=True)
                finally:
                    cap.release()

        else:  # Demo Mode
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e293b,#0f172a);
                 border-radius:16px; padding:20px; text-align:center; margin-bottom:16px;'>
                <div style='font-size:0.85rem; color:#94a3b8; margin-bottom:4px'>
                    DEMO MODE – {'Real YOLO Inference' if use_real_yolo and YOLO_AVAILABLE else 'AI Mock Inference'}
                </div>
                <div style='font-size:1rem; color:#e2e8f0; font-weight:600'>
                    {'Using trained YOLOv8 model' if use_real_yolo and YOLO_AVAILABLE else 'Simulating YOLO object detection'} on a virtual road feed
                </div>
            </div>""", unsafe_allow_html=True)

            demo_frame_holder = st.empty()
            demo_alert_holder = st.empty()
            demo_stats_holder = st.empty()

            if "demo_running" not in st.session_state:
                st.session_state.demo_running = False

            btn_lbl = "⏹ Stop Detection" if st.session_state.demo_running else "▶️ Start Demo Detection"
            if st.button(btn_lbl, use_container_width=False):
                st.session_state.demo_running = not st.session_state.demo_running

            if st.session_state.demo_running:
                for frame_n in range(200):
                    # Synthesize a fake road frame
                    bg = np.zeros((360, 640, 3), dtype=np.uint8)
                    bg[:] = (30, 30, 40)
                    # road
                    cv2.rectangle(bg, (160, 0), (480, 360), (55, 55, 65), -1)
                    # lane markings
                    for y in range(0, 360, 40):
                        cv2.rectangle(bg, (318, y), (322, y + 20), (200, 200, 100), -1)

                    detections = get_detections(bg, use_real_yolo)
                    annotated  = draw_detections(bg, detections)

                    # overlay info
                    cv2.putText(annotated, f"VigiRoad AI | Frame {frame_n}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                    cv2.putText(annotated, f"Conf threshold: {confidence_threshold:.0%}",
                                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    demo_frame_holder.image(frame_rgb, use_container_width=True,
                                            caption=f"Frame {frame_n} | {len(detections)} detection(s)")

                    for d in detections:
                        if d["confidence"] >= confidence_threshold:
                            loc = get_random_tn_location()
                            log = f"Frame {frame_n}: {d['label']} | {d['confidence']:.0%} | {loc['name']}"
                            if auto_report:
                                damage_rate = d.get("damage_rate", 0)
                                add_report(
                                    "accident" if d["label"] == "Accident" else "pothole",
                                    d["severity"], loc["name"], loc["lat"], loc["lon"],
                                    damage_rate=damage_rate
                                )
                            if log not in st.session_state.detection_log:
                                st.session_state.detection_log.append(log)

                            if d["label"] == "Accident":
                                demo_alert_holder.error(f"🚨 ACCIDENT DETECTED at {loc['name']}! Police & Ambulance alerted.")
                            else:
                                demo_alert_holder.warning(f"🕳️ {d['severity'].upper()} POTHOLE at {loc['name']}")

                    import time
                    time.sleep(0.18)

                    if not st.session_state.demo_running:
                        break

                st.session_state.demo_running = False
                st.info("Demo stopped. Check dashboards for logged reports.")