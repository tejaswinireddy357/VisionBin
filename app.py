import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.classifier import classify_waste, WASTE_CATEGORIES
from src.logger import log_detection, get_logs, get_statistics, clear_logs

st.set_page_config(page_title="VisionBin", page_icon="♻️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #111210;
    font-family: 'Space Grotesk', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px; }

/* ── HEADER ── */
.vb-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e1f1c;
    margin-bottom: 2rem;
}
.vb-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.vb-logo-mark {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #c8f04a, #78c832);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
}
.vb-logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 800;
    color: #f0f0ee;
    letter-spacing: -0.02em;
}
.vb-logo-text span { color: #c8f04a; }
.vb-status {
    display: flex; align-items: center; gap: 0.5rem;
    background: #1a1c18; border: 1px solid #2a2d26;
    border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.75rem; color: #6b7066; letter-spacing: 0.05em;
}
.vb-status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #c8f04a;
    box-shadow: 0 0 8px #c8f04a;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1f1c !important;
    gap: 0 !important;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a4d46 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.8rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #c8f04a !important;
    border-bottom: 2px solid #c8f04a !important;
    background: transparent !important;
}

/* ── CAMERA AREA ── */
.cam-wrap {
    position: relative;
    background: #0d0e0c;
    border: 1px solid #1e1f1c;
    border-radius: 20px;
    overflow: hidden;
    padding: 1.5rem;
}
.cam-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #3a3d36; margin-bottom: 1rem;
}
.cam-ring {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 120px; height: 120px;
    border: 2px dashed #c8f04a33;
    border-radius: 50%;
    pointer-events: none;
    animation: spin 8s linear infinite;
}
@keyframes spin { to { transform: translate(-50%, -50%) rotate(360deg); } }

/* ── RESULT PANEL ── */
.result-wrap {
    background: #0d0e0c;
    border: 1px solid #1e1f1c;
    border-radius: 20px;
    padding: 2rem;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.result-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 1rem;
    color: #2a2d26;
}
.result-empty-icon {
    font-size: 3.5rem;
    filter: grayscale(1) opacity(0.3);
}
.result-empty-text {
    font-size: 0.9rem;
    color: #3a3d36;
    line-height: 1.6;
}
.result-detected {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}
.result-type-block {
    text-align: center;
    padding: 2rem 1rem;
    background: #111210;
    border-radius: 16px;
    border: 1px solid #1e1f1c;
    position: relative;
    overflow: hidden;
}
.result-type-block::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, var(--accent-color) 0%, transparent 70%);
    opacity: 0.06;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-type-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: var(--accent-color);
    letter-spacing: -0.02em;
}
.result-confidence {
    font-size: 0.8rem; color: #4a4d46;
    margin-top: 0.3rem; letter-spacing: 0.05em;
}
.confidence-bar {
    height: 3px;
    background: #1e1f1c;
    border-radius: 999px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: var(--accent-color);
    border-radius: 999px;
    transition: width 0.6s ease;
}
.result-meta {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}
.result-meta-row {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    padding: 0.8rem;
    background: #111210;
    border-radius: 10px;
    border: 1px solid #1a1c18;
}
.result-meta-icon { font-size: 1rem; margin-top: 0.1rem; flex-shrink: 0; }
.result-meta-content { flex: 1; }
.result-meta-label {
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #3a3d36; margin-bottom: 0.2rem;
}
.result-meta-value { font-size: 0.85rem; color: #9a9d96; line-height: 1.4; }

/* ── SCAN BUTTON ── */
div.stButton > button {
    background: #c8f04a !important;
    color: #111210 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 0 0 #c8f04a44 !important;
}
div.stButton > button:hover {
    background: #d4f75e !important;
    box-shadow: 0 0 24px #c8f04a44 !important;
    transform: translateY(-1px) !important;
}

/* ── SAVE BUTTON ── */
div.stButton > button[kind="secondary"] {
    background: #1a1c18 !important;
    color: #c8f04a !important;
    border: 1px solid #2a2d26 !important;
}

/* ── STAT CARDS ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: #0d0e0c;
    border: 1px solid #1e1f1c;
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #2a2d26; }
.stat-emoji { font-size: 1.5rem; margin-bottom: 0.5rem; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem; font-weight: 800;
    color: #c8f04a; line-height: 1;
}
.stat-label { font-size: 0.75rem; color: #4a4d46; margin-top: 0.4rem; letter-spacing: 0.05em; }

/* ── HISTORY ── */
.history-list { display: flex; flex-direction: column; gap: 0.5rem; }
.history-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.2rem;
    background: #0d0e0c;
    border: 1px solid #1a1c18;
    border-radius: 12px;
    transition: border-color 0.15s;
}
.history-row:hover { border-color: #2a2d26; }
.history-left { display: flex; align-items: center; gap: 0.8rem; }
.history-emoji { font-size: 1.1rem; }
.history-type { font-size: 0.9rem; font-weight: 600; color: #c8c9c4; }
.history-time { font-size: 0.75rem; color: #3a3d36; }

/* ── GUIDE CARDS ── */
.guide-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}
.guide-card {
    background: #0d0e0c;
    border: 1px solid #1e1f1c;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
}
.guide-card:hover {
    border-color: #2a2d26;
    transform: translateY(-2px);
}
.guide-emoji { font-size: 2rem; margin-bottom: 0.8rem; }
.guide-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800;
    color: #c8f04a; margin-bottom: 0.8rem;
}
.guide-row {
    font-size: 0.8rem; color: #6b7066;
    margin-bottom: 0.4rem; line-height: 1.5;
}
.guide-row b { color: #9a9d96; }

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #3a3d36; margin-bottom: 1.2rem;
}

/* ── STREAMLIT OVERRIDES ── */
.stAlert { background: #0d0e0c !important; border: 1px solid #1e1f1c !important; color: #6b7066 !important; }
div[data-testid="stCameraInput"] > div { background: transparent !important; border: none !important; }
div[data-testid="stCameraInput"] video { border-radius: 12px; }
div[data-testid="stCameraInput"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="vb-header">
    <div class="vb-logo">
        <div class="vb-logo-mark">♻️</div>
        <div class="vb-logo-text">Vision<span>Bin</span></div>
    </div>
    <div class="vb-status">
        <div class="vb-status-dot"></div>
        SYSTEM ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATE ──
if "last_detection" not in st.session_state:
    st.session_state.last_detection = None

# Accent color map per waste type
ACCENT_COLORS = {
    "Plastic":  "#60a5fa",
    "Paper":    "#fbbf24",
    "Metal":    "#94a3b8",
    "Glass":    "#34d399",
    "Organic":  "#86efac",
    "E-Waste":  "#f87171",
}

tab1, tab2, tab3 = st.tabs(["SCAN", "STATS", "GUIDE"])

# ══════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="cam-wrap"><div class="cam-label">Camera Feed</div>', unsafe_allow_html=True)
        photo = st.camera_input("", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if photo:
            pil_image = Image.open(photo)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            with st.spinner("Scanning..."):
                waste_type, confidence, info = classify_waste(frame)
                st.session_state.last_detection = {
                    "type": waste_type,
                    "confidence": confidence,
                    "info": info
                }

    with col2:
        if st.session_state.last_detection:
            d = st.session_state.last_detection
            accent = ACCENT_COLORS.get(d['type'], '#c8f04a')
            st.markdown(f"""
            <div class="result-wrap">
                <div class="sec-label">Detected Waste</div>
                <div class="result-detected">
                    <div class="result-type-block" style="--accent-color:{accent}">
                        <div class="result-emoji">{d['info']['emoji']}</div>
                        <div class="result-type-name">{d['type']}</div>
                        <div class="result-confidence">CONFIDENCE · {d['confidence']}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{d['confidence']}%; background:{accent};"></div>
                        </div>
                    </div>
                    <div class="result-meta">
                        <div class="result-meta-row">
                            <div class="result-meta-icon">🗑️</div>
                            <div class="result-meta-content">
                                <div class="result-meta-label">Bin</div>
                                <div class="result-meta-value">{d['info']['bin']}</div>
                            </div>
                        </div>
                        <div class="result-meta-row">
                            <div class="result-meta-icon">💡</div>
                            <div class="result-meta-content">
                                <div class="result-meta-label">Disposal Tip</div>
                                <div class="result-meta-value">{d['info']['tip']}</div>
                            </div>
                        </div>
                        <div class="result-meta-row">
                            <div class="result-meta-icon">📦</div>
                            <div class="result-meta-content">
                                <div class="result-meta-label">Examples</div>
                                <div class="result-meta-value">{d['info']['examples']}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Save to History →"):
                log_detection(d['type'], d['confidence'])
                st.success("Saved!")
        else:
            st.markdown("""
            <div class="result-wrap">
                <div class="sec-label">Detection Result</div>
                <div class="result-empty">
                    <div class="result-empty-icon">🎯</div>
                    <div class="result-empty-text">
                        Point your camera at a waste item<br>and take a photo to identify it
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — STATS
# ══════════════════════════════════════════
with tab2:
    stats = get_statistics()
    logs = get_logs()

    total = sum(stats.values()) if stats else 0
    st.markdown(f"""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800; color:#f0f0ee; line-height:1;">
            {total}
        </div>
        <div style="font-size:0.8rem; color:#3a3d36; letter-spacing:0.1em; margin-top:0.3rem;">TOTAL SCANS LOGGED</div>
    </div>
    """, unsafe_allow_html=True)

    if not stats:
        st.markdown('<div style="color:#3a3d36; font-size:0.9rem;">No scans saved yet — scan waste and hit Save to History.</div>', unsafe_allow_html=True)
    else:
        # Stat cards
        cols = st.columns(len(stats))
        for col, (waste_type, count) in zip(cols, stats.items()):
            emoji = WASTE_CATEGORIES.get(waste_type, {}).get("emoji", "🗑️")
            accent = ACCENT_COLORS.get(waste_type, "#c8f04a")
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-emoji">{emoji}</div>
                    <div class="stat-num" style="color:{accent};">{count}</div>
                    <div class="stat-label">{waste_type.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<br><div class="sec-label">Recent History</div>', unsafe_allow_html=True)
        st.markdown('<div class="history-list">', unsafe_allow_html=True)
        for log in reversed(logs[-12:]):
            emoji = WASTE_CATEGORIES.get(log['waste_type'], {}).get("emoji", "🗑️")
            accent = ACCENT_COLORS.get(log['waste_type'], "#c8f04a")
            st.markdown(f"""
            <div class="history-row">
                <div class="history-left">
                    <div class="history-emoji">{emoji}</div>
                    <div class="history-type" style="color:{accent};">{log['waste_type']}</div>
                </div>
                <div class="history-time">{log['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear all history"):
            clear_logs()
            st.rerun()

# ══════════════════════════════════════════
# TAB 3 — GUIDE
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">Waste Categories</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-grid">', unsafe_allow_html=True)
    for waste_type, info in WASTE_CATEGORIES.items():
        accent = ACCENT_COLORS.get(waste_type, "#c8f04a")
        st.markdown(f"""
        <div class="guide-card">
            <div class="guide-emoji">{info['emoji']}</div>
            <div class="guide-name" style="color:{accent};">{waste_type}</div>
            <div class="guide-row"><b>Bin</b> · {info['bin']}</div>
            <div class="guide-row"><b>Examples</b> · {info['examples']}</div>
            <div class="guide-row"><b>Tip</b> · {info['tip']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)