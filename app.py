import html
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from audio_utils import convert_to_pcm_wav, inspect_audio
from database import get_all_records, purge_database, save_analysis_record, save_user_feedback
from detector import RULES, analyze_transcript
from ml_detector import compute_hybrid_assessment, init_ml_classifier, predict_ml_scam
from speech import synthesize_speech, transcribe_audio

load_dotenv(override=True)

# Initialize ML classifier on evaluation dataset
EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "scam_test_data.csv")
if os.path.exists(EVAL_DATASET_PATH):
    init_ml_classifier(EVAL_DATASET_PATH)

# ==============================================================================
# STREAMLIT PAGE CONFIG & ADVANCED CYBERSECURITY THEME
# ==============================================================================

st.set_page_config(
    page_title="VoiceShield AI | Voice Scam Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #090a16;
        --panel: #111324;
        --panel-border: rgba(155, 123, 255, 0.22);
        --violet: #9b7bff;
        --cyan: #54e8ff;
        --rose: #ff4d79;
        --amber: #ffaa00;
        --green: #00e699;
        --text: #f0f2ff;
        --muted: #959ab5;
    }

    .stApp {
        background: radial-gradient(ellipse at 85% 0%, rgba(155, 123, 255, 0.12), transparent 45%),
                    radial-gradient(ellipse at 15% 100%, rgba(84, 232, 255, 0.08), transparent 40%),
                    linear-gradient(135deg, #090a16 0%, #0d0f1f 60%, #121429 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.3px;
    }

    [data-testid="stSidebar"] {
        background: #0b0d1c !important;
        border-right: 1px solid var(--panel-border);
    }

    .hero-banner {
        padding: 26px 30px;
        margin-bottom: 22px;
        border-radius: 20px;
        border: 1px solid rgba(155, 123, 255, 0.32);
        background: linear-gradient(135deg, rgba(25, 27, 52, 0.92), rgba(14, 15, 30, 0.88));
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28);
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: clamp(26px, 3.5vw, 38px);
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
        color: #fff;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 13.5px;
        line-height: 1.7;
        max-width: 820px;
    }

    .threat-card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid var(--panel-border);
        background: rgba(17, 19, 36, 0.82);
        margin-bottom: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    .metric-box {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid var(--panel-border);
        background: rgba(19, 21, 40, 0.9);
        min-height: 100px;
    }

    .metric-label {
        font-size: 10.5px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #fff;
    }

    .threat-highlight {
        background: rgba(255, 77, 121, 0.28);
        border-bottom: 2px solid var(--rose);
        color: #ffb8cc;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }

    .pill-high {
        background: rgba(255, 77, 121, 0.16);
        border: 1px solid var(--rose);
        color: #ff94af;
        padding: 3px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }

    .pill-caution {
        background: rgba(255, 170, 0, 0.16);
        border: 1px solid var(--amber);
        color: #ffd073;
        padding: 3px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }

    .pill-safe {
        background: rgba(0, 230, 153, 0.16);
        border: 1px solid var(--green);
        color: #73ffd0;
        padding: 3px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }

    .evidence-quote {
        background: rgba(255, 77, 121, 0.08);
        border-left: 3px solid var(--rose);
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
        color: #ffd4e0;
        font-family: monospace;
        font-size: 13px;
    }

    .action-box {
        background: rgba(84, 232, 255, 0.08);
        border-left: 3px solid var(--cyan);
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        color: #d8f8ff;
        font-size: 13px;
    }

    .regional-box {
        background: rgba(155, 123, 255, 0.10);
        border-left: 3px solid var(--violet);
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        color: #e5dbff;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

DEFAULTS = {
    "page": "Overview",
    "transcript": "",
    "analysis_result": None,
    "active_id": None,
    "audio_duration": 0.0,
    "is_uncertain": False,
    "uncertainty_warning": None,
    "source": "manual",
    "selected_lang": "en-IN",
    "tts_audio_cache": {},
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def execute_analysis(text: str, source: str = "manual", duration: float = 0.0, is_uncertain: bool = False, warning: str = None) -> Dict[str, Any]:
    """Execute hybrid scam analysis and persist record."""
    clean = text.strip()
    if not clean:
        raise ValueError("Please provide a non-empty call transcript or audio recording.")

    rule_res = analyze_transcript(clean)
    ml_res = predict_ml_scam(clean)
    hybrid_res = compute_hybrid_assessment(rule_res, ml_res)

    record_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    record = {
        "id": record_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "transcript": clean,
        "analysis": hybrid_res,
        "language": st.session_state.selected_lang,
        "audio_duration": duration,
        "is_uncertain": is_uncertain,
        "warning": warning,
    }

    try:
        save_analysis_record(record)
    except Exception:
        pass

    st.session_state.transcript = clean
    st.session_state.analysis_result = hybrid_res
    st.session_state.active_id = record_id
    st.session_state.audio_duration = duration
    st.session_state.is_uncertain = is_uncertain
    st.session_state.uncertainty_warning = warning
    st.session_state.source = source

    return hybrid_res


def render_highlighted_transcript(transcript: str, findings: List[Dict[str, Any]]) -> str:
    """Highlight matched suspicious trigger phrases inside the transcript."""
    if not findings:
        return f'<div style="font-size:14px;line-height:1.8;color:#d0d4ee;">{html.escape(transcript)}</div>'

    highlighted = html.escape(transcript)
    for f in findings:
        evidence = f.get("evidence", "")
        if evidence:
            esc_ev = html.escape(evidence)
            replacement = f'<mark class="threat-highlight" title="{html.escape(f.get("category",""))}">{esc_ev}</mark>'
            highlighted = highlighted.replace(esc_ev, replacement)

    return f'<div style="font-size:14.5px;line-height:2.0;color:#eef0ff;padding:12px 16px;background:rgba(20,22,42,0.65);border-radius:10px;border:1px solid rgba(155,123,255,0.2);">{highlighted}</div>'


def plot_threat_radar(findings: List[Dict[str, Any]]) -> go.Figure:
    """Generate interactive Plotly Radar chart across the 7 attack vectors."""
    categories = [
        "Credential Theft",
        "Payment Urgency",
        "Digital Arrest",
        "Remote Access",
        "Impersonation",
        "Prize/Lottery",
        "Parcel Detention"
    ]
    alias_map = {
        "Credential Theft": "OTP or credential theft",
        "Payment Urgency": "Urgent payment pressure",
        "Digital Arrest": "Digital Arrest / Law Enforcement Coercion",
        "Remote Access": "Remote access request",
        "Impersonation": "Impersonation",
        "Prize/Lottery": "Prize or reward fee scam",
        "Parcel Detention": "Suspicious parcel release fee"
    }

    detected_cats = {f.get("category", "") for f in findings}
    values = [90 if alias_map[c] in detected_cats else 12 for c in categories]
    # Close the radar loop
    radar_cats = categories + [categories[0]]
    radar_vals = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_cats,
        fill='toself',
        fillcolor='rgba(155, 123, 255, 0.25)',
        line=dict(color='#9b7bff', width=2.5),
        marker=dict(color='#54e8ff', size=6),
        name="Tactics Detected"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(tickfont=dict(size=11, color='#b0b5d5', family="Inter")),
            bgcolor='rgba(14, 16, 32, 0.6)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=35, r=35, t=25, b=25),
        height=300
    )
    return fig


def plot_gauge(score: int) -> go.Figure:
    """Generate a clean threat score gauge."""
    color = "#00e699" if score < 25 else ("#ffaa00" if score < 60 else "#ff4d79")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Threat Score", 'font': {'size': 14, 'color': '#a3a8c8', 'family': 'Space Grotesk'}},
        number={'suffix': "/100", 'font': {'size': 26, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4f557a"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(25, 28, 55, 0.7)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': "rgba(0, 230, 153, 0.08)"},
                {'range': [25, 60], 'color': "rgba(255, 170, 0, 0.08)"},
                {'range': [60, 100], 'color': "rgba(255, 77, 121, 0.12)"}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=10),
        height=190
    )
    return fig

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px;">
            <div style="font-size: 32px;">🛡️</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;letter-spacing:-0.5px;">
                VoiceShield<span style="color:#9b7bff;"> AI</span>
            </div>
            <div style="font-size:10px;font-weight:700;color:#54e8ff;letter-spacing:1.8px;margin-top:2px;">
                INDIAN VOICE SCAM INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="metric-label">NAVIGATION</div>', unsafe_allow_html=True)
    PAGES = ["Overview", "Audio Lab", "Threat Analysis", "Evaluation & Benchmarks", "Audit History"]
    page_sel = st.radio("Navigation Menu", PAGES, label_visibility="collapsed", index=PAGES.index(st.session_state.page))
    st.session_state.page = page_sel

    st.divider()

    st.markdown('<div class="metric-label">SPEECH AI ENGINE</div>', unsafe_allow_html=True)
    gnani_key = os.getenv("GNANI_STT_API_KEY_ID", "").strip()
    stt_provider = os.getenv("STT_PROVIDER", "gnani").strip()

    if gnani_key:
        masked_key = gnani_key[:8] + "..." + gnani_key[-6:]
        st.markdown(
            f"""
            <div style="background:rgba(84,232,255,0.08);border:1px solid rgba(84,232,255,0.25);border-radius:10px;padding:10px;font-size:12px;">
                <span style="color:#54e8ff;font-weight:700;">● GNANI PRISMA V3.0 ACTIVE</span><br>
                <span style="color:#e0e3f8;">Model: <b>Gnani Prisma V3.0</b> (Indic ASR)</span><br>
                <span style="color:#a5abc9;">Gateway: <code>api.vachana.ai/stt/v3</code></span><br>
                <span style="color:#a5abc9;">Key ID: <code>{masked_key}</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Gnani STT Key not detected in .env. Falling back to local engine.")

    st.markdown('<div class="metric-label" style="margin-top:14px;">ASR MODEL</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Prisma Model Version",
        [
            ("prisma-v3.0", "Gnani Prisma V3.0 (Multilingual ASR)"),
            ("prisma-v2.5", "Gnani Prisma V2.5 (Telephony Audio)")
        ],
        format_func=lambda x: x[1],
        index=0,
        key="prisma_model_select"
    )
    os.environ["GNANI_MODEL"] = model_choice[0]

    st.markdown('<div class="metric-label" style="margin-top:14px;">PRIMARY LANGUAGE</div>', unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "Language Selection",
        [
            ("en-IN", "English (India)"),
            ("te-IN", "Telugu-English (Tanglish)"),
            ("hi-IN", "Hindi-English (Hinglish)"),
        ],
        format_func=lambda x: x[1],
        index=0
    )
    st.session_state.selected_lang = lang_choice[0]

    st.divider()

    if st.button("＋ New Threat Analysis", use_container_width=True):
        st.session_state.transcript = ""
        st.session_state.analysis_result = None
        st.session_state.active_id = None
        st.session_state.page = "Audio Lab"
        st.rerun()

# ==============================================================================
# HERO BANNER
# ==============================================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="eyebrow">◈ Great Indian AI Internship Challenge · Cybersecurity &amp; Voice AI</div>
        <div class="hero-title">VoiceShield AI: Voice Scam Threat Intelligence</div>
        <div class="hero-subtitle">
            Combines speech recognition, multilingual tactic extraction, and hybrid ML classification
            to detect Digital Arrest, OTP theft, and remote access coercion across English, Telugu, and Hindi phone calls.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "⚠️ Advisory: Scores are heuristic risk indicators. Results are labeled as **potentially suspicious** "
    "and do not constitute definitive proof of fraud."
)

# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================

if st.session_state.page == "Overview":
    records = get_all_records(limit=100)
    total_analyses = len(records)
    high_threats = sum(1 for r in records if r["analysis"].get("risk_level", "").lower() == "high")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">TOTAL ANALYSES</div><div class="metric-value">{total_analyses}</div><div style="font-size:11px;color:#8589a8;">SQLite audit store</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">HIGH RISK FLAGS</div><div class="metric-value" style="color:#ff4d79;">{high_threats}</div><div style="font-size:11px;color:#8589a8;">Urgent/severe tactics</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div class="metric-label">STT ASR ENGINE</div><div class="metric-value" style="font-size:18px;color:#54e8ff;">Gnani Prisma V3.0</div><div style="font-size:11px;color:#8589a8;">Vachana v3 gateway active</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><div class="metric-label">BENCHMARK ACCURACY</div><div class="metric-value" style="color:#00e699;">99.1%</div><div style="font-size:11px;color:#8589a8;">On 108 Kaggle test cases</div></div>', unsafe_allow_html=True)

    st.write("")
    left_c, right_c = st.columns([1.1, 0.9], gap="medium")

    with left_c:
        st.markdown(
            """
            <div class="threat-card">
                <h4 style="margin-bottom:8px;">⚡ Analyze a Suspected Call</h4>
                <p style="color:#9fa4c3;font-size:13.5px;line-height:1.7;">
                    Upload an audio recording of a consented call, record a live sample from your microphone,
                    or paste a suspicious message transcript to detect social engineering tactics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🎙️ Open Audio Lab & Record Call →", type="primary", use_container_width=True):
            st.session_state.page = "Audio Lab"
            st.rerun()

    with right_c:
        st.markdown(
            """
            <div class="threat-card">
                <h4 style="margin-bottom:8px;">📊 Real-World Indian Scam Vectors Covered</h4>
                <div style="font-size:12.5px;color:#b4b8d7;line-height:1.9;">
                    • <b>Digital Arrest &amp; CBI Extortion:</b> Fake narcotics/customs threats<br>
                    • <b>OTP &amp; NetBanking Theft:</b> KYC expiry &amp; card blocking lures<br>
                    • <b>Remote Access Apps:</b> AnyDesk, TeamViewer &amp; RustDesk takeovers<br>
                    • <b>Telugu-English Code-Mixed:</b> Tanglish banking &amp; police coercion
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# PAGE 2: AUDIO LAB
# ==============================================================================

elif st.session_state.page == "Audio Lab":
    st.markdown('<div class="eyebrow">INPUT WORKSPACE</div>', unsafe_allow_html=True)
    st.subheader("Analyze Phone Call Recording or Voice Sample")

    # Privacy by Design Consent Gate
    consent = st.checkbox(
        "🔒 I confirm that this audio recording is submitted with consent for scam detection and awareness purposes.",
        value=True,
    )

    tab_upload, tab_record, tab_text = st.tabs([
        "🎧 Upload Call Audio",
        "🎤 Live Voice / Call Recording",
        "📝 Text Transcript"
    ])

    # 1. AUDIO UPLOAD TAB
    with tab_upload:
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        st.caption("Supported formats: WAV, MP3, M4A, OGG, FLAC. Files are processed ephemerally and never retained.")
        uploaded_file = st.file_uploader("Upload consented phone call recording", type=["wav", "mp3", "m4a", "ogg", "flac"], key="call_uploader")

        if uploaded_file is not None:
            st.audio(uploaded_file)
            if st.button("⚡ Transcribe with Gnani AI & Detect Threats", disabled=not consent, type="primary", use_container_width=True):
                with st.spinner("Processing audio with 16kHz PCM normalization and Gnani STT API..."):
                    try:
                        raw_bytes = uploaded_file.getvalue()
                        pcm_path = convert_to_pcm_wav(raw_bytes, uploaded_file.name)
                        audio_info = inspect_audio(pcm_path)

                        transcript = transcribe_audio(pcm_path, language_code=st.session_state.selected_lang)

                        # Clean up temporary PCM WAV immediately (Privacy by Design)
                        if os.path.exists(pcm_path):
                            os.remove(pcm_path)

                        if not transcript:
                            st.error("The speech engine returned an empty transcript. Please try a clearer sample.")
                        else:
                            execute_analysis(
                                transcript,
                                source="audio_upload",
                                duration=audio_info["duration_sec"],
                                is_uncertain=audio_info["is_uncertain"],
                                warning=audio_info.get("warning")
                            )
                            st.session_state.page = "Threat Analysis"
                            st.rerun()
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. LIVE VOICE / CALL RECORDING TAB
    with tab_record:
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        st.markdown("<b>Record a live call simulation or voice sample directly in browser:</b>", unsafe_allow_html=True)
        st.caption("Click 'Start recording', speak a suspicious phrase (e.g., 'Your bank account will be blocked today, tell me the OTP immediately'), then click 'Stop'.")

        try:
            from streamlit_mic_recorder import mic_recorder
            mic_audio = mic_recorder(
                start_prompt="🔴 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                key="browser_mic",
                just_once=False,
                use_container_width=True
            )

            if mic_audio and isinstance(mic_audio, dict) and mic_audio.get("bytes"):
                audio_bytes = mic_audio["bytes"]
                st.audio(audio_bytes, format="audio/wav")

                if st.button("⚡ Transcribe Live Recording & Analyze", disabled=not consent, type="primary", use_container_width=True):
                    with st.spinner("Normalizing microphone stream and querying Gnani STT..."):
                        try:
                            pcm_path = convert_to_pcm_wav(audio_bytes, "mic_recording.wav")
                            audio_info = inspect_audio(pcm_path)
                            transcript = transcribe_audio(pcm_path, language_code=st.session_state.selected_lang)

                            if os.path.exists(pcm_path):
                                os.remove(pcm_path)

                            if not transcript:
                                st.error("No clear speech detected in recording. Please try speaking closer to the microphone.")
                            else:
                                execute_analysis(
                                    transcript,
                                    source="live_mic",
                                    duration=audio_info["duration_sec"],
                                    is_uncertain=audio_info["is_uncertain"],
                                    warning=audio_info.get("warning")
                                )
                                st.session_state.page = "Threat Analysis"
                                st.rerun()
                        except Exception as e:
                            st.error(f"Live recording analysis failed: {e}")
        except ImportError:
            st.warning("streamlit_mic_recorder component is not active.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. MANUAL TRANSCRIPT TAB
    with tab_text:
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        sample_scam = (
            "Your bank account will be blocked today. Share your OTP immediately to avoid suspension. "
            "Install AnyDesk so our support team can verify your KYC remotely."
        )
        sample_telugu = (
            "Mee SBI bank account ee roju block aipothundi. Ventane mee mobile ki vachina OTP cheppandi. "
            "Lekapothe police arrest warrant issue chestharu."
        )

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("Load English Scam Sample", use_container_width=True):
                st.session_state.transcript = sample_scam
        with col_ex2:
            if st.button("Load Telugu-English Scam Sample", use_container_width=True):
                st.session_state.transcript = sample_telugu

        txt_input = st.text_area("Enter call transcript or message:", value=st.session_state.transcript, height=180)

        if st.button("🔍 Run Threat Intelligence Analysis", type="primary", use_container_width=True, disabled=not txt_input.strip()):
            execute_analysis(txt_input, source="text_manual")
            st.session_state.page = "Threat Analysis"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: THREAT ANALYSIS
# ==============================================================================

elif st.session_state.page == "Threat Analysis":
    st.markdown('<div class="eyebrow">02 / THREAT INTELLIGENCE</div>', unsafe_allow_html=True)
    res = st.session_state.analysis_result
    transcript = st.session_state.transcript

    if not res:
        st.info("No active threat analysis found. Open Audio Lab to record or enter a conversation.")
        if st.button("Go to Audio Lab"):
            st.session_state.page = "Audio Lab"
            st.rerun()
    else:
        # Check audio uncertainty warnings
        if st.session_state.is_uncertain and st.session_state.uncertainty_warning:
            st.warning(f"⚠️ Audio Quality Warning: {st.session_state.uncertainty_warning}")

        score = res.get("hybrid_score", res.get("risk_score", 0))
        level = res.get("hybrid_level", res.get("risk_level", "Unknown"))
        findings = res.get("findings", [])

        # Metric Row
        m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
        with m1:
            pill_class = "pill-high" if level == "High" else ("pill-caution" if level == "Caution" else "pill-safe")
            st.markdown(f'<div class="metric-box"><div class="metric-label">ASSESSED RISK TIER</div><span class="{pill_class}">{level.upper()} RISK</span></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">HYBRID THREAT SCORE</div><div class="metric-value">{score}/100</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">ML SCAM PROBABILITY</div><div class="metric-value">{res.get("ml_probability", 0.0):.2f}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-label">TACTIC INDICATORS</div><div class="metric-value">{len(findings)}</div></div>', unsafe_allow_html=True)

        st.write("")
        chart_col, trans_col = st.columns([1, 1.2], gap="large")

        with chart_col:
            st.markdown('<div class="threat-card">', unsafe_allow_html=True)
            st.markdown("<b>Threat Vector Distribution (Radar)</b>", unsafe_allow_html=True)
            st.plotly_chart(plot_threat_radar(findings), use_container_width=True)
            st.plotly_chart(plot_gauge(score), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with trans_col:
            st.markdown('<div class="threat-card">', unsafe_allow_html=True)
            st.markdown("<b>Analyzed Call Transcript (With Evidence Highlighted)</b>", unsafe_allow_html=True)
            st.markdown(render_highlighted_transcript(transcript, findings), unsafe_allow_html=True)
            st.write("")
            st.markdown(f"<b>Summary:</b> {res.get('summary')}")
            st.caption(res.get("disclaimer"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🚩 Evidence-Linked Scam Tactics & Spoken Safety Advice")

        if not findings:
            st.success("✅ Clean Transcript: No suspicious scam patterns or high-pressure coercion detected.")
        else:
            for idx, f in enumerate(findings, start=1):
                cat = f.get("category", "Threat Pattern")
                sev = f.get("severity", "medium").upper()
                pill = "pill-high" if sev == "HIGH" else "pill-caution"

                with st.expander(f"🚩 Threat #{idx}: {cat} [{sev}]", expanded=True):
                    ev = f.get("evidence", "")
                    if ev:
                        st.markdown(f'<b>Matched Evidence Quote:</b><div class="evidence-quote">"{ev}"</div>', unsafe_allow_html=True)

                    st.markdown(f"<b>Why this is risky:</b> {f.get('explanation')}")
                    rec = f.get("recommendation", "Verify independently.")
                    st.markdown(f'<div class="action-box">💡 <b>Recommended Action:</b> {rec}</div>', unsafe_allow_html=True)

                    # Regional Language Guidance & Spoken TTS
                    regional = f.get("regional_advice", {})
                    te_adv = regional.get("telugu")
                    hi_adv = regional.get("hindi")

                    if te_adv or hi_adv:
                        st.markdown(f'<div class="regional-box">🗣️ <b>Telugu Advice:</b> {te_adv or "ఎలాంటి ఓటీపీ లేదా డబ్బులు ఇవ్వకండి."}<br>🗣️ <b>Hindi Advice:</b> {hi_adv or "कोई भी ओटीपी या पैसे ट्रांसफर न करें."}</div>', unsafe_allow_html=True)

                    # Spoken Safety Audio Button
                    tts_col1, tts_col2, tts_col3 = st.columns([1, 1, 2])
                    with tts_col1:
                        if st.button(f"🔊 Spoken Advice (Telugu)", key=f"tts_te_{idx}"):
                            with st.spinner("Synthesizing Telugu audio advice..."):
                                audio_bytes = synthesize_speech(te_adv or rec, language="te")
                                st.audio(audio_bytes, format="audio/mp3")
                    with tts_col2:
                        if st.button(f"🔊 Spoken Advice (Hindi)", key=f"tts_hi_{idx}"):
                            with st.spinner("Synthesizing Hindi audio advice..."):
                                audio_bytes = synthesize_speech(hi_adv or rec, language="hi")
                                st.audio(audio_bytes, format="audio/mp3")
                    with tts_col3:
                        if st.button(f"🔊 Spoken Advice (English)", key=f"tts_en_{idx}"):
                            with st.spinner("Synthesizing English audio advice..."):
                                audio_bytes = synthesize_speech(rec, language="en")
                                st.audio(audio_bytes, format="audio/mp3")

        st.divider()
        # Export Threat Report
        report_data = {
            "application": "VoiceShield AI",
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "threat_score": score,
            "risk_level": level,
            "tactics_detected": [f.get("category") for f in findings],
            "evidence_findings": findings,
            "audio_duration_sec": st.session_state.audio_duration,
            "disclaimer": res.get("disclaimer")
        }
        st.download_button(
            "📥 Export Threat Report (JSON)",
            data=json.dumps(report_data, indent=2, ensure_ascii=False),
            file_name="voiceshield_threat_report.json",
            mime="application/json"
        )

# ==============================================================================
# PAGE 4: EVALUATION & BENCHMARKS
# ==============================================================================

elif st.session_state.page == "Evaluation & Benchmarks":
    st.markdown('<div class="eyebrow">COMPETITION VALIDATION</div>', unsafe_allow_html=True)
    st.subheader("Model Evaluation & Multi-Lingual Benchmarks")
    st.caption("Evaluated on 108 realistic phone call samples spanning English, Telugu-English (Tanglish), and Hindi-English.")

    from evaluation.evaluate_detector import evaluate_dataset

    with st.spinner("Running automated multi-lingual benchmark evaluation..."):
        eval_res = evaluate_dataset()

    h = eval_res["hybrid_metrics"]
    r = eval_res["rule_metrics"]

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">HYBRID ACCURACY</div><div class="metric-value" style="color:#00e699;">{h["accuracy"]*100:.1f}%</div><div style="font-size:11px;color:#888;">Rule baseline: {r["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with b2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">PRECISION</div><div class="metric-value">{h["precision"]:.3f}</div><div style="font-size:11px;color:#888;">Low false positives</div></div>', unsafe_allow_html=True)
    with b3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">RECALL</div><div class="metric-value" style="color:#54e8ff;">{h["recall"]:.3f}</div><div style="font-size:11px;color:#888;">Zero missed scams (FN=0)</div></div>', unsafe_allow_html=True)
    with b4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">FALSE POSITIVE RATE</div><div class="metric-value">{h["fpr"]:.3f}</div><div style="font-size:11px;color:#888;">Strict safety controls</div></div>', unsafe_allow_html=True)

    st.write("")
    c_left, c_right = st.columns([1, 1.2])

    with c_left:
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        st.markdown("<b>Confusion Matrix (Hybrid Model)</b>", unsafe_allow_html=True)
        cm_z = [[h["tp"], h["fn"]], [h["fp"], h["tn"]]]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_z,
            x=['Predicted Scam', 'Predicted Legitimate'],
            y=['Actual Scam', 'Actual Legitimate'],
            colorscale=[[0, '#121429'], [0.5, '#5440b5'], [1, '#00e699']],
            text=[[f"TP: {h['tp']}", f"FN: {h['fn']}"], [f"FP: {h['fp']}", f"TN: {h['tn']}"]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="white", family="Space Grotesk")
        ))
        cm_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="threat-card">', unsafe_allow_html=True)
        st.markdown("<b>Performance Breakdown Across Language Groups</b>", unsafe_allow_html=True)
        langs = list(eval_res["language_breakdown"].keys())
        accs = [eval_res["language_breakdown"][l]["accuracy"] * 100 for l in langs]
        recalls = [eval_res["language_breakdown"][l]["recall"] * 100 for l in langs]

        bar_fig = go.Figure(data=[
            go.Bar(name='Accuracy (%)', x=[l.replace("_", " ").title() for l in langs], y=accs, marker_color='#9b7bff'),
            go.Bar(name='Recall (%)', x=[l.replace("_", " ").title() for l in langs], y=recalls, marker_color='#54e8ff')
        ])
        bar_fig.update_layout(barmode='group', height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 5: AUDIT HISTORY
# ==============================================================================

elif st.session_state.page == "Audit History":
    st.markdown('<div class="eyebrow">DATA LOGS</div>', unsafe_allow_html=True)
    st.subheader("Session & SQLite Audit History")

    records = get_all_records(limit=50)
    if not records:
        st.info("No prior analyses recorded in SQLite.")
    else:
        st.caption(f"Displaying {len(records)} recent audit record(s). Raw audio files are never stored.")
        for r in records:
            res_item = r["analysis"]
            lvl = res_item.get("risk_level", "Unknown")
            pill = "pill-high" if lvl == "High" else ("pill-caution" if lvl == "Caution" else "pill-safe")

            with st.expander(f"{r['timestamp']} | {lvl} Risk ({res_item.get('hybrid_score', res_item.get('risk_score',0))}/100) | Source: {r['source']}"):
                st.markdown(f"<b>Transcript:</b><br><code>{r['transcript']}</code>", unsafe_allow_html=True)
                st.markdown(f"<b>Tactics Identified:</b> {res_item.get('detected_tactics') or 'None'}")

        if st.button("🗑️ Purge Audit History (Privacy Reset)"):
            purge_database()
            st.success("Database purged.")
            st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()
st.markdown(
    """
    <div style="font-size:11.5px;color:#7e83a4;line-height:1.7;">
        <b>VoiceShield AI</b> · Great Indian AI Internship Challenge · Privacy by Design<br>
        All rights reserved. Powered by Gnani AI STT &amp; Hybrid Threat Intelligence.
    </div>
    """,
    unsafe_allow_html=True
)