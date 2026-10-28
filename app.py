
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

BG = "#080B18"
PURPLE = "#9B7BFF"
CYAN = "#46E6D5"
PINK = "#FF6FAE"
GOLD = "#FFC66D"
WHITE = "#F4F6FF"
MUTED = "#929BB8"

# =========================================================
# FUTURISTIC UI
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --purple: #9B7BFF;
        --cyan: #46E6D5;
        --pink: #FF6FAE;
        --white: #F4F6FF;
        --muted: #929BB8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 12% 4%, rgba(100,65,220,.16), transparent 32%),
          radial-gradient(ellipse at 88% 12%, rgba(38,218,207,.09), transparent 28%),
          #080B18;
        color: var(--white);
    }

    [data-testid="stHeader"] {
        background: rgba(8,11,24,.82);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#10152B,#090C1B);
        border-right: 1px solid rgba(155,123,255,.20);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F4F6FF !important;
        letter-spacing: -.035em;
    }

    p, label { color: #D5DCF2; }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -1.4px;
        color: white;
    }

    .brand span { color: var(--purple); }

    .brand-caption {
        color: var(--muted);
        font-size: .68rem;
        text-transform: uppercase;
        letter-spacing: .13em;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: .70rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 300px;
        padding: 32px;
        border-radius: 25px;
        border: 1px solid rgba(155,123,255,.30);
        background:
          radial-gradient(circle at 82% 48%, rgba(116,81,239,.18), transparent 30%),
          linear-gradient(120deg,rgba(28,27,66,.96),rgba(12,18,39,.97));
        margin-bottom: 22px;
    }

    .hero-copy {
        position: relative;
        z-index: 2;
        max-width: 67%;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem,4vw,3.15rem);
        font-weight: 700;
        letter-spacing: -.055em;
        line-height: 1.07;
        color: white;
        margin-bottom: 14px;
    }

    .gradient-text {
        background: linear-gradient(90deg,#B5A0FF,#55F0DE);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        color: #B6C0DE;
        font-size: .91rem;
        line-height: 1.8;
        max-width: 530px;
    }

    .hero-chip {
        display: inline-block;
        padding: 8px 12px;
        border: 1px solid rgba(70,230,213,.28);
        background: rgba(70,230,213,.07);
        color: #9BF7EB;
        border-radius: 999px;
        font-size: .69rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Holographic orb */
    .orb-stage {
        position: absolute;
        width: 270px;
        height: 270px;
        right: 2%;
        top: 50%;
        transform: translateY(-50%);
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
    }

    .orb-glow {
        position: absolute;
        width: 190px;
        height: 190px;
        border-radius: 50%;
        background: rgba(117,80,217,.18);
        filter: blur(35px);
        animation: glowPulse 4s ease-in-out infinite;
    }

    .orb-halo {
        position: absolute;
        width: 235px;
        height: 235px;
        border-radius: 50%;
        border: 1px solid rgba(155,123,255,.4);
        box-shadow: 0 0 35px rgba(155,123,255,.15);
        animation: haloPulse 5s ease-in-out infinite;
    }

    .orb-ring {
        position: absolute;
        width: 205px;
        height: 205px;
        border-radius: 50%;
        border: 2px solid transparent;
        border-top-color: var(--cyan);
        border-right-color: var(--purple);
        filter: drop-shadow(0 0 8px rgba(70,230,213,.5));
        animation: orbit 8s linear infinite;
    }

    .orb-ring.two {
        width: 170px;
        height: 170px;
        border-top-color: var(--pink);
        border-right-color: var(--cyan);
        animation-duration: 6s;
        animation-direction: reverse;
    }

    .orb-core {
        position: relative;
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: radial-gradient(
          circle at 32% 25%,#F0E9FF 0%,#B9A1FF 10%,
          #7550D9 29%,#39267F 53%,#10162D 75%
        );
        box-shadow:
          0 0 28px rgba(155,123,255,.72),
          0 0 70px rgba(117,80,217,.4),
          inset -12px -14px 25px rgba(0,0,0,.65),
          inset 8px 7px 20px rgba(255,255,255,.18);
        animation: floatOrb 4s ease-in-out infinite;
    }

    .orb-core:after {
        content: "";
        position: absolute;
        inset: 17px 20px 27px 19px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.27);
        transform: rotate(-24deg);
    }

    @keyframes orbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes floatOrb {
        0%,100% { transform: translateY(0); }
        50% { transform: translateY(-9px); }
    }

    @keyframes haloPulse {
        0%,100% { opacity: .55; transform: scale(.97); }
        50% { opacity: 1; transform: scale(1.04); }
    }

    @keyframes glowPulse {
        0%,100% { opacity: .55; }
        50% { opacity: 1; }
    }

    .metric-card {
        min-height: 132px;
        padding: 19px;
        border-radius: 17px;
        background: linear-gradient(145deg,rgba(28,35,66,.88),rgba(15,20,40,.92));
        border: 1px solid rgba(155,123,255,.20);
        box-shadow: 0 8px 28px rgba(0,0,0,.12);
        transition: transform .22s ease,border-color .22s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(70,230,213,.48);
    }

    .metric-label {
        color: #A8B2D0;
        font-size: .70rem;
        font-weight: 600;
        letter-spacing: .055em;
        margin-bottom: 12px;
    }

    .metric-value {
        color: white;
        font-family: 'Space Grotesk',sans-serif;
        font-size: clamp(1.35rem,2vw,1.9rem);
        font-weight: 700;
        letter-spacing: -.04em;
    }

    .metric-foot {
        color: #8490B1;
        font-size: .72rem;
        margin-top: 9px;
    }

    .section-heading {
        font-family: 'Space Grotesk',sans-serif;
        font-size: 1.12rem;
        font-weight: 600;
        color: #F4F6FF;
        margin: 13px 0 4px;
    }

    .section-caption {
        color: #929BB8;
        font-size: .80rem;
        margin-bottom: 13px;
    }

    .glass-card {
        background: rgba(17,23,44,.76);
        border: 1px solid rgba(155,123,255,.18);
        border-radius: 17px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .insight-card {
        border-left: 3px solid var(--cyan);
        border-radius: 0 12px 12px 0;
        padding: 13px 15px;
        margin: 10px 0;
        background: rgba(70,230,213,.055);
        color: #DCE3F9;
        font-size: .84rem;
        line-height: 1.65;
    }

    .muted {
        color: #929BB8;
        font-size: .77rem;
    }

    .risk-box {
        border-radius: 18px;
        padding: 23px;
        min-height: 180px;
        background: rgba(17,23,44,.84);
    }

    .risk-high {
        border: 1px solid rgba(255,111,174,.48);
        background: linear-gradient(140deg,rgba(255,111,174,.12),rgba(17,23,44,.8));
    }

    .risk-medium {
        border: 1px solid rgba(255,198,109,.45);
        background: linear-gradient(140deg,rgba(255,198,109,.10),rgba(17,23,44,.8));
    }

    .risk-low {
        border: 1px solid rgba(70,230,213,.4);
        background: linear-gradient(140deg,rgba(70,230,213,.10),rgba(17,23,44,.8));
    }

    .risk-label {
        font-size: .74rem;
        letter-spacing: .13em;
        font-weight: 700;
    }

    .risk-number {
        font-family: 'Space Grotesk',sans-serif;
        font-size: 2.7rem;
        font-weight: 700;
        margin: 8px 0;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        border: 1px solid rgba(155,123,255,.42);
        background: linear-gradient(110deg,#7558DD,#5140AE);
        color: white;
        font-weight: 600;
        padding: .62rem 1rem;
        transition: all .2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--cyan);
        color: white;
        box-shadow: 0 0 18px rgba(155,123,255,.2);
    }

    [data-testid="stMetric"] {
        background: rgba(20,25,47,.7);
        border: 1px solid rgba(155,123,255,.16);
        padding: 13px;
        border-radius: 13px;
    }

    [data-testid="stMetricLabel"] { color: #A8B2D0 !important; }
    [data-testid="stMetricValue"] { color: #F4F6FF !important; }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #12182D;
        border-color: rgba(155,123,255,.25);
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #6F7897;
        font-size: .73rem;
        padding: 24px 0 8px;
        border-top: 1px solid rgba(155,123,255,.12);
        margin-top: 32px;
        line-height: 1.9;
    }

    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 850px) {
        .hero-copy { max-width: 100%; }
        .orb-stage { opacity: .22; right: -80px; }
    }

    @media (prefers-reduced-motion: reduce) {
        .orb-halo,.orb-ring,.orb-core,.orb-glow {
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA AND EXISTING TRAINED MODEL
# =========================================================
@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )
    return data

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Check dataset/customer_churn.csv exists.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run python train_model.py first.")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

if "Churn" not in df.columns:
    st.error("Your dataset must contain a Churn column.")
    st.stop()

df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

feature_columns = [
    col for col in df.columns
    if col not in ["Churn", "customerID"]
]

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_customers * 100 if total_customers else 0

avg_monthly = (
    df["MonthlyCharges"].mean()
    if "MonthlyCharges" in df.columns else 0
)

# =========================================================
# REUSABLE HELPERS
# =========================================================
def section_title(title, caption=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<div class="section-caption">{caption}</div>',
            unsafe_allow_html=True,
        )

def metric_card(label, value, note, accent=PURPLE):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
            <div class="metric-foot">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_chart(fig, height=330, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans,sans-serif", color="#B8C1DF", size=12),
        margin=dict(l=10, r=15, t=35, b=10),
        hoverlabel=dict(
            bgcolor="#151A30", bordercolor=PURPLE, font_color="#FFFFFF"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ) if legend else dict(visible=False),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(146,155,184,.10)", zeroline=False)
    return fig

def churn_rate_by(data, column):
    return (
        data.groupby(column, dropna=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="Churn rate")
    )

def positive_probability(pipeline, frame):
    probabilities = pipeline.predict_proba(frame)[0]

    classes = getattr(pipeline, "classes_", None)
    if classes is None and hasattr(pipeline, "named_steps"):
        final_estimator = list(pipeline.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)

    if classes is not None:
        for i, label in enumerate(classes):
            if str(label).strip().lower() in ["yes", "1", "true"]:
                return float(probabilities[i])

    # This is a fallback for the usual binary Yes/No churn dataset.
    return float(probabilities[-1])

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0 5px 24px">
            <div class="brand">Churn<span>IQ</span> ◈</div>
            <div class="brand-caption">Customer Churn Prediction System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**WORKSPACE**")

    page = st.radio(
        "Navigation",
        [
            "Churn Intelligence Dashboard",
            "Customer Churn Analysis",
            "Churn Prediction",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="color:#46E6D5;font-weight:700;font-size:.82rem">
                ● &nbsp; Model loaded
            </div>
            <div class="muted" style="margin-top:8px">
                Saved ML pipeline ready
            </div>
            <div class="muted" style="margin-top:5px">
                {len(feature_columns)} customer features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("PYTHON · STREAMLIT · PLOTLY")
    st.caption("ChurnIQ • Customer analytics")

# =========================================================
# PAGE 1: CHURN INTELLIGENCE DASHBOARD
# =========================================================
if page == "Churn Intelligence Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-copy">
                <div class="eyebrow">◈ &nbsp; CUSTOMER CHURN PREDICTION SYSTEM</div>
                <div class="hero-title">
                    Understand churn.<br>
                    <span class="gradient-text">Predict what comes next.</span>
                </div>
                <div class="hero-description">
                    Explore customer behaviour, identify churn patterns,
                    and use machine learning predictions to support
                    smarter customer-retention decisions.
                </div>
                <span class="hero-chip">✦ &nbsp; MACHINE LEARNING INSIGHTS</span>
            </div>
            <div class="orb-stage">
                <div class="orb-glow"></div>
                <div class="orb-halo"></div>
                <div class="orb-ring"></div>
                <div class="orb-ring two"></div>
                <div class="orb-core"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Customer pulse")
    st.caption("Summary of the customer records currently loaded.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL CUSTOMERS", f"{total_customers:,}",
                    "Customer records", PURPLE)
    with c2:
        metric_card("CHURNED CUSTOMERS", f"{churned:,}",
                    "Marked as churned", PINK)
    with c3:
        metric_card("CHURN RATE", f"{churn_rate:.2f}%",
                    "Observed in the dataset", CYAN)
    with c4:
        metric_card("AVG. MONTHLY CHARGE", f"${avg_monthly:,.2f}",
                    "Average customer charge", GOLD)

    st.write("")
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        section_title(
            "Customer retention balance",
            "Distribution of churned and retained customers.",
        )

        fig = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[retained, churned],
            hole=.76,
            sort=False,
            marker=dict(
                colors=[CYAN, PINK],
                line=dict(color=BG, width=5),
            ),
            textinfo="percent",
            hovertemplate="%{label}: %{value:,}<br>%{percent}<extra></extra>",
        ))

        fig.add_annotation(
            x=.5, y=.5,
            text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px'>CHURN</span>",
            showarrow=False,
            font=dict(size=22, color=WHITE),
        )

        st.plotly_chart(style_chart(fig, 320), use_container_width=True)

    with right:
        section_title(
            "Churn by contract",
            "Compare the observed churn rate across contract types.",
        )

        if "Contract" in df.columns:
            summary = churn_rate_by(df, "Contract").sort_values("Churn rate")
            fig = px.bar(
                summary,
                x="Churn rate",
                y="Contract",
                orientation="h",
                text=summary["Churn rate"].map(lambda v: f"{v:.1f}%"),
                color="Churn rate",
                color_continuous_scale=[CYAN, PURPLE, PINK],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(
                style_chart(fig, 320, False), use_container_width=True
            )
        else:
            st.info("Contract data is unavailable.")

    st.write("")
    section_title(
        "Customer galaxy · Interactive 3D",
        "Rotate, zoom and hover to explore tenure, monthly charges and total charges.",
    )

    needed = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]

    if all(c in df.columns for c in needed):
        cloud = df[needed].copy()

        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            cloud[col] = pd.to_numeric(cloud[col], errors="coerce")

        cloud = cloud.dropna(subset=needed[:-1])

        if len(cloud) > 1800:
            cloud = cloud.sample(1800, random_state=42)

        fig3d = px.scatter_3d(
            cloud,
            x="tenure",
            y="MonthlyCharges",
            z="TotalCharges",
            color="Churn",
            color_discrete_map={"Yes": PINK, "No": CYAN},
            opacity=.78,
            labels={
                "tenure": "Tenure (months)",
                "MonthlyCharges": "Monthly charges",
                "TotalCharges": "Total charges",
            },
        )

        fig3d.update_traces(marker=dict(size=3.5))
        fig3d.update_layout(
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#C7D0ED"),
            margin=dict(l=0, r=0, t=15, b=0),
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Tenure", gridcolor="rgba(155,123,255,.14)"),
                yaxis=dict(title="Monthly charges", gridcolor="rgba(70,230,213,.13)"),
                zaxis=dict(title="Total charges", gridcolor="rgba(155,123,255,.14)"),
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
            ),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
        st.caption(
            "Each point is a customer record. This visualization explores "
            "the dataset; it is not itself a model prediction."
        )
    else:
        st.info(
            "The 3D chart requires tenure, MonthlyCharges, TotalCharges and Churn."
        )

    st.write("")
    section_title(
        "Customer signals",
        "Observed patterns to investigate further, not proof of causation.",
    )

    a, b, c = st.columns(3)

    with a:
        if "Contract" in df.columns:
            rates = churn_rate_by(df, "Contract")
            highest = rates.loc[rates["Churn rate"].idxmax()]
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="eyebrow">CONTRACT SIGNAL</div>
                    <h3>{highest['Contract']}</h3>
                    <div style="font-size:1.8rem;color:#FF6FAE;font-weight:700">
                        {highest['Churn rate']:.1f}%
                    </div>
                    <div class="muted">Highest observed contract churn rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with b:
        if "tenure" in df.columns:
            early = df[pd.to_numeric(df["tenure"], errors="coerce") <= 12]
            if not early.empty:
                early_rate = (early["Churn"] == "Yes").mean() * 100
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="eyebrow">EARLY CUSTOMER JOURNEY</div>
                        <h3>First 12 months</h3>
                        <div style="font-size:1.8rem;color:#46E6D5;font-weight:700">
                            {early_rate:.1f}%
                        </div>
                        <div class="muted">Observed churn rate in this group</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with c:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="eyebrow">RETENTION OPPORTUNITY</div>
                <h3>Customers retained</h3>
                <div style="font-size:1.8rem;color:#9B7BFF;font-weight:700">
                    {retained:,}
                </div>
                <div class="muted">
                    {100 - churn_rate:.1f}% of the loaded customer base
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE 2: CUSTOMER CHURN ANALYSIS
# =========================================================
elif page == "Customer Churn Analysis":

    st.markdown(
        '<div class="eyebrow">CUSTOMER BEHAVIOUR LAB</div>',
        unsafe_allow_html=True,
    )
    st.title("Customer Churn Analysis")
    st.caption(
        "Explore customer segments, compare churn rates and inspect the records."
    )

    filtered = df.copy()

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        if "Contract" in filtered.columns:
            contracts = sorted(filtered["Contract"].dropna().unique().tolist())
            selected_contract = st.selectbox(
                "Contract type", ["All"] + contracts
            )
            if selected_contract != "All":
                filtered = filtered[filtered["Contract"] == selected_contract]

    with filter2:
        if "InternetService" in filtered.columns:
            services = sorted(
                filtered["InternetService"].dropna().unique().tolist()
            )
            selected_service = st.selectbox(
                "Internet service", ["All"] + services
            )
            if selected_service != "All":
                filtered = filtered[
                    filtered["InternetService"] == selected_service
                ]

    with filter3:
        if "SeniorCitizen" in filtered.columns:
            selected_senior = st.selectbox(
                "Senior citizen status", ["All", "Yes", "No"]
            )
            if selected_senior != "All":
                target = 1 if selected_senior == "Yes" else 0
                filtered = filtered[filtered["SeniorCitizen"] == target]

    filtered_count = len(filtered)
    filtered_churn = int((filtered["Churn"] == "Yes").sum())
    filtered_rate = (
        filtered_churn / filtered_count * 100 if filtered_count else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered customers", f"{filtered_count:,}")
    m2.metric("Churned in segment", f"{filtered_churn:,}")
    m3.metric("Segment churn rate", f"{filtered_rate:.2f}%")

    st.write("")
    left, right = st.columns(2)

    with left:
        section_title("Churn by tenure", "Churn rates grouped by customer tenure.")

        if "tenure" in filtered.columns and not filtered.empty:
            temp = filtered.copy()
            temp["Tenure group"] = pd.cut(
                pd.to_numeric(temp["tenure"], errors="coerce"),
                bins=[-1, 12, 24, 48, 72],
                labels=["0–12", "13–24", "25–48", "49–72"],
            )
            temp = temp.dropna(subset=["Tenure group"])

            if not temp.empty:
                summary = churn_rate_by(temp, "Tenure group")
                fig = px.bar(
                    summary,
                    x="Tenure group",
                    y="Churn rate",
                    text=summary["Churn rate"].map(lambda v: f"{v:.1f}%"),
                    color="Churn rate",
                    color_continuous_scale=[CYAN, PURPLE, PINK],
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(
                    style_chart(fig, 350, False), use_container_width=True
                )
        else:
            st.info("No tenure data available for this segment.")

    with right:
        section_title(
            "Monthly charges",
            "Compare monthly charges for churned and retained customers.",
        )

        if "MonthlyCharges" in filtered.columns and not filtered.empty:
            fig = px.box(
                filtered,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                color_discrete_map={"Yes": PINK, "No": CYAN},
                points="outliers",
            )
            st.plotly_chart(
                style_chart(fig, 350), use_container_width=True
            )
        else:
            st.info("No monthly charge data available.")

    st.write("")
    section_title("Customer records", "A preview of the filtered dataset.")

    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button(
        "Download filtered customer data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="churn_filtered_customers.csv",
        mime="text/csv",
    )

# =========================================================
# PAGE 3: CHURN PREDICTION
# =========================================================
elif page == "Churn Prediction":

    st.markdown(
        '<div class="eyebrow">MACHINE LEARNING · CUSTOMER RISK ASSESSMENT</div>',
        unsafe_allow_html=True,
    )
    st.title("Churn Prediction")
    st.caption(
        "Enter customer details to estimate churn probability using your saved model."
    )

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:1.05rem;font-weight:700;color:#F4F6FF">
                ◈ &nbsp; AI Risk Studio
            </div>
            <div class="muted" style="margin-top:6px;line-height:1.7">
                Complete the customer profile below. The model will evaluate
                the supplied attributes and return a churn-risk estimate.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use the original preprocessor's learned feature names so the
    # prediction input matches the saved pipeline.
    try:
        preprocessor = model.named_steps["preprocessor"]
        model_features = list(preprocessor.feature_names_in_)
    except (AttributeError, KeyError):
        model_features = list(feature_columns)

    excluded = {"customerID", "Churn"}
    model_features = [
        col for col in model_features
        if col not in excluded and col in df.columns
    ]

    categorical_columns = [
        col for col in model_features
        if (
            pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
        )
    ]

    numeric_columns = [
        col for col in model_features
        if col not in categorical_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    input_values = {}

    # Group fields into readable sections.
    with st.form("customer_churn_prediction_form"):

        section_title(
            "01 · Customer profile",
            "Basic customer and service details.",
        )

        profile_cols = st.columns(2)

        with profile_cols[0]:
            for col in categorical_columns:
                options = sorted(df[col].dropna().astype(str).unique().tolist())

                if not options:
                    continue

                # Split fields into two balanced columns.
                if categorical_columns.index(col) % 2 == 0:
                    input_values[col] = st.selectbox(
                        col,
                        options,
                        key=f"prediction_{col}",
                    )

        with profile_cols[1]:
            for col in categorical_columns:
                options = sorted(df[col].dropna().astype(str).unique().tolist())

                if not options:
                    continue

                if categorical_columns.index(col) % 2 == 1:
                    input_values[col] = st.selectbox(
                        col,
                        options,
                        key=f"prediction_{col}",
                    )

        st.write("")
        section_title(
            "02 · Numeric attributes",
            "Enter the customer's numeric information.",
        )

        numeric_cols_ui = st.columns(2)

        for i, col in enumerate(numeric_columns):
            with numeric_cols_ui[i % 2]:
                series = pd.to_numeric(df[col], errors="coerce").dropna()

                if series.empty:
                    input_values[col] = 0
                    continue

                min_value = float(series.min())
                max_value = float(series.max())
                median_value = float(series.median())

                if col in ["tenure", "SeniorCitizen"]:
                    input_values[col] = st.number_input(
                        col,
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(round(median_value)),
                        step=1,
                        key=f"prediction_{col}",
                    )
                else:
                    input_values[col] = st.number_input(
                        col,
                        min_value=min_value,
                        max_value=max_value,
                        value=float(round(median_value, 2)),
                        step=1.0,
                        key=f"prediction_{col}",
                    )

        submitted = st.form_submit_button(
            "✦  Analyse Churn Risk",
            use_container_width=True,
        )

    if submitted:
        try:
            # Reconstruct input in the exact feature order expected by the model.
            prediction_row = {}

            for col in model_features:
                if col in input_values:
                    prediction_row[col] = input_values[col]
                elif col in df.columns:
                    prediction_row[col] = df[col].mode().iloc[0]

            input_frame = pd.DataFrame(
                [prediction_row],
                columns=model_features,
            )

            # Ensure numeric columns have numeric values.
            for col in numeric_columns:
                if col in input_frame.columns:
                    input_frame[col] = pd.to_numeric(
                        input_frame[col], errors="coerce"
                    )

            # Match the string categorical representation used in the dataset.
            for col in categorical_columns:
                if col in input_frame.columns:
                    input_frame[col] = input_frame[col].astype(str)

            probability = positive_probability(model, input_frame)
            prediction = model.predict(input_frame)[0]

            # For the common Yes/No churn target, probability is the
            # estimated probability of churn. Prediction is the model's class.
            is_churn = str(prediction).strip().lower() in [
                "yes", "1", "true"
            ]

            if probability >= 0.65:
                risk = "HIGH"
                risk_class = "risk-high"
                risk_color = PINK
                action_title = "Prioritise retention outreach"
                actions = [
                    "Contact the customer proactively to understand concerns.",
                    "Review contract flexibility and service experience.",
                    "Offer a relevant retention option where appropriate.",
                ]
            elif probability >= 0.35:
                risk = "MODERATE"
                risk_class = "risk-medium"
                risk_color = GOLD
                action_title = "Monitor and engage"
                actions = [
                    "Check customer satisfaction and recent service issues.",
                    "Share relevant plan or support options.",
                    "Monitor engagement and reassess risk later.",
                ]
            else:
                risk = "LOW"
                risk_class = "risk-low"
                risk_color = CYAN
                action_title = "Maintain customer experience"
                actions = [
                    "Continue reliable service and support.",
                    "Share relevant benefits without unnecessary offers.",
                    "Reassess risk if customer circumstances change.",
                ]

            st.write("")
            st.markdown("---")
            section_title(
                "Prediction result",
                "Model output for the customer profile you submitted.",
            )

            result_col, gauge_col = st.columns([1, 1.05], gap="large")

            with result_col:
                st.markdown(
                    f"""
                    <div class="risk-box {risk_class}">
                        <div class="risk-label" style="color:{risk_color}">
                            {risk} CHURN RISK
                        </div>
                        <div class="risk-number" style="color:{risk_color}">
                            {probability * 100:.1f}%
                        </div>
                        <div style="color:#D8DFF6;font-weight:600">
                            Estimated churn probability
                        </div>
                        <div class="muted" style="margin-top:10px">
                            Predicted class: {"Churn" if is_churn else "No Churn"}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with gauge_col:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                        "font": {"size": 34, "color": WHITE},
                    },
                    title={
                        "text": "CHURN PROBABILITY",
                        "font": {"size": 12, "color": MUTED},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": MUTED,
                            "tickfont": {"color": MUTED},
                        },
                        "bar": {"color": risk_color, "thickness": .24},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 35], "color": "rgba(70,230,213,.13)"},
                            {"range": [35, 65], "color": "rgba(255,198,109,.13)"},
                            {"range": [65, 100], "color": "rgba(255,111,174,.13)"},
                        ],
                        "threshold": {
                            "line": {"color": WHITE, "width": 3},
                            "thickness": .75,
                            "value": probability * 100,
                        },
                    },
                ))

                gauge.update_layout(
                    height=250,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": WHITE},
                    margin=dict(l=25, r=25, t=50, b=10),
                )

                st.plotly_chart(gauge, use_container_width=True)

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="eyebrow">RECOMMENDED NEXT STEP</div>
                    <h3>{action_title}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for action in actions:
                st.markdown(
                    f'<div class="insight-card">✦ &nbsp; {action}</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "Risk bands are illustrative dashboard thresholds, not calibrated "
                "business rules. A probability estimate is not a guarantee."
            )

        except Exception as exc:
            st.error("The prediction could not be completed.")
            st.exception(exc)
            st.info(
                "If this happens, share the complete error message. "
                "We can adjust the input form to match your saved model."
            )

# =========================================================
# PAGE 4: MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.markdown(
        '<div class="eyebrow">MODEL EVALUATION & PERFORMANCE</div>',
        unsafe_allow_html=True,
    )
    st.title("Model Performance")
    st.caption(
        "Review model details and evaluate its predictions on a held-out split."
    )

    try:
        classifier = list(model.named_steps.values())[-1]
        classifier_name = type(classifier).__name__
    except Exception:
        classifier_name = type(model).__name__

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="eyebrow">LOADED MODEL</div>
            <h2>{classifier_name}</h2>
            <div class="muted">
                Using the existing saved pipeline. No retraining is performed
                by this dashboard page.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "The metrics below are calculated when this page is opened, using a "
        "new stratified hold-out split. They may differ from the original "
        "training report. The saved model is not retrained."
    )

    try:
        X = df[feature_columns].copy()
        y = df["Churn"].copy()

        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(
                X["TotalCharges"], errors="coerce"
            )

        # Keep feature rows with complete input data.
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        X_train, X_test, y_train, y_test = __import__(
            "sklearn.model_selection", fromlist=["train_test_split"]
        ).train_test_split(
            X,
            y,
            test_size=.20,
            random_state=42,
            stratify=y,
        )

        predictions = model.predict(X_test)
        probabilities = positive_probability if False else None

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(
                list(model.named_steps.values())[-1], "classes_", None
            )

        churn_index = None
        if classes is not None:
            for i, label in enumerate(classes):
                if str(label).strip().lower() in ["yes", "1", "true"]:
                    churn_index = i
                    break

        if hasattr(model, "predict_proba"):
            probability_matrix = model.predict_proba(X_test)
            if churn_index is None:
                churn_index = probability_matrix.shape[1] - 1
            churn_probabilities = probability_matrix[:, churn_index]
        else:
            churn_probabilities = None

        # Convert actual and predicted churn labels to 0 and 1.
        y_true = (
            y_test.astype(str)
            .str.strip()
            .str.lower()
            .map({"no": 0, "yes": 1, "0": 0, "1": 1})
            .astype(int)
        )

        y_pred = (
            pd.Series(predictions, index=y_test.index)
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"no": 0, "yes": 1, "0": 0, "1": 1})
            .astype(int)
        )



        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        m2.metric("Precision", f"{precision * 100:.2f}%")
        m3.metric("Recall", f"{recall * 100:.2f}%")
        m4.metric("F1 score", f"{f1:.3f}")

        if churn_probabilities is not None and y_true.nunique() == 2:
            auc = roc_auc_score(y_true, churn_probabilities)
            st.metric("ROC-AUC", f"{auc:.4f}")

        st.write("")
        left, right = st.columns(2)

        with left:
            section_title("Confusion matrix", "Predicted versus actual churn labels.")

            cm = __import__(
                "sklearn.metrics", fromlist=["confusion_matrix"]
            ).confusion_matrix(y_true, y_pred, labels=[0, 1])

            cm_fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                color_continuous_scale=[
                    "#11172C", "#6550B8", "#46E6D5"
                ],
                aspect="auto",
            )

            cm_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(
                style_chart(cm_fig, 360, False), use_container_width=True
            )

        with right:
            section_title("Metric interpretation", "What the measures tell you.")

            explanations = [
                ("Accuracy", "Share of all predictions that are correct."),
                ("Precision", "Of customers predicted to churn, how many actually churned."),
                ("Recall", "Of actual churners, how many the model detected."),
                ("F1 score", "Balance between precision and recall."),
                ("ROC-AUC", "How well churners and retained customers are ranked."),
            ]

            for name, description in explanations:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>{name}</b><br>{description}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error("Could not calculate evaluation metrics for this model.")
        st.exception(exc)
        st.info(
            "Your model and dataset may use different feature columns. "
            "Share the error message if you need help matching them."
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <b style="color:#B5A0FF">ChurnIQ</b> · Customer Churn Prediction System<br>
        Built with Python · Streamlit · Scikit-learn · Plotly<br>
        <span>Predictions support analysis and should be validated before business decisions.</span>
    </div>
    """,
    unsafe_allow_html=True,
)