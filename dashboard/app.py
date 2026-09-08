
from pathlib import Path
import sys, pickle, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from data_preprocessing import clean_data, add_business_features
from predict import predict, explain_customer, recommendation

st.set_page_config(page_title="ChurnIQ | Business Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ---------- Professional Dark Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg: #070a0f;
    --panel: #0e131b;
    --panel-2: #121923;
    --border: rgba(148,163,184,.14);
    --text: #f4f7fb;
    --muted: #91a0b5;
    --accent: #7c83ff;
    --accent-2: #22d3ee;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #fb7185;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 0%, rgba(124,131,255,.10), transparent 27%),
        radial-gradient(circle at 92% 12%, rgba(34,211,238,.07), transparent 25%),
        var(--bg);
}

[data-testid="stHeader"] {
    background: rgba(7,10,15,.72);
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(14,19,27,.98), rgba(8,11,17,.98));
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

.block-container {
    padding: 2rem 2.4rem 3rem;
    max-width: 1540px;
}

h1 {
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -.045em !important;
    color: var(--text) !important;
    margin-bottom: .3rem !important;
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 750 !important;
    letter-spacing: -.03em !important;
}

h3 {
    font-size: 1.18rem !important;
    font-weight: 700 !important;
}

p, label, [data-testid="stCaptionContainer"] {
    font-size: 1rem;
}

[data-testid="stSidebar"] h2 {
    font-size: 1.5rem !important;
}

[data-testid="stSidebar"] .stCaption {
    color: var(--muted);
    font-size: .88rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px;
    min-height: 165px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(124,131,255,.22);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(124,131,255,.13), rgba(14,19,27,.82) 48%, rgba(34,211,238,.07)),
        rgba(14,19,27,.92);
    box-shadow: 0 25px 70px rgba(0,0,0,.30);
    animation: fadeUp .65s ease both;
}

.hero::before {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -70px;
    top: -110px;
    border-radius: 50%;
    background: rgba(124,131,255,.18);
    filter: blur(35px);
    animation: floatGlow 5s ease-in-out infinite;
}

.hero::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    left: 42%;
    bottom: -130px;
    border-radius: 50%;
    background: rgba(34,211,238,.10);
    filter: blur(30px);
}

.hero-title {
    position: relative;
    z-index: 1;
    font-size: 2.45rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.055em;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #ffffff, #bfc5ff 48%, #8be9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.muted {
    position: relative;
    z-index: 1;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
    max-width: 850px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(18,25,35,.96), rgba(10,14,21,.96));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 19px 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    animation: fadeUp .55s ease both;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(124,131,255,.35);
    box-shadow: 0 18px 42px rgba(0,0,0,.28);
}

[data-testid="stMetricLabel"] {
    font-size: .86rem !important;
    font-weight: 600 !important;
    color: #9eabc0 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #f7f9ff !important;
    letter-spacing: -.035em;
}

[data-testid="stMetricDelta"] {
    font-size: .84rem !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
}

div[data-testid="stFileUploader"] {
    background: rgba(14,19,27,.85);
    border: 1px dashed rgba(124,131,255,.38);
    border-radius: 17px;
    padding: 12px;
    transition: border-color .2s ease, background .2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(34,211,238,.65);
    background: rgba(18,25,35,.95);
}

.stButton > button, .stDownloadButton > button {
    border: 1px solid rgba(124,131,255,.30) !important;
    border-radius: 11px !important;
    background: linear-gradient(135deg, rgba(124,131,255,.17), rgba(34,211,238,.07)) !important;
    color: #f4f7fb !important;
    font-weight: 650 !important;
    min-height: 42px;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    border-color: rgba(124,131,255,.60) !important;
    box-shadow: 0 9px 25px rgba(0,0,0,.25);
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #101722 !important;
    border-color: var(--border) !important;
    border-radius: 11px !important;
}

[data-baseweb="select"] * {
    font-size: .96rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: rgba(14,19,27,.7);
    padding: 6px;
    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 9px 17px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: rgba(124,131,255,.16);
}

[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid var(--border);
}

.sidebar-brand {
    padding: 10px 10px 17px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.04em;
    background: linear-gradient(90deg,#fff,#aeb5ff,#8be9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-sub {
    margin-top: 5px;
    color: #8290a6;
    font-size: .78rem;
    line-height: 1.5;
}

.section-kicker {
    color: #8f9db2;
    text-transform: uppercase;
    letter-spacing: .13em;
    font-size: .72rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.badge-high { color: var(--red); font-weight: 800; }
.badge-med { color: var(--yellow); font-weight: 800; }
.badge-low { color: var(--green); font-weight: 800; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes floatGlow {
    0%,100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(-15px,18px) scale(1.08); }
}

@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1rem 2rem; }
    .hero { padding: 26px 24px; min-height: 145px; }
    .hero-title { font-size: 1.85rem; }
    h1 { font-size: 1.9rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.45rem !important; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default():
    p = ROOT/"data/processed/customer_churn_processed.csv"
    if not p.exists():
        p = ROOT/"data/raw/customer_churn.csv"
    return add_business_features(clean_data(pd.read_csv(p)))

@st.cache_data
def load_metrics():
    return pd.read_csv(ROOT/"models/model_metrics.csv")

@st.cache_data
def load_importance():
    return pd.read_csv(ROOT/"models/feature_importance.csv").head(12)

df = load_default()
metrics = load_metrics()
importance = load_importance()

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-logo">◈ Churn-IQ</div>
    <div class="sidebar-sub">Customer Churn Prediction<br>& Business Intelligence</div>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Executive Overview","Customer Analysis","Churn Analysis","Prediction","Model Lab","Data Explorer"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.info("Built with Python • Pandas • Scikit-learn • XGBoost • Plotly • Streamlit • SQL")

# ---------- Common calculations ----------
churn_rate = (df["Churn"]=="Yes").mean()
try:
    with open(ROOT/"models/model_metadata.json") as f:
        meta=json.load(f)
    best_model=meta["best_model"]
except:
    best_model="Best trained model"

try:
    scored = predict(df)
except Exception:
    scored = df.copy()
    scored["Churn_Probability"] = np.nan
    scored["Risk_Level"] = "UNKNOWN"
    scored["Estimated_Revenue_At_Risk"] = 0

high = scored[scored["Risk_Level"]=="HIGH"]
revenue_risk = scored["Estimated_Revenue_At_Risk"].sum()

if page == "Executive Overview":
    st.markdown("<div class=\"section-kicker\"></div>", unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-title">Customer Churn Intelligence</div><div class="muted">Turn customer behavior into retention decisions — predict risk, explain drivers, and prioritize action.</div></div>', unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Current Churn Rate", f"{churn_rate*100:.1f}%")
    c3.metric("High-Risk Customers", f"{len(high):,}")
    c4.metric("Revenue at Risk", f"${revenue_risk:,.0f}")
    st.write("")
    a,b = st.columns([1.25,1])
    with a:
        st.subheader("Churn by Contract")
        x = df.groupby("Contract_Type", as_index=False).agg(Churn_Rate=("Churn", lambda s:(s=="Yes").mean()))
        x["Churn_Rate"]*=100
        fig=px.bar(x,x="Contract_Type",y="Churn_Rate",text="Churn_Rate",template="plotly_dark")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(height=360,margin=dict(l=12,r=12,t=28,b=12),yaxis_title="Churn %",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#cbd5e1"))
        st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    with b:
        st.subheader("Risk Distribution")
        x=scored["Risk_Level"].value_counts().reindex(["LOW","MEDIUM","HIGH"]).fillna(0).reset_index()
        x.columns=["Risk","Customers"]
        fig=px.pie(x,names="Risk",values="Customers",hole=.62,template="plotly_dark")
        fig.update_layout(height=360,margin=dict(l=12,r=12,t=28,b=12),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#cbd5e1"))
        st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    st.subheader("Top Retention Priorities")
    display_cols=["Customer_ID","Contract_Type","Account_Age_Months","Monthly_Charges","Support_Calls","Churn_Probability","Risk_Level","Estimated_Revenue_At_Risk"]
    st.dataframe(scored.sort_values("Churn_Probability",ascending=False)[display_cols].head(15),width="stretch",hide_index=True)

elif page == "Customer Analysis":
    st.markdown("<div class=\"section-kicker\">Customer intelligence</div>", unsafe_allow_html=True)
    st.title("Customer Analysis")
    a,b,c=st.columns(3)
    with a: seg=st.selectbox("Contract",["All"]+sorted(df.Contract_Type.unique().tolist()))
    with b: gender=st.selectbox("Gender",["All"]+sorted(df.Gender.unique().tolist()))
    with c: internet=st.selectbox("Internet Service",["All"]+sorted(df.Internet_Service.unique().tolist()))
    f=df.copy()
    if seg!="All": f=f[f.Contract_Type==seg]
    if gender!="All": f=f[f.Gender==gender]
    if internet!="All": f=f[f.Internet_Service==internet]
    a,b=st.columns(2)
    with a:
        fig=px.histogram(f,x="Account_Age_Months",color="Churn",nbins=24,barmode="overlay",template="plotly_dark")
        fig.update_layout(title="Tenure Distribution",height=380,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#cbd5e1"))
        st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    with b:
        fig=px.scatter(f,x="Monthly_Charges",y="Total_Charges",color="Churn",size="Support_Calls",hover_data=["Customer_ID"],template="plotly_dark")
        fig.update_layout(title="Spending Behavior",height=380,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#cbd5e1"))
        st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    st.subheader("Segment Summary")
    segdf=f.groupby(["Contract_Type","Internet_Service"],as_index=False).agg(
        Customers=("Customer_ID","count"), Churn_Rate=("Churn",lambda s:(s=="Yes").mean()),
        Avg_Monthly_Charges=("Monthly_Charges","mean"), Avg_Support_Calls=("Support_Calls","mean")
    )
    segdf["Churn_Rate"]*=100
    st.dataframe(segdf.round(2),width="stretch",hide_index=True)

elif page == "Churn Analysis":
    st.markdown("<div class=\"section-kicker\">Risk & drivers</div>", unsafe_allow_html=True)
    st.title("Churn Analysis")
    a,b=st.columns(2)
    with a:
        x=df.groupby("Contract_Type",as_index=False).agg(Churn_Rate=("Churn",lambda s:(s=="Yes").mean()))
        x["Churn_Rate"]*=100
        st.plotly_chart(px.bar(x,x="Contract_Type",y="Churn_Rate",color="Contract_Type",template="plotly_dark",title="Churn by Contract Type"),width="stretch")
    with b:
        x=df.copy()
        x["Tenure_Band"]=pd.cut(x["Account_Age_Months"],[0,6,12,24,48,72],labels=["0–6","7–12","13–24","25–48","49–72"])
        x=x.groupby("Tenure_Band",observed=True,as_index=False).agg(Churn_Rate=("Churn",lambda s:(s=="Yes").mean()))
        x["Churn_Rate"]*=100
        st.plotly_chart(px.line(x,x="Tenure_Band",y="Churn_Rate",markers=True,template="plotly_dark",title="Churn by Customer Tenure"),width="stretch")
    st.subheader("Top Churn Drivers")
    fig=px.bar(importance.sort_values("Importance"),x="Importance",y="Feature",orientation="h",template="plotly_dark")
    fig.update_layout(height=440,margin=dict(l=12,r=12,t=28,b=12),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#cbd5e1"))
    st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    st.caption("Feature importance is calculated from actual model experiments using permutation importance on the held-out test set.")

elif page == "Prediction":
    st.markdown("<div class=\"section-kicker\">Predictive analytics</div>", unsafe_allow_html=True)
    st.title("Individual Churn Prediction")
    uploaded=st.file_uploader("Upload customer CSV",type=["csv"])
    if uploaded:
        user_df=pd.read_csv(uploaded)
        user_df=add_business_features(clean_data(user_df))
    else:
        user_df=df.copy()
    try:
        scored_user=predict(user_df)
        st.success(f"Scored {len(scored_user):,} customers.")
        cols=["Customer_ID","Churn_Probability","Risk_Level","Estimated_Revenue_At_Risk"]
        st.dataframe(scored_user.sort_values("Churn_Probability",ascending=False)[cols].head(100),width="stretch",hide_index=True)
        st.download_button("Download scored customers",scored_user.to_csv(index=False).encode(),"churn_predictions.csv","text/csv")
        selected=st.selectbox("Explain a customer",scored_user["Customer_ID"].tolist())
        row=scored_user[scored_user.Customer_ID==selected].iloc[0].to_dict()
        p=float(row["Churn_Probability"]); level=row["Risk_Level"]
        st.markdown(f"### {selected} · <span class='badge-{level.lower() if level.lower() in ['high','med','low'] else 'low'}'>{level}</span>",unsafe_allow_html=True)
        st.metric("Churn Probability",f"{p*100:.1f}%")
        reasons=explain_customer(row)
        st.markdown("**Major Risk Factors**")
        for i,(r,sev) in enumerate(reasons,1): st.write(f"{i}. {r} — {sev}")
        st.info(recommendation(level,reasons))
    except Exception as e:
        st.error(f"Prediction could not be generated. Check that the uploaded file contains the required case-study fields. Details: {e}")

elif page == "Model Lab":
    st.markdown("<div class=\"section-kicker\">Model performance</div>", unsafe_allow_html=True)
    st.title("Model Lab")
    st.caption("Actual held-out test-set experiments — no invented metric values.")
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),width="stretch",hide_index=True)
    fig=px.bar(metrics,x="Model",y="ROC_AUC",text="ROC_AUC",template="plotly_dark",title="ROC-AUC Comparison")
    fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
    st.plotly_chart(fig,width="stretch", config={"displaylogo": False, "responsive": True})
    st.write(f"**Best model by ROC-AUC:** {best_model}")
    st.write("Evaluation includes Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC, as required by the project specification.")

elif page == "Data Explorer":
    st.markdown("<div class=\"section-kicker\">Dataset inspection</div>", unsafe_allow_html=True)
    st.title("Data Explorer")
    st.caption("Cleaning, feature engineering and dataset inspection")
    st.write("Rows:", len(df), "· Columns:", len(df.columns))
    st.dataframe(df.head(100),width="stretch",hide_index=True)
    st.download_button("Download processed dataset",df.to_csv(index=False).encode(),"customer_churn_processed.csv","text/csv")
    st.subheader("Missing Values")
    miss=df.isna().sum().reset_index()
    miss.columns=["Feature","Missing"]
    st.dataframe(miss[miss.Missing>0],width="stretch",hide_index=True)
