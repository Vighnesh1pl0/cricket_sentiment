import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup
st.set_page_config(page_title="🏏 Cricket Sentiment Analyzer", layout="wide")
analyzer = SentimentIntensityAnalyzer()
model = joblib.load("momentum_model.pkl")
df = pd.read_csv("rr_vs_dc_ipl2026.csv")

# Cricket score function
def cricket_score(line):
    keywords = {"SIX": 0.5, "FOUR": 0.3, "WICKET": -0.5, "OUT": -0.5, "DOT": -0.2}
    score = 0
    for word, value in keywords.items():
        if word in line.upper():
            score += value
    return score

# Preprocess data
df["vader_score"] = df["commentary"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["cricket_score"] = df["commentary"].apply(cricket_score)
df["final_score"] = df["vader_score"] + df["cricket_score"]

# Header
st.title("🏏 Cricket Commentary Sentiment Analyzer")
st.markdown("**RR vs DC — IPL 2026 | May 1, Jaipur**")
st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Live Analyzer", "📊 Match Dashboard", "🤖 Momentum Predictor"])

# ── TAB 1 — Live Analyzer ──────────────────────────
with tab1:
    st.subheader("Paste any cricket commentary line")
    user_input = st.text_area("Commentary:", height=100, 
                               placeholder="e.g. Rohit smashes it for a massive SIX!")
    
    if st.button("Analyze 🔍"):
        if user_input:
            vader = analyzer.polarity_scores(user_input)["compound"]
            cricket = cricket_score(user_input)
            final = vader + cricket

            col1, col2, col3 = st.columns(3)
            col1.metric("VADER Score", f"{vader:.2f}")
            col2.metric("Cricket Score", f"{cricket:.2f}")
            col3.metric("Final Score", f"{final:.2f}")

            st.divider()
            if final >= 0.3:
                st.success("🟢 Batting team dominating!")
            elif final <= -0.3:
                st.error("🔴 Bowling team under pressure!")
            else:
                st.warning("🟡 Even contest!")

            # Gauge bar
            st.markdown("**Sentiment Meter:**")
            normalized = (final + 1.5) / 3.0
            normalized = max(0.0, min(1.0, normalized))
            st.progress(normalized)
        else:
            st.warning("Please enter some commentary!")

# ── TAB 2 — Match Dashboard ────────────────────────
with tab2:
    st.subheader("Over-wise Sentiment — RR vs DC IPL 2026")

    over_sentiment = df.groupby(["innings", "over"])["final_score"].mean().reset_index()
    over_sentiment["team"] = over_sentiment["innings"].map({1: "RR Batting", 2: "DC Batting"})

    fig = px.line(over_sentiment, x="over", y="final_score",
                  color="team",
                  markers=True,
                  title="Match Momentum — Over by Over",
                  labels={"final_score": "Sentiment Score", "over": "Over Number"},
                  color_discrete_map={"RR Batting": "#1f77b4", "DC Batting": "#ff7f0e"})
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Ball by Ball Data")
    innings_filter = st.selectbox("Select Innings:", [1, 2], 
                                   format_func=lambda x: "RR Batting" if x==1 else "DC Batting")
    filtered = df[df["innings"] == innings_filter][["over", "ball", "batsman", "bowler", "commentary", "final_score"]]
    st.dataframe(filtered, use_container_width=True)

# ── TAB 3 — Momentum Predictor ─────────────────────
with tab3:
    st.subheader("🤖 Predict Momentum Shift")
    st.markdown("Enter over statistics to predict if a momentum shift will occur:")

    col1, col2 = st.columns(2)
    with col1:
        avg_sent = st.slider("Avg Sentiment Score", -1.5, 1.5, 0.0, 0.1)
        wicket_count = st.slider("Wickets in Over", 0, 6, 0)
    with col2:
        boundary_count = st.slider("Boundaries (4s + 6s)", 0, 6, 0)
        dot_count = st.slider("Dot Balls", 0, 6, 0)

    if st.button("Predict Momentum 🤖"):
        sample = pd.DataFrame({
            "avg_sentiment": [avg_sent],
            "wicket_count": [wicket_count],
            "boundary_count": [boundary_count],
            "dot_count": [dot_count]
        })
        prediction = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]

        st.divider()
        if prediction == 1:
            st.error(f"🔴 Momentum Shift Likely! Confidence: {proba[1]*100:.1f}%")
        else:
            st.success(f"🟢 No Momentum Shift! Confidence: {proba[0]*100:.1f}%")

        st.markdown("**Input Summary:**")
        st.json({
            "avg_sentiment": avg_sent,
            "wicket_count": wicket_count,
            "boundary_count": boundary_count,
            "dot_count": dot_count,
            "prediction": int(prediction)
        })