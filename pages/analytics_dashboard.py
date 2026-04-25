"""
- Average royalty rate
- Distribution of deal scores
- Most common red flags
- Best/worst contracts
- Clause frequency
- Artist-friendly vs label-friendly breakdown
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Analytics Dashboard")

path = Path("data/extracted_contract_features.csv")

if not path.exists():
    st.warning("No analyzed contracts yet. Use the Deal Analyzer first.")
    st.stop()

df = pd.read_csv(path)

st.metric("Contracts analyzed", len(df))
st.metric("Average score", round(df["artist_friendliness_score"].mean(), 1))
st.metric("Average royalty rate", round(df["royalty_rate"].mean(), 3))

st.subheader("Risk Breakdown")
st.bar_chart(df["risk_level"].value_counts())

st.subheader("Deal Scores")
st.bar_chart(df["artist_friendliness_score"])

st.subheader("Analyzed Contracts")
st.dataframe(df)