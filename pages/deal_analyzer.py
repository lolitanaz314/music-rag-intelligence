"""
select document → extract features → score deal → save row to CSV
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.scoring.deal_score import score_contract

st.title("Deal Analyzer")

st.write("Analyze music contracts for artist-friendliness.")

document_name = st.text_input("Document name")

royalty_rate = st.number_input("Royalty rate", min_value=0.0, max_value=1.0, value=0.15)
advance_amount = st.number_input("Advance amount", min_value=0.0, value=0.0)
recoupable = st.checkbox("Recoupable advance?")
term_years = st.number_input("Term length in years", min_value=0.0, value=1.0)
ownership = st.text_input("Ownership / masters language")
audit_rights = st.checkbox("Audit rights included?")
exclusivity = st.checkbox("Exclusivity restriction?")

if st.button("Score Deal"):
    features = {
        "document_name": document_name,
        "royalty_rate": royalty_rate,
        "advance_amount": advance_amount,
        "recoupable": recoupable,
        "term_years": term_years,
        "ownership": ownership,
        "audit_rights": audit_rights,
        "exclusivity": exclusivity,
    }

    score = score_contract(features)
    row = {**features, **score}

    st.metric("Artist-Friendliness Score", score["artist_friendliness_score"])
    st.write("Risk level:", score["risk_level"])
    st.write("Reasons:", score["score_reasons"])

    output_path = Path("data/extracted_contract_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df = pd.read_csv(output_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(output_path, index=False)

    st.success("Saved analysis.")