"""Minimal deploy-pipeline skeleton. Not the real UI (that's Phase 5) —
this exists only to prove requirements/memory/cold-start/artifact-loading
work end to end on Streamlit Community Cloud before building anything real
on top of it.
"""

import streamlit as st

import recommend

st.title("GlowMatch — deploy skeleton")

HARDCODED_PRODUCT_ID = "P420652"

st.write(f"Recommendations for product `{HARDCODED_PRODUCT_ID}`:")

results = recommend.recommend_by_item(HARDCODED_PRODUCT_ID, n=5)
st.dataframe(results[["product_id", "product_name", "blended_score", "label"]])
