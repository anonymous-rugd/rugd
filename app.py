"""Interactive OpenCity result explorer. Run: streamlit run app.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ShiftGuard-RUGD Traffic", layout="wide")
st.title("ShiftGuard-RUGD: cross-city severe-congestion research demo")
st.caption("Explores frozen held-out predictions; it is not operational traffic guidance.")
root = Path(st.sidebar.text_input("Result directory", "outputs/studies/final_study/main"))
results_path = root / "results_all.csv"
if not results_path.exists():
    st.info("Run the experiment and reporting commands from README.md first.")
    st.stop()

results = pd.read_csv(results_path)
cities = sorted(results.target_city.unique())
city = st.sidebar.selectbox("Held-out city", cities)
available = results[results.target_city == city]
model = st.sidebar.selectbox("Model", sorted(available.model.unique()),
                             index=max(0, len(available.model.unique()) - 1))
seeds = sorted(available[available.model == model].seed.unique())
seed = st.sidebar.selectbox("Seed", seeds)

st.subheader("Cross-city benchmark")
metric = st.selectbox("Ranking metric", ["severe_auprc", "macro_f1", "nll", "ece"])
chart = results.groupby(["target_city", "model"], as_index=False)[metric].mean()
st.plotly_chart(px.bar(chart, x="model", y=metric, color="target_city", barmode="group"),
                use_container_width=True)

run = root / city / model / f"seed_{seed}"
prediction_path = run / "predictions.npz"
if prediction_path.exists():
    data = np.load(prediction_path)
    y, p, uncertainty = data["targets"], data["probabilities"], data["uncertainty"]
    time_ids = data["time_ids"]
    unique_times = np.unique(time_ids)
    selected = st.slider("Held-out timestamp index", 0, len(unique_times) - 1, 0)
    mask = time_ids == unique_times[selected]
    severe = p[mask, 2]
    frame = pd.DataFrame({"node": np.arange(mask.sum()), "severe_probability": severe,
                          "predicted": p[mask].argmax(1), "observed": y[mask],
                          "uncertainty": uncertainty[mask]})
    threshold = st.slider("Show nodes above severe probability", 0.0, 1.0, .5, .01)
    shown = frame[frame.severe_probability >= threshold].sort_values("severe_probability",
                                                                     ascending=False)
    a, b = st.columns(2)
    a.metric("Evaluated nodes", f"{len(frame):,}")
    b.metric("Flagged nodes", f"{len(shown):,}")
    st.plotly_chart(px.scatter(frame, x="node", y="severe_probability", color="uncertainty",
                               hover_data=["predicted", "observed"]), use_container_width=True)
    st.dataframe(shown.head(250), use_container_width=True)

robustness = root / "robustness_all.csv"
if robustness.exists():
    st.subheader("Distribution-shift stress tests")
    robust = pd.read_csv(robustness)
    subset = robust[(robust.target_city == city) & (robust.model == model)]
    st.plotly_chart(px.line(subset, x="severity", y="severe_auprc", color="corruption",
                            markers=True, line_group="seed"), use_container_width=True)

st.warning("Do not select models or claims from this dashboard. Use the registered aggregate and "
           "paired statistical analysis produced by the main study launcher.")
