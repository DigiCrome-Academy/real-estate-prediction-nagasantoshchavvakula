"""
Real Estate Price Prediction Dashboard

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob as _glob
import sys
import os
import tempfile
import gdown

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import (
    load_housing_data,
    preprocess_features,
    split_data,
    create_feature_engineering,
)
from src.ensemble import load_model
from src.recommendation import (
    compute_property_similarity,
    content_based_recommend,
    knn_recommend,
)
from src.clustering import cluster_with_pca

st.set_page_config(
    page_title="Real Estate Price Prediction Engine",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Real Estate Price Prediction Engine")
st.markdown("---")

# =============================================================================
# Sidebar
# =============================================================================
page = st.sidebar.selectbox(
    "Navigate",
    ["Price Prediction", "Property Recommendations", "Market Segmentation"]
)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def get_data():
    df = load_housing_data()
    df_eng = create_feature_engineering(df)

    X_scaled, y, feature_names, scaler = preprocess_features(df_eng)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    return df_eng, X_scaled, y, feature_names, scaler, X_train, X_test, y_train, y_test


# ✅ YOUR GOOGLE DRIVE FOLDER ID
GDRIVE_FOLDER_ID = "16JdHlwA-BJpjEQbKpzN7W0WN4_fHaKGg"

_MODEL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "real_estate_models")


@st.cache_resource(show_spinner="Downloading models from Google Drive...")
def get_models():
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)

    voting_path = os.path.join(_MODEL_CACHE_DIR, "voting_ensemble.joblib")
    stacking_path = os.path.join(_MODEL_CACHE_DIR, "stacking_ensemble.joblib")

    if not (os.path.exists(voting_path) and os.path.exists(stacking_path)):
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=_MODEL_CACHE_DIR,
            quiet=True,
            use_cookies=False,
        )

        # Flatten structure
        for src in _glob.glob(os.path.join(_MODEL_CACHE_DIR, "**", "*.joblib"), recursive=True):
            dst = os.path.join(_MODEL_CACHE_DIR, os.path.basename(src))
            if src != dst:
                os.replace(src, dst)

    voting = load_model(voting_path) if os.path.exists(voting_path) else None
    stacking = load_model(stacking_path) if os.path.exists(stacking_path) else None

    return voting, stacking


df, X_scaled, y, feature_names, scaler, X_train, X_test, y_train, y_test = get_data()
voting_model, stacking_model = get_models()

# =============================================================================
# PRICE PREDICTION
# =============================================================================
if page == "Price Prediction":
    st.header("💰 Price Prediction")
    st.write("Select values for the below features to predict estimated price")

    available = {}
    if stacking_model:
        available["Stacking Ensemble"] = stacking_model
    if voting_model:
        available["Voting Ensemble"] = voting_model

    if not available:
        st.error("Models not loaded. Check Google Drive permissions.")
        st.stop()

    model_name = st.sidebar.selectbox("Model", list(available.keys()))
    model = available[model_name]

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.slider("Median Income", 0.5, 15.0, 3.5)
        house_age = st.slider("House Age", 1, 52, 20)
        ave_rooms = st.slider("Average Rooms", 1.0, 20.0, 5.0)
        ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)

    with col2:
        population = st.slider("Population", 5, 5000, 1200)
        ave_occup = st.slider("Average Occupancy", 1.0, 10.0, 2.5)
        latitude = st.slider("Latitude", 32.5, 42.0, 36.0)
        longitude = st.slider("Longitude", -124.5, -114.0, -120.0)

    if st.button("Predict Price", type="primary"):

        # Feature engineering (same as training)
        rooms_per_household = ave_rooms * ave_occup
        bedrooms_ratio = ave_bedrms / ave_rooms if ave_rooms != 0 else 0
        population_density = population / ave_occup if ave_occup != 0 else 0

        raw_input = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude,
            rooms_per_household, bedrooms_ratio, population_density
        ]])

        scaled_input = scaler.transform(raw_input)
        prediction = model.predict(scaled_input)[0]

        st.success(f"Estimated Price: ${prediction * 100000:,.0f}")

# =============================================================================
# PROPERTY RECOMMENDATIONS
# =============================================================================
elif page == "Property Recommendations":
    st.header("🔍 Property Recommendations")
    st.write("Find similar properties based on features.")

    # Subsample for speed
    SAMPLE_SIZE = 500
    X_sample = X_scaled[:SAMPLE_SIZE]
    df_sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)

    # Sidebar controls
    method = st.sidebar.radio(
        "Recommendation Method",
        ["Content-Based (Cosine)", "KNN"]
    )

    n_recs = st.sidebar.slider(
        "Number of Recommendations",
        3, 20, 5
    )

    # Property selection
    property_idx = st.number_input(
        f"Select Property Index (0 – {SAMPLE_SIZE - 1})",
        min_value=0,
        max_value=SAMPLE_SIZE - 1,
        value=0,
        step=1
    )

    # Show selected property
    st.subheader("Selected Property Features")
    selected = df_sample.iloc[property_idx]
    st.dataframe(selected.to_frame().T, use_container_width=True)

    # Recommendation logic
    if st.button("Find Similar Properties", type="primary"):

        if method == "Content-Based (Cosine)":
            sim_matrix = compute_property_similarity(X_sample, metric='cosine')

            recs = content_based_recommend(
                property_idx,
                sim_matrix,
                n_recommendations=n_recs
            )

            rec_indices = [r['property_index'] for r in recs]
            scores = [r['similarity_score'] for r in recs]
            score_label = "Similarity Score"

        else:
            recs = knn_recommend(
                X_sample,
                property_idx,
                n_recommendations=n_recs
            )

            rec_indices = [r['property_index'] for r in recs]
            scores = [r['distance'] for r in recs]
            score_label = "Distance"

        # Build result table
        rec_df = df_sample.iloc[rec_indices].copy()

        rec_df.insert(0, score_label, [round(s, 4) for s in scores])
        rec_df.insert(0, "Property Index", rec_indices)

        st.subheader(f"Top {len(recs)} Similar Properties")
        st.dataframe(rec_df, use_container_width=True)

# =============================================================================
# MARKET SEGMENTATION
# =============================================================================
elif page == "Market Segmentation":
    st.header("📊 Market Segmentation")
    st.write("Explore property market segments identified by clustering.")

    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

    # Subsample for performance
    SAMPLE_SIZE = 2000
    X_sample = X_scaled[:SAMPLE_SIZE]
    df_sample = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)

    with st.spinner("Running PCA + K-Means clustering..."):
        result = cluster_with_pca(X_sample, n_clusters=n_clusters, n_components=2)

    pca_data = result["pca_data"]
    labels = result["labels"]
    silhouette = result.get("silhouette", None)

    # ✅ Silhouette score (if available)
    if silhouette is not None:
        st.metric(
            "Silhouette Score",
            f"{silhouette:.4f}",
            help="Higher is better (range: -1 to 1)"
        )

    # =========================
    # PCA Scatter Plot
    # =========================
    st.subheader("PCA 2D Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=10
    )

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="upper right"
    )
    ax.add_artist(legend)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"K-Means Clusters (k={n_clusters})")

    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # Cluster Statistics
    # =========================
    st.subheader("Cluster Statistics")

    df_stats = df_sample.copy()
    df_stats["Cluster"] = labels

    cluster_stats = (
        df_stats.groupby("Cluster")
        .agg(
            Count=("MedHouseVal", "count"),
            Avg_MedHouseVal=("MedHouseVal", "mean"),
            Avg_MedInc=("MedInc", "mean"),
            Avg_HouseAge=("HouseAge", "mean"),
            Avg_AveRooms=("AveRooms", "mean"),
        )
        .round(3)
        .reset_index()
    )

    # Convert to dollar format
    cluster_stats["Avg_MedHouseVal ($)"] = (
        cluster_stats["Avg_MedHouseVal"] * 100000
    ).map("${:,.0f}".format)

    cluster_stats.drop(columns=["Avg_MedHouseVal"], inplace=True)

    st.dataframe(cluster_stats, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("Built as part of the Real Estate Price Prediction Engine project. Uses the California Housing dataset from scikit-learn.")