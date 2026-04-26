"""Streamlit application for upload, classification, depth map, and analytics."""

from datetime import datetime
from datetime import timezone
from pathlib import Path

import streamlit as st

from database import fetch_uploads_df
from database import init_db
from database import insert_upload_record
from models.classifier import classify_image
from models.depth_estimator import estimate_depth_like_map
from models.depth_estimator import save_depth_output
from utils.analytics import compute_metrics
from utils.analytics import plot_top_classes
from utils.analytics import plot_uploads_over_time
from utils.analytics import prepare_dataframe
from utils.image_utils import bgr_to_rgb
from utils.image_utils import ensure_directories
from utils.image_utils import pil_to_bgr
from utils.image_utils import save_uploaded_file

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DEPTH_DIR = BASE_DIR / "data" / "depth_outputs"
DEMO_DIR = BASE_DIR / "assets" / "demo_images"


def setup_environment() -> None:
    """Create required folders and initialize the database."""
    ensure_directories([UPLOAD_DIR, DEPTH_DIR, DEMO_DIR])
    init_db()


def initialize_ui_state() -> None:
    """Initialize session-state keys used across pages."""
    st.session_state.setdefault("uploaded_image", None)
    st.session_state.setdefault("uploaded_filename", "")
    st.session_state.setdefault("saved_image_path", "")
    st.session_state.setdefault("predictions", [])
    st.session_state.setdefault("depth_colored", None)
    st.session_state.setdefault("depth_output_path", "")


def run_analysis() -> None:
    """Run image classification and depth map generation for uploaded image."""
    uploaded_image = st.session_state.get("uploaded_image")
    saved_image_path = st.session_state.get("saved_image_path")

    if uploaded_image is None or not saved_image_path:
        st.warning("Please upload an image first.")
        return

    try:
        with st.spinner("Running classifier and depth estimation..."):
            predictions = classify_image(uploaded_image, top_k=3)

            image_bgr = pil_to_bgr(uploaded_image)
            depth_colored, _ = estimate_depth_like_map(image_bgr)
            depth_output_path = save_depth_output(
                depth_colored,
                DEPTH_DIR,
                Path(saved_image_path).name,
            )

        top_prediction = predictions[0]
        insert_upload_record(
            image_filename=Path(saved_image_path).name,
            upload_time=datetime.now(timezone.utc).isoformat(),
            predicted_class=top_prediction["class_name"],
            confidence=top_prediction["confidence"],
            depth_output_path=str(Path(depth_output_path).relative_to(BASE_DIR)),
        )

        st.session_state["predictions"] = predictions
        st.session_state["depth_colored"] = depth_colored
        st.session_state["depth_output_path"] = depth_output_path

        st.success("Analysis completed successfully.")
        st.info("Record saved in SQLite database.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def render_upload_page() -> None:
    """Render upload page where the user selects an image."""
    st.subheader("Upload")
    st.write("Upload a JPG or PNG image and run AI analysis.")

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("No image selected yet.")
        return

    try:
        saved_image_path, pil_image = save_uploaded_file(uploaded_file, UPLOAD_DIR)
    except Exception as exc:
        st.error(f"Could not save uploaded image: {exc}")
        return

    st.session_state["uploaded_image"] = pil_image
    st.session_state["uploaded_filename"] = Path(saved_image_path).name
    st.session_state["saved_image_path"] = saved_image_path

    left_col, right_col = st.columns([2, 1], gap="large")
    with left_col:
        st.image(
            pil_image,
            caption=f"Uploaded image: {Path(saved_image_path).name}",
            width="stretch",
        )
    with right_col:
        st.markdown("### File Info")
        st.write(f"Name: {Path(saved_image_path).name}")
        st.write(f"Type: {Path(saved_image_path).suffix.lower()}")

    st.write("")
    if st.button("Analyze Image", type="primary", use_container_width=True):
        run_analysis()


def render_results_page() -> None:
    """Render page that shows analysis outputs and predictions."""
    st.subheader("Results")

    predictions = st.session_state.get("predictions", [])
    depth_colored = st.session_state.get("depth_colored")
    uploaded_image = st.session_state.get("uploaded_image")
    depth_output_path = st.session_state.get("depth_output_path", "")

    if not predictions:
        st.info("No analysis results yet. Go to Upload and click Analyze Image.")
        return

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown("### Uploaded Image")
        st.image(uploaded_image, width="stretch")
    with right_col:
        st.markdown("### Depth Map")
        st.image(
            bgr_to_rgb(depth_colored),
            caption=Path(depth_output_path).name if depth_output_path else "Depth map",
            width="stretch",
        )

    st.write("")
    st.markdown("### Top 3 Predictions")
    cards = st.columns(3, gap="medium")
    for index, item in enumerate(predictions[:3], start=1):
        with cards[index - 1]:
            st.markdown(f"**#{index} {item['class_name']}**")
            st.progress(float(item["confidence"]))
            st.caption(f"Confidence: {item['confidence'] * 100:.2f}%")


def render_analytics_page() -> None:
    """Render analytics dashboard from saved database records."""
    st.subheader("Analytics")

    try:
        raw_df = fetch_uploads_df()
    except Exception as exc:
        st.error(f"Could not load data from database: {exc}")
        return

    df = prepare_dataframe(raw_df)
    metrics = compute_metrics(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Uploaded Images", metrics["total_uploads"])
    col2.metric("Average Confidence", f"{metrics['avg_confidence'] * 100:.2f}%")

    most_common = "N/A"
    if not metrics["top_classes"].empty:
        most_common = metrics["top_classes"].index[0]
    col3.metric("Most Common Class", most_common)

    st.markdown("### Most Common Predicted Classes")
    st.pyplot(plot_top_classes(df), clear_figure=True)

    st.markdown("### Uploads Over Time")
    st.pyplot(plot_uploads_over_time(df), clear_figure=True)

    if not df.empty:
        st.markdown("### Recent Records")
        columns_to_show = [
            "image_filename",
            "upload_time",
            "predicted_class",
            "confidence",
            "depth_output_path",
        ]
        st.dataframe(df[columns_to_show].head(20), width="stretch")


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="AI Vision System with User-Generated Data",
        layout="wide",
    )
    st.title("AI Vision System with User-Generated Data")
    st.write(
        "Upload an image, classify it with a pretrained CNN, "
        "generate a depth-like map, and track insights over time."
    )

    setup_environment()
    initialize_ui_state()

    st.sidebar.title("Menu")
    page = st.sidebar.radio(
        "Navigate",
        ["Upload", "Results", "Analytics"],
        index=0,
    )

    if page == "Upload":
        render_upload_page()
    elif page == "Results":
        render_results_page()
    else:
        render_analytics_page()


if __name__ == "__main__":
    main()
