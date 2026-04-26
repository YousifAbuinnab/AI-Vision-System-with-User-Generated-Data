from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from database import fetch_uploads_df, init_db, insert_upload_record
from models.classifier import classify_image
from models.depth_estimator import estimate_depth_like_map, save_depth_output
from utils.analytics import compute_metrics, plot_top_classes, plot_uploads_over_time, prepare_dataframe
from utils.image_utils import bgr_to_rgb, ensure_directories, pil_to_bgr, save_uploaded_file

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DEPTH_DIR = BASE_DIR / "data" / "depth_outputs"
DEMO_DIR = BASE_DIR / "assets" / "demo_images"


def setup_environment() -> None:
    """Initialize folders and database needed by the app."""
    ensure_directories([UPLOAD_DIR, DEPTH_DIR, DEMO_DIR])
    init_db()


def render_upload_and_analysis() -> None:
    st.subheader("1) Upload Image")
    uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to start analysis.")
        return

    try:
        saved_image_path, pil_image = save_uploaded_file(uploaded_file, UPLOAD_DIR)
    except Exception as exc:
        st.error(f"Failed to read/save uploaded image: {exc}")
        return

    st.image(pil_image, caption=f"Uploaded image: {Path(saved_image_path).name}", width="stretch")

    st.subheader("2) Run AI Analysis")
    if st.button("Analyze Image", type="primary"):
        try:
            with st.spinner("Running classifier and depth estimation..."):
                predictions = classify_image(pil_image, top_k=3)

                image_bgr = pil_to_bgr(pil_image)
                depth_colored, _ = estimate_depth_like_map(image_bgr)
                depth_output_path = save_depth_output(depth_colored, DEPTH_DIR, Path(saved_image_path).name)

            st.success("Analysis completed.")

            st.markdown("### Top 3 Predicted Classes")
            for idx, item in enumerate(predictions, start=1):
                st.write(f"{idx}. {item['class_name']} ({item['confidence'] * 100:.2f}%)")

            st.markdown("### Depth Map (Stereo-inspired Placeholder)")
            st.image(
                bgr_to_rgb(depth_colored),
                caption=Path(depth_output_path).name,
                width="stretch",
            )

            top_prediction = predictions[0]
            insert_upload_record(
                image_filename=Path(saved_image_path).name,
                upload_time=datetime.now(timezone.utc).isoformat(),
                predicted_class=top_prediction["class_name"],
                confidence=top_prediction["confidence"],
                depth_output_path=str(Path(depth_output_path).relative_to(BASE_DIR)),
            )
            st.info("Metadata saved to SQLite database.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


def render_dashboard() -> None:
    st.subheader("Analytics Dashboard")

    try:
        raw_df = fetch_uploads_df()
    except Exception as exc:
        st.error(f"Could not load analytics data: {exc}")
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
    fig_classes = plot_top_classes(df)
    st.pyplot(fig_classes, clear_figure=True)

    st.markdown("### Uploads Over Time")
    fig_trend = plot_uploads_over_time(df)
    st.pyplot(fig_trend, clear_figure=True)

    if not df.empty:
        st.markdown("### Recent Upload Records")
        preview_cols = [
            "image_filename",
            "upload_time",
            "predicted_class",
            "confidence",
            "depth_output_path",
        ]
        st.dataframe(df[preview_cols].head(20), width="stretch")


def main() -> None:
    st.set_page_config(page_title="AI Vision System with User-Generated Data", layout="wide")
    st.title("AI Vision System with User-Generated Data")

    st.write(
        "Upload an image, classify it with a pretrained CNN, generate a depth-like map, and track insights from user-generated data."
    )

    setup_environment()

    tab1, tab2 = st.tabs(["Upload and Analyze", "Analytics Dashboard"])

    with tab1:
        render_upload_and_analysis()

    with tab2:
        render_dashboard()


if __name__ == "__main__":
    main()
