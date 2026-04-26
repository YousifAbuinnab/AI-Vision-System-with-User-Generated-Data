"""Analytics and charting helpers for dashboard visualizations."""

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data types and handle missing rows safely."""
    if df.empty:
        return df

    prepared = df.copy()
    prepared["upload_time"] = pd.to_datetime(prepared["upload_time"], errors="coerce")
    prepared = prepared.dropna(subset=["upload_time"])

    return prepared


def compute_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """Compute dashboard metrics from upload records."""
    total_uploads = int(len(df))
    avg_confidence = float(df["confidence"].mean()) if total_uploads > 0 else 0.0
    top_classes = df["predicted_class"].value_counts().head(5) if total_uploads > 0 else pd.Series(dtype=int)

    return {
        "total_uploads": total_uploads,
        "avg_confidence": avg_confidence,
        "top_classes": top_classes,
    }


def plot_uploads_over_time(df: pd.DataFrame):
    """Create a time-series chart of upload count over days."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if df.empty:
        ax.text(0.5, 0.5, "No upload data yet", ha="center", va="center")
        ax.axis("off")
        return fig

    trend = df.set_index("upload_time").resample("D").size()
    ax.plot(trend.index, trend.values, marker="o", linewidth=2.4, color="#1f77b4")
    ax.set_title("Uploads Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Uploads")
    ax.grid(alpha=0.35, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_top_classes(df: pd.DataFrame, top_n: int = 5):
    """Create a bar chart of most common predicted classes."""
    fig, ax = plt.subplots(figsize=(10, 4))

    if df.empty:
        ax.text(0.5, 0.5, "No prediction data yet", ha="center", va="center")
        ax.axis("off")
        return fig

    top_classes = df["predicted_class"].value_counts().head(top_n)
    bars = ax.bar(top_classes.index, top_classes.values, color="#2ca02c")
    ax.set_title(f"Top {top_n} Predicted Classes")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.35, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    return fig
