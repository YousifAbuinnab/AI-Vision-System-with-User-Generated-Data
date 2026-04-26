"""SQLite helper functions for storing and reading upload records."""

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "vision_data.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return an open SQLite connection and ensure parent directory exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create uploads table when database is initialized."""
    query = """
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT NOT NULL,
        upload_time TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        depth_output_path TEXT NOT NULL
    );
    """

    with get_connection(db_path) as conn:
        conn.execute(query)
        conn.commit()


def insert_upload_record(
    image_filename: str,
    upload_time: str,
    predicted_class: str,
    confidence: float,
    depth_output_path: str,
    db_path: Path = DB_PATH,
) -> None:
    """Insert one analyzed image record into uploads table."""
    query = """
    INSERT INTO uploads (
        image_filename,
        upload_time,
        predicted_class,
        confidence,
        depth_output_path
    )
    VALUES (?, ?, ?, ?, ?);
    """

    with get_connection(db_path) as conn:
        conn.execute(
            query,
            (image_filename, upload_time, predicted_class, confidence, depth_output_path),
        )
        conn.commit()


def fetch_uploads_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Fetch upload records ordered by newest first."""
    query = "SELECT * FROM uploads ORDER BY upload_time DESC;"

    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)
