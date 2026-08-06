# report_history.py

"""
Modul untuk menyimpan riwayat laporan harian ke database SQLite ringan
(sqlite3 built-in di Python — tanpa dependency eksternal tambahan).
Memungkinkan pelacakan akurasi rekomendasi AI dari waktu ke waktu.
"""

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

from gemini_analyzer import AnalysisReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "report_history.db"
JAKARTA_TZ = ZoneInfo("Asia/Jakarta")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_wib TEXT NOT NULL,
    market_summary TEXT NOT NULL,
    news_highlights_json TEXT NOT NULL,
    recommendations_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success'
);
"""


@contextmanager
def _get_connection():
    """Context manager agar koneksi SQLite selalu ditutup dengan benar, bahkan jika terjadi error di tengah operasi."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Membuat tabel `reports` jika belum ada. Idempotent — aman dipanggil berkali-kali."""
    with _get_connection() as conn:
        conn.execute(SCHEMA)


def save_report(report: AnalysisReport, status: str = "success") -> None:
    """
    Menyimpan satu laporan analisis (data terstruktur, bukan teks bebas)
    sebagai satu baris riwayat.

    Args:
        report: Objek AnalysisReport hasil gemini_analyzer.analyze_market().
        status: Penanda keandalan siklus ('success' saat ini — kegagalan
                Gemini ditangani sebagai fallback di main.py dan sengaja
                TIDAK disimpan ke tabel ini, karena tidak ada data
                terstruktur valid untuk disimpan pada kasus tersebut).
    """
    init_db()

    timestamp_wib = datetime.now(JAKARTA_TZ).isoformat()
    news_highlights_json = json.dumps(report.news_highlights, ensure_ascii=False)
    recommendations_json = json.dumps(
        [rec.model_dump() for rec in report.recommendations], ensure_ascii=False
    )

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reports (timestamp_wib, market_summary, news_highlights_json, recommendations_json, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp_wib, report.market_summary, news_highlights_json, recommendations_json, status),
        )

    logger.info(f"Laporan berhasil disimpan ke riwayat (status: {status}).")


def get_recent_reports(limit: int = 7) -> List[dict]:
    """Mengambil N laporan terakhir dari database, diurutkan dari yang terbaru."""
    init_db()

    with _get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "timestamp_wib": row["timestamp_wib"],
            "market_summary": row["market_summary"],
            "news_highlights": json.loads(row["news_highlights_json"]),
            "recommendations": json.loads(row["recommendations_json"]),
            "status": row["status"],
        }
        for row in rows
    ]


def export_to_csv(output_path: str = "report_history_export.csv", limit: int = 90) -> str:
    """
    Mengekspor riwayat ke CSV datar (satu baris per rekomendasi
    instrumen) agar mudah dibuka di Excel/Google Sheets untuk mengecek
    akurasi rekomendasi AI secara manual dari waktu ke waktu.
    """
    reports = get_recent_reports(limit=limit)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_wib", "instrument", "action", "confidence", "reason", "status"])
        for report in reports:
            for rec in report["recommendations"]:
                writer.writerow([
                    report["timestamp_wib"],
                    rec["instrument"],
                    rec["action"],
                    rec["confidence"],
                    rec["reason"],
                    report["status"],
                ])

    logger.info(f"Riwayat berhasil diekspor ke: {output_path}")
    return output_path


if __name__ == "__main__":
    # Self-test: jalankan `python report_history.py` untuk melihat
    # ringkasan riwayat yang tersimpan sejauh ini.
    init_db()
    recent = get_recent_reports(limit=5)

    if not recent:
        print("Belum ada riwayat laporan tersimpan.")
    else:
        print(f"Menampilkan {len(recent)} laporan terakhir:\n")
        for r in recent:
            print(
                f"[{r['timestamp_wib']}] Status: {r['status']} | "
                f"Rekomendasi: {len(r['recommendations'])} instrumen"
            )