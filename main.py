# main.py

"""
Orchestrator utama "AI Finance Department Agent".

Alur eksekusi:
    1. Tarik data pasar (indeks & saham individual)
    2. Tarik berita keuangan terbaru (Indonesia & global)
    3. Kirim data ke Gemini AI untuk dianalisis (structured output + retry otomatis)
    4. Simpan laporan ke riwayat (SQLite) — best-effort, tidak boleh gagalkan pengiriman
    5. Format & kirim laporan ke Telegram

Prinsip desain: setiap tahap diisolasi dalam try/except sendiri. Kegagalan
di satu tahap TIDAK BOLEH membuat seluruh proses mati diam-diam.
"""

import logging
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from config import Config, ConfigError
from market_data import get_all_market_data, format_market_summary
from news_fetcher import fetch_all_news, format_news_summary
from gemini_analyzer import analyze_market, format_report_for_telegram, GeminiAnalysisError
from telegram_sender import send_telegram_message
from report_history import save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

JAKARTA_TZ = ZoneInfo("Asia/Jakarta")


def _current_timestamp_wib() -> str:
    return datetime.now(JAKARTA_TZ).strftime("%A, %d %B %Y - %H:%M WIB")


def _notify_failure(stage: str, error: Exception) -> None:
    message = (
        f"🚨 *AI Finance Agent - GAGAL BERJALAN*\n\n"
        f"Waktu: {_current_timestamp_wib()}\n"
        f"Tahap gagal: *{stage}*\n"
        f"Detail: `{str(error)[:300]}`\n\n"
        f"Periksa `agent.log` di server untuk detail lengkap."
    )
    try:
        sent = send_telegram_message(message)
        if not sent:
            logger.error("Notifikasi kegagalan JUGA gagal dikirim ke Telegram.")
    except Exception as notify_exc:
        logger.error(f"Tidak bisa mengirim notifikasi kegagalan ke Telegram: {notify_exc}")


def run_daily_report() -> bool:
    logger.info("=== Memulai siklus AI Finance Department Agent ===")

    try:
        Config.validate()
    except ConfigError as exc:
        logger.critical(f"Konfigurasi dasar tidak valid, agen dihentikan: {exc}")
        return False

    # --- Tahap 1: Data pasar (WAJIB) ---
    try:
        market_indices = get_all_market_data()
        if not market_indices:
            raise RuntimeError("Seluruh ticker gagal diambil (list kosong).")
        market_summary = format_market_summary(market_indices)
        logger.info("Tahap 1 (Market Data): berhasil.")
    except Exception as exc:
        logger.error(f"Tahap 1 (Market Data) gagal total: {exc}\n{traceback.format_exc()}")
        _notify_failure("Pengambilan Data Pasar", exc)
        return False

    # --- Tahap 2: Berita (OPSIONAL) ---
    try:
        news_items = fetch_all_news(limit_per_source=5)
        news_summary = format_news_summary(news_items)
        logger.info(f"Tahap 2 (News): berhasil, {len(news_items)} berita diambil.")
    except Exception as exc:
        logger.warning(f"Tahap 2 (News) gagal, melanjutkan tanpa data berita: {exc}")
        news_summary = "Tidak ada data berita yang tersedia hari ini (gagal diambil)."

    # --- Tahap 3: Analisis Gemini AI (structured output + retry otomatis) ---
    analysis_report = None
    try:
        analysis_report = analyze_market(market_summary, news_summary)
        final_report_text = format_report_for_telegram(analysis_report)
        logger.info("Tahap 3 (Gemini Analysis): berhasil.")
    except GeminiAnalysisError as exc:
        logger.error(f"Tahap 3 (Gemini Analysis) gagal: {exc}")
        final_report_text = (
            f"⚠️ *Analisis Gemini AI gagal dijalankan hari ini.*\n"
            f"Alasan: `{str(exc)[:200]}`\n\n"
            f"Berikut data mentah yang berhasil dikumpulkan:\n\n"
            f"📈 *DATA PASAR*\n{market_summary}\n\n"
            f"📰 *BERITA*\n{news_summary}"
        )

    # --- Tahap 4: Simpan ke riwayat (best-effort) ---
    if analysis_report is not None:
        try:
            save_report(analysis_report, status="success")
            logger.info("Tahap 4 (Simpan Riwayat): berhasil.")
        except Exception as exc:
            # Kegagalan simpan riwayat TIDAK boleh menghentikan pengiriman
            # laporan ke Telegram — riwayat bersifat "nice-to-have".
            logger.warning(f"Tahap 4 (Simpan Riwayat) gagal, melanjutkan tanpa menyimpan: {exc}")

    # --- Tahap 5: Kirim laporan akhir ---
    header = f"🤖 *AI Finance Department Agent*\n📅 {_current_timestamp_wib()}\n"
    full_message = f"{header}\n{final_report_text}"

    try:
        sent = send_telegram_message(full_message)
        if not sent:
            logger.error("Tahap 5 (Kirim Telegram): sebagian/seluruh pesan gagal terkirim.")
            return False
        logger.info("Tahap 5 (Kirim Telegram): berhasil. Siklus selesai.")
        return True
    except Exception as exc:
        logger.error(f"Tahap 5 (Kirim Telegram) gagal total: {exc}")
        return False


if __name__ == "__main__":
    success = run_daily_report()
    sys.exit(0 if success else 1)