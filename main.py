# main.py

"""
Orchestrator utama "AI Finance Department Agent".

Alur eksekusi:
    1. Tarik data pasar (IHSG & indeks global)
    2. Tarik berita keuangan terbaru (Indonesia & global)
    3. Kirim data ke Gemini AI untuk dianalisis
    4. Format laporan akhir
    5. Kirim laporan ke Telegram

Prinsip desain: setiap tahap diisolasi dalam try/except sendiri. Kegagalan
di satu tahap TIDAK BOLEH membuat seluruh proses mati diam-diam — sistem
harus selalu memberi sinyal ke pengguna (via Telegram jika memungkinkan,
via log lokal jika tidak) tentang apa yang terjadi.
"""

import logging
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from config import Config, ConfigError
from market_data import get_all_market_data, format_market_summary
from news_fetcher import fetch_all_news, format_news_summary
from gemini_analyzer import analyze_market, GeminiAnalysisError
from telegram_sender import send_telegram_message

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
    """Menggunakan zoneinfo (built-in Python 3.9+) — tanpa dependency tambahan."""
    return datetime.now(JAKARTA_TZ).strftime("%A, %d %B %Y - %H:%M WIB")


def _notify_failure(stage: str, error: Exception) -> None:
    """
    Mengirim notifikasi kegagalan ke Telegram agar pengguna tahu agen
    gagal berjalan (bukan diam tanpa jejak). Jika pengiriman notifikasi
    ini sendiri gagal, kegagalan dicatat ke log lokal sebagai upaya
    terakhir (last resort) — dan di GitHub Actions, exit code non-nol
    akan tetap menandai run tersebut sebagai "failed" di tab Actions.
    """
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
            logger.error(
                "Notifikasi kegagalan JUGA gagal dikirim ke Telegram. "
                "Ini kegagalan kritis yang butuh pengecekan manual."
            )
    except Exception as notify_exc:
        logger.error(
            f"Tidak bisa mengirim notifikasi kegagalan ke Telegram sama "
            f"sekali: {notify_exc}"
        )


def run_daily_report() -> bool:
    """
    Menjalankan satu siklus penuh laporan harian.

    Returns:
        True jika laporan berhasil terkirim (laporan lengkap maupun
        laporan fallback), False jika seluruh pipeline gagal total.
    """
    logger.info("=== Memulai siklus AI Finance Department Agent ===")

    # --- Tahap 0: Validasi konfigurasi dasar ---
    try:
        Config.validate()
    except ConfigError as exc:
        # Tidak bisa notifikasi Telegram karena kredensial Telegram
        # sendiri yang bermasalah — hanya bisa dicatat di log.
        logger.critical(f"Konfigurasi dasar tidak valid, agen dihentikan: {exc}")
        return False

    # --- Tahap 1: Tarik data pasar (WAJIB — kegagalan total = abort) ---
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

    # --- Tahap 2: Tarik berita (OPSIONAL — kegagalan tetap lanjut) ---
    try:
        news_items = fetch_all_news(limit_per_source=5)
        news_summary = format_news_summary(news_items)
        logger.info(f"Tahap 2 (News): berhasil, {len(news_items)} berita diambil.")
    except Exception as exc:
        logger.warning(f"Tahap 2 (News) gagal, melanjutkan tanpa data berita: {exc}")
        news_summary = "Tidak ada data berita yang tersedia hari ini (gagal diambil)."

    # --- Tahap 3: Analisis Gemini AI (fallback ke laporan mentah jika gagal) ---
    try:
        final_report = analyze_market(market_summary, news_summary)
        logger.info("Tahap 3 (Gemini Analysis): berhasil.")
    except GeminiAnalysisError as exc:
        logger.error(f"Tahap 3 (Gemini Analysis) gagal: {exc}")
        # Graceful degradation: pengguna tetap menerima data mentah
        # daripada tidak menerima apa pun sama sekali.
        final_report = (
            f"⚠️ *Analisis Gemini AI gagal dijalankan hari ini.*\n"
            f"Alasan: `{str(exc)[:200]}`\n\n"
            f"Berikut data mentah yang berhasil dikumpulkan:\n\n"
            f"📈 *DATA PASAR*\n{market_summary}\n\n"
            f"📰 *BERITA*\n{news_summary}"
        )

    # --- Tahap 4: Susun & kirim laporan akhir ---
    header = f"🤖 *AI Finance Department Agent*\n📅 {_current_timestamp_wib()}\n"
    full_message = f"{header}\n{final_report}"

    try:
        sent = send_telegram_message(full_message)
        if not sent:
            logger.error("Tahap 4 (Kirim Telegram): sebagian/seluruh pesan gagal terkirim.")
            return False
        logger.info("Tahap 4 (Kirim Telegram): berhasil. Siklus selesai.")
        return True
    except Exception as exc:
        logger.error(f"Tahap 4 (Kirim Telegram) gagal total: {exc}")
        return False


if __name__ == "__main__":
    # Exit code penting: GitHub Actions & systemd membaca ini untuk
    # menandai run sebagai sukses (0) atau gagal (bukan 0).
    success = run_daily_report()
    sys.exit(0 if success else 1)