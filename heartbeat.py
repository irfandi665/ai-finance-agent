# heartbeat.py

"""
Modul heartbeat mingguan: mengirim notifikasi singkat ke Telegram untuk
memastikan pengguna tahu agen masih berjalan normal. Berguna khususnya
untuk mendeteksi kasus GitHub Actions yang otomatis dinonaktifkan
setelah 60 hari repository tidak ada commit/aktivitas.
"""

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram_sender import send_telegram_message
from report_history import get_recent_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JAKARTA_TZ = ZoneInfo("Asia/Jakarta")


def send_heartbeat() -> bool:
    """
    Mengirim pesan heartbeat mingguan berisi status kesehatan agen,
    termasuk jumlah laporan sukses dalam 7 entri terakhir di database
    sebagai indikasi tambahan bahwa siklus harian benar-benar berjalan
    (bukan cuma proses heartbeat itu sendiri yang hidup).
    """
    timestamp = datetime.now(JAKARTA_TZ).strftime("%A, %d %B %Y - %H:%M WIB")

    try:
        recent_reports = get_recent_reports(limit=7)
        success_count = sum(1 for r in recent_reports if r["status"] == "success")
    except Exception as exc:
        logger.warning(f"Gagal membaca riwayat untuk heartbeat: {exc}")
        success_count = 0

    message = (
        f"💚 *AI Finance Agent - Heartbeat Mingguan*\n\n"
        f"Waktu cek: {timestamp}\n"
        f"Laporan sukses (7 entri terakhir di database): *{success_count}*\n\n"
    )

    if success_count == 0:
        message += (
            "⚠️ *Perhatian:* Tidak ada laporan sukses tercatat baru-baru ini. "
            "Kemungkinan scheduler/GitHub Actions berhenti berjalan — "
            "segera periksa secara manual."
        )
    else:
        message += "Agen terdeteksi masih berjalan normal. ✅"

    sent = send_telegram_message(message)
    logger.info("Heartbeat berhasil dikirim." if sent else "Heartbeat gagal dikirim ke Telegram.")
    return sent


if __name__ == "__main__":
    success = send_heartbeat()
    sys.exit(0 if success else 1)