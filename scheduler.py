# scheduler.py

"""
Scheduler untuk menjalankan main.py setiap hari jam 07:00 (waktu lokal
server) secara berulang. Cocok dijalankan sebagai proses long-running
di WSL/Linux.

CATATAN PENTING: Library `schedule` menggunakan timezone LOKAL SERVER,
bukan timezone eksplisit. Pastikan timezone server sudah diset ke
Asia/Jakarta (lihat Langkah 2.1), atau sesuaikan SCHEDULE_TIME secara
manual jika server berjalan di timezone lain (misal UTC).
"""

import logging
import time

import schedule

from main import run_daily_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SCHEDULE_TIME = "07:00"  # Waktu lokal server — pastikan server sudah WIB


def job() -> None:
    logger.info(f"Menjalankan job terjadwal pada {SCHEDULE_TIME}...")
    try:
        run_daily_report()
    except Exception as exc:
        # Safety net terakhir: apa pun yang lolos dari penanganan error
        # di main.py tidak boleh membuat proses scheduler ini mati,
        # karena scheduler harus tetap hidup untuk job besok.
        logger.critical(f"Job gagal dengan error tak terduga: {exc}")


def start_scheduler() -> None:
    schedule.every().day.at(SCHEDULE_TIME).do(job)
    logger.info(
        f"Scheduler aktif. Laporan akan dikirim setiap hari jam "
        f"{SCHEDULE_TIME} (waktu lokal server)."
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()