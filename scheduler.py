# scheduler.py

"""
Scheduler untuk menjalankan main.py setiap hari jam 07:00, DAN mengirim
heartbeat mingguan setiap Senin jam 08:00 (waktu lokal server — pastikan
timezone server sudah WIB, lihat Fase 3 Langkah 2.1).
"""

import logging
import time

import schedule

from main import run_daily_report
from heartbeat import send_heartbeat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAILY_REPORT_TIME = "07:00"
HEARTBEAT_TIME = "08:00"


def daily_job() -> None:
    logger.info(f"Menjalankan laporan harian terjadwal pada {DAILY_REPORT_TIME}...")
    try:
        run_daily_report()
    except Exception as exc:
        logger.critical(f"Job laporan harian gagal dengan error tak terduga: {exc}")


def heartbeat_job() -> None:
    logger.info("Menjalankan heartbeat mingguan...")
    try:
        send_heartbeat()
    except Exception as exc:
        logger.critical(f"Job heartbeat gagal dengan error tak terduga: {exc}")


def start_scheduler() -> None:
    schedule.every().day.at(DAILY_REPORT_TIME).do(daily_job)
    schedule.every().monday.at(HEARTBEAT_TIME).do(heartbeat_job)

    logger.info(
        f"Scheduler aktif. Laporan harian: {DAILY_REPORT_TIME}, "
        f"Heartbeat mingguan: Senin {HEARTBEAT_TIME} (waktu lokal server)."
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()