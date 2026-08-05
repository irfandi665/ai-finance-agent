# telegram_sender.py

import requests
import logging
from config import Config

# Setup logging dasar agar setiap kegagalan pengiriman tercatat jelas
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TELEGRAM_API_BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096  # Batas hard-limit dari Telegram API


def _split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list:
    """
    Memecah teks panjang menjadi beberapa bagian (chunk) agar tidak
    melebihi batas karakter Telegram, tanpa memotong kata di tengah.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        # Jika menambahkan baris ini masih dalam batas, tambahkan
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Mengirim pesan ke Telegram menggunakan Bot API.

    Args:
        message: Isi pesan yang akan dikirim. Mendukung format Markdown
                 (contoh: *bold*, _italic_, `code`).
        parse_mode: Mode parsing Telegram, default "Markdown".

    Returns:
        True jika seluruh bagian pesan berhasil terkirim, False jika ada
        kegagalan pada salah satu bagian.
    """
    url = TELEGRAM_API_BASE_URL.format(token=Config.TELEGRAM_BOT_TOKEN)
    chunks = _split_message(message)
    all_success = True

    for index, chunk in enumerate(chunks, start=1):
        payload = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            logger.info(
                f"Pesan bagian {index}/{len(chunks)} berhasil dikirim ke Telegram."
            )
        except requests.exceptions.HTTPError as http_err:
            logger.error(
                f"Gagal mengirim pesan bagian {index}/{len(chunks)}. "
                f"HTTP Error: {http_err} | Response: {response.text}"
            )
            all_success = False
        except requests.exceptions.RequestException as req_err:
            logger.error(
                f"Gagal mengirim pesan bagian {index}/{len(chunks)}. "
                f"Kesalahan jaringan: {req_err}"
            )
            all_success = False

    return all_success


if __name__ == "__main__":
    # Blok pengujian mandiri (self-test).
    # Jalankan langsung: python telegram_sender.py
    # untuk memverifikasi BOT_TOKEN & CHAT_ID sudah benar sebelum lanjut ke Fase 2.
    test_message = (
        "✅ *Tes Koneksi Berhasil*\n\n"
        "Modul `telegram_sender.py` pada *AI Finance Department Agent* "
        "sudah terhubung dengan benar ke bot Telegram Anda.\n\n"
        "_Fase 1 siap dilanjutkan ke Fase 2._"
    )

    success = send_telegram_message(test_message)

    if success:
        print("✅ Pesan tes berhasil dikirim. Periksa Telegram Anda.")
    else:
        print("❌ Pengiriman gagal. Periksa log error di atas dan validasi .env Anda.")