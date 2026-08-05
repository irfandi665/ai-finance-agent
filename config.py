# config.py

import os
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Exception khusus jika ada credential yang hilang atau tidak valid."""
    pass


class Config:
    """
    Kelas terpusat untuk mengakses seluruh credentials & konfigurasi proyek.
    """

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    @classmethod
    def validate(cls) -> None:
        """
        Memvalidasi credential WAJIB untuk operasi dasar (pengiriman Telegram).
        Dipanggil di awal eksekusi seluruh entry point.
        """
        required_vars = {
            "TELEGRAM_BOT_TOKEN": cls.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": cls.TELEGRAM_CHAT_ID,
        }
        missing = [name for name, value in required_vars.items() if not value]

        if missing:
            raise ConfigError(
                f"Konfigurasi tidak lengkap. Variabel berikut belum diisi "
                f"di file .env: {', '.join(missing)}."
            )

    @classmethod
    def validate_gemini(cls) -> None:
        """
        Memvalidasi credential khusus untuk fitur analisis Gemini AI.
        Dipanggil secara terpisah oleh gemini_analyzer.py, bukan oleh
        validate() inti, agar setiap modul hanya gagal-fast atas
        credential yang relevan dengan fungsinya masing-masing.
        """
        if not cls.GEMINI_API_KEY:
            raise ConfigError(
                "GEMINI_API_KEY belum diisi di file .env. "
                "Dapatkan API key gratis di https://aistudio.google.com/app/apikey"
            )


# Validasi dasar otomatis dijalankan saat modul ini di-import
Config.validate()