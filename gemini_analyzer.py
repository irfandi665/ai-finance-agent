# gemini_analyzer.py

"""
Modul untuk mengirimkan data pasar + berita ke Gemini API (Google GenAI SDK)
dan menghasilkan analisis serta rekomendasi Buy/Sell/Hold yang terstruktur.

PENTING: Modul ini menggunakan package `google-genai` (SDK resmi terbaru),
BUKAN `google-generativeai` yang sudah deprecated per 30 November 2025.
"""

import logging

from google import genai
from google.genai import types

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Model Gemini yang dipakai. gemini-2.5-flash dipilih karena cepat,
# berkualitas baik untuk analisis terstruktur, dan tersedia di free tier
# Google AI Studio. Bisa diganti jika model lain tersedia di akun Anda.
GEMINI_MODEL_NAME = "gemini-3.6-flash"

DISCLAIMER_TEXT = (
    "\n\n⚠️ *DISCLAIMER:* Laporan ini dihasilkan secara otomatis oleh AI "
    "berdasarkan data pasar & berita publik. *Bukan* nasihat finansial "
    "resmi. Selalu lakukan riset mandiri (DYOR) dan konsultasi dengan "
    "penasihat keuangan berlisensi sebelum mengambil keputusan investasi."
)


class GeminiAnalysisError(Exception):
    """Exception khusus jika proses analisis Gemini gagal total."""
    pass


def _build_analysis_prompt(market_summary: str, news_summary: str) -> str:
    """
    Menyusun prompt terstruktur untuk Gemini. Dirancang agar output
    objektif berbasis data (bukan spekulasi bebas), memakai format
    Markdown dasar yang kompatibel Telegram, dan konsisten strukturnya
    setiap hari agar cepat dibaca di pagi hari.
    """
    return f"""Anda adalah seorang analis keuangan profesional yang membuat ringkasan pagi untuk seorang investor individu.

DATA PASAR HARI INI:
{market_summary}

BERITA KEUANGAN TERBARU:
{news_summary}

TUGAS ANDA:
Buat laporan analisis pasar pagi dalam Bahasa Indonesia dengan struktur PERSIS seperti berikut:

📈 *RINGKASAN PASAR*
(2-3 kalimat merangkum kondisi IHSG dan indeks global berdasarkan data di atas)

📰 *SOROTAN BERITA*
(2-4 poin berita paling berpengaruh terhadap pasar, jelaskan singkat dampaknya)

🎯 *REKOMENDASI*
(Untuk setiap instrumen yang datanya tersedia, berikan baris terpisah dengan format:
- [Nama Instrumen]: [BUY/SELL/HOLD] — [alasan singkat berbasis data, maksimal 1 kalimat] (Keyakinan: Rendah/Sedang/Tinggi))

ATURAN PENTING:
1. Rekomendasi HARUS didasarkan pada data pasar dan berita yang diberikan di atas — jangan mengarang data atau berspekulasi di luar informasi yang tersedia.
2. Jika data suatu instrumen tidak cukup untuk kesimpulan yang bertanggung jawab, tulis "HOLD — data belum cukup untuk kesimpulan kuat" daripada memaksakan Buy/Sell.
3. Gunakan bahasa lugas, profesional, TIDAK bombastis (hindari kata seperti "pasti", "dijamin", "meledak").
4. Gunakan HANYA format Markdown dasar yang didukung Telegram: tanda bintang tunggal (*teks*) untuk bold, garis bawah tunggal (_teks_) untuk italic. JANGAN gunakan tabel, heading (#), atau bullet bertingkat.
5. JANGAN menyertakan disclaimer di akhir — disclaimer ditambahkan otomatis oleh sistem.
6. Jawab HANYA dengan isi laporan sesuai struktur di atas, tanpa kalimat pembuka/penutup tambahan."""


def analyze_market(market_summary: str, news_summary: str) -> str:
    """
    Mengirim data pasar & berita ke Gemini API dan mengembalikan laporan
    analisis yang sudah diformat, lengkap dengan disclaimer otomatis.

    Raises:
        GeminiAnalysisError: jika API key belum diset atau pemanggilan
        API gagal.
    """
    Config.validate_gemini()

    prompt = _build_analysis_prompt(market_summary, news_summary)

    try:
        client = genai.Client(api_key=Config.GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,       # Rendah agar analisis konsisten & tidak liar
                max_output_tokens=1024,
            ),
        )

        analysis_text = (response.text or "").strip()

        if not analysis_text:
            raise GeminiAnalysisError(
                "Gemini API mengembalikan respons kosong. "
                "Kemungkinan konten diblokir oleh safety filter."
            )

        logger.info("Analisis Gemini berhasil dihasilkan.")
        return analysis_text + DISCLAIMER_TEXT

    except GeminiAnalysisError:
        raise
    except Exception as exc:
        logger.error(f"Gagal memanggil Gemini API: {exc}")
        raise GeminiAnalysisError(f"Pemanggilan Gemini API gagal: {exc}") from exc


if __name__ == "__main__":
    # Self-test: jalankan `python gemini_analyzer.py` untuk memverifikasi
    # GEMINI_API_KEY valid, menggunakan data dummy tanpa perlu menarik
    # data pasar/berita asli terlebih dahulu.
    dummy_market = (
        "🟢 IHSG (Indonesia) (^JKSE): 7,250.10 (+35.20 / +0.49%) | Volume: 1,200,000,000\n"
        "🔴 Dow Jones (AS) (^DJI): 39,800.55 (-120.30 / -0.30%) | Volume: 350,000,000"
    )
    dummy_news = (
        "\n--- Berita Indonesia ---\n"
        "- [CNBC Indonesia] Bank Indonesia tahan suku bunga acuan di 6%\n"
        "\n--- Berita Global ---\n"
        "- [Investing.com] The Fed sinyalkan potensi pemangkasan suku bunga Q4"
    )

    print("Mengirim data ke Gemini AI untuk dianalisis...\n")
    try:
        result = analyze_market(dummy_market, dummy_news)
        print(result)
    except GeminiAnalysisError as err:
        print(f"❌ {err}")