# gemini_analyzer.py

"""
Modul untuk mengirimkan data pasar + berita ke Gemini API (Google GenAI SDK)
dan menghasilkan analisis & rekomendasi Buy/Sell/Hold yang TERSTRUKTUR dan
TERVALIDASI (structured output via Pydantic schema), lengkap dengan retry
otomatis (exponential backoff) untuk menangani rate limit (HTTP 429) pada
jam-jam sibuk.

PENTING: Modul ini menggunakan package `google-genai` (SDK resmi terbaru),
BUKAN `google-generativeai` yang sudah deprecated per 30 November 2025.
"""

import logging
from typing import List, Literal, Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Disesuaikan dengan model yang aktif dipakai (lihat catatan di respons).
# gemini-2.5-flash dijadwalkan shutdown 16 Okt 2026 — gemini-3.6-flash
# adalah penggantinya yang sudah GA dan lebih efisien token.
GEMINI_MODEL_NAME = "gemini-3.5-flash"

DISCLAIMER_TEXT = (
    "\n\n⚠️ *DISCLAIMER:* Laporan ini dihasilkan secara otomatis oleh AI "
    "berdasarkan data pasar & berita publik. *Bukan* nasihat finansial "
    "resmi. Selalu lakukan riset mandiri (DYOR) dan konsultasi dengan "
    "penasihat keuangan berlisensi sebelum mengambil keputusan investasi."
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Model 2.5+/3.x adalah "reasoning model": token untuk proses berpikir
# internal ikut dipotong dari max_output_tokens. Untuk tugas ekstraksi
# terstruktur seperti ini (bukan penalaran matematis/logis kompleks),
# thinking tidak diperlukan — menonaktifkannya membuat output lebih
# cepat, lebih murah, dan yang terpenting TIDAK terpotong tak terduga.
GEMINI_THINKING_BUDGET = 0
GEMINI_MAX_OUTPUT_TOKENS = 4096


class RecommendationItem(BaseModel):
    """Satu baris rekomendasi untuk satu instrumen."""
    instrument: str = Field(description="Nama instrumen, misal 'IHSG' atau 'BBCA.JK'")
    action: Literal["BUY", "SELL", "HOLD"]
    reason: str = Field(description="Alasan singkat berbasis data, maksimal satu kalimat")
    confidence: Literal["Rendah", "Sedang", "Tinggi"]


class AnalysisReport(BaseModel):
    """Skema output terstruktur dari Gemini — dipaksa via response_schema."""
    market_summary: str = Field(description="2-3 kalimat ringkasan kondisi pasar")
    news_highlights: List[str] = Field(description="2-4 poin berita paling berpengaruh")
    recommendations: List[RecommendationItem]


class GeminiAnalysisError(Exception):
    """Exception khusus jika proses analisis Gemini gagal total."""
    pass


def _is_retryable_error(exc: BaseException) -> bool:
    """Predikat tenacity: hanya retry untuk error transien (429 & 5xx)."""
    return (
        isinstance(exc, genai_errors.APIError)
        and getattr(exc, "code", None) in RETRYABLE_STATUS_CODES
    )


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini_with_retry(client: genai.Client, prompt: str):
    """
    Wrapper pemanggilan Gemini API dengan retry otomatis. thinking_budget
    diset ke 0 agar seluruh kuota max_output_tokens dipakai untuk output
    JSON aktual, bukan proses berpikir internal yang tidak diperlukan
    untuk tugas ekstraksi terstruktur seperti ini.
    """
    return client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=AnalysisReport,
        ),
    )


def _build_analysis_prompt(market_summary: str, news_summary: str) -> str:
    """Prompt tidak berubah — format sepenuhnya dikontrol via response_schema."""
    return f"""Anda adalah seorang analis keuangan profesional yang membuat ringkasan pagi untuk seorang investor individu.

DATA PASAR HARI INI:
{market_summary}

BERITA KEUANGAN TERBARU:
{news_summary}

TUGAS ANDA:
Hasilkan analisis pasar pagi berdasarkan data di atas:

1. market_summary: 2-3 kalimat merangkum kondisi indeks dan saham yang dipantau berdasarkan data yang diberikan.
2. news_highlights: 2-4 poin berita paling berpengaruh terhadap pasar, masing-masing satu kalimat singkat yang menjelaskan dampaknya.
3. recommendations: untuk SETIAP instrumen yang datanya tersedia di atas, berikan satu entri berisi instrument, action, reason, dan confidence.

ATURAN PENTING:
- Rekomendasi HARUS didasarkan pada data yang diberikan di atas — jangan mengarang data atau berspekulasi di luar informasi yang tersedia.
- Jika data suatu instrumen tidak cukup untuk kesimpulan yang bertanggung jawab, gunakan action "HOLD" dengan reason "data belum cukup untuk kesimpulan kuat" dan confidence "Rendah" — jangan memaksakan BUY/SELL.
- Gunakan bahasa lugas, profesional, TIDAK bombastis (hindari kata seperti "pasti", "dijamin", "meledak")."""


def analyze_market(market_summary: str, news_summary: str) -> AnalysisReport:
    """
    Mengirim data pasar & berita ke Gemini API dan mengembalikan objek
    AnalysisReport yang sudah tervalidasi tipenya.

    Raises:
        GeminiAnalysisError: jika API key belum diset, seluruh percobaan
        retry habis, output terpotong (MAX_TOKENS), atau JSON gagal
        divalidasi terhadap skema.
    """
    Config.validate_gemini()

    prompt = _build_analysis_prompt(market_summary, news_summary)
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    try:
        response = _call_gemini_with_retry(client, prompt)
    except genai_errors.APIError as exc:
        logger.error(f"Gemini API gagal setelah seluruh percobaan retry: {exc}")
        raise GeminiAnalysisError(
            f"Gemini API gagal setelah beberapa kali percobaan (kemungkinan "
            f"rate limit/gangguan server): {exc}"
        ) from exc
    except Exception as exc:
        logger.error(f"Gagal memanggil Gemini API: {exc}")
        raise GeminiAnalysisError(f"Pemanggilan Gemini API gagal: {exc}") from exc

    # Deteksi dini output terpotong SEBELUM mencoba parsing, agar pesan
    # error jelas ("naikkan max_output_tokens") alih-alih error pydantic
    # yang membingungkan seperti yang terjadi sebelumnya.
    finish_reason = None
    if response.candidates:
        finish_reason = getattr(response.candidates[0], "finish_reason", None)

    if finish_reason is not None and "MAX_TOKENS" in str(finish_reason).upper():
        raise GeminiAnalysisError(
            f"Respons Gemini terpotong karena mencapai batas "
            f"max_output_tokens ({GEMINI_MAX_OUTPUT_TOKENS}). Naikkan "
            f"nilai GEMINI_MAX_OUTPUT_TOKENS di gemini_analyzer.py, atau "
            f"kurangi jumlah instrumen di MONITORED_INSTRUMENTS."
        )

    report: Optional[AnalysisReport] = response.parsed

    if report is None:
        try:
            report = AnalysisReport.model_validate_json(response.text)
        except (ValidationError, ValueError) as exc:
            raise GeminiAnalysisError(
                f"Respons Gemini tidak sesuai skema yang diharapkan: {exc}"
            ) from exc

    if not report.recommendations:
        raise GeminiAnalysisError("Gemini tidak menghasilkan rekomendasi apa pun (list kosong).")

    logger.info(
        f"Analisis Gemini berhasil: {len(report.recommendations)} rekomendasi dihasilkan."
    )
    return report


def format_report_for_telegram(report: AnalysisReport) -> str:
    """Mengubah AnalysisReport menjadi teks Markdown siap kirim ke Telegram."""
    lines = ["📈 *RINGKASAN PASAR*", report.market_summary, ""]

    lines.append("📰 *SOROTAN BERITA*")
    for point in report.news_highlights:
        lines.append(f"• {point}")
    lines.append("")

    lines.append("🎯 *REKOMENDASI*")
    action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    for rec in report.recommendations:
        emoji = action_emoji.get(rec.action, "⚪")
        lines.append(
            f"{emoji} *{rec.instrument}*: {rec.action} — {rec.reason} "
            f"(Keyakinan: {rec.confidence})"
        )

    body = "\n".join(lines)
    return body + DISCLAIMER_TEXT


if __name__ == "__main__":
    dummy_market = (
        "\n--- Indeks ---\n"
        "🟢 IHSG (Indonesia) (^JKSE): 7,250.10 (+35.20 / +0.49%) | Volume: 1,200,000,000\n"
        "🔴 Dow Jones (AS) (^DJI): 39,800.55 (-120.30 / -0.30%) | Volume: 350,000,000\n"
        "\n--- Saham ---\n"
        "🟢 Bank Central Asia (BBCA.JK): 9,850.00 (+50.00 / +0.51%) | Volume: 45,000,000"
    )
    dummy_news = (
        "\n--- Berita Indonesia ---\n"
        "- [CNBC Indonesia] Bank Indonesia tahan suku bunga acuan di 6%\n"
        "\n--- Berita Global ---\n"
        "- [Investing.com] The Fed sinyalkan potensi pemangkasan suku bunga Q4"
    )

    print("Mengirim data ke Gemini AI untuk dianalisis (structured output)...\n")
    try:
        result = analyze_market(dummy_market, dummy_news)
        print(format_report_for_telegram(result))
    except GeminiAnalysisError as err:
        print(f"❌ {err}")