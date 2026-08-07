# interactive_bot.py

"""
Bot Telegram interaktif dua arah untuk AI Finance Department Agent, dengan:
1. MEMORI PERCAKAPAN per chat_id (multi-turn context)
2. AKSES DATA PASAR REAL-TIME via Function Calling manual — Gemini bisa
   memanggil get_stock_price() yang menarik data live dari Yahoo Finance
   (reuse market_data.get_index_data() dari Fase 3).

Arsitektur: python-telegram-bot v20+ (async, ApplicationBuilder).

Jalankan dengan:
    python interactive_bot.py
"""

import asyncio
import logging
import sys
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from config import Config, ConfigError
from market_data import get_index_data

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("interactive_bot.log", encoding="utf-8"),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Konfigurasi Gemini ---
GEMINI_MODEL_NAME = "gemini-3.5-flash"
GEMINI_MAX_OUTPUT_TOKENS = 1024
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_HISTORY_TURNS = 10

# Batas ronde function-calling per satu pesan user (misal: user minta
# bandingkan 2 saham sekaligus butuh 2 ronde). Mencegah infinite loop
# jika model terus-menerus meminta tool call tanpa henti.
MAX_FUNCTION_CALL_ROUNDS = 3

SYSTEM_PROMPT = """Anda adalah seorang Analis Keuangan Senior di sebuah firma investasi ternama di Wall Street, sedang berdiskusi langsung dengan seorang klien individu melalui chat.

GAYA KOMUNIKASI:
- Profesional namun ramah, seperti berbicara dengan klien yang Anda hormati — bukan kaku seperti robot.
- Jawaban ringkas dan padat (maksimal 4-6 kalimat untuk pertanyaan sederhana), kecuali klien meminta penjelasan mendalam.
- Manfaatkan riwayat percakapan sebelumnya untuk memberi jawaban yang nyambung dengan konteks.
- Gunakan istilah keuangan yang tepat, tapi jelaskan singkat jika istilahnya teknis.

AKSES DATA PASAR REAL-TIME:
- Anda memiliki akses ke fungsi get_stock_price untuk mengambil harga saham/indeks TERKINI langsung dari Yahoo Finance.
- WAJIB panggil fungsi ini setiap kali klien menanyakan harga, performa, atau kondisi terkini suatu saham/indeks tertentu — JANGAN PERNAH menjawab dari ingatan/asumsi tanpa memanggil fungsi ini terlebih dahulu.
- Jika fungsi mengembalikan error (misal ticker tidak ditemukan), sampaikan dengan jujur ke klien dan tanyakan konfirmasi kode ticker yang benar — jangan mengarang angka.

ATURAN PENTING:
- Jangan pernah memberikan kepastian ("pasti naik", "dijamin untung") — gunakan bahasa probabilistik yang bertanggung jawab ("berpotensi", "perlu dipantau", "berdasarkan data terkini").
- Jika pertanyaan di luar topik keuangan/ekonomi/investasi, arahkan dengan sopan kembali ke topik Anda sebagai analis keuangan.
- Gunakan HANYA format Markdown dasar Telegram: *teks* untuk bold, _teks_ untuk italic. Jangan gunakan heading (#) atau tabel."""

DISCLAIMER_FOOTER = "\n\n_⚠️ Bukan nasihat finansial resmi. DYOR sebelum mengambil keputusan investasi._"


class GeminiChatError(Exception):
    """Exception khusus jika Gemini API gagal merespons chat interaktif."""
    pass


# --- Definisi Tool: get_stock_price ---
# Skema JSON manual (bukan auto-generate dari docstring) agar deskripsi
# untuk model bisa ditulis sangat presisi, terutama soal format ticker
# yang sering jadi sumber kesalahan (BBCA.JK vs BBCA, ^JKSE vs JKSE).
GET_STOCK_PRICE_DECLARATION = types.FunctionDeclaration(
    name="get_stock_price",
    description=(
        "Mengambil data harga saham/indeks REAL-TIME (harga penutupan "
        "terakhir, perubahan harian, dan volume) langsung dari Yahoo "
        "Finance. WAJIB dipanggil setiap kali user menanyakan harga, "
        "performa, atau kondisi terkini dari suatu saham atau indeks."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Kode ticker Yahoo Finance. Untuk saham Indonesia, WAJIB "
                    "tambahkan akhiran '.JK' (contoh: 'BBCA.JK' untuk Bank "
                    "Central Asia, 'TLKM.JK' untuk Telkom Indonesia). Untuk "
                    "indeks, gunakan awalan '^' (contoh: '^JKSE' untuk IHSG, "
                    "'^DJI' untuk Dow Jones, '^GSPC' untuk S&P 500). Untuk "
                    "saham Amerika, gunakan kode biasa (contoh: 'AAPL')."
                ),
            }
        },
        "required": ["ticker"],
    },
)

MARKET_DATA_TOOL = types.Tool(function_declarations=[GET_STOCK_PRICE_DECLARATION])


async def _execute_get_stock_price(ticker: str) -> dict:
    """
    Wrapper async untuk get_index_data() dari market_data.py — fungsi
    SYNC yang melakukan panggilan jaringan blocking ke Yahoo Finance.

    Dijalankan via asyncio.to_thread() agar TIDAK memblokir event loop
    bot: tanpa ini, seluruh bot akan "membeku" (tidak bisa merespons
    user lain) selama request yfinance sedang berlangsung.

    Returns:
        Dict berisi data harga (jika sukses) atau dict {"error": ...}
        (jika ticker tidak ditemukan) — keduanya JSON-serializable,
        siap dikirim balik ke Gemini sebagai function response.
    """
    logger.info(f"Menjalankan get_stock_price(ticker={ticker!r})...")
    data = await asyncio.to_thread(get_index_data, ticker)

    if data is None:
        return {
            "error": (
                f"Data untuk ticker '{ticker}' tidak ditemukan atau gagal "
                f"diambil. Periksa apakah format ticker sudah benar "
                f"(misal '.JK' untuk saham Indonesia)."
            )
        }

    return {
        "ticker": data.ticker,
        "name": data.name,
        "last_close": data.last_close,
        "previous_close": data.previous_close,
        "change_value": data.change_value,
        "change_percent": data.change_percent,
        "volume": data.volume,
    }


async def _call_tool(function_name: str, function_args: dict) -> dict:
    """
    Dispatcher sederhana: memetakan nama fungsi yang diminta Gemini ke
    implementasi Python-nya. Menambah tool baru di masa depan cukup
    dengan menambah satu entri di sini (Open/Closed Principle).
    """
    if function_name == "get_stock_price":
        return await _execute_get_stock_price(**function_args)

    logger.warning(f"Gemini meminta fungsi yang tidak dikenali: {function_name}")
    return {"error": f"Fungsi '{function_name}' tidak dikenali oleh sistem."}


class ChatHistoryManager:
    """Menyimpan riwayat percakapan IN-MEMORY, di-keyed per chat_id, dengan batas maksimum jumlah turn."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self._max_turns = max_turns
        self._store: Dict[int, List[types.Content]] = {}

    def get_history(self, chat_id: int) -> List[types.Content]:
        return self._store.get(chat_id, [])

    def add_exchange(self, chat_id: int, user_text: str, model_text: str) -> None:
        history = self._store.setdefault(chat_id, [])
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=model_text)]))

        max_entries = self._max_turns * 2
        if len(history) > max_entries:
            self._store[chat_id] = history[-max_entries:]

    def reset(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)


def _split_text(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """Memecah teks panjang agar tidak melebihi batas 4096 karakter Telegram."""
    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


async def _reply_safe(update: Update, text: str) -> None:
    """Mengirim balasan dengan fallback Markdown → teks polos, dan pemecahan otomatis jika terlalu panjang."""
    chunks = _split_text(text)
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except BadRequest as exc:
            logger.warning(f"Gagal kirim dengan Markdown, fallback ke teks polos: {exc}")
            await update.message.reply_text(chunk)


async def get_gemini_response(
    user_message: str,
    gemini_client: genai.Client,
    history: List[types.Content],
) -> str:
    """
    Mengirim pesan user + riwayat percakapan ke Gemini, dengan akses tool
    get_stock_price untuk data pasar real-time. Menjalankan loop
    function-calling manual: jika Gemini meminta tool, kita eksekusi lalu
    kirim hasilnya kembali, sampai Gemini memberi jawaban teks final.

    Returns:
        Teks balasan final MENTAH (tanpa disclaimer — ditambahkan terpisah
        saat pengiriman ke Telegram).

    Raises:
        GeminiChatError: jika API gagal, respons kosong, atau loop
        function-calling melebihi MAX_FUNCTION_CALL_ROUNDS.
    """
    try:
        user_content = types.Content(role="user", parts=[types.Part(text=user_message)])
        contents: List[types.Content] = history + [user_content]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=[MARKET_DATA_TOOL],
        )

        for round_number in range(1, MAX_FUNCTION_CALL_ROUNDS + 1):
            response = await gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=config,
            )

            candidate_content = response.candidates[0].content
            function_call_parts = [
                part for part in candidate_content.parts if part.function_call is not None
            ]

            # Tidak ada tool call diminta -> ini jawaban final dari Gemini.
            if not function_call_parts:
                text = (response.text or "").strip()
                if not text:
                    raise GeminiChatError("Gemini mengembalikan respons kosong.")
                return text

            logger.info(
                f"Ronde {round_number}: Gemini meminta "
                f"{[p.function_call.name for p in function_call_parts]}"
            )

            # Sertakan turn "model" berisi permintaan tool call ke dalam
            # riwayat percakapan SEMENTARA (khusus untuk turn ini).
            contents.append(candidate_content)

            # Eksekusi SEMUA tool call yang diminta (bisa lebih dari satu
            # dalam satu ronde, misal saat user minta bandingkan 2 saham).
            response_parts = []
            for part in function_call_parts:
                fc = part.function_call
                result = await _call_tool(fc.name, dict(fc.args))
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )

            contents.append(types.Content(role="tool", parts=response_parts))
            # Loop berlanjut: Gemini akan dipanggil ULANG dengan hasil tool
            # sudah tersedia di contents, agar bisa menyusun jawaban final.

        raise GeminiChatError(
            f"Gemini masih meminta pemanggilan fungsi setelah "
            f"{MAX_FUNCTION_CALL_ROUNDS} ronde — kemungkinan ada masalah data."
        )

    except genai_errors.APIError as exc:
        logger.error(f"Gemini API error saat memproses chat: {exc}")
        raise GeminiChatError(f"Gemini API error: {exc}") from exc
    except GeminiChatError:
        raise
    except Exception as exc:
        logger.error(f"Kesalahan tak terduga saat memanggil Gemini: {exc}")
        raise GeminiChatError(f"Kesalahan tak terduga: {exc}") from exc


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler untuk perintah /start — pesan sambutan & panduan singkat."""
    welcome_message = (
        "👋 Halo! Saya *AI Finance Assistant* Anda.\n\n"
        "Saya bisa mengambil harga saham/indeks *real-time* langsung dari "
        "Yahoo Finance dan mengingat konteks obrolan kita.\n\n"
        "Contoh pertanyaan:\n"
        "• _Berapa harga BBCA sekarang?_\n"
        "• _Bandingkan performa BBCA dan TLKM hari ini_\n"
        "• _Apa dampak The Fed naikkan suku bunga?_\n\n"
        "Ketik /reset kapan saja untuk memulai topik baru dari nol."
    )
    await _reply_safe(update, welcome_message)
    logger.info(f"User {update.effective_user.id} memulai sesi via /start.")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler untuk perintah /reset — menghapus riwayat percakapan chat ini."""
    chat_id = update.effective_chat.id
    history_manager: ChatHistoryManager = context.bot_data["history_manager"]
    history_manager.reset(chat_id)

    await update.message.reply_text("🔄 Riwayat percakapan telah dihapus. Mari mulai topik baru!")
    logger.info(f"Riwayat chat {chat_id} direset via /reset.")


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler utama untuk pesan teks — dengan memori percakapan & akses data real-time."""
    user_message = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    logger.info(f"Pesan masuk dari user {user_id}: {user_message[:80]}")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    gemini_client: genai.Client = context.bot_data["gemini_client"]
    history_manager: ChatHistoryManager = context.bot_data["history_manager"]

    try:
        history = history_manager.get_history(chat_id)
        reply_text = await get_gemini_response(user_message, gemini_client, history)

        history_manager.add_exchange(chat_id, user_message, reply_text)

        await _reply_safe(update, reply_text + DISCLAIMER_FOOTER)
        logger.info(f"Balasan berhasil dikirim ke user {user_id}.")

    except GeminiChatError as exc:
        logger.error(f"Gagal mendapatkan respons Gemini untuk user {user_id}: {exc}")
        await update.message.reply_text(
            "⚠️ Maaf, sistem analisis AI sedang sibuk atau mengalami "
            "gangguan sementara (bisa jadi data pasar sedang tidak dapat "
            "diakses). Silakan coba lagi dalam beberapa saat."
        )
    except Exception as exc:
        logger.critical(f"Error tak terduga di handle_text_message (user {user_id}): {exc}")
        await update.message.reply_text(
            "⚠️ Terjadi kesalahan tak terduga di sistem kami. "
            "Tim teknis akan segera memeriksanya."
        )


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Error handler tingkat aplikasi untuk kegagalan di luar try/except lokal."""
    logger.error(f"Update yang menyebabkan error: {update}", exc_info=context.error)


def main() -> None:
    """Inisialisasi dan menjalankan bot dalam mode polling."""
    try:
        Config.validate()
        Config.validate_gemini()
    except ConfigError as exc:
        logger.critical(f"Konfigurasi tidak valid, bot tidak bisa dijalankan: {exc}")
        sys.exit(1)

    gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

    application = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()
    application.bot_data["gemini_client"] = gemini_client
    application.bot_data["history_manager"] = ChatHistoryManager(max_turns=MAX_HISTORY_TURNS)

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(global_error_handler)

    logger.info("Bot interaktif AI Finance Agent (memori + real-time data) mulai berjalan...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()