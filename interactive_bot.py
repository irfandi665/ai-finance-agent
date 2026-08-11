# interactive_bot.py

"""
Bot Telegram interaktif untuk AI Finance Department Agent, dengan:
1. MEMORI PERCAKAPAN per chat_id
2. AKSES DATA PASAR REAL-TIME (harga) via Function Calling manual
3. ANALISIS TEKNIKAL REAL-TIME (SMA/EMA/RSI/MACD/Bollinger) via tool
   get_technical_analysis (technical_analysis.py)
4. ANALISIS GAMBAR & DOKUMEN — user bisa kirim screenshot chart (analisis
   teknikal visual + konfirmasi angka riil) atau PDF laporan keuangan
   (analisis fundamental)
5. DYNAMIC MODEL SWITCHER via /model (tombol) + rate-limit fallback otomatis
6. PROFESSIONAL HTML FORMATTING dengan sanitasi anti-crash

Arsitektur: python-telegram-bot v20+ (async, ApplicationBuilder).

Jalankan dengan:
    python interactive_bot.py
"""

import asyncio
import logging
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
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
from technical_analysis import get_technical_analysis

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


# --- Konfigurasi Umum ---
GEMINI_MAX_OUTPUT_TOKENS = 3072  # naik dari 1536 — provenance + rumus + daily & weekly + completeness check butuh ruang lebih
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_HISTORY_TURNS = 10
MAX_FUNCTION_CALL_ROUNDS = 4     # naik dari 3 — mengakomodasi pemanggilan daily + weekly + harga sekaligus
RATE_LIMIT_COOLDOWN_SECONDS = 60

# Batas ukuran file yang diterima — sedikit di bawah limit download bot
# Telegram (20MB) untuk memberi headroom, dan tetap aman untuk inline
# request Gemini (bukan File API).
MAX_MEDIA_FILE_BYTES = 15 * 1024 * 1024
ALLOWED_DOCUMENT_MIME_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/webp"}


@dataclass
class ModelOption:
    id: str
    label: str
    rpm_limit: int
    rpd_limit: int


# ⚠️ PENTING: verifikasi ulang ketersediaan model di
# https://ai.google.dev/gemini-api/docs/models dan limit RPM/RPD akun
# Anda di https://aistudio.google.com/app/apikey.
AVAILABLE_MODELS: List[ModelOption] = [
    ModelOption(id="gemini-3.6-flash", label="Gemini 3.6 Flash", rpm_limit=10, rpd_limit=250),
    ModelOption(id="gemini-3.5-flash", label="Gemini 3.5 Flash", rpm_limit=10, rpd_limit=250),
    ModelOption(id="gemini-3.5-flash-lite", label="Gemini 3.5 Flash-Lite", rpm_limit=15, rpd_limit=1000),
    ModelOption(id="gemini-3.1-pro", label="Gemini 3.1 Pro", rpm_limit=5, rpd_limit=100),
    ModelOption(id="gemini-3.1-flash-lite", label="Gemini 3.1 Flash-Lite", rpm_limit=15, rpd_limit=1000),
    ModelOption(id="gemini-2.5-pro", label="Gemini 2.5 Pro", rpm_limit=5, rpd_limit=100),
    ModelOption(id="gemini-2.5-flash", label="Gemini 2.5 Flash", rpm_limit=10, rpd_limit=250),
    ModelOption(id="gemini-2.5-flash-lite", label="Gemini 2.5 Flash-Lite", rpm_limit=15, rpd_limit=1000),
]

DEFAULT_MODEL_ID = AVAILABLE_MODELS[0].id

SYSTEM_PROMPT = """Anda adalah seorang Analis Keuangan Senior di sebuah firma investasi ternama di Wall Street, sedang berdiskusi langsung dengan seorang klien individu melalui chat.

GAYA KOMUNIKASI:
- Profesional namun ramah — bukan kaku seperti robot.
- Manfaatkan riwayat percakapan sebelumnya untuk jawaban yang nyambung dengan konteks.
- Gunakan istilah keuangan yang tepat, jelaskan singkat jika teknis.

AKSES DATA PASAR REAL-TIME:
- Gunakan get_stock_price untuk harga TERKINI (harga, perubahan, volume) dari Yahoo Finance.
- WAJIB panggil setiap kali klien menanyakan harga/performa suatu saham/indeks — JANGAN pernah menjawab dari ingatan/asumsi.

ANALISIS TEKNIKAL BERBASIS DATA REAL:
- Gunakan get_technical_analysis untuk mendapatkan SMA, EMA, RSI, MACD, Bollinger Bands, dan support/resistance 3 bulan yang DIHITUNG dari data historis sungguhan.
- WAJIB panggil fungsi ini setiap kali diminta analisis teknikal, sinyal beli/jual berbasis indikator, atau menyebut istilah seperti RSI/MACD/support/resistance untuk suatu ticker. JANGAN PERNAH mengarang nilai indikator dari ingatan.

ANALISIS GAMBAR & DOKUMEN:
- Jika user mengirim GAMBAR (misal screenshot chart candlestick): lakukan analisis teknikal VISUAL (identifikasi pola chart seperti head-and-shoulders/double top-bottom, arah tren, level support/resistance yang terlihat di gambar). Jika ada ticker/nama saham yang disebutkan di caption atau terlihat di gambar, WAJIB panggil get_technical_analysis untuk MENGONFIRMASI dengan angka riil — jangan hanya mengandalkan interpretasi visual untuk angka presisi seperti RSI.
- Jika user mengirim DOKUMEN (misal laporan keuangan PDF): lakukan analisis FUNDAMENTAL berdasarkan isi dokumen tersebut — ekstrak metrik penting (pendapatan, laba bersih, margin, rasio utang, pertumbuhan YoY) yang BENAR-BENAR TERTULIS di dokumen. Jangan mengarang angka yang tidak ada di dalamnya.

FORMAT OUTPUT (SANGAT PENTING — WAJIB DIIKUTI):
Gunakan HANYA tag HTML berikut yang didukung Telegram: <b>teks</b> (bold — untuk angka penting, harga, persentase, sub-judul), <i>teks</i> (italic — catatan kecil), <u>teks</u> (underline — penekanan khusus), <code>teks</code> (kode ticker, misal <code>BBCA.JK</code>).
JANGAN PERNAH memakai tag lain seperti <div>, <table>, <h1>, <ul>, <li>, <p>, <br> — Telegram TIDAK mendukungnya dan pesan akan gagal terkirim.

Untuk pertanyaan/analisis (bukan basa-basi singkat), susun seperti laporan profesional:

📊 <b>[Judul Singkat Topik]</b>
──────────────────
[paragraf ringkasan 1-2 kalimat]

<b>Poin Kunci:</b>
- [poin 1, angka penting dalam <b>bold</b>]
- [poin 2]
- [poin 3, maksimal 4-5 poin]
──────────────────

Untuk sapaan/basa-basi singkat, jawab singkat tanpa struktur laporan ini.

ATURAN LAIN:
- Jangan pernah memberi kepastian ("pasti naik", "dijamin untung") — gunakan bahasa probabilistik ("berpotensi", "perlu dipantau").
- Jika di luar topik keuangan, arahkan sopan kembali ke topik Anda."""

DISCLAIMER_FOOTER = "\n\n<i>⚠️ Bukan nasihat finansial resmi. DYOR sebelum mengambil keputusan investasi.</i>"


class GeminiChatError(Exception):
    """Exception khusus jika Gemini API gagal merespons chat interaktif."""
    pass


# ============================================================
# STATE MANAGEMENT: Rate-Limit Tracking (global, per model)
# ============================================================

class ModelManager:
    """Melacak penggunaan (RPM/RPD) dan status rate-limit tiap model Gemini secara global, serta fallback reaktif saat API mengembalikan 429."""

    def __init__(self, models: List[ModelOption]) -> None:
        if not models:
            raise ValueError("AVAILABLE_MODELS tidak boleh kosong.")
        self._models = models
        self._request_log: Dict[str, deque] = {m.id: deque() for m in models}
        self._daily_count: Dict[str, Tuple[str, int]] = {m.id: (self._today(), 0) for m in models}
        self._cooldown_until: Dict[str, float] = {m.id: 0.0 for m in models}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_option(self, model_id: str) -> Optional[ModelOption]:
        return next((m for m in self._models if m.id == model_id), None)

    def _prune_and_count_rpm(self, model_id: str) -> int:
        now = time.time()
        log = self._request_log[model_id]
        while log and now - log[0] > 60:
            log.popleft()
        return len(log)

    def _get_rpd_count(self, model_id: str) -> int:
        today = self._today()
        stored_date, count = self._daily_count[model_id]
        if stored_date != today:
            self._daily_count[model_id] = (today, 0)
            return 0
        return count

    def _has_headroom(self, option: ModelOption) -> bool:
        if time.time() < self._cooldown_until[option.id]:
            return False
        if self._prune_and_count_rpm(option.id) >= option.rpm_limit:
            return False
        if self._get_rpd_count(option.id) >= option.rpd_limit:
            return False
        return True

    def get_active_model(self, preferred: Optional[str] = None) -> str:
        if preferred:
            option = self._get_option(preferred)
            if option and self._has_headroom(option):
                return option.id
        for option in self._models:
            if self._has_headroom(option):
                return option.id
        return self._models[0].id

    def get_next_fallback(self, current_model: str) -> Optional[str]:
        ids = [m.id for m in self._models]
        try:
            idx = ids.index(current_model)
        except ValueError:
            return ids[0] if ids else None
        return ids[idx + 1] if idx + 1 < len(ids) else None

    def record_request(self, model_id: str) -> None:
        self._request_log.setdefault(model_id, deque()).append(time.time())
        today = self._today()
        stored_date, count = self._daily_count.get(model_id, (today, 0))
        if stored_date != today:
            count = 0
        self._daily_count[model_id] = (today, count + 1)

    def mark_rate_limited(self, model_id: str) -> None:
        self._cooldown_until[model_id] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS
        logger.warning(f"Model '{model_id}' terkena rate limit (429) — cooldown {RATE_LIMIT_COOLDOWN_SECONDS}s.")

    def get_status_text(self) -> str:
        lines = ["📊 <b>Status Penggunaan Model Gemini (Global)</b>\n"]
        for option in self._models:
            rpm_used = self._prune_and_count_rpm(option.id)
            rpd_used = self._get_rpd_count(option.id)
            in_cooldown = time.time() < self._cooldown_until[option.id]
            icon = "🧊" if in_cooldown else "✅"
            lines.append(
                f"{icon} <b>{option.label}</b> (<code>{option.id}</code>)\n"
                f"     RPM: {rpm_used}/{option.rpm_limit} | RPD: {rpd_used}/{option.rpd_limit}"
            )
        lines.append(
            "\n<i>Catatan: angka limit adalah perkiraan lokal. Cek limit "
            "sebenarnya di aistudio.google.com/app/apikey.</i>"
        )
        return "\n".join(lines)


class UserModelPreferenceStore:
    """Menyimpan preferensi model Gemini PER USER (in-memory, di-keyed oleh user_id)."""

    def __init__(self, default_model_id: str) -> None:
        self._default_model_id = default_model_id
        self._preferences: Dict[int, str] = {}

    def get(self, user_id: int) -> str:
        return self._preferences.get(user_id, self._default_model_id)

    def set(self, user_id: int, model_id: str) -> None:
        self._preferences[user_id] = model_id


class ChatHistoryManager:
    """Menyimpan riwayat percakapan IN-MEMORY per chat_id, dengan batas maksimum jumlah turn."""

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


# ============================================================
# HTML SANITIZATION
# ============================================================

_ALLOWED_HTML_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}
_HTML_TAG_PATTERN = re.compile(r"</?([a-zA-Z0-9]+)(\s+[^>]*)?>")


def _sanitize_html_for_telegram(text: str) -> str:
    """Membuang tag HTML di luar whitelist, mempertahankan teks di dalamnya."""
    def _strip_disallowed(match) -> str:
        tag_name = match.group(1).lower()
        return match.group(0) if tag_name in _ALLOWED_HTML_TAGS else ""
    return _HTML_TAG_PATTERN.sub(_strip_disallowed, text)


def _strip_all_html_tags(text: str) -> str:
    """Fallback terakhir: buang SEMUA tag HTML."""
    return _HTML_TAG_PATTERN.sub("", text)


def _split_text(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
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
    """Sanitasi tag HTML tak dikenal → split otomatis → fallback teks polos jika Telegram tetap menolak."""
    sanitized = _sanitize_html_for_telegram(text)
    chunks = _split_text(sanitized)
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except BadRequest as exc:
            logger.warning(f"Gagal kirim dengan HTML, fallback ke teks polos: {exc}")
            await update.message.reply_text(_strip_all_html_tags(chunk))


# ============================================================
# TOOLS: get_stock_price & get_technical_analysis
# ============================================================

GET_STOCK_PRICE_DECLARATION = types.FunctionDeclaration(
    name="get_stock_price",
    description=(
        "Mengambil data harga saham/indeks REAL-TIME (harga penutupan "
        "terakhir, perubahan harian, volume) dari Yahoo Finance. WAJIB "
        "dipanggil setiap kali user menanyakan harga/performa terkini."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": (
                    "Kode ticker Yahoo Finance. Saham Indonesia WAJIB akhiran "
                    "'.JK' (contoh 'BBCA.JK'). Indeks pakai awalan '^' (contoh "
                    "'^JKSE'). Saham AS pakai kode biasa (contoh 'AAPL')."
                ),
            }
        },
        "required": ["ticker"],
    },
)

GET_TECHNICAL_ANALYSIS_DECLARATION = types.FunctionDeclaration(
    name="get_technical_analysis",
    description=(
        "Menghitung indikator analisis TEKNIKAL REAL-TIME (SMA, EMA, RSI, "
        "MACD, Bollinger Bands, support/resistance 3 bulan) dari data "
        "historis saham/indeks. WAJIB dipanggil setiap kali user meminta "
        "analisis teknikal atau sinyal beli/jual berbasis indikator — "
        "jangan pernah mengarang nilai indikator."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Format ticker sama seperti get_stock_price (contoh 'BBCA.JK', '^JKSE', 'AAPL').",
            }
        },
        "required": ["ticker"],
    },
)

FINANCE_TOOLS = types.Tool(
    function_declarations=[GET_STOCK_PRICE_DECLARATION, GET_TECHNICAL_ANALYSIS_DECLARATION]
)


async def _execute_get_stock_price(ticker: str) -> dict:
    logger.info(f"Menjalankan get_stock_price(ticker={ticker!r})...")
    data = await asyncio.to_thread(get_index_data, ticker)
    if data is None:
        return {"error": f"Data untuk ticker '{ticker}' tidak ditemukan atau gagal diambil."}
    return {
        "ticker": data.ticker,
        "name": data.name,
        "last_close": data.last_close,
        "previous_close": data.previous_close,
        "change_value": data.change_value,
        "change_percent": data.change_percent,
        "volume": data.volume,
    }


async def _execute_get_technical_analysis(ticker: str) -> dict:
    logger.info(f"Menjalankan get_technical_analysis(ticker={ticker!r})...")
    result = await asyncio.to_thread(get_technical_analysis, ticker)
    if result is None:
        return {"error": f"Data historis untuk '{ticker}' tidak cukup/gagal diambil untuk analisis teknikal."}
    return asdict(result)


async def _call_tool(function_name: str, function_args: dict) -> dict:
    """Dispatcher: memetakan nama fungsi yang diminta Gemini ke implementasinya."""
    if function_name == "get_stock_price":
        return await _execute_get_stock_price(**function_args)
    if function_name == "get_technical_analysis":
        return await _execute_get_technical_analysis(**function_args)
    logger.warning(f"Gemini meminta fungsi yang tidak dikenali: {function_name}")
    return {"error": f"Fungsi '{function_name}' tidak dikenali oleh sistem."}


# ============================================================
# AI CALLER
# ============================================================

async def get_gemini_response(
    user_message: str,
    gemini_client: genai.Client,
    history: List[types.Content],
    model_manager: ModelManager,
    preferred_model: str,
    media_parts: Optional[List[types.Part]] = None,
) -> str:
    """
    Mengirim pesan user (+ opsional media gambar/dokumen) & riwayat ke
    Gemini, dengan tool real-time dan model switching otomatis.
    """
    user_parts: List[types.Part] = [types.Part(text=user_message)]
    if media_parts:
        user_parts.extend(media_parts)

    user_content = types.Content(role="user", parts=user_parts)
    base_contents: List[types.Content] = history + [user_content]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        tools=[FINANCE_TOOLS],
    )

    current_model = model_manager.get_active_model(preferred=preferred_model)
    models_tried: List[str] = []

    while True:
        models_tried.append(current_model)
        contents = list(base_contents)

        try:
            for round_number in range(1, MAX_FUNCTION_CALL_ROUNDS + 1):
                response = await gemini_client.aio.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )

                candidate_content = response.candidates[0].content
                function_call_parts = [
                    part for part in candidate_content.parts if part.function_call is not None
                ]

                if not function_call_parts:
                    text = (response.text or "").strip()
                    if not text:
                        raise GeminiChatError("Gemini mengembalikan respons kosong.")
                    model_manager.record_request(current_model)
                    return text

                logger.info(
                    f"[{current_model}] Ronde {round_number}: Gemini meminta "
                    f"{[p.function_call.name for p in function_call_parts]}"
                )

                contents.append(candidate_content)
                response_parts = []
                for part in function_call_parts:
                    fc = part.function_call
                    result = await _call_tool(fc.name, dict(fc.args))
                    response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
                contents.append(types.Content(role="tool", parts=response_parts))

            model_manager.record_request(current_model)
            raise GeminiChatError(
                f"Model '{current_model}' masih meminta function call setelah {MAX_FUNCTION_CALL_ROUNDS} ronde."
            )

        except genai_errors.APIError as exc:
            is_rate_limited = getattr(exc, "code", None) == 429
            if is_rate_limited:
                model_manager.mark_rate_limited(current_model)
                next_model = model_manager.get_next_fallback(current_model)
                if next_model and next_model not in models_tried:
                    logger.warning(f"'{current_model}' kena rate limit, beralih ke '{next_model}'...")
                    current_model = next_model
                    continue
                raise GeminiChatError(
                    f"Seluruh model dalam fallback chain terkena rate limit ({', '.join(models_tried)})."
                ) from exc
            logger.error(f"Gemini API error ({current_model}): {exc}")
            raise GeminiChatError(f"Gemini API error: {exc}") from exc

        except GeminiChatError:
            raise
        except Exception as exc:
            logger.error(f"Kesalahan tak terduga saat memanggil Gemini ({current_model}): {exc}")
            raise GeminiChatError(f"Kesalahan tak terduga: {exc}") from exc


# ============================================================
# TELEGRAM HANDLERS — Teks
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "👋 Halo! Saya <b>AI Finance Assistant</b> Anda.\n\n"
        "Saya bisa:\n"
        "• Ambil harga & analisis teknikal <b>real-time</b> (SMA/RSI/MACD)\n"
        "• Menganalisis <b>screenshot chart</b> yang Anda kirim\n"
        "• Menganalisis <b>PDF laporan keuangan</b> yang Anda kirim\n"
        "• Mengingat konteks obrolan kita\n\n"
        "Perintah tersedia:\n"
        "• /model — pilih model Gemini yang dipakai\n"
        "• /reset — hapus riwayat percakapan\n"
        "• /model_status — lihat status rate-limit tiap model\n\n"
        "Contoh: kirim <i>screenshot chart BBCA</i>, atau ketik "
        "<i>\"analisis teknikal TLKM\"</i>."
    )
    await _reply_safe(update, welcome_message)
    logger.info(f"User {update.effective_user.id} memulai sesi via /start.")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history_manager: ChatHistoryManager = context.bot_data["history_manager"]
    history_manager.reset(chat_id)
    await update.message.reply_text("🔄 Riwayat percakapan telah dihapus. Mari mulai topik baru!")
    logger.info(f"Riwayat chat {chat_id} direset via /reset.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]
    current_model = preference_store.get(update.effective_user.id)

    keyboard_rows = []
    for i in range(0, len(AVAILABLE_MODELS), 2):
        row = []
        for option in AVAILABLE_MODELS[i:i + 2]:
            label = f"✅ {option.label}" if option.id == current_model else option.label
            row.append(InlineKeyboardButton(label, callback_data=f"set_model:{option.id}"))
        keyboard_rows.append(row)

    reply_markup = InlineKeyboardMarkup(keyboard_rows)

    await update.message.reply_text(
        f"🤖 <b>Pilih Model Gemini</b>\n\n"
        f"Model aktif Anda saat ini: <code>{current_model}</code>\n\n"
        f"Ketuk salah satu tombol di bawah untuk beralih model:",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML,
    )


async def model_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = update.effective_user.id

    try:
        if not query.data or not query.data.startswith("set_model:"):
            await query.answer("Data tombol tidak valid.", show_alert=True)
            return

        selected_id = query.data.split("set_model:", 1)[1]
        option = next((m for m in AVAILABLE_MODELS if m.id == selected_id), None)

        if option is None:
            await query.answer("Model tidak dikenali.", show_alert=True)
            return

        preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]
        preference_store.set(user_id, option.id)

        await query.answer(f"Model diubah ke {option.label}")
        await query.edit_message_text(
            f"✅ Model berhasil diubah menjadi: <b>{option.label}</b>\n<code>{option.id}</code>",
            parse_mode=ParseMode.HTML,
        )
        logger.info(f"User {user_id} mengganti preferensi model ke '{option.id}'.")

    except BadRequest as exc:
        logger.warning(f"Gagal mengedit pesan setelah callback /model (user {user_id}): {exc}")
    except Exception as exc:
        logger.error(f"Error tak terduga di model_button_callback (user {user_id}): {exc}")
        try:
            await query.answer("Terjadi kesalahan, coba lagi.", show_alert=True)
        except Exception:
            pass


async def model_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model_manager: ModelManager = context.bot_data["model_manager"]
    await _reply_safe(update, model_manager.get_status_text())


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    logger.info(f"Pesan masuk dari user {user_id}: {user_message[:80]}")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    gemini_client: genai.Client = context.bot_data["gemini_client"]
    history_manager: ChatHistoryManager = context.bot_data["history_manager"]
    model_manager: ModelManager = context.bot_data["model_manager"]
    preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]

    try:
        history = history_manager.get_history(chat_id)
        preferred_model = preference_store.get(user_id)

        reply_text = await get_gemini_response(
            user_message, gemini_client, history, model_manager, preferred_model
        )

        history_manager.add_exchange(chat_id, user_message, reply_text)
        await _reply_safe(update, reply_text + DISCLAIMER_FOOTER)
        logger.info(f"Balasan berhasil dikirim ke user {user_id}.")

    except GeminiChatError as exc:
        logger.error(f"Gagal mendapatkan respons Gemini untuk user {user_id}: {exc}")
        await update.message.reply_text(
            "⚠️ Maaf, sistem analisis AI sedang sibuk atau mengalami gangguan. "
            "Coba lagi sebentar, atau ketik /model untuk mencoba model lain."
        )
    except Exception as exc:
        logger.critical(f"Error tak terduga di handle_text_message (user {user_id}): {exc}")
        await update.message.reply_text("⚠️ Terjadi kesalahan tak terduga. Tim teknis akan segera memeriksanya.")


# ============================================================
# TELEGRAM HANDLERS — Media (Gambar & Dokumen)
# ============================================================

async def _process_media_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_part: types.Part,
    caption_text: str,
) -> None:
    """
    Logika BERSAMA untuk memproses pesan bermedia (foto/dokumen): panggil
    Gemini dengan media + teks, simpan RINGKASAN TEKS ke riwayat (media
    mentah TIDAK disimpan agar ChatHistoryManager tetap ringan), lalu
    kirim balasan. Dipisah dari handler agar tidak duplikasi logika
    antara handle_photo_message dan handle_document_message (DRY).
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    gemini_client: genai.Client = context.bot_data["gemini_client"]
    history_manager: ChatHistoryManager = context.bot_data["history_manager"]
    model_manager: ModelManager = context.bot_data["model_manager"]
    preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]

    try:
        history = history_manager.get_history(chat_id)
        preferred_model = preference_store.get(user_id)

        reply_text = await get_gemini_response(
            caption_text,
            gemini_client,
            history,
            model_manager,
            preferred_model,
            media_parts=[media_part],
        )

        history_manager.add_exchange(chat_id, f"[Mengirim file/gambar] {caption_text}", reply_text)
        await _reply_safe(update, reply_text + DISCLAIMER_FOOTER)
        logger.info(f"Analisis media berhasil dikirim ke user {user_id}.")

    except GeminiChatError as exc:
        logger.error(f"Gagal menganalisis media untuk user {user_id}: {exc}")
        await update.message.reply_text(
            "⚠️ Maaf, sistem gagal menganalisis file ini (mungkin sedang sibuk). Coba lagi sebentar."
        )
    except Exception as exc:
        logger.critical(f"Error tak terduga saat memproses media (user {user_id}): {exc}")
        await update.message.reply_text("⚠️ Terjadi kesalahan tak terduga saat memproses file.")


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler untuk PESAN BERISI FOTO (misal screenshot chart candlestick)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # Ambil resolusi TERTINGGI yang tersedia (elemen terakhir di list photo)
        photo = update.message.photo[-1]

        if photo.file_size and photo.file_size > MAX_MEDIA_FILE_BYTES:
            await update.message.reply_text("⚠️ Ukuran gambar terlalu besar (maks 15MB).")
            return

        telegram_file = await context.bot.get_file(photo.file_id)
        file_bytes = bytes(await telegram_file.download_as_bytearray())

        caption = update.message.caption or (
            "Tolong analisis chart/gambar ini dari sudut pandang teknikal. "
            "Jika ada nama saham/ticker yang terlihat atau disebutkan, sertakan analisisnya."
        )

        logger.info(f"Menerima foto dari user {user_id} ({len(file_bytes)} bytes).")

        await _process_media_and_reply(
            update=update,
            context=context,
            media_part=types.Part.from_bytes(data=file_bytes, mime_type="image/jpeg"),
            caption_text=caption,
        )

    except Exception as exc:
        logger.error(f"Gagal memproses foto dari user {user_id}: {exc}")
        await update.message.reply_text("⚠️ Gagal memproses gambar. Silakan coba lagi.")


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler untuk PESAN BERISI DOKUMEN (PDF laporan keuangan, atau gambar dikirim sebagai file)."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    document = update.message.document

    if document.mime_type not in ALLOWED_DOCUMENT_MIME_TYPES:
        await update.message.reply_text(
            "⚠️ Format file belum didukung. Saat ini hanya PDF, PNG, JPEG, dan WEBP."
        )
        return

    if document.file_size and document.file_size > MAX_MEDIA_FILE_BYTES:
        await update.message.reply_text("⚠️ Ukuran file terlalu besar (maks 15MB).")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        telegram_file = await context.bot.get_file(document.file_id)
        file_bytes = bytes(await telegram_file.download_as_bytearray())

        is_pdf = document.mime_type == "application/pdf"
        default_prompt = (
            "Tolong lakukan analisis fundamental dari laporan/dokumen ini. "
            "Ekstrak metrik keuangan penting (pendapatan, laba bersih, margin, "
            "rasio utang, pertumbuhan) jika tersedia di dokumen."
            if is_pdf else
            "Tolong analisis chart/gambar ini dari sudut pandang teknikal."
        )
        caption = update.message.caption or default_prompt

        logger.info(f"Menerima dokumen ({document.mime_type}) dari user {user_id} ({len(file_bytes)} bytes).")

        await _process_media_and_reply(
            update=update,
            context=context,
            media_part=types.Part.from_bytes(data=file_bytes, mime_type=document.mime_type),
            caption_text=caption,
        )

    except Exception as exc:
        logger.error(f"Gagal memproses dokumen dari user {user_id}: {exc}")
        await update.message.reply_text("⚠️ Gagal memproses dokumen. Silakan coba lagi.")


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    application.bot_data["model_manager"] = ModelManager(AVAILABLE_MODELS)
    application.bot_data["preference_store"] = UserModelPreferenceStore(DEFAULT_MODEL_ID)

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("model_status", model_status_command))
    application.add_handler(CallbackQueryHandler(model_button_callback, pattern="^set_model:"))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler_error_handler = None  # no-op, dihapus jika ada versi lama
    application.add_error_handler(global_error_handler)

    logger.info(
        f"Bot mulai berjalan. {len(AVAILABLE_MODELS)} model, "
        f"tools: get_stock_price, get_technical_analysis, media analysis aktif."
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()