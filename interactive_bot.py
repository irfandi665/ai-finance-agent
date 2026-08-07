# interactive_bot.py

"""
Bot Telegram interaktif untuk AI Finance Department Agent, dengan:
1. MEMORI PERCAKAPAN per chat_id
2. AKSES DATA PASAR REAL-TIME via Function Calling manual
3. DYNAMIC MODEL SWITCHER — perintah /model menampilkan 8 pilihan model
   Gemini via InlineKeyboardMarkup, preferensi disimpan per user_id
4. RATE-LIMIT FALLBACK OTOMATIS — reaktif (429) & proaktif (tracking lokal)
5. PROFESSIONAL HTML FORMATTING — output Gemini diformat sebagai laporan
   rapi (heading, bullet, bold, divider) via ParseMode.HTML, dengan
   sanitasi tag untuk mencegah crash saat parsing di sisi Telegram.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
GEMINI_MAX_OUTPUT_TOKENS = 1024
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_HISTORY_TURNS = 10
MAX_FUNCTION_CALL_ROUNDS = 3
RATE_LIMIT_COOLDOWN_SECONDS = 60


@dataclass
class ModelOption:
    """
    Definisi satu model Gemini: id resmi, label tampilan, dan limit lokal
    (rpm/rpd). SATU sumber kebenaran dipakai baik untuk tombol /model
    maupun tracking rate-limit ModelManager — mencegah duplikasi data
    yang bisa tidak sinkron.
    """
    id: str
    label: str
    rpm_limit: int
    rpd_limit: int


# ⚠️ PENTING: daftar & angka limit di bawah adalah PERKIRAAN per Agustus
# 2026 berdasarkan dokumentasi publik Google. VERIFIKASI ULANG ketersediaan
# model di https://ai.google.dev/gemini-api/docs/models dan limit RPM/RPD
# akun Anda di https://aistudio.google.com/app/apikey — keduanya bisa
# berubah tanpa pemberitahuan. Model *-pro mungkin butuh billing aktif.
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
- Anda punya akses ke fungsi get_stock_price untuk data harga TERKINI dari Yahoo Finance.
- WAJIB panggil fungsi ini setiap kali klien menanyakan harga/performa suatu saham/indeks — JANGAN pernah menjawab dari ingatan/asumsi.
- Jika fungsi error, sampaikan jujur, jangan mengarang angka.

FORMAT OUTPUT (SANGAT PENTING — WAJIB DIIKUTI):
Gunakan HANYA tag HTML berikut yang didukung Telegram: <b>teks</b> (bold — untuk angka penting, harga, persentase, dan sub-judul), <i>teks</i> (italic — untuk catatan kecil), <u>teks</u> (underline — sesekali untuk penekanan), <code>teks</code> (untuk kode ticker, misal <code>BBCA.JK</code>).
JANGAN PERNAH memakai tag lain seperti <div>, <table>, <h1>, <ul>, <li>, <p>, <br> — Telegram TIDAK mendukungnya dan pesan akan gagal terkirim.

Untuk pertanyaan analitis (bukan basa-basi singkat), susun jawaban seperti laporan profesional:

📊 <b>[Judul Singkat Topik]</b>
──────────────────
[paragraf ringkasan 1-2 kalimat]

<b>Poin Kunci:</b>
- [poin 1, angka penting dalam <b>bold</b>]
- [poin 2]
- [poin 3, maksimal 4 poin]
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
    """
    Melacak penggunaan (RPM/RPD) dan status rate-limit tiap model Gemini
    secara GLOBAL, serta fallback reaktif saat API mengembalikan 429
    sungguhan. Terpisah dari UserModelPreferenceStore (di bawah) yang
    menyimpan PILIHAN model tiap user — Single Responsibility: satu
    kelas urus rate-limit, satu kelas urus preferensi personal.
    """

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
        """
        1. Jika user punya preferensi DAN model itu masih ada headroom lokal, pakai itu.
        2. Jika tidak, pilih model pertama di AVAILABLE_MODELS yang masih ada headroom.
        3. Jika semua kehabisan headroom, tetap coba model pertama sebagai usaha terakhir.
        """
        if preferred:
            option = self._get_option(preferred)
            if option and self._has_headroom(option):
                return option.id

        for option in self._models:
            if self._has_headroom(option):
                return option.id

        return self._models[0].id

    def get_next_fallback(self, current_model: str) -> Optional[str]:
        """Model SETELAH current_model di AVAILABLE_MODELS, atau None jika sudah yang terakhir."""
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
            "\n<i>Catatan: angka limit adalah perkiraan lokal, bisa diubah "
            "di kode. Cek limit sebenarnya di aistudio.google.com/app/apikey.</i>"
        )
        return "\n".join(lines)


# ============================================================
# STATE MANAGEMENT: Preferensi Model Per-User (untuk /model)
# ============================================================

class UserModelPreferenceStore:
    """
    Menyimpan preferensi model Gemini PER USER (state management
    in-memory sederhana, di-keyed oleh user_id Telegram). Sama seperti
    ChatHistoryManager, state ini RESET saat proses bot di-restart.
    """

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
# HTML SANITIZATION (pertahanan anti-crash Telegram)
# ============================================================

_ALLOWED_HTML_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}
_HTML_TAG_PATTERN = re.compile(r"</?([a-zA-Z0-9]+)(\s+[^>]*)?>")


def _sanitize_html_for_telegram(text: str) -> str:
    """
    LAPIS PERTAHANAN PERTAMA: meski system prompt sudah menginstruksikan
    Gemini untuk HANYA memakai tag aman, LLM sesekali tetap bisa "lupa"
    dan menghasilkan tag lain (misal <h1>, <li>, <div>). Fungsi ini
    membuang tag APA PUN di luar whitelist, TANPA menghapus teks di
    dalamnya — hanya tag pembungkusnya yang dibuang.
    """
    def _strip_disallowed(match) -> str:
        tag_name = match.group(1).lower()
        return match.group(0) if tag_name in _ALLOWED_HTML_TAGS else ""

    return _HTML_TAG_PATTERN.sub(_strip_disallowed, text)


def _strip_all_html_tags(text: str) -> str:
    """LAPIS PERTAHANAN KEDUA (fallback terakhir): buang SEMUA tag HTML — dipakai saat Telegram tetap menolak parsing meski sudah disanitasi (misal tag tidak tertutup dengan benar)."""
    return _HTML_TAG_PATTERN.sub("", text)


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
    """
    Mengirim balasan dengan tiga lapis perlindungan: (1) sanitasi tag
    HTML tak dikenal, (2) split otomatis jika >4096 karakter, (3)
    fallback ke teks polos jika Telegram TETAP menolak parsing.
    """
    sanitized = _sanitize_html_for_telegram(text)
    chunks = _split_text(sanitized)

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except BadRequest as exc:
            logger.warning(f"Gagal kirim dengan HTML, fallback ke teks polos: {exc}")
            await update.message.reply_text(_strip_all_html_tags(chunk))


# ============================================================
# TOOL: get_stock_price (data pasar real-time)
# ============================================================

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
                    "tambahkan akhiran '.JK' (contoh: 'BBCA.JK'). Untuk "
                    "indeks, gunakan awalan '^' (contoh: '^JKSE' untuk IHSG). "
                    "Untuk saham Amerika, gunakan kode biasa (contoh: 'AAPL')."
                ),
            }
        },
        "required": ["ticker"],
    },
)

MARKET_DATA_TOOL = types.Tool(function_declarations=[GET_STOCK_PRICE_DECLARATION])


async def _execute_get_stock_price(ticker: str) -> dict:
    """Wrapper async untuk get_index_data() — dijalankan via asyncio.to_thread() agar tidak memblokir event loop bot."""
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


async def _call_tool(function_name: str, function_args: dict) -> dict:
    """Dispatcher: memetakan nama fungsi yang diminta Gemini ke implementasinya."""
    if function_name == "get_stock_price":
        return await _execute_get_stock_price(**function_args)
    logger.warning(f"Gemini meminta fungsi yang tidak dikenali: {function_name}")
    return {"error": f"Fungsi '{function_name}' tidak dikenali oleh sistem."}


# ============================================================
# AI CALLER: pemanggilan Gemini (terpisah dari handler Telegram)
# ============================================================

async def get_gemini_response(
    user_message: str,
    gemini_client: genai.Client,
    history: List[types.Content],
    model_manager: ModelManager,
    preferred_model: str,
) -> str:
    """
    Mengirim pesan user + riwayat ke Gemini, dengan tool real-time DAN
    model switching otomatis. Dimulai dari preferred_model (pilihan user
    via /model); jika model itu (atau model fallback berikutnya) kena
    429, otomatis lanjut ke model berikutnya di AVAILABLE_MODELS —
    TANPA mengubah preferensi tersimpan user (fallback ini hanya berlaku
    untuk satu kali exchange ini).
    """
    user_content = types.Content(role="user", parts=[types.Part(text=user_message)])
    base_contents: List[types.Content] = history + [user_content]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        tools=[MARKET_DATA_TOOL],
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
# TELEGRAM HANDLERS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "👋 Halo! Saya <b>AI Finance Assistant</b> Anda.\n\n"
        "Saya bisa mengambil harga saham/indeks <b>real-time</b>, mengingat "
        "konteks obrolan, dan Anda bisa memilih model Gemini sendiri.\n\n"
        "Perintah tersedia:\n"
        "• /model — pilih model Gemini yang dipakai\n"
        "• /reset — hapus riwayat percakapan\n"
        "• /model_status — lihat status rate-limit tiap model\n\n"
        "Contoh pertanyaan:\n"
        "• <i>Berapa harga BBCA sekarang?</i>\n"
        "• <i>Apa dampak The Fed naikkan suku bunga?</i>"
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
    """
    Handler /model — menampilkan 8 pilihan model Gemini sebagai
    INLINE KEYBOARD (tombol interaktif). Klik tombol ditangani oleh
    model_button_callback() via CallbackQueryHandler di bawah.
    """
    preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]
    current_model = preference_store.get(update.effective_user.id)

    # --- Penyusunan Inline Keyboard: 2 kolom x 4 baris ---
    # Setiap tombol membawa callback_data unik "set_model:<id>" agar
    # handler bisa tahu persis model mana yang diklik. Model yang
    # sedang aktif untuk user ini ditandai ✅ di labelnya.
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
    """
    Handler CALLBACK QUERY — menangkap klik tombol dari /model.
    Alur: validasi data tombol -> update UserModelPreferenceStore ->
    kirim notifikasi pop-up (answerCallbackQuery) -> edit pesan asal
    menjadi konfirmasi teks.
    """
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

        # --- STATE UPDATE: simpan preferensi model baru untuk user ini ---
        preference_store: UserModelPreferenceStore = context.bot_data["preference_store"]
        preference_store.set(user_id, option.id)

        # Pop-up notifikasi kecil di UI Telegram (bukan pesan chat baru)
        await query.answer(f"Model diubah ke {option.label}")

        # Edit pesan asal (yang berisi tombol) menjadi teks konfirmasi
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
    """Handler utama untuk pesan teks — memori, data real-time, dan model sesuai preferensi user."""
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
            "⚠️ Maaf, sistem analisis AI sedang sibuk atau mengalami gangguan "
            "sementara. Coba lagi sebentar, atau ketik /model untuk mencoba "
            "model lain, atau /model_status untuk cek kondisinya."
        )
    except Exception as exc:
        logger.critical(f"Error tak terduga di handle_text_message (user {user_id}): {exc}")
        await update.message.reply_text(
            "⚠️ Terjadi kesalahan tak terduga di sistem kami. Tim teknis akan segera memeriksanya."
        )


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

    # --- Inisialisasi seluruh state management di bot_data ---
    application.bot_data["gemini_client"] = gemini_client
    application.bot_data["history_manager"] = ChatHistoryManager(max_turns=MAX_HISTORY_TURNS)
    application.bot_data["model_manager"] = ModelManager(AVAILABLE_MODELS)
    application.bot_data["preference_store"] = UserModelPreferenceStore(DEFAULT_MODEL_ID)

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("model_status", model_status_command))
    # Pattern regex membatasi handler ini HANYA menangkap callback dari
    # tombol /model kita (callback_data diawali "set_model:") — mencegah
    # bentrok jika suatu saat ada CallbackQueryHandler lain di masa depan.
    application.add_handler(CallbackQueryHandler(model_button_callback, pattern="^set_model:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(global_error_handler)

    logger.info(f"Bot mulai berjalan. {len(AVAILABLE_MODELS)} model tersedia via /model.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()