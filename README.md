# 🤖 AI Finance Department Agent

Bot & agen otomatis untuk **analisis pasar keuangan harian** yang menggabungkan **data pasar real-time (Yahoo Finance)**, **berita terbaru (RSS Indonesia & Global)**, dan **analisis AI (Google Gemini)** menjadi laporan rekomendasi **Buy / Sell / Hold** yang terstruktur — lalu mengirimkannya langsung ke **Telegram**.

Dirancang dengan prinsip **fault isolation**: kegagalan pada satu tahap tidak akan menggagalkan seluruh proses.

---

## ✨ Fitur Utama

### 📈 Laporan Harian Terjadwal (main.py)
- **Data Pasar Real-time** — mengambil harga indeks (IHSG, Dow Jones, S&P 500, Nasdaq) & saham individual (BBCA, BBRI, TLKM, AAPL) via `yfinance`.
- **Berita Keuangan** — mengumpulkan berita terbaru dari sumber RSS Indonesia (CNBC Indonesia, Bisnis.com) & Global (Investing.com, Yahoo Finance).
- **Analisis AI Terstruktur** — mengirim data ke Gemini AI dengan **structured output** (Pydantic schema) + **retry otomatis** (exponential backoff) untuk menangani rate limit.
- **Riwayat SQLite** — menyimpan laporan harian (tanpa dependency eksternal) untuk pelacakan akurasi rekomendasi dari waktu ke waktu, dengan ekspor ke CSV.
- **Notifikasi Kegagalan** — mengirim peringatan otomatis ke Telegram jika salah satu tahap gagal.

### 🗣️ Bot Telegram Interaktif (interactive_bot.py)
- **Memori percakapan** per `chat_id` (in-memory, max 10 turn).
- **Data pasar real-time** melalui *Function Calling* (manual) ke Yahoo Finance.
- **Dynamic Model Switcher** — perintah `/model` menampilkan 8 pilihan model Gemini via tombol inline, preferensi disimpan per user.
- **Rate-limit fallback otomatis** — reaktif (HTTP 429) & proaktif (tracking RPM/RPD lokal).
- **HTML formatting aman** — output Gemini diformat rapi dengan 3 lapis perlindungan anti-crash (sanitasi tag, split pesan, fallback teks polos).

### 💚 Heartbeat Mingguan (heartbeat.py)
- Mengirim notifikasi status kesehatan agen setiap minggu untuk mendeteksi GitHub Actions yang otomatis nonaktif setelah 60 hari tanpa aktivitas.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| **Bahasa** | Python 3.11+ |
| **Data Pasar** | `yfinance` |
| **Berita** | `feedparser` (RSS) |
| **AI/LLM** | Google GenAI SDK (`google-genai`) — Gemini |
| **Bot Telegram** | `python-telegram-bot` v20+ (async) |
| **Validasi Data** | `pydantic` v2 (structured output) |
| **Retry** | `tenacity` |
| **Penjadwalan** | `schedule` |
| **Database** | SQLite (`sqlite3` built-in) |
| **CI/CD** | GitHub Actions |

---

## 📁 Struktur Direktori

```
ai-finance-agent/
├── main.py               # Orkestrator utama laporan harian
├── config.py             # Konfigurasi & validasi credential (.env)
├── market_data.py        # Pengambilan data pasar (yfinance)
├── news_fetcher.py       # Pengambilan berita (RSS)
├── gemini_analyzer.py    # Analisis Gemini (structured output + retry)
├── telegram_sender.py    # Pengiriman pesan ke Telegram
├── report_history.py     # Riwayat laporan (SQLite + CSV)
├── heartbeat.py          # Heartbeat status mingguan
├── scheduler.py          # Penjadwalan harian & mingguan
├── interactive_bot.py    # Bot Telegram interaktif
├── requirements.txt      # Dependensi Python
├── .env                  # Credential (RAHASIA — jangan commit)
├── .gitignore
└── .github/workflows/
    └── weekly-heartbeat.yml  # GitHub Actions heartbeat mingguan
```

---

## 🚀 Instalasi

### Prasyarat
- Python 3.11+
- `pip` / `venv`

### Langkah

1. **Clone repository:**
   ```bash
   git clone <url-repo-anda> ai-finance-agent
   cd ai-finance-agent
   ```

2. **Buat & aktifkan virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate  # Linux / macOS
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan file `.env`** (isi sesuai credential Anda):
   ```env
   # Telegram Bot — dapatkan dari @BotFather
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234_ghIkl...
   # Chat ID Telegram tujuan (bisa user ID atau group ID)
   TELEGRAM_CHAT_ID=123456789

   # Google Gemini API — gratis di https://aistudio.google.com/app/apikey
   GEMINI_API_KEY=AIza...
   ```

---

## ⚙️ Cara Penggunaan

### 1. Laporan Harian (sekali jalan)
```bash
python main.py
```

### 2. Scheduler Otomatis (laporan harian 07:00 + heartbeat Senin 08:00)
```bash
python scheduler.py
```

### 3. Bot Telegram Interaktif
```bash
python interactive_bot.py
```

### 4. Self-test per Modul
Setiap modul dapat diuji mandiri:
```bash
python market_data.py      # uji pengambilan data pasar
python news_fetcher.py     # verifikasi sumber RSS aktif
python telegram_sender.py  # uji koneksi ke bot Telegram
python report_history.py   # lihat riwayat tersimpan
```

---

## 📱 Perintah Bot Telegram

| Perintah | Fungsi |
|---|---|
| `/start` | Memulai sesi & menampilkan daftar perintah |
| `/model` | Memilih model Gemini (8 pilihan via tombol inline) |
| `/model_status` | Melihat status rate-limit tiap model |
| `/reset` | Menghapus riwayat percakapan |

**Contoh pertanyaan:**
- *"Berapa harga BBCA sekarang?"* — bot otomatis mengambil harga real-time via function calling.
- *"Apa dampak The Fed menaikkan suku bunga?"*

---

## 🧠 Arsitektur Alur Laporan Harian

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Pasar (WAJIB) ── yfinance → indeks & saham          │
│ 2. Berita (OPSIONAL) ── RSS → berita Indonesia & Global     │
│ 3. Analisis Gemini ── structured output + retry otomatis    │
│ 4. Simpan Riwayat ── SQLite (best-effort, tidak memblokir)  │
│ 5. Kirim ke Telegram ── format Markdown + auto-split        │
└─────────────────────────────────────────────────────────────┘
```

**Prinsip desain penting:**
- Setiap tahap diisolasi dalam `try/except` masing-masing.
- Kegagalan di satu tahap **TIDAK** menggagalkan seluruh proses.
- Tahap berita & penyimpanan riwayat bersifat *optional* / *best-effort*.

---

## 🐳 Deployment Otomatis (GitHub Actions)

Workflow `weekly-heartbeat.yml` berjalan setiap **Senin 08:00 WIB** untuk mengirim heartbeat. Siapkan **Repository Secrets** berikut di GitHub:

| Secret | Deskripsi |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram |
| `TELEGRAM_CHAT_ID` | Chat ID tujuan |

> Anda dapat menambahkan workflow tambahan untuk menjalankan `main.py` secara berkala dengan cron yang berbeda.

---

## 🗂️ Konfigurasi Kustom

### Menambah Instrumen yang Dipantau
Edit `MONITORED_INSTRUMENTS` di `market_data.py`:
```python
MONITORED_INSTRUMENTS = {
    "Indeks": { "^JKSE": "IHSG", ... },
    "Saham": { "BBCA.JK": "Bank Central Asia", "TLKM.JK": "Telkom", ... },
}
```
> Cukup tambahkan baris — tidak perlu menyentuh logika pengambilan data (Open/Closed Principle).

### Mengubah Sumber Berita
Edit `NEWS_SOURCES` di `news_fetcher.py` — tambah/ganti URL RSS sesuai kebutuhan.

### Mengubah Model / Parameter Gemini
Edit `gemini_analyzer.py`:
- `GEMINI_MODEL_NAME` — pilih model Gemini yang aktif.
- `GEMINI_MAX_OUTPUT_TOKENS` — naikkan jika output sering terpotong (MAX_TOKENS).
- `GEMINI_THINKING_BUDGET` — set 0 untuk output lebih cepat & murah.

---

## ⚠️ Disclaimer

Laporan dan rekomendasi dihasilkan **secara otomatis oleh AI** berdasarkan data pasar & berita publik. Ini **bukan nasihat finansial resmi**. Selalu lakukan riset mandiri (DYOR) dan konsultasikan dengan penasihat keuangan berlisensi sebelum mengambil keputusan investasi.

---

## 📄 Lisensi

Proyek ini didistribusikan untuk penggunaan pribadi. Silakan hubungi pemilik repository untuk informasi lisensi lebih lanjut.
