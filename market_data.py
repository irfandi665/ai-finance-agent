# market_data.py

"""
Modul untuk menarik data pasar (indeks & saham individual) menggunakan
yfinance. Bertanggung jawab tunggal: mengambil & memformat data harga.
Tidak menangani analisis atau pengiriman pesan (Single Responsibility).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Daftar instrumen yang dipantau, dikelompokkan per kategori. Menambah
# saham baru cukup tambahkan baris di kategori "Saham" — tidak perlu
# menyentuh logika pengambilan data sama sekali (Open/Closed Principle).
MONITORED_INSTRUMENTS = {
    "Indeks": {
        "^JKSE": "IHSG (Indonesia)",
        "^DJI": "Dow Jones (AS)",
        "^GSPC": "S&P 500 (AS)",
        "^IXIC": "Nasdaq Composite (AS)",
    },
    "Saham": {
        "BBCA.JK": "Bank Central Asia",
        "BBRI.JK": "Bank Rakyat Indonesia",
        "TLKM.JK": "Telkom Indonesia",
        "AAPL": "Apple Inc.",
    },
}


@dataclass
class IndexData:
    """Representasi data satu instrumen (indeks atau saham) pada satu titik waktu."""
    ticker: str
    name: str
    category: str
    last_close: float
    previous_close: float
    change_value: float
    change_percent: float
    volume: int

    @property
    def trend_emoji(self) -> str:
        if self.change_percent > 0:
            return "🟢"
        elif self.change_percent < 0:
            return "🔴"
        return "⚪"


def get_index_data(
    ticker: str, name: Optional[str] = None, category: str = "Indeks"
) -> Optional[IndexData]:
    """
    Mengambil harga penutupan terakhir, penutupan sebelumnya, perubahan
    (nilai & persen), dan volume untuk satu ticker (indeks maupun saham
    individual — yfinance memperlakukan keduanya dengan API yang sama).
    """
    display_name = name or ticker

    try:
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="5d", interval="1d")

        if history.empty or len(history) < 2:
            logger.warning(
                f"Data historis untuk {ticker} ({display_name}) tidak cukup "
                f"untuk menghitung perubahan. Melewati ticker ini."
            )
            return None

        last_row = history.iloc[-1]
        previous_row = history.iloc[-2]

        last_close = float(last_row["Close"])
        previous_close = float(previous_row["Close"])
        volume = int(last_row["Volume"])

        change_value = last_close - previous_close
        change_percent = (
            (change_value / previous_close) * 100 if previous_close != 0 else 0.0
        )

        return IndexData(
            ticker=ticker,
            name=display_name,
            category=category,
            last_close=round(last_close, 2),
            previous_close=round(previous_close, 2),
            change_value=round(change_value, 2),
            change_percent=round(change_percent, 2),
            volume=volume,
        )

    except Exception as exc:
        logger.error(f"Gagal mengambil data untuk {ticker} ({display_name}): {exc}")
        return None


def get_all_market_data() -> List[IndexData]:
    """
    Mengambil data seluruh instrumen di MONITORED_INSTRUMENTS (indeks +
    saham). Kegagalan pada satu ticker tidak menghentikan pengambilan
    ticker lain (fault isolation) — penting karena semakin banyak
    instrumen dipantau, semakin besar kemungkinan satu-dua di antaranya
    gagal diambil pada hari tertentu.

    Returns:
        List[IndexData] — hanya berisi ticker yang berhasil diambil.
    """
    results: List[IndexData] = []

    for category, tickers in MONITORED_INSTRUMENTS.items():
        for ticker, name in tickers.items():
            data = get_index_data(ticker, name, category=category)
            if data is not None:
                results.append(data)
            else:
                logger.warning(f"{name} ({ticker}) dilewati karena data tidak tersedia.")

    if not results:
        logger.error("Tidak ada data pasar yang berhasil diambil sama sekali.")

    return results


def format_market_summary(index_list: List[IndexData]) -> str:
    """
    Memformat list IndexData menjadi teks ringkas, dikelompokkan per
    kategori (Indeks/Saham), siap dipakai sebagai konteks prompt Gemini
    AI maupun tampilan di Telegram.
    """
    if not index_list:
        return "Data pasar tidak tersedia saat ini."

    grouped: dict = {}
    for data in index_list:
        grouped.setdefault(data.category, []).append(data)

    lines = []
    for category, items in grouped.items():
        lines.append(f"\n--- {category} ---")
        for data in items:
            sign_value = "+" if data.change_value >= 0 else ""
            sign_pct = "+" if data.change_percent >= 0 else ""
            lines.append(
                f"{data.trend_emoji} {data.name} ({data.ticker}): "
                f"{data.last_close:,.2f} "
                f"({sign_value}{data.change_value:,.2f} / {sign_pct}{data.change_percent:.2f}%) "
                f"| Volume: {data.volume:,}"
            )

    return "\n".join(lines).strip()


if __name__ == "__main__":
    print("Mengambil data pasar (indeks & saham)...\n")
    market_data = get_all_market_data()

    if market_data:
        print(format_market_summary(market_data))
    else:
        print("❌ Gagal mengambil data pasar. Periksa koneksi internet Anda.")