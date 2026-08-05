# news_fetcher.py

"""
Modul untuk menarik berita keuangan terbaru dari sumber RSS gratis
(Indonesia & global) menggunakan feedparser.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from time import mktime
from typing import List, Optional

import feedparser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Daftar sumber RSS. Menambah/mengganti sumber cukup ubah list ini
# (Open/Closed Principle) — tidak perlu menyentuh logika fetch-nya.
#
# CATATAN PENTING: URL RSS pihak ketiga dapat berubah sewaktu-waktu tanpa
# pemberitahuan. Jika suatu saat sebuah sumber berhenti mengembalikan
# data, jalankan `python news_fetcher.py` langsung untuk mendiagnosis
# sumber mana yang bermasalah, lalu ganti URL-nya di sini.
NEWS_SOURCES = [
    {
        "name": "CNBC Indonesia - Market",
        "url": "https://www.cnbcindonesia.com/market/rss",
        "category": "Indonesia",
    },
    {
        "name": "Bisnis.com - Ekonomi Bisnis",
        "url": "https://www.bisnis.com/rss",
        "category": "Indonesia",
    },
    {
        "name": "Investing.com - Financial News",
        "url": "https://www.investing.com/rss/news.rss",
        "category": "Global",
    },
    {
        "name": "Yahoo Finance - Top News",
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "Global",
    },
]


@dataclass
class NewsItem:
    """Representasi satu berita hasil parsing RSS."""
    title: str
    link: str
    source: str
    category: str
    published: Optional[datetime]

    @property
    def published_display(self) -> str:
        if self.published:
            return self.published.strftime("%d %b %Y %H:%M")
        return "Waktu tidak diketahui"


def fetch_news_from_source(source: dict, limit: int = 5) -> List[NewsItem]:
    """
    Mengambil dan mem-parsing berita dari satu sumber RSS.

    Kegagalan pada satu sumber (URL mati, timeout, format rusak) tidak
    boleh menghentikan seluruh proses — dicatat sebagai warning dan
    mengembalikan list kosong, agar sumber lain tetap bisa diproses.
    """
    items: List[NewsItem] = []

    try:
        feed = feedparser.parse(source["url"])

        if feed.bozo and not feed.entries:
            logger.warning(
                f"Sumber '{source['name']}' gagal di-parse atau tidak "
                f"mengembalikan entri (bozo_exception: {feed.get('bozo_exception')})."
            )
            return items

        for entry in feed.entries[:limit]:
            published_dt = None
            if getattr(entry, "published_parsed", None):
                published_dt = datetime.fromtimestamp(mktime(entry.published_parsed))

            items.append(
                NewsItem(
                    title=entry.get("title", "Tanpa judul").strip(),
                    link=entry.get("link", ""),
                    source=source["name"],
                    category=source["category"],
                    published=published_dt,
                )
            )

        logger.info(f"Berhasil mengambil {len(items)} berita dari '{source['name']}'.")

    except Exception as exc:
        logger.error(f"Gagal mengambil berita dari '{source['name']}': {exc}")

    return items


def fetch_all_news(limit_per_source: int = 5) -> List[NewsItem]:
    """
    Mengambil berita dari seluruh sumber di NEWS_SOURCES, digabung menjadi
    satu list dan diurutkan dari yang terbaru (jika tanggal tersedia).
    """
    all_items: List[NewsItem] = []

    for source in NEWS_SOURCES:
        all_items.extend(fetch_news_from_source(source, limit=limit_per_source))

    all_items.sort(key=lambda item: item.published or datetime.min, reverse=True)

    if not all_items:
        logger.error("Tidak ada berita yang berhasil diambil dari sumber manapun.")

    return all_items


def format_news_summary(news_items: List[NewsItem]) -> str:
    """
    Memformat list NewsItem menjadi teks ringkas terstruktur per kategori,
    siap dipakai sebagai konteks prompt Gemini AI.
    """
    if not news_items:
        return "Tidak ada berita terbaru yang tersedia saat ini."

    grouped: dict = {"Indonesia": [], "Global": []}
    for item in news_items:
        grouped.setdefault(item.category, []).append(item)

    lines = []
    for category, items in grouped.items():
        if not items:
            continue
        lines.append(f"\n--- Berita {category} ---")
        for item in items:
            lines.append(f"- [{item.source}] {item.title}")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    # Self-test: jalankan `python news_fetcher.py` untuk memverifikasi
    # seluruh sumber RSS masih aktif dan dapat di-parse dengan benar.
    print("Mengambil berita dari seluruh sumber RSS...\n")
    news = fetch_all_news(limit_per_source=3)

    if news:
        print(format_news_summary(news))
        print(f"\nTotal berita berhasil diambil: {len(news)}")
    else:
        print("❌ Gagal mengambil berita dari semua sumber. Periksa NEWS_SOURCES.")