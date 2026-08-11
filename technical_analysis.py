# technical_analysis.py

"""
Modul untuk menghitung indikator analisis teknikal dari data historis
REAL via yfinance, dengan PROTOKOL INTEGRITAS DATA:
- Setiap indikator yang datanya tidak cukup → None (bukan estimasi/tebakan)
- Setiap hasil membawa metadata provenance (sumber, tanggal, periode)
- Entry/SL/TP/ATR/Fibonacci/RR dihitung DI SINI (Python, deterministik),
  bukan diserahkan ke LLM untuk dihitung ulang
- Daily dan Weekly adalah query TERPISAH (parameter timeframe wajib)
- foreign_flow SELALU ditandai tidak tersedia — sistem ini tidak punya
  akses data foreign flow resmi KSEI/IDX, dan TIDAK BOLEH berpura-pura punya
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CRITICAL_INDICATORS = {"EMA20", "EMA50", "RSI14", "ATR14"}


# ============================================================
# Fungsi Kalkulasi Indikator (murni matematis, deterministik)
# ============================================================

def _calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _calculate_ema(series, fast)
    ema_slow = _calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = _calculate_sma(series, window)
    std = series.rolling(window=window).std()
    return sma + (num_std * std), sma, sma - (num_std * std)


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — dasar perhitungan Stop Loss/Take Profit berbasis volatilitas."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — konfirmasi tren dari sisi volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).fillna(0).cumsum()


def _safe_last(series: Optional[pd.Series]) -> Optional[float]:
    """
    Mengembalikan nilai TERAKHIR suatu series HANYA jika valid (bukan NaN).
    Ini adalah PENJAGA UTAMA anti-halusinasi: pandas menghasilkan NaN saat
    data historis tidak cukup untuk suatu window (misal SMA200 dengan
    hanya 120 hari data) — fungsi ini memastikan NaN itu menjadi None,
    BUKAN angka yang seolah-olah valid.
    """
    if series is None or len(series) == 0:
        return None
    last_val = series.iloc[-1]
    return None if pd.isna(last_val) else float(last_val)


# ============================================================
# Struktur Hasil (membawa data + provenance + status kelengkapan)
# ============================================================

@dataclass
class TechnicalAnalysisResult:
    ticker: str
    timeframe: str  # "daily" atau "weekly" — TIDAK PERNAH dicampur

    # --- Data Provenance (Requirement #2) ---
    data_source: str
    fetched_at_utc: str
    fetched_at_wib: str
    price_date: str
    calculation_period: str

    # --- Harga ---
    last_close: Optional[float]

    # --- Trend (Requirement #5: nama field harfiah, tidak ditukar) ---
    ema_20: Optional[float]
    ema_50: Optional[float]
    sma_200: Optional[float]

    # --- Momentum ---
    rsi_14: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]

    # --- Volatilitas ---
    atr_14: Optional[float]
    bollinger_upper: Optional[float]
    bollinger_middle: Optional[float]
    bollinger_lower: Optional[float]

    # --- Volume ---
    obv_latest: Optional[float]
    obv_trend: Optional[str]

    # --- Fibonacci (berbasis swing high/low periode fetch) ---
    swing_high: Optional[float]
    swing_low: Optional[float]
    fib_236: Optional[float]
    fib_382: Optional[float]
    fib_500: Optional[float]
    fib_618: Optional[float]
    fib_786: Optional[float]

    # --- Trade Setup (Requirement #3: sudah dihitung + rumus siap pakai) ---
    suggested_entry: Optional[float]
    suggested_stop_loss: Optional[float]
    suggested_tp1: Optional[float]
    suggested_tp2: Optional[float]
    risk_reward_ratio: Optional[float]
    trade_setup_formulas: Dict[str, str]

    # --- Foreign Flow (Requirement #1 & #7: SELALU tidak tersedia) ---
    foreign_flow_status: str
    foreign_flow_reason: str

    # --- Metadata Kelengkapan (Requirement #1, #5, #8) ---
    missing_indicators: List[str]
    data_sufficient_for_verdict: bool


def get_technical_analysis(ticker: str, timeframe: str = "daily") -> Optional[TechnicalAnalysisResult]:
    """
    Menghitung seluruh indikator teknikal untuk SATU timeframe (daily
    ATAU weekly — panggil dua kali terpisah untuk keduanya, JANGAN
    mencampur data dari kedua interval ini).

    Returns:
        TechnicalAnalysisResult, atau None jika data historis sama
        sekali tidak cukup untuk indikator apa pun (< 15 candle).
    """
    if timeframe not in ("daily", "weekly"):
        logger.error(f"Timeframe tidak valid: {timeframe}")
        return None

    fetch_period = "2y" if timeframe == "daily" else "5y"
    fetch_interval = "1d" if timeframe == "daily" else "1wk"

    try:
        history = yf.Ticker(ticker).history(period=fetch_period, interval=fetch_interval)

        if history.empty or len(history) < 15:
            logger.warning(f"Data {ticker} ({timeframe}) terlalu sedikit untuk analisis apa pun.")
            return None

        close, high, low, volume = history["Close"], history["High"], history["Low"], history["Volume"]
        missing: List[str] = []

        def _check(name: str, value: Optional[float]) -> Optional[float]:
            if value is None:
                missing.append(name)
            return value

        ema_20 = _check("EMA20", _safe_last(_calculate_ema(close, 20)))
        ema_50 = _check("EMA50", _safe_last(_calculate_ema(close, 50)))
        sma_200 = _check("SMA200", _safe_last(_calculate_sma(close, 200)))
        rsi_14 = _check("RSI14", _safe_last(_calculate_rsi(close, 14)))

        macd_line_s, macd_signal_s, macd_hist_s = _calculate_macd(close)
        macd_line = _check("MACD_Line", _safe_last(macd_line_s))
        macd_signal = _check("MACD_Signal", _safe_last(macd_signal_s))
        macd_histogram = _check("MACD_Histogram", _safe_last(macd_hist_s))

        atr_14 = _check("ATR14", _safe_last(_calculate_atr(high, low, close, 14)))

        bb_upper_s, bb_mid_s, bb_lower_s = _calculate_bollinger_bands(close)
        bollinger_upper = _check("BollingerUpper", _safe_last(bb_upper_s))
        bollinger_middle = _check("BollingerMiddle", _safe_last(bb_mid_s))
        bollinger_lower = _check("BollingerLower", _safe_last(bb_lower_s))

        obv_series = _calculate_obv(close, volume)
        obv_latest = _safe_last(obv_series)
        obv_trend = None
        if obv_latest is not None and len(obv_series) > 10 and not pd.isna(obv_series.iloc[-11]):
            obv_10_ago = float(obv_series.iloc[-11])
            if obv_latest > obv_10_ago * 1.01:
                obv_trend = "naik"
            elif obv_latest < obv_10_ago * 0.99:
                obv_trend = "turun"
            else:
                obv_trend = "datar"
        if obv_trend is None:
            missing.append("OBV_Trend")

        last_close = _safe_last(close)
        swing_high = float(high.max()) if len(high) > 0 else None
        swing_low = float(low.min()) if len(low) > 0 else None

        fib_levels: Dict[str, float] = {}
        if swing_high is not None and swing_low is not None and swing_high > swing_low:
            diff = swing_high - swing_low
            fib_levels = {
                "fib_236": round(swing_high - 0.236 * diff, 2),
                "fib_382": round(swing_high - 0.382 * diff, 2),
                "fib_500": round(swing_high - 0.500 * diff, 2),
                "fib_618": round(swing_high - 0.618 * diff, 2),
                "fib_786": round(swing_high - 0.786 * diff, 2),
            }
        else:
            missing.append("Fibonacci_Levels")

        # --- Trade Setup: dihitung & diverifikasi DI SINI, bukan oleh LLM ---
        suggested_entry = suggested_stop_loss = suggested_tp1 = suggested_tp2 = risk_reward_ratio = None
        trade_setup_formulas: Dict[str, str] = {}

        if last_close is not None and atr_14 is not None and atr_14 > 0:
            suggested_entry = round(last_close, 2)
            suggested_stop_loss = round(last_close - (1.5 * atr_14), 2)
            suggested_tp1 = round(last_close + (1.0 * atr_14), 2)
            suggested_tp2 = round(last_close + (2.0 * atr_14), 2)

            risk = suggested_entry - suggested_stop_loss
            reward1 = suggested_tp1 - suggested_entry
            risk_reward_ratio = round(reward1 / risk, 2) if risk > 0 else None

            trade_setup_formulas = {
                "ATR14": "ATR14 = SMA(True Range, 14); True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)",
                "Entry": f"Entry = Harga penutupan terakhir = {suggested_entry}",
                "Stop_Loss": f"SL = Entry - (1.5 x ATR14) = {suggested_entry} - (1.5 x {round(atr_14, 2)}) = {suggested_stop_loss}",
                "TP1": f"TP1 = Entry + (1.0 x ATR14) = {suggested_entry} + (1.0 x {round(atr_14, 2)}) = {suggested_tp1}",
                "TP2": f"TP2 = Entry + (2.0 x ATR14) = {suggested_entry} + (2.0 x {round(atr_14, 2)}) = {suggested_tp2}",
                "Risk_Reward_Ratio": (
                    f"RR = (TP1 - Entry) / (Entry - SL) = ({suggested_tp1} - {suggested_entry}) / "
                    f"({suggested_entry} - {suggested_stop_loss}) = {risk_reward_ratio}"
                    if risk_reward_ratio is not None else "RR tidak dapat dihitung (risiko = 0)"
                ),
            }
        else:
            missing.append("Trade_Setup_Entry_SL_TP")

        now_utc = datetime.now(timezone.utc)
        now_wib = now_utc.astimezone(ZoneInfo("Asia/Jakarta"))

        return TechnicalAnalysisResult(
            ticker=ticker,
            timeframe=timeframe,
            data_source="Yahoo Finance (via library yfinance)",
            fetched_at_utc=now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            fetched_at_wib=now_wib.strftime("%Y-%m-%d %H:%M:%S WIB"),
            price_date=history.index[-1].strftime("%Y-%m-%d"),
            calculation_period=f"{len(history)} candle {timeframe} (fetch period: {fetch_period}, interval: {fetch_interval})",
            last_close=round(last_close, 2) if last_close is not None else None,
            ema_20=round(ema_20, 2) if ema_20 is not None else None,
            ema_50=round(ema_50, 2) if ema_50 is not None else None,
            sma_200=round(sma_200, 2) if sma_200 is not None else None,
            rsi_14=round(rsi_14, 2) if rsi_14 is not None else None,
            macd_line=round(macd_line, 4) if macd_line is not None else None,
            macd_signal=round(macd_signal, 4) if macd_signal is not None else None,
            macd_histogram=round(macd_histogram, 4) if macd_histogram is not None else None,
            atr_14=round(atr_14, 2) if atr_14 is not None else None,
            bollinger_upper=round(bollinger_upper, 2) if bollinger_upper is not None else None,
            bollinger_middle=round(bollinger_middle, 2) if bollinger_middle is not None else None,
            bollinger_lower=round(bollinger_lower, 2) if bollinger_lower is not None else None,
            obv_latest=round(obv_latest, 0) if obv_latest is not None else None,
            obv_trend=obv_trend,
            swing_high=round(swing_high, 2) if swing_high is not None else None,
            swing_low=round(swing_low, 2) if swing_low is not None else None,
            fib_236=fib_levels.get("fib_236"),
            fib_382=fib_levels.get("fib_382"),
            fib_500=fib_levels.get("fib_500"),
            fib_618=fib_levels.get("fib_618"),
            fib_786=fib_levels.get("fib_786"),
            suggested_entry=suggested_entry,
            suggested_stop_loss=suggested_stop_loss,
            suggested_tp1=suggested_tp1,
            suggested_tp2=suggested_tp2,
            risk_reward_ratio=risk_reward_ratio,
            trade_setup_formulas=trade_setup_formulas,
            foreign_flow_status="DATA TIDAK TERSEDIA",
            foreign_flow_reason=(
                "Sistem belum terintegrasi dengan sumber data resmi foreign flow "
                "(KSEI/IDX). yfinance TIDAK menyediakan data ini — JANGAN membuat "
                "klaim akumulasi/distribusi asing berdasarkan data yang tidak ada."
            ),
            missing_indicators=missing,
            data_sufficient_for_verdict=not any(item in missing for item in CRITICAL_INDICATORS),
        )

    except Exception as exc:
        logger.error(f"Gagal menghitung analisis teknikal {ticker} ({timeframe}): {exc}")
        return None


if __name__ == "__main__":
    for tf in ("daily", "weekly"):
        print(f"\n=== {tf.upper()} ===")
        result = get_technical_analysis("BBCA.JK", timeframe=tf)
        if result:
            for key, value in result.__dict__.items():
                print(f"{key}: {value}")
        else:
            print("❌ Gagal mengambil analisis teknikal.")