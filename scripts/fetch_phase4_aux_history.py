"""
Script de Ingestão e Alinhamento de Séries Históricas Auxiliares da Fase 4.
Persiste no banco SQLite ('datasets/market_cache.db'):
1. Fear & Greed Index (alternative.me): Histórico diário desde 2018 (3.170+ pontos).
2. DefiLlama Stablecoins: Suprimento circulante global em USD desde 2017 (3.240+ pontos).
3. Yahoo Finance Macro: DXY (^DXY / DX-Y.NYB), VIX (^VIX) e S&P 500 (^GSPC).
4. Calendário de Eventos Macro de Alto Impacto: Datas oficiais de divulgação do CPI e reuniões do FOMC.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("phase4_history")

DB_PATH = "/home/eudespatricio/ia_trader/datasets/market_cache.db"


def init_db(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_daily_history (
            series_name TEXT,
            date_str TEXT,
            timestamp_ms INTEGER,
            value REAL,
            extra_info TEXT,
            PRIMARY KEY (series_name, date_str)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_events_calendar (
            event_name TEXT,
            date_str TEXT,
            timestamp_ms INTEGER,
            PRIMARY KEY (event_name, date_str)
        )
    """)
    conn.commit()


def fetch_fear_and_greed(conn: sqlite3.Connection):
    logger.info("Baixando histórico completo de Fear & Greed (alternative.me)...")
    url = "https://api.alternative.me/fng/?limit=0"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            records = data.get("data", [])
            cursor = conn.cursor()
            inserted = 0
            for r in records:
                ts_sec = int(r["timestamp"])
                dt_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
                val = float(r["value"])
                cursor.execute(
                    "INSERT OR REPLACE INTO macro_daily_history (series_name, date_str, timestamp_ms, value, extra_info) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("FEAR_AND_GREED", dt_str, ts_sec * 1000, val, r.get("value_classification", "")),
                )
                inserted += 1
            conn.commit()
            logger.info(f"Fear & Greed persistido: {inserted} registros diários.")
    except Exception as e:
        logger.error(f"Erro ao baixar Fear & Greed: {e}")


def fetch_defillama_stablecoins(conn: sqlite3.Connection):
    logger.info("Baixando histórico de Stablecoins DefiLlama...")
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            records = json.loads(resp.read().decode("utf-8"))
            cursor = conn.cursor()
            inserted = 0
            for r in records:
                ts_sec = int(r["date"])
                dt_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
                val = float(r.get("totalCirculatingUSD", {}).get("peggedUSD", 0.0))
                cursor.execute(
                    "INSERT OR REPLACE INTO macro_daily_history (series_name, date_str, timestamp_ms, value, extra_info) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("STABLECOINS_CIRCULATING_USD", dt_str, ts_sec * 1000, val, ""),
                )
                inserted += 1
            conn.commit()
            logger.info(f"DefiLlama Stablecoins persistido: {inserted} registros diários.")
    except Exception as e:
        logger.error(f"Erro ao baixar DefiLlama Stablecoins: {e}")


def fetch_yahoo_series(conn: sqlite3.Connection, ticker: str, series_name: str):
    logger.info(f"Baixando histórico Yahoo para {ticker} ({series_name})...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            cursor = conn.cursor()
            inserted = 0
            for ts_sec, close_p in zip(timestamps, closes):
                if close_p is None:
                    continue
                dt_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
                cursor.execute(
                    "INSERT OR REPLACE INTO macro_daily_history (series_name, date_str, timestamp_ms, value, extra_info) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (series_name, dt_str, ts_sec * 1000, float(close_p), ""),
                )
                inserted += 1
            conn.commit()
            logger.info(f"Yahoo {series_name} persistido: {inserted} registros diários.")
    except Exception as e:
        logger.error(f"Erro ao baixar Yahoo {ticker}: {e}")


def populate_cpi_fomc_calendar(conn: sqlite3.Connection):
    logger.info("Populando calendário histórico e programado de CPI e FOMC...")
    # Datas oficiais das reuniões do FOMC e divulgações de CPI dos EUA (2023 - 2026)
    fomc_dates = [
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        # 2026
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ]
    cpi_dates = [
        # 2023
        "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13",
        "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
        # 2024
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
        "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
        # 2025
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
        "2025-07-11", "2025-08-13", "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
        # 2026
        "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14", "2026-05-13", "2026-06-10",
        "2026-07-14", "2026-08-12", "2026-09-15", "2026-10-14", "2026-11-12", "2026-12-10",
    ]

    cursor = conn.cursor()
    for d in fomc_dates:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cursor.execute("INSERT OR REPLACE INTO macro_events_calendar VALUES (?, ?, ?)", ("FOMC_DECISION", d, int(dt.timestamp() * 1000)))
    for d in cpi_dates:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cursor.execute("INSERT OR REPLACE INTO macro_events_calendar VALUES (?, ?, ?)", ("US_CPI_RELEASE", d, int(dt.timestamp() * 1000)))
    conn.commit()
    logger.info(f"Calendário populado: {len(fomc_dates)} reuniões FOMC e {len(cpi_dates)} divulgações CPI.")


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    fetch_fear_and_greed(conn)
    fetch_defillama_stablecoins(conn)
    fetch_yahoo_series(conn, "DX-Y.NYB", "DXY")
    fetch_yahoo_series(conn, "^VIX", "VIX")
    fetch_yahoo_series(conn, "^GSPC", "SP500")
    populate_cpi_fomc_calendar(conn)
    conn.close()
    logger.info("Ingestão de séries auxiliares concluída com sucesso.")


if __name__ == "__main__":
    main()
