"""
Script de Download Massivo e Ingestão de Histórico Real da Bitget (Fase 3 Re-auditoria).
Baixa velas reais e paginadas da API pública de futuros USDT da Bitget para
BTCUSDT, ETHUSDT e SOLUSDT nos timeframes: 15m, 1h, 4h e 1d.
Persiste no banco SQLite local (/datasets/market_cache.db) com validação de integridade.
"""

import os
import sys
import time
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

# Raiz do projeto
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from data.data_loader import DataLoader
from data.data_integrations import CandleData, BitgetConnector


TARGETS = [
    # (symbol, timeframe, granularity, max_batches_of_200, label)
    # 1d: ~1000-2000 velas (máximo disponível)
    ("BTCUSDT", "1d", "1D", 15, "1d BTC"),
    ("ETHUSDT", "1d", "1D", 15, "1d ETH"),
    ("SOLUSDT", "1d", "1D", 15, "1d SOL"),
    # 4h: ~6000-8000 velas (~3-4 anos)
    ("BTCUSDT", "4h", "4H", 40, "4h BTC"),
    ("ETHUSDT", "4h", "4H", 35, "4h ETH"),
    ("SOLUSDT", "4h", "4H", 30, "4h SOL"),
    # 1h: ~18000-22000 velas (~2-2.5 anos)
    ("BTCUSDT", "1h", "1H", 110, "1h BTC"),
    ("ETHUSDT", "1h", "1H", 90, "1h ETH"),
    ("SOLUSDT", "1h", "1H", 70, "1h SOL"),
    # 15m: ~35000 velas (~1 ano)
    ("BTCUSDT", "15m", "15m", 180, "15m BTC"),
    ("ETHUSDT", "15m", "15m", 140, "15m ETH"),
    ("SOLUSDT", "15m", "15m", 120, "15m SOL"),
]


async def fetch_target(b: BitgetConnector, loader: DataLoader, symbol: str, timeframe: str, granularity: str, max_batches: int):
    print(f"\n>>> Iniciando download: {symbol} [{timeframe}] (Meta: até {max_batches * 200} velas)...")
    
    # 1. Pega vela mais recente primeiro
    recent_endpoint = f"{b.BASE_URL}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol.upper(),
        "productType": "USDT-FUTURES",
        "granularity": granularity,
        "limit": "100",
    }
    recent_data = await b.http.get_json(recent_endpoint, params=params)
    recent_raw = recent_data.get("data", []) if recent_data and isinstance(recent_data, dict) else []
    
    oldest_ts = int(time.time() * 1000)
    if recent_raw:
        candles_recent = []
        for c in recent_raw:
            try:
                candles_recent.append(CandleData(
                    timestamp=int(c[0]),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                    quote_volume=float(c[6]) if len(c) > 6 else float(c[5]) * float(c[4]),
                    symbol=symbol.upper(),
                    timeframe=timeframe.lower(),
                ))
            except Exception:
                pass
        candles_recent.sort(key=lambda x: x.timestamp)
        if candles_recent:
            loader._persist_candles(candles_recent)
            oldest_ts = candles_recent[0].timestamp

    # 2. Paginação retroativa com history-candles
    total_added = 0
    history_endpoint = f"{b.BASE_URL}/api/v2/mix/market/history-candles"
    
    consecutive_empty = 0
    for batch_i in range(max_batches):
        p = {
            "symbol": symbol.upper(),
            "productType": "USDT-FUTURES",
            "granularity": granularity,
            "limit": "200",
            "endTime": str(oldest_ts - 1),
        }
        data = await b.http.get_json(history_endpoint, params=p)
        raw = data.get("data", []) if data and isinstance(data, dict) else []
        
        if not raw:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print(f"    Sem mais histórico para {symbol} [{timeframe}] no batch {batch_i + 1}.")
                break
            await asyncio.sleep(0.15)
            continue
        
        consecutive_empty = 0
        batch_candles = []
        for c in raw:
            try:
                batch_candles.append(CandleData(
                    timestamp=int(c[0]),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                    quote_volume=float(c[6]) if len(c) > 6 else float(c[5]) * float(c[4]),
                    symbol=symbol.upper(),
                    timeframe=timeframe.lower(),
                ))
            except Exception:
                pass
                
        if not batch_candles:
            break
            
        batch_candles.sort(key=lambda x: x.timestamp)
        oldest_ts = batch_candles[0].timestamp
        loader._persist_candles(batch_candles)
        total_added += len(batch_candles)
        
        if (batch_i + 1) % 15 == 0 or (batch_i + 1) == max_batches:
            oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"    Batch {batch_i + 1}/{max_batches} | Velas baixadas: {total_added} | Ponto mais antigo: {oldest_dt}")
            
        await asyncio.sleep(0.04)


async def main():
    loader = DataLoader()
    b = loader.manager.bitget
    
    start_time = time.time()
    print("=================================================================")
    print("INICIANDO DOWNLOAD MASSIVO DE DADOS HISTÓRICOS REAIS DA BITGET")
    print("=================================================================")
    
    for symbol, tf, gran, max_b, label in TARGETS:
        await fetch_target(b, loader, symbol, tf, gran, max_b)
        
    await loader.close()
    elapsed = time.time() - start_time
    print(f"\nDownloads concluídos em {elapsed:.1f}s. Validando banco de dados...")
    
    # Validação e Relatório de Integridade
    conn = sqlite3.connect(loader.db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, timeframe, count(*), min(timestamp), max(timestamp)
        FROM candles
        GROUP BY symbol, timeframe
        ORDER BY symbol, 
                 CASE timeframe 
                     WHEN '15m' THEN 1 
                     WHEN '1h' THEN 2 
                     WHEN '4h' THEN 3 
                     WHEN '1d' THEN 4 
                     ELSE 5 
                 END;
    """)
    rows = cur.fetchall()
    
    print("\n=========================================================================================")
    print("RELATÓRIO DE COBERTURA HISTÓRICA REAL NO CACHE SQLITE (/datasets/market_cache.db)")
    print("=========================================================================================")
    print(f"{'SÍMBOLO':<10} | {'TIMEFRAME':<10} | {'Nº VELAS':<10} | {'DATA INICIAL':<20} | {'DATA FINAL':<20}")
    print("-" * 85)
    
    for sym, tf, count, min_ts, max_ts in rows:
        dt_start = datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        dt_end = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"{sym:<10} | {tf:<10} | {count:<10} | {dt_start:<20} | {dt_end:<20}")
        
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
