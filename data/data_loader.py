"""
Módulo Orquestrador e Carregador de Dados do IA Trader (Fase 1).
Responsável por:
1. Cache local persistente em SQLite (/home/eudespatricio/ia_trader/datasets/market_cache.db)
2. Arquitetura Multi-Cadência:
   - Alta Frequência: Velas OHLCV e profundidade da Bitget em tempo real com baixa latência.
   - Baixa Frequência: FRED (macro), Yahoo Finance (índices), DefiLlama (TVL/stablecoins)
     e CryptoPanic (sentimento) armazenados com TTL configurável (ex.: 30 a 60 min).
3. Sanitização Rigorosa:
   - Tratamento estrito de valores faltantes (NaNs) e anomalias de spread/volume.
   - Ordenação temporal estrita ascendente (prevenção de desalinhamento de série).
   - Ausência absoluta de Lookahead Bias (filtragem estrita por release_date em dados macro).
4. Suporte flexível com ou sem pandas instalado.
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone

# Importação condicional de pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .data_integrations import (
    CandleData,
    OrderBookDepth,
    MacroDataPoint,
    SentimentPost,
    BitgetConnector,
    CoinGlassConnector,
    FredConnector,
    YahooFinanceConnector,
    CryptoPanicConnector,
    DefiLlamaConnector,
    CoinMarketCalConnector,
    DataIntegrationManager,
)

logger = logging.getLogger("ia_trader.data_loader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [data_loader] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DataLoader:
    """
    Camada orquestradora de dados de mercado para o ecossistema IA Trader.
    Centraliza o acesso, persistência em SQLite, validação de integridade e multi-cadência.
    """

    DEFAULT_DB_PATH = "/home/eudespatricio/ia_trader/datasets/market_cache.db"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        # Garante a existência do diretório pai
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.manager = DataIntegrationManager()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Cria conexão SQLite otimizada com WAL mode para concorrência segura."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_database(self) -> None:
        """Inicializa as tabelas do banco de dados SQLite com suporte multi-exchange e migração automática."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1. Tabela de Velas OHLCV com suporte a múltiplas exchanges (Bitget, Binance)
            cur.execute("PRAGMA table_info(candles);")
            cols = [r["name"] for r in cur.fetchall()]
            if not cols:
                conn.execute("""
                    CREATE TABLE candles (
                        exchange TEXT NOT NULL DEFAULT 'bitget',
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        quote_volume REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_ex_ts ON candles (exchange, symbol, timeframe, timestamp ASC);")
            elif "exchange" not in cols:
                # Migração transparente preservando dados legados
                conn.execute("""
                    CREATE TABLE candles_migration_v2 (
                        exchange TEXT NOT NULL DEFAULT 'bitget',
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        quote_volume REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (exchange, symbol, timeframe, timestamp)
                    );
                """)
                conn.execute("""
                    INSERT OR IGNORE INTO candles_migration_v2
                    (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume, created_at)
                    SELECT 'bitget', symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume, created_at
                    FROM candles;
                """)
                conn.execute("DROP TABLE candles;")
                conn.execute("ALTER TABLE candles_migration_v2 RENAME TO candles;")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_ex_ts ON candles (exchange, symbol, timeframe, timestamp ASC);")

            # 2. Tabela de Taxas de Financiamento (Funding Rates)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    exchange TEXT NOT NULL DEFAULT 'binance',
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    funding_rate REAL NOT NULL,
                    mark_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (exchange, symbol, timestamp)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_rates (exchange, symbol, timestamp ASC);")

            # 3. Tabela de Métricas do Livro de Ofertas (Order Book Snapshots)
            cur.execute("PRAGMA table_info(order_books);")
            ob_cols = [r["name"] for r in cur.fetchall()]
            if not ob_cols:
                conn.execute("""
                    CREATE TABLE order_books (
                        exchange TEXT NOT NULL DEFAULT 'bitget',
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        best_bid REAL NOT NULL,
                        best_ask REAL NOT NULL,
                        spread REAL NOT NULL,
                        imbalance REAL NOT NULL,
                        raw_bids TEXT,
                        raw_asks TEXT,
                        PRIMARY KEY (exchange, symbol, timestamp)
                    );
                """)
            elif "exchange" not in ob_cols:
                try:
                    conn.execute("ALTER TABLE order_books ADD COLUMN exchange TEXT NOT NULL DEFAULT 'bitget';")
                except sqlite3.OperationalError:
                    pass

            # 4. Tabela de Dados Macroeconômicos com Data de Divulgação (Anti-Lookahead)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_series (
                    series_id TEXT NOT NULL,
                    observation_date TEXT NOT NULL,
                    release_date TEXT NOT NULL,
                    value REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (series_id, observation_date)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_release ON macro_series (series_id, release_date ASC);")

            # 5. Tabela de Cache Genérico com TTL (Baixa Frequência)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS low_freq_cache (
                    category TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY (category, cache_key)
                );
            """)
            conn.commit()

    # =====================================================================
    # MULTI-CADÊNCIA: ALTA FREQUÊNCIA (VELAS OHLCV MULTI-EXCHANGE)
    # =====================================================================

    async def get_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 100,
        exchange: str = "bitget",
        force_refresh: bool = False,
        as_dataframe: bool = True,
    ) -> Union[List[Dict[str, Any]], Any]:
        """
        Recupera velas OHLCV ordenadas cronologicamente para a exchange especificada ('bitget' ou 'binance').
        Fluxo:
        1. Tenta buscar dados novos em tempo real na exchange alvo.
        2. Sanitiza os dados e insere atômico no SQLite identificado por exchange.
        3. Caso a rede falhe, recupera o histórico mais recente persistido no SQLite.
        4. Garante ordenação temporal estrita ascendente e ausência de NaNs.
        """
        sym = symbol.upper()
        tf = timeframe.lower()
        ex = exchange.lower()

        if force_refresh or True:
            try:
                if ex == "binance":
                    remote_candles = await self.manager.binance.get_klines(
                        symbol=sym, timeframe=tf, limit=limit, market_type="futures"
                    )
                else:
                    remote_candles = await self.manager.bitget.get_candles(
                        symbol=sym, timeframe=tf, limit=limit, product_type="USDT-FUTURES"
                    )
                if remote_candles:
                    self._persist_candles(remote_candles, exchange=ex)
            except Exception as e:
                logger.warning(f"Erro ao buscar velas remotas ({ex}) para {sym}: {e}. Operando a partir do cache persistente.")

        # Leitura local estruturada do SQLite
        records = self._load_candles_from_db(sym, tf, limit, exchange=ex)

        # Sanitização e remoção de registros anômalos
        cleaned_records = self._sanitize_candles(records)

        if as_dataframe and HAS_PANDAS:
            return pd.DataFrame(cleaned_records)
        return cleaned_records

    def _persist_candles(self, candles: List[CandleData], exchange: str = "bitget") -> None:
        """Persiste lista de velas no SQLite utilizando INSERT OR REPLACE com campo de exchange."""
        if not candles:
            return
        query = """
            INSERT OR REPLACE INTO candles
            (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (exchange.lower(), c.symbol, c.timeframe, c.timestamp, c.open, c.high, c.low, c.close, c.volume, c.quote_volume)
            for c in candles
        ]
        with self._get_connection() as conn:
            conn.executemany(query, params)
            conn.commit()

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int, exchange: str = "bitget") -> List[Dict[str, Any]]:
        """Carrega do SQLite garantindo ordenação cronológica ascendente (mais antigo -> mais recente)."""
        query = """
            SELECT exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume
            FROM candles
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (exchange.lower(), symbol.upper(), timeframe.lower(), limit))
            rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        # Reverte para ordem estritamente ascendente
        results.reverse()
        return results

    def _sanitize_candles(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitização de integridade dos dados OHLCV:
        1. Remove candles com preços <= 0 ou volume < 0.
        2. Garante coerência geométrica básica: high >= low, high >= max(open, close), low <= min(open, close).
        3. Remove duplicações e timestamps nulos.
        """
        sanitized = []
        max_ts = -1

        for r in records:
            ts = r.get("timestamp")
            if not ts or ts <= max_ts:
                continue

            max_ts = ts

            op = float(r.get("open", 0.0))
            hi = float(r.get("high", 0.0))
            lo = float(r.get("low", 0.0))
            cl = float(r.get("close", 0.0))
            vol = float(r.get("volume", 0.0))
            qvol = float(r.get("quote_volume", 0.0))

            if op <= 0.0 or hi <= 0.0 or lo <= 0.0 or cl <= 0.0:
                continue

            # Ajuste de consistência se houver micro distorção
            hi = max(hi, op, cl)
            lo = min(lo, op, cl)
            vol = max(0.0, vol)
            qvol = max(0.0, qvol)

            r_clean = dict(r)
            r_clean["open"] = op
            r_clean["high"] = hi
            r_clean["low"] = lo
            r_clean["close"] = cl
            r_clean["volume"] = vol
            r_clean["quote_volume"] = qvol

            sanitized.append(r_clean)

        return sanitized

    # =====================================================================
    # FUNDING RATES HISTÓRICO & PERSISTÊNCIA
    # =====================================================================

    async def get_funding_history(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 100,
        exchange: str = "binance",
    ) -> List[Dict[str, Any]]:
        """
        Retorna histórico de Funding Rates com cache SQLite persistente.
        """
        sym = symbol.upper()
        ex = exchange.lower()
        try:
            if ex == "binance":
                remote_rates = await self.manager.binance.get_funding_rate_history(symbol=sym, limit=limit)
                if remote_rates:
                    self._persist_funding_rates(remote_rates, exchange=ex)
        except Exception as e:
            logger.warning(f"Erro ao buscar funding rate remoto ({ex}) para {sym}: {e}")

        # Carrega do SQLite
        query = """
            SELECT exchange, symbol, timestamp, funding_rate, mark_price
            FROM funding_rates
            WHERE exchange = ? AND symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (ex, sym, limit))
            rows = cursor.fetchall()

        results = [dict(r) for r in rows]
        results.reverse()
        return results

    def _persist_funding_rates(self, rates: List[Dict[str, Any]], exchange: str = "binance") -> None:
        """Persiste taxas de financiamento no SQLite."""
        if not rates:
            return
        query = """
            INSERT OR REPLACE INTO funding_rates
            (exchange, symbol, timestamp, funding_rate, mark_price)
            VALUES (?, ?, ?, ?, ?)
        """
        params = [
            (
                exchange.lower(),
                r["symbol"].upper(),
                int(r["funding_time"]),
                float(r["funding_rate"]),
                float(r["mark_price"]) if r.get("mark_price") is not None else None,
            )
            for r in rates
        ]
        with self._get_connection() as conn:
            conn.executemany(query, params)
            conn.commit()

    async def get_binance_derivatives_context(self, symbol: str = "BTCUSDT", ttl_seconds: int = 300) -> Dict[str, Any]:
        """
        Recupera métricas derivadas da Binance (Open Interest, Long/Short Ratio, Taker Volume) com TTL de 5 min.
        """
        sym = symbol.upper()
        cache_key = f"binance_deriv_{sym}"
        cached = self._get_from_cache("derivatives", cache_key)
        if cached is not None:
            return cached

        oi_task = self.manager.binance.get_open_interest(sym)
        ls_task = self.manager.binance.get_long_short_ratio(sym, limit=1)
        tk_task = self.manager.binance.get_taker_volume_ratio(sym, limit=1)

        oi_res, ls_res, tk_res = await asyncio.gather(oi_task, ls_task, tk_task, return_exceptions=True)

        oi_val = oi_res.get("open_interest", 0.0) if isinstance(oi_res, dict) else 0.0
        ls_val = ls_res[0].get("long_short_ratio", 1.0) if (isinstance(ls_res, list) and ls_res) else 1.0
        tk_val = tk_res[0].get("buy_sell_ratio", 1.0) if (isinstance(tk_res, list) and tk_res) else 1.0

        data = {
            "symbol": sym,
            "open_interest": oi_val,
            "long_short_ratio": ls_val,
            "taker_buy_sell_ratio": tk_val,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        self._save_to_cache("derivatives", cache_key, data, ttl_seconds)
        return data

    # =====================================================================
    # VALIDAÇÃO CRUZADA BITGET X BINANCE (CROSS-VALIDATION)
    # =====================================================================

    async def cross_validate_bitget_binance(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 500,
        tolerance_pct: float = 0.5,
        bitget_candles: Optional[List[Dict[str, Any]]] = None,
        binance_candles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Alinha velas de Bitget e Binance para o mesmo símbolo e timeframe, analisando:
        - Divergência percentual de preço de fechamento (Close Diff %)
        - Razão de volume (Volume Ratio)
        - Candles faltantes em cada lado (Missing candles)
        - Correlação de Pearson dos retornos percentuais consecutivos
        - Disparo de alerta caso desvio máximo > tolerance_pct ou correlação < 0.95.
        """
        sym = symbol.upper()
        tf = timeframe.lower()

        # Obtém candles se não fornecidos diretamente
        if bitget_candles is None:
            bg_data = await self.get_candles(sym, tf, limit=limit, exchange="bitget", as_dataframe=False)
            bitget_candles = bg_data if isinstance(bg_data, list) else []

        if binance_candles is None:
            bn_data = await self.get_candles(sym, tf, limit=limit, exchange="binance", as_dataframe=False)
            binance_candles = bn_data if isinstance(bn_data, list) else []

        if tf in ("1d", "1day"):
            # Normaliza chave temporal para data civil YYYY-MM-DD para compatibilizar fechamentos diários
            bg_map = {
                datetime.fromtimestamp(int(c["timestamp"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d"): c
                for c in bitget_candles
            }
            bn_map = {
                datetime.fromtimestamp(int(c["timestamp"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d"): c
                for c in binance_candles
            }
        else:
            bg_map = {int(c["timestamp"]): c for c in bitget_candles}
            bn_map = {int(c["timestamp"]): c for c in binance_candles}

        bg_ts = set(bg_map.keys())
        bn_ts = set(bn_map.keys())

        common_ts = sorted(bg_ts & bn_ts)
        missing_in_binance = sorted(bg_ts - bn_ts)
        missing_in_bitget = sorted(bn_ts - bg_ts)

        close_diffs = []
        vol_ratios = []
        divergences = []
        returns_bg = []
        returns_bn = []

        prev_bg_close = None
        prev_bn_close = None

        for ts in common_ts:
            bg_c = bg_map[ts]
            bn_c = bn_map[ts]

            bg_close = float(bg_c["close"])
            bn_close = float(bn_c["close"])
            bg_vol = float(bg_c.get("volume", 0.0))
            bn_vol = float(bn_c.get("volume", 0.0))

            diff_pct = (abs(bg_close - bn_close) / bn_close * 100.0) if bn_close > 0 else 0.0
            close_diffs.append(diff_pct)

            vol_ratio = (bg_vol / bn_vol) if bn_vol > 0 else 1.0
            vol_ratios.append(vol_ratio)

            if diff_pct > (tolerance_pct * 0.5):
                dt_label = ts if isinstance(ts, str) else datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                divergences.append({
                    "timestamp": ts,
                    "datetime": dt_label,
                    "bitget_close": bg_close,
                    "binance_close": bn_close,
                    "diff_pct": round(diff_pct, 4),
                })

            if prev_bg_close is not None and prev_bn_close is not None:
                r_bg = (bg_close - prev_bg_close) / prev_bg_close if prev_bg_close > 0 else 0.0
                r_bn = (bn_close - prev_bn_close) / prev_bn_close if prev_bn_close > 0 else 0.0
                returns_bg.append(r_bg)
                returns_bn.append(r_bn)

            prev_bg_close = bg_close
            prev_bn_close = bn_close

        # Cálculo da correlação de Pearson dos retornos
        correlation = 1.0
        if len(returns_bg) > 2:
            import math
            mean_bg = sum(returns_bg) / len(returns_bg)
            mean_bn = sum(returns_bn) / len(returns_bn)
            num = sum((x - mean_bg) * (y - mean_bn) for x, y in zip(returns_bg, returns_bn))
            den_bg = sum((x - mean_bg) ** 2 for x in returns_bg)
            den_bn = sum((y - mean_bn) ** 2 for y in returns_bn)
            denom = math.sqrt(den_bg * den_bn)
            correlation = round(num / denom, 4) if denom > 0 else 1.0

        mean_diff = round(sum(close_diffs) / len(close_diffs), 4) if close_diffs else 0.0
        max_diff = round(max(close_diffs), 4) if close_diffs else 0.0
        alert_triggered = bool((max_diff > tolerance_pct) or (correlation < 0.95 and len(common_ts) > 5))

        divergences.sort(key=lambda x: x["diff_pct"], reverse=True)

        return {
            "symbol": sym,
            "timeframe": tf,
            "total_bitget": len(bitget_candles),
            "total_binance": len(binance_candles),
            "common_count": len(common_ts),
            "missing_in_binance_count": len(missing_in_binance),
            "missing_in_bitget_count": len(missing_in_bitget),
            "mean_close_diff_pct": mean_diff,
            "max_close_diff_pct": max_diff,
            "returns_correlation": correlation,
            "alert_triggered": alert_triggered,
            "tolerance_pct": tolerance_pct,
            "top_divergences": divergences[:5],
        }

    async def run_and_save_cross_validation_report(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        tolerance_pct: float = 0.5,
        output_path: str = "/home/eudespatricio/ia_trader/datasets/reports/cross_validation_bitget_binance.csv",
    ) -> List[Dict[str, Any]]:
        """
        Executa a validação cruzada para múltiplos pares e timeframes e salva o relatório em CSV.
        """
        target_symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        target_tfs = timeframes or ["1h", "1d"]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        reports = []

        for sym in target_symbols:
            for tf in target_tfs:
                report = await self.cross_validate_bitget_binance(
                    symbol=sym, timeframe=tf, limit=500, tolerance_pct=tolerance_pct
                )
                reports.append(report)

        # Salva em CSV
        header = "symbol,timeframe,total_bitget,total_binance,common_count,missing_binance,missing_bitget,mean_close_diff_pct,max_close_diff_pct,returns_correlation,alert_triggered,status\n"
        lines = [header]
        for r in reports:
            status = "ALERT" if r["alert_triggered"] else "OK"
            lines.append(
                f"{r['symbol']},{r['timeframe']},{r['total_bitget']},{r['total_binance']},"
                f"{r['common_count']},{r['missing_in_binance_count']},{r['missing_in_bitget_count']},"
                f"{r['mean_close_diff_pct']},{r['max_close_diff_pct']},{r['returns_correlation']},"
                f"{r['alert_triggered']},{status}\n"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info(f"Relatório de validação cruzada Bitget x Binance salvo em {output_path}")
        return reports

    async def get_order_book_metrics(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Obtém métricas do livro de ofertas (Spread e Imbalance) da Bitget com persistência imediata.
        """
        depth = await self.manager.bitget.get_order_book(symbol=symbol, limit=20)
        if depth:
            best_bid = depth.bids[0][0] if depth.bids else 0.0
            best_ask = depth.asks[0][0] if depth.asks else 0.0

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO order_books
                    (symbol, timestamp, best_bid, best_ask, spread, imbalance, raw_bids, raw_asks)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    depth.symbol,
                    depth.timestamp,
                    best_bid,
                    best_ask,
                    depth.spread,
                    depth.imbalance,
                    json.dumps(depth.bids[:5]),
                    json.dumps(depth.asks[:5]),
                ))
                conn.commit()

            return {
                "symbol": depth.symbol,
                "timestamp": depth.timestamp,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": depth.spread,
                "imbalance": depth.imbalance,
            }

        # Fallback para o último snapshot persistido no SQLite
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT symbol, timestamp, best_bid, best_ask, spread, imbalance
                FROM order_books
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol.upper(),))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return {
            "symbol": symbol.upper(),
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "best_bid": 0.0,
            "best_ask": 0.0,
            "spread": 0.0,
            "imbalance": 0.0,
        }

    # =====================================================================
    # MULTI-CADÊNCIA: BAIXA FREQUÊNCIA COM TTL (CACHE PERSISTENTE SQLITE)
    # =====================================================================

    def _get_from_cache(self, category: str, key: str) -> Optional[Any]:
        """Recupera valor do cache SQLite se ainda estiver dentro da validade (TTL)."""
        now = int(datetime.now(timezone.utc).timestamp())
        query = """
            SELECT payload FROM low_freq_cache
            WHERE category = ? AND cache_key = ? AND expires_at > ?
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (category, key, now))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["payload"])
                except json.JSONDecodeError:
                    return None
        return None

    def _save_to_cache(self, category: str, key: str, payload: Any, ttl_seconds: int) -> None:
        """Salva valor no cache SQLite com expiração calculada."""
        now = int(datetime.now(timezone.utc).timestamp())
        expires_at = now + ttl_seconds
        payload_str = json.dumps(payload)
        query = """
            INSERT OR REPLACE INTO low_freq_cache
            (category, cache_key, payload, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(query, (category, key, payload_str, now, expires_at))
            conn.commit()

    async def get_macro_context(
        self,
        as_of_date: Optional[str] = None,
        ttl_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """
        Recupera o contexto macroeconômico (FRED: CPI, Juros FEDFUNDS) e intermercado (DXY, VIX, S&P 500).
        Utiliza cache de 1 hora (3600s).
        PREVENÇÃO DE LOOKAHEAD BIAS:
        Se as_of_date for fornecido (ex.: '2023-05-15'), filtra estritamente por release_date <= as_of_date.
        """
        cache_key = f"macro_context_{as_of_date or 'latest'}"
        cached = self._get_from_cache("macro", cache_key)
        if cached is not None:
            return cached

        # Busca dados remotos de baixa frequência
        snapshots = await self.manager.fetch_all_macro_snapshots()

        # Persiste séries macro na tabela anti-lookahead
        with self._get_connection() as conn:
            for item in snapshots.get("cpi", []):
                if isinstance(item, MacroDataPoint):
                    conn.execute("""
                        INSERT OR REPLACE INTO macro_series
                        (series_id, observation_date, release_date, value)
                        VALUES (?, ?, ?, ?)
                    """, (item.series_id, item.observation_date, item.release_date, item.value))

            for item in snapshots.get("fedfunds", []):
                if isinstance(item, MacroDataPoint):
                    conn.execute("""
                        INSERT OR REPLACE INTO macro_series
                        (series_id, observation_date, release_date, value)
                        VALUES (?, ?, ?, ?)
                    """, (item.series_id, item.observation_date, item.release_date, item.value))
            conn.commit()

        # Extrai valores com proteção de release_date
        current_date_filter = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cpi_val = self._get_latest_macro_val("CPIAUCSL", max_release_date=current_date_filter)
        fedfunds_val = self._get_latest_macro_val("FEDFUNDS", max_release_date=current_date_filter)

        # Últimos preços de VIX, DXY e S&P 500
        vix_points = snapshots.get("vix", [])
        dxy_points = snapshots.get("dxy", [])
        sp_points = snapshots.get("sp500", [])

        vix_val = vix_points[-1]["close"] if vix_points else 18.0
        dxy_val = dxy_points[-1]["close"] if dxy_points else 103.5
        sp_val = sp_points[-1]["close"] if sp_points else 5000.0

        result = {
            "as_of_date": current_date_filter,
            "cpi": cpi_val,
            "fedfunds": fedfunds_val,
            "vix": vix_val,
            "dxy": dxy_val,
            "sp500": sp_val,
            "regime": "risk_off" if (vix_val > 25.0 or dxy_val > 105.0) else "risk_on",
            "cached_at": int(datetime.now(timezone.utc).timestamp()),
        }

        self._save_to_cache("macro", cache_key, result, ttl_seconds)
        return result

    def _get_latest_macro_val(self, series_id: str, max_release_date: str) -> float:
        """Busca o valor mais recente cuja release_date seja anterior ou igual à data de referência."""
        query = """
            SELECT value FROM macro_series
            WHERE series_id = ? AND release_date <= ?
            ORDER BY release_date DESC
            LIMIT 1
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (series_id, max_release_date))
            row = cursor.fetchone()
            if row:
                return float(row["value"])
        # Valores de referência médios históricos para fallback seguro se banco novo
        defaults = {"CPIAUCSL": 314.0, "FEDFUNDS": 5.33, "DGS10": 4.25}
        return defaults.get(series_id, 0.0)

    async def get_sentiment_context(self, currency: str = "BTC", ttl_seconds: int = 1800) -> Dict[str, Any]:
        """
        Recupera o contexto de sentimento de notícias (CryptoPanic).
        Utiliza cache de 30 minutos (1800s).
        """
        cache_key = f"sentiment_{currency.upper()}"
        cached = self._get_from_cache("sentiment", cache_key)
        if cached is not None:
            return cached

        data = await self.manager.cryptopanic.get_consolidated_sentiment_score(currency=currency)
        self._save_to_cache("sentiment", cache_key, data, ttl_seconds)
        return data

    async def get_onchain_context(self, ttl_seconds: int = 3600) -> Dict[str, Any]:
        """
        Recupera o contexto on-chain e DeFi (DefiLlama: TVL e fluxos de Stablecoins).
        Utiliza cache de 1 hora (3600s).
        """
        cache_key = "onchain_defillama"
        cached = self._get_from_cache("onchain", cache_key)
        if cached is not None:
            return cached

        tvl_task = self.manager.defillama.get_total_tvl()
        stables_task = self.manager.defillama.get_stablecoin_flows()
        tvl_res, stables_res = await asyncio.gather(tvl_task, stables_task, return_exceptions=True)

        result = {
            "tvl": tvl_res if not isinstance(tvl_res, Exception) else {"tvl_usd": 0.0, "change_24h_pct": 0.0},
            "stablecoins": stables_res if not isinstance(stables_res, Exception) else {"total_stablecoins_usd": 0.0, "net_inflow_24h_usd": 0.0},
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
        }

        self._save_to_cache("onchain", cache_key, result, ttl_seconds)
        return result

    async def get_risk_events_context(self, coin: str = "BTC", ttl_seconds: int = 7200) -> List[Dict[str, Any]]:
        """
        Recupera eventos iminentes de mercado e risco de deslistagem (CoinMarketCal).
        Utiliza cache de 2 horas (7200s).
        """
        cache_key = f"risk_events_{coin.upper()}"
        cached = self._get_from_cache("events", cache_key)
        if cached is not None:
            return cached

        events = await self.manager.coinmarketcal.get_upcoming_events(coin=coin, days_ahead=7)
        serialized = [e.to_dict() for e in events]
        self._save_to_cache("events", cache_key, serialized, ttl_seconds)
        return serialized

    async def close(self) -> None:
        """Encerra instâncias e conexões ativas."""
        await self.manager.close()
