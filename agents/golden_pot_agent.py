"""
golden_pot_agent.py - Agente Scanner de Mercado & Caçador de Oportunidades ("Golden Pots").
Fase 4 & Expansão de Universo do IA Trader.

Missão:
Escanear continuamente o universo de contratos perpétuos USDT (Binance / Bitget)
com custo zero de dados, aplicando filtros rigorosos de liquidez institucional
para identificar em tempo real as melhores oportunidades de:
1. SCALP_REVERSION (15m): Reversão à média por esticamento elástico de VWAP e exaustão de RSI.
2. SWING_BREAKOUT (1h/4h): Rompimento de consolidação de 24h com surto de volume e momentum limpo.

Critérios Institucionais:
- Liquidez Mínima: Volume 24h >= $20,000,000 USD (elimina shitcoins, baixa liquidez e risco de slippage).
- Exclusão de Stablecoins (USDC, FDUSD, EUR, TUSD, etc.).
- Cálculo Instantâneo sem overhead: Usa 24hr Tickers agregados das exchanges em chamada única (~500ms).
- Persistência em cache SQLite (`golden_pots` em `market_cache.db`) para leitura ultra-rápida no Dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from db_conn import connect as _db_connect
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from data.data_integrations import BinanceConnector, BitgetConnector, AsyncHttpClient

logger = logging.getLogger("ia_trader.golden_pot_agent")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [golden_pot] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Pares que devem ser sempre excluídos da varredura (stablecoins, índices ou moedas sintéticas)
STABLECOIN_EXCLUSIONS = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT",
    "USDPUSDT", "AEURUSDT", "DAIUSDT", "USDTUSDC", "USDTCAD"
}


@dataclass
class GoldenPotOpportunity:
    """Contrato estruturado de uma oportunidade detectada pelo GoldenPotAgent."""
    symbol: str
    setup_type: str                   # 'SCALP_REVERSION' ou 'SWING_BREAKOUT'
    direction: str                    # 'BUY' ou 'SELL'
    score: float                      # Score ponderado [0.0, 100.0]
    last_price: float
    dist_vwap_pct: float              # Desvio percentual em relação ao VWAP 24h
    volume_24h_usd: float             # Volume em dólares nas últimas 24h
    price_change_pct_24h: float       # Variação em 24h
    range_position_pct: float         # Posição percentual no range 24h (0% = mínima, 100% = máxima)
    amplitude_pct: float              # Amplitude máxima-mínima 24h
    suggested_timeframe: str          # '15m' para scalp, '1h'/'4h' para swing
    rationale: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoldenPotAgent:
    """
    Agente Scanner Autônomo para Descoberta de Oportunidades ("Golden Pots").
    Opera em multi-pares com atualização assíncrona periódica e persistência SQLite.
    """

    def __init__(
        self,
        min_volume_usd: float = 20_000_000.0,
        db_path: Optional[str] = None,
        binance_connector: Optional[BinanceConnector] = None,
        bitget_connector: Optional[BitgetConnector] = None,
        cache_ttl_seconds: int = 180,  # TTL de 3 minutos para cache em memória
    ):
        self.min_volume_usd = float(min_volume_usd)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "market_cache.db"
        )
        self.binance = binance_connector or BinanceConnector()
        self.bitget = bitget_connector or BitgetConnector()

        self._cached_scan: Optional[Dict[str, Any]] = None
        self._last_scan_time: float = 0.0
        self._init_database_table()

    async def close(self) -> None:
        """Encerra graciosamente as conexões HTTP subjacentes."""
        await self.binance.http.close()
        await self.bitget.http.close()

    def _get_connection(self) -> sqlite3.Connection:

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = _db_connect(self.db_path, timeout=10.0)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_database_table(self) -> None:
        """Cria tabela SQLite para histórico e snapshots dos Golden Pots."""
        with self._get_connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS golden_pots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    score REAL NOT NULL,
                    last_price REAL NOT NULL,
                    dist_vwap_pct REAL NOT NULL,
                    volume_24h_usd REAL NOT NULL,
                    price_change_pct_24h REAL NOT NULL,
                    range_position_pct REAL NOT NULL,
                    amplitude_pct REAL NOT NULL,
                    suggested_timeframe TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_golden_pots_ts ON golden_pots(timestamp)")
            con.commit()

    async def fetch_market_universe_tickers(self) -> List[Dict[str, Any]]:
        """
        Coleta dados 24h de todo o universo de futuros USDT em chamada única.
        Tenta Binance Futuros como fonte primária; usa Bitget Futuros como fallback.
        """
        # 1. Fonte Primária: Binance Futuros (/fapi/v1/ticker/24hr)
        try:
            url = f"{self.binance.FUTURES_URL}/fapi/v1/ticker/24hr"
            data = await self.binance.http.get_json(url)
            if data and isinstance(data, list) and len(data) > 0:
                normalized = []

                for item in data:
                    sym = str(item.get("symbol", "")).upper()
                    if not sym.endswith("USDT") or sym in STABLECOIN_EXCLUSIONS:
                        continue
                    try:
                        last_p = float(item.get("lastPrice", 0.0))
                        vwap = float(item.get("weightedAvgPrice", 0.0))
                        high_p = float(item.get("highPrice", 0.0))
                        low_p = float(item.get("lowPrice", 0.0))
                        vol_usd = float(item.get("quoteVolume", 0.0))
                        chg_pct = float(item.get("priceChangePercent", 0.0))
                        if last_p <= 0.0 or vol_usd < self.min_volume_usd:
                            continue
                        normalized.append({
                            "symbol": sym,
                            "last_price": last_p,
                            "vwap_24h": vwap if vwap > 0 else last_p,
                            "high_24h": high_p,
                            "low_24h": low_p,
                            "volume_24h_usd": vol_usd,
                            "price_change_pct": chg_pct,
                            "source": "binance",
                        })
                    except (ValueError, TypeError):
                        continue
                if normalized:
                    return normalized
        except Exception as e:
            logger.warning(f"Falha na consulta primária Binance Tickers: {e}. Acionando fallback Bitget.")

        # 2. Fonte Secundária: Bitget Futuros (/api/v2/mix/market/tickers)
        try:
            url = f"{self.bitget.BASE_URL}/api/v2/mix/market/tickers?productType=USDT-FUTURES"
            res = await self.bitget.http.get_json(url)
            items = res.get("data", []) if isinstance(res, dict) else []
            if items:
                normalized = []
                for item in items:
                    sym = str(item.get("symbol", "")).upper()
                    if not sym.endswith("USDT") or sym in STABLECOIN_EXCLUSIONS:
                        continue
                    try:
                        last_p = float(item.get("lastPr", 0.0))
                        high_p = float(item.get("high24h", 0.0))
                        low_p = float(item.get("low24h", 0.0))
                        vol_usd = float(item.get("usdtVolume", 0.0))
                        chg_pct = float(item.get("change24h", 0.0)) * 100.0 # Bitget fornece decimal
                        vwap = (high_p + low_p + last_p) / 3.0 # Aproximação se não houver VWAP nativo
                        if last_p <= 0.0 or vol_usd < self.min_volume_usd:
                            continue
                        normalized.append({
                            "symbol": sym,
                            "last_price": last_p,
                            "vwap_24h": vwap,
                            "high_24h": high_p,
                            "low_24h": low_p,
                            "volume_24h_usd": vol_usd,
                            "price_change_pct": chg_pct,
                            "source": "bitget",
                        })
                    except (ValueError, TypeError):
                        continue
                if normalized:
                    return normalized
        except Exception as e:
            logger.error(f"Falha também no fallback Bitget Tickers: {e}")

        return []

    def evaluate_ticker_opportunity(self, item: Dict[str, Any]) -> Optional[GoldenPotOpportunity]:
        """
        Analisa um ativo específico e determina se ele configura um Golden Pot
        para Scalp Reversion (15m) ou Swing Breakout (1h/4h).
        """
        sym = item["symbol"]
        last_p = item["last_price"]
        vwap = item["vwap_24h"]
        high_p = item["high_24h"]
        low_p = item["low_24h"]
        vol_usd = item["volume_24h_usd"]
        chg_pct = item["price_change_pct"]

        # 1. Métricas Derivadas
        dist_vwap_pct = ((last_p - vwap) / vwap) * 100.0 if vwap > 0 else 0.0
        range_span = max(high_p - low_p, 1e-8)
        range_pos_pct = ((last_p - low_p) / range_span) * 100.0
        amplitude_pct = ((high_p - low_p) / low_p) * 100.0 if low_p > 0 else 0.0

        # Normalização de Liquidez (volume acima de $100M ganha bônus de score)
        vol_score = min(vol_usd / 200_000_000.0, 1.0) * 25.0

        # =====================================================================
        # SETUP 1: SWING BREAKOUT (Momentum / Expansão Direcional)
        # Avaliado PRIMEIRO para que ativos em rompimento forte nos extremos de 24h
        # não sejam classificados prematuramente como exaustão (SCALP_REVERSION).
        # =====================================================================
        is_bull_breakout = (range_pos_pct >= 90.0 and chg_pct >= 3.5)
        is_bear_breakout = (range_pos_pct <= 10.0 and chg_pct <= -3.5)

        if is_bull_breakout or is_bear_breakout:
            direction = "BUY" if is_bull_breakout else "SELL"
            momentum_score = min(abs(chg_pct) * 5.0, 50.0)
            proximity_score = 25.0 if (range_pos_pct >= 95.0 or range_pos_pct <= 5.0) else 15.0
            total_score = round(min(momentum_score + proximity_score + vol_score, 98.0), 1)

            action_text = "Rompimento de máxima de 24h" if direction == "BUY" else "Perda de suporte na mínima de 24h"
            rationale = (
                f"Swing 1h/4h Momentum: {sym} em forte expansão ({chg_pct:+.2f}% em 24h). "
                f"{action_text} com volume institucional de ${vol_usd/1e6:.1f}M USD."
            )

            return GoldenPotOpportunity(
                symbol=sym,
                setup_type="SWING_BREAKOUT",
                direction=direction,
                score=total_score,
                last_price=last_p,
                dist_vwap_pct=round(dist_vwap_pct, 2),
                volume_24h_usd=vol_usd,
                price_change_pct_24h=round(chg_pct, 2),
                range_position_pct=round(range_pos_pct, 1),
                amplitude_pct=round(amplitude_pct, 2),
                suggested_timeframe="1h",
                rationale=rationale,
            )

        # =====================================================================
        # SETUP 2: SCALP REVERSION (Esticamento Elástico de VWAP)
        # =====================================================================
        if abs(dist_vwap_pct) >= 2.5 and amplitude_pct >= 4.0:
            direction = "BUY" if dist_vwap_pct < 0 else "SELL"
            stretch_score = min(abs(dist_vwap_pct) * 12.0, 50.0)
            amp_score = min(amplitude_pct * 2.0, 25.0)
            total_score = round(min(stretch_score + amp_score + vol_score, 99.0), 1)

            side_text = "Sobrevenda extrema abaixo do VWAP (retração rápida)" if direction == "BUY" else "Sobrecompra esticada acima do VWAP (exaustão)"
            rationale = (
                f"Scalp 15m Cirúrgico: {sym} descolou {dist_vwap_pct:+.2f}% do VWAP 24h. "
                f"{side_text} com ${vol_usd/1e6:.1f}M de volume."
            )

            return GoldenPotOpportunity(
                symbol=sym,
                setup_type="SCALP_REVERSION",
                direction=direction,
                score=total_score,
                last_price=last_p,
                dist_vwap_pct=round(dist_vwap_pct, 2),
                volume_24h_usd=vol_usd,
                price_change_pct_24h=round(chg_pct, 2),
                range_position_pct=round(range_pos_pct, 1),
                amplitude_pct=round(amplitude_pct, 2),
                suggested_timeframe="15m",
                rationale=rationale,
            )

        return None

    async def scan_market(self, force_refresh: bool = False, top_n: int = 5) -> Dict[str, Any]:
        """
        Executa a varredura completa do mercado e retorna o ranking dos Golden Pots.
        Respeita o TTL de cache em memória para evitar requisições redundantes.
        """
        now = time.time()
        if not force_refresh and self._cached_scan and (now - self._last_scan_time < self.cache_ttl_seconds):
            return self._cached_scan

        tickers = await self.fetch_market_universe_tickers()
        if not tickers:
            logger.warning("Nenhum ticker retornado na varredura de mercado.")
            if self._cached_scan:
                return self._cached_scan
            return {
                "timestamp": int(now * 1000),
                "total_scanned": 0,
                "liquid_symbols_count": 0,
                "top_scalps": [],
                "top_swings": [],
                "all_pots": [],
            }

        opportunities: List[GoldenPotOpportunity] = []
        for item in tickers:
            opp = self.evaluate_ticker_opportunity(item)
            if opp:
                opportunities.append(opp)

        # Ordenação por score decrescente
        scalp_pots = [o for o in opportunities if o.setup_type == "SCALP_REVERSION"]
        swing_pots = [o for o in opportunities if o.setup_type == "SWING_BREAKOUT"]

        scalp_pots.sort(key=lambda x: x.score, reverse=True)
        swing_pots.sort(key=lambda x: x.score, reverse=True)
        opportunities.sort(key=lambda x: x.score, reverse=True)

        top_scalps = [o.to_dict() for o in scalp_pots[:top_n]]
        top_swings = [o.to_dict() for o in swing_pots[:top_n]]
        all_pots = [o.to_dict() for o in opportunities[:top_n * 2]]

        result = {
            "timestamp": int(now * 1000),
            "total_scanned": len(tickers),
            "liquid_symbols_count": len(tickers),
            "top_scalps_count": len(top_scalps),
            "top_swings_count": len(top_swings),
            "top_scalps": top_scalps,
            "top_swings": top_swings,
            "all_pots": all_pots,
        }

        # Persiste os melhores Golden Pots no SQLite
        self._persist_pots_snapshot(opportunities[:top_n * 2])

        self._cached_scan = result
        self._last_scan_time = now
        logger.info(
            f"Varredura concluída: {len(tickers)} pares líquidos analisados | "
            f"{len(scalp_pots)} Golden Pots de Scalp | {len(swing_pots)} Golden Pots de Swing."
        )
        return result

    def _persist_pots_snapshot(self, pots: List[GoldenPotOpportunity]) -> None:
        """Grava os Golden Pots identificados no SQLite para telemetria sem lock."""
        if not pots:
            return
        try:
            with self._get_connection() as con:
                # Mantém apenas os últimos 500 registros para evitar crescimento desnecessário
                con.execute("DELETE FROM golden_pots WHERE timestamp < ?", (int(time.time() * 1000) - 86400000,))
                for pot in pots:
                    con.execute("""
                        INSERT INTO golden_pots (
                            symbol, setup_type, direction, score, last_price, dist_vwap_pct,
                            volume_24h_usd, price_change_pct_24h, range_position_pct, amplitude_pct,
                            suggested_timeframe, rationale, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pot.symbol, pot.setup_type, pot.direction, pot.score, pot.last_price,
                        pot.dist_vwap_pct, pot.volume_24h_usd, pot.price_change_pct_24h,
                        pot.range_position_pct, pot.amplitude_pct, pot.suggested_timeframe,
                        pot.rationale, pot.timestamp
                    ))
                con.commit()
        except Exception as e:
            logger.warning(f"Erro ao persistir golden_pots no SQLite: {e}")
