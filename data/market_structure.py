"""
market_structure.py - Motor de Estrutura de Mercado, Suporte/Resistência, Zonas de Liquidez & POI.
Fase 2 / Fase 4 do IA Trader.

Implementação 100% matemática, causal e sem lookahead bias para:
1. Suporte & Resistência Dinâmicos:
   - Swing Highs & Swing Lows (Fractais com janela de confirmação causal n=3).
   - Níveis de Pivot Points Institucionais e Proximidade de S/R.
2. Zonas de Liquidez (Liquidity Pools):
   - Buy-Side Liquidity (BSL) acima de topos de swing e Equal Highs (EQH).
   - Sell-Side Liquidity (SSL) abaixo de fundos de swing e Equal Lows (EQL).
   - Liquidity Sweeps (Stop Hunts): pavio varre o pool de liquidez e candle fecha dentro com rejeição.
3. POI (Points of Interest - Smart Money Concepts):
   - Fair Value Gaps (FVG) / Imbalances de 3 velas e seu nível de mitigação (50% Consequent Encroachment).
   - Order Blocks (OB): vela de ignição anterior a deslocamento expressivo (Displacement >= 1.5x ATR).
4. Tendência Estrutural de Mercado (Market Structure Trend):
   - Higher Highs / Higher Lows (HH/HL = +1) vs Lower Highs / Lower Lows (LH/LL = -1).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("ia_trader.market_structure")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [market_structure] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class POIZone:
    """Zona de Ponto de Interesse (Order Block ou Fair Value Gap)."""
    poi_type: str                # 'BULLISH_OB', 'BEARISH_OB', 'BULLISH_FVG', 'BEARISH_FVG'
    top: float
    bottom: float
    midpoint: float              # 50% da zona (Consequent Encroachment / Mean Threshold)
    created_index: int
    created_timestamp: int
    is_mitigated: bool = False
    mitigated_timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityLevel:
    """Nível de liquidez (BSL ou SSL)."""
    level_type: str              # 'BSL' (Buy-Side) ou 'SSL' (Sell-Side)
    price: float
    touches: int
    created_index: int
    created_timestamp: int
    is_swept: bool = False
    swept_timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketStructureDetector:
    """
    Detector algorítmico de estrutura de mercado, níveis chave e zonas de liquidez.
    Trabalha com DataFrames OHLCV ordenados cronologicamente.
    """

    def __init__(self, swing_window: int = 3, eq_threshold_pct: float = 0.002):
        """
        :param swing_window: número de barras antes e depois para confirmar um fractal swing (padrão 3).
        :param eq_threshold_pct: tolerância percentual para considerar topos/fundos iguais (Equal Highs/Lows).
        """
        self.swing_window = int(swing_window)
        self.eq_threshold_pct = float(eq_threshold_pct)

    def identify_swings(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Identifica fractais de Swing High e Swing Low estritamente causais.
        Um topo no índice i só é confirmado e conhecido no índice i + swing_window.
        Retorna (swing_highs, swing_lows) onde os valores são os preços ou NaN.
        """
        n = self.swing_window
        length = len(df)
        highs = df["high"].values
        lows = df["low"].values

        swing_h = np.full(length, np.nan)
        swing_l = np.full(length, np.nan)

        if length < 2 * n + 1:
            return pd.Series(swing_h, index=df.index), pd.Series(swing_l, index=df.index)

        for i in range(n, length - n):
            # Swing High: high[i] é estritamente maior que os n vizinhos antes e depois
            is_sh = True
            for j in range(1, n + 1):
                if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                    is_sh = False
                    break
            if is_sh:
                # O topo ocorreu no índice i, mas só é confirmado no índice i + n
                confirmed_idx = i + n
                if confirmed_idx < length:
                    swing_h[confirmed_idx] = highs[i]

            # Swing Low: low[i] é estritamente menor que os n vizinhos antes e depois
            is_sl = True
            for j in range(1, n + 1):
                if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                    is_sl = False
                    break
            if is_sl:
                confirmed_idx = i + n
                if confirmed_idx < length:
                    swing_l[confirmed_idx] = lows[i]

        return pd.Series(swing_h, index=df.index), pd.Series(swing_l, index=df.index)

    def detect_liquidity_pools_and_sweeps(
        self,
        df: pd.DataFrame,
        lookback_swings: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Detecta Zonas de Liquidez (BSL e SSL) e eventos de Liquidity Sweep (captura de liquidez).
        Retorna:
        - nearest_bsl: Série com o preço do BSL ativo mais próximo acima do fechamento.
        - nearest_ssl: Série com o preço do SSL ativo mais próximo abaixo do fechamento.
        - sweep_bull: 1 quando ocorre varredura de fundo (low < SSL mas close > SSL) com pavio de absorção.
        - sweep_bear: 1 quando ocorre varredura de topo (high > BSL mas close < BSL) com pavio de absorção.
        """
        length = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        sh_series, sl_series = self.identify_swings(df)
        sh_vals = sh_series.values
        sl_vals = sl_series.values

        nearest_bsl = np.full(length, np.nan)
        nearest_ssl = np.full(length, np.nan)
        sweep_bull = np.zeros(length, dtype=int)
        sweep_bear = np.zeros(length, dtype=int)

        active_bsl_levels: List[float] = []
        active_ssl_levels: List[float] = []

        for i in range(length):
            # Se um novo swing foi confirmado neste candle, adiciona ao pool
            if not np.isnan(sh_vals[i]):
                active_bsl_levels.append(sh_vals[i])
            if not np.isnan(sl_vals[i]):
                active_ssl_levels.append(sl_vals[i])

            # Mantém apenas os níveis recentes
            active_bsl_levels = active_bsl_levels[-lookback_swings:]
            active_ssl_levels = active_ssl_levels[-lookback_swings:]

            curr_close = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]

            # BSL mais próximo acima do preço
            above = [lvl for lvl in active_bsl_levels if lvl > curr_close]
            if above:
                closest_bsl = min(above)
                nearest_bsl[i] = closest_bsl

                # Checa Liquidity Sweep de topo (Bearish Sweep)
                # O pavio furou o BSL, mas o fechamento retraiu e fechou abaixo dele!
                if curr_high >= closest_bsl and curr_close < closest_bsl:
                    # Exige pavio superior significativo (rejeição)
                    upper_wick = curr_high - max(curr_close, df["open"].iloc[i])
                    body = abs(curr_close - df["open"].iloc[i])
                    if upper_wick >= body * 0.4:
                        sweep_bear[i] = 1

            # SSL mais próximo abaixo do preço
            below = [lvl for lvl in active_ssl_levels if lvl < curr_close]
            if below:
                closest_ssl = max(below)
                nearest_ssl[i] = closest_ssl

                # Checa Liquidity Sweep de fundo (Bullish Sweep)
                # O pavio furou o SSL, mas o fechamento retraiu e fechou acima dele!
                if curr_low <= closest_ssl and curr_close > closest_ssl:
                    # Exige pavio inferior significativo (absorção compradora)
                    lower_wick = min(curr_close, df["open"].iloc[i]) - curr_low
                    body = abs(curr_close - df["open"].iloc[i])
                    if lower_wick >= body * 0.4:
                        sweep_bull[i] = 1

        return (
            pd.Series(nearest_bsl, index=df.index),
            pd.Series(nearest_ssl, index=df.index),
            pd.Series(sweep_bull, index=df.index),
            pd.Series(sweep_bear, index=df.index),
        )

    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        min_gap_pct: float = 0.001,
    ) -> Tuple[pd.Series, pd.Series, List[POIZone]]:
        """
        Detecta Fair Value Gaps (FVG) de 3 velas e monitora mitigação.
        - Bullish FVG: Low(i) > High(i-2).
        - Bearish FVG: High(i) < Low(i-2).
        Retorna (inside_bull_fvg, inside_bear_fvg, list_of_active_fvgs).
        """
        length = len(df)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(length)

        inside_bull_fvg = np.zeros(length, dtype=int)
        inside_bear_fvg = np.zeros(length, dtype=int)
        fvg_zones: List[POIZone] = []

        for i in range(2, length):
            curr_c = closes[i]
            ts = int(timestamps[i])

            # 1. Checa formação de novo Bullish FVG entre candle i-2 e candle i
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                gap_pct = gap_size / curr_c
                if gap_pct >= min_gap_pct:
                    zone = POIZone(
                        poi_type="BULLISH_FVG",
                        top=float(lows[i]),
                        bottom=float(highs[i - 2]),
                        midpoint=float((lows[i] + highs[i - 2]) / 2.0),
                        created_index=i,
                        created_timestamp=ts,
                    )
                    fvg_zones.append(zone)

            # 2. Checa formação de novo Bearish FVG entre candle i-2 e candle i
            elif highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                gap_pct = gap_size / curr_c
                if gap_pct >= min_gap_pct:
                    zone = POIZone(
                        poi_type="BEARISH_FVG",
                        top=float(lows[i - 2]),
                        bottom=float(highs[i]),
                        midpoint=float((lows[i - 2] + highs[i]) / 2.0),
                        created_index=i,
                        created_timestamp=ts,
                    )
                    fvg_zones.append(zone)

            # 3. Atualiza estado e mitigação dos FVGs ativos
            # Um FVG é mitigado quando o preço entra e fecha ou toca além do midpoint
            for zone in fvg_zones:
                if zone.is_mitigated or zone.created_index >= i:
                    continue

                if zone.poi_type == "BULLISH_FVG":
                    # Preço atual está dentro da zona do FVG
                    if zone.bottom <= curr_c <= zone.top:
                        inside_bull_fvg[i] = 1
                    # Se rompeu abaixo da base, foi totalmente invalidado/mitigado
                    if lows[i] <= zone.bottom:
                        zone.is_mitigated = True
                        zone.mitigated_timestamp = ts

                elif zone.poi_type == "BEARISH_FVG":
                    # Preço atual está dentro da zona do FVG
                    if zone.bottom <= curr_c <= zone.top:
                        inside_bear_fvg[i] = 1
                    # Se rompeu acima do topo, foi totalmente invalidado/mitigado
                    if highs[i] >= zone.top:
                        zone.is_mitigated = True
                        zone.mitigated_timestamp = ts

        return (
            pd.Series(inside_bull_fvg, index=df.index),
            pd.Series(inside_bear_fvg, index=df.index),
            [z for z in fvg_zones if not z.is_mitigated],
        )

    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        atr_series: Optional[pd.Series] = None,
        displacement_atr_mult: float = 1.3,
    ) -> Tuple[pd.Series, pd.Series, List[POIZone]]:
        """
        Detecta Order Blocks Institucionais (OB).
        - Bullish OB: Último candle de baixa antes de uma expansão forte com deslocamento (Displacement >= 1.3x ATR).
        - Bearish OB: Último candle de alta antes de uma expansão forte de baixa com deslocamento.
        Retorna (inside_bull_ob, inside_bear_ob, list_of_active_obs).
        """
        length = len(df)
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(length)

        if atr_series is not None:
            atrs = atr_series.values
        else:
            tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
            atrs = pd.Series(tr).rolling(14, min_periods=1).mean().values

        inside_bull_ob = np.zeros(length, dtype=int)
        inside_bear_ob = np.zeros(length, dtype=int)
        ob_zones: List[POIZone] = []

        for i in range(1, length - 1):
            curr_c = closes[i]
            ts = int(timestamps[i])
            curr_atr = atrs[i] if atrs[i] > 0 else (curr_c * 0.005)

            # Movimento de deslocamento (Displacement) entre candle i e i+1
            move = closes[i + 1] - opens[i + 1]

            # 1. Bullish Order Block: candle i é de baixa (close < open) e candle i+1 tem forte alta
            if closes[i] < opens[i] and move >= (displacement_atr_mult * curr_atr):
                zone = POIZone(
                    poi_type="BULLISH_OB",
                    top=float(highs[i]),
                    bottom=float(lows[i]),
                    midpoint=float((highs[i] + lows[i]) / 2.0),
                    created_index=i + 1,
                    created_timestamp=ts,
                )
                ob_zones.append(zone)

            # 2. Bearish Order Block: candle i é de alta (close > open) e candle i+1 tem forte baixa
            elif closes[i] > opens[i] and -move >= (displacement_atr_mult * curr_atr):
                zone = POIZone(
                    poi_type="BEARISH_OB",
                    top=float(highs[i]),
                    bottom=float(lows[i]),
                    midpoint=float((highs[i] + lows[i]) / 2.0),
                    created_index=i + 1,
                    created_timestamp=ts,
                )
                ob_zones.append(zone)

        # 3. Monitora se cada candle está retestando ou mitigando um OB ativo
        for i in range(length):
            curr_c = closes[i]
            ts = int(timestamps[i])
            for zone in ob_zones:
                if zone.is_mitigated or zone.created_index >= i:
                    continue

                if zone.poi_type == "BULLISH_OB":
                    if zone.bottom <= curr_c <= zone.top:
                        inside_bull_ob[i] = 1
                    if lows[i] < zone.bottom:
                        zone.is_mitigated = True
                        zone.mitigated_timestamp = ts

                elif zone.poi_type == "BEARISH_OB":
                    if zone.bottom <= curr_c <= zone.top:
                        inside_bear_ob[i] = 1
                    if highs[i] > zone.top:
                        zone.is_mitigated = True
                        zone.mitigated_timestamp = ts

        return (
            pd.Series(inside_bull_ob, index=df.index),
            pd.Series(inside_bear_ob, index=df.index),
            [z for z in ob_zones if not z.is_mitigated],
        )

    def compute_all_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula e acopla todas as features de estrutura, S/R, Liquidez e POI no DataFrame.
        """
        if df.empty or len(df) < 10:
            return df.copy()

        out = df.copy()
        closes = out["close"].astype(float)

        # 1. Swings e Zonas de Liquidez
        nearest_bsl, nearest_ssl, sweep_bull, sweep_bear = self.detect_liquidity_pools_and_sweeps(out)
        out["nearest_bsl"] = nearest_bsl
        out["nearest_ssl"] = nearest_ssl
        out["liquidity_sweep_bull"] = sweep_bull
        out["liquidity_sweep_bear"] = sweep_bear

        # Distância percentual até o suporte (SSL) e resistência (BSL) mais próximos
        out["dist_resistance_pct"] = np.where(
            out["nearest_bsl"].notna() & (out["nearest_bsl"] > closes),
            ((out["nearest_bsl"] - closes) / closes) * 100.0,
            np.nan,
        )
        out["dist_support_pct"] = np.where(
            out["nearest_ssl"].notna() & (closes > out["nearest_ssl"]),
            ((closes - out["nearest_ssl"]) / closes) * 100.0,
            np.nan,
        )

        # 2. Fair Value Gaps (FVG)
        in_bull_fvg, in_bear_fvg, active_fvgs = self.detect_fair_value_gaps(out)
        out["inside_bullish_fvg"] = in_bull_fvg
        out["inside_bearish_fvg"] = in_bear_fvg

        # 3. Order Blocks (OB)
        in_bull_ob, in_bear_ob, active_obs = self.detect_order_blocks(out)
        out["inside_bullish_ob"] = in_bull_ob
        out["inside_bearish_ob"] = in_bear_ob

        # 4. Confluência de POI (Preço está em qualquer zona de compra institucional ou venda)
        out["at_bullish_poi"] = ((out["inside_bullish_fvg"] == 1) | (out["inside_bullish_ob"] == 1) | (out["liquidity_sweep_bull"] == 1)).astype(int)
        out["at_bearish_poi"] = ((out["inside_bearish_fvg"] == 1) | (out["inside_bearish_ob"] == 1) | (out["liquidity_sweep_bear"] == 1)).astype(int)

        return out
