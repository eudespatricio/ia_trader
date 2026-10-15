"""
scripts/run_paper_simulation.py - Daemon Autônomo de Simulação em Tempo Real (Paper Trading).
Executa o ciclo contínuo de trading simulado com microestrutura institucional:
1. Carry Arbitrage em STANDBY (conforme diretriz estratégica).
2. Foco Total em:
   - Scalper Cirúrgico Intradiário (15m): saídas parciais (TP1 50% + SL Breakeven + TP2),
     execução Maker (0.02%) e cota diária rigorosa (máximo 5 scalps/dia).
   - Swing Trade de Momentum (1h/4h): rompimentos com surto de volume e tendência macro.
   - Radar Golden Pots integrado para descoberta dinâmica dos melhores pares do mercado.
3. Persistência contínua em SQLite (market_cache.db) refletida no Dashboard Web em tempo real.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from agents.golden_pot_agent import GoldenPotAgent
from signal_generator import TradeSignal
from smart_trade import PaperTradingEngine
from leverage_policy import decide_leverage

logger = logging.getLogger("ia_trader.paper_simulation")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [paper_daemon] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


SWING_CANDLE_LIMIT = 400          # velas 1h buscadas (sma_200 + folga após dropna)
SWING_MIN_WARMUP_CANDLES = 250    # mínimo bruto para sma_200 válida em ao menos 50 linhas

# --- GOLDEN_POT v2 (ADDENDUM E5, 2026-10-15) -------------------------------------------------
GP_BAR_MS = 15 * 60 * 1000          # duração da vela 15m (chave de "1 entrada por vela" e cooldown)
GP_CANDLE_LIMIT = 400               # velas 15m buscadas (binance -> fallback bitget)
GP_MIN_WARMUP_CANDLES = 250         # mínimo bruto; além disso exige sma_200 válida (exclui listagens novas)
GP_MAX_CANDIDATES = 4               # candidatos do scanner avaliados por ciclo
GP_COOLDOWN_BARS = 16               # sem reentrada no símbolo por N barras 15m após qualquer fechamento
GP_MAX_ENTRIES_PER_BAR = 1          # no máximo 1 nova entrada GP por vela 15m (qualquer símbolo)
GP_MIN_COMMITTEE_CONFIDENCE = 0.70  # mesma barra de confiança do swing
GP_KNIFE_RSI_LOW = 20.0             # RSI abaixo => "faca caindo" (LONG) se close < sma_200
GP_KNIFE_RSI_HIGH = 80.0            # simétrico (SHORT) se close > sma_200
GP_KNIFE_ATR_MOVE = 3.0             # movimento > X ATR na última hora também caracteriza tendência forte
GP_KNIFE_LOOKBACK_BARS = 4          # 4 barras de 15m = 1h
GP_TP1_ATR_MULT = 1.2
GP_TP2_ATR_MULT = 2.5
GP_SL_ATR_MULT = 1.0
GP_LOW_LIQUIDITY_USD = 50_000_000.0  # volume 24h abaixo disso => alavancagem forçada a 1x
GP_QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "PERP")
# Lista de exclusão configurável: ações/ETFs/índices tokenizados (perps TradFi) - fora do universo GP,
# pois não seguem a dinâmica cripto 24/7 (gaps de sessão, liquidez fina fora do pregão).
# Casamento por ativo-base exato (após remover o sufixo da cotação). Edite aqui para ampliar.
GP_EXCLUDED_BASE_ASSETS = frozenset({
    "SKHYNIX", "SKHY", "SAMSUNG", "KORU", "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX", "AMD", "INTC",
    "COIN", "MSTR", "HOOD", "PLTR", "BABA", "TSM", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO",
})


class PaperTradingDaemon:
    """
    Daemon de execução contínua em Paper Trading.
    Supervisiona posições abertas, varre o mercado e despacha ordens simuladas.
    """

    def __init__(
        self,
        poll_interval_seconds: int = 15,
        engine: Optional[PaperTradingEngine] = None,
        loader: Optional[DataLoader] = None,
        fe: Optional[FeatureEngineer] = None,
        golden_pot_agent: Optional[GoldenPotAgent] = None,
    ):
        self.poll_interval = poll_interval_seconds
        self.engine = engine or PaperTradingEngine()
        self.loader = loader or DataLoader()
        self.fe = fe or FeatureEngineer()
        self.scanner = golden_pot_agent or GoldenPotAgent(min_volume_usd=20_000_000.0)
        self.is_running = False

        # Pares fixos monitorados continuamente
        self.core_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def _compute_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features com warmup explícito: descarta as linhas em que sma_200 ainda não existia
        (o fillna(0.0) de drop_na=False as marca com 0.0). Evita o dropna global de
        compute_all_features(drop_na=True), que zera TODO o DataFrame em tendências
        monotônicas (colunas de estrutura nearest_ssl/bsl ficam NaN em toda a janela).
        """
        feat = self.fe.compute_all_features(df, drop_na=False)
        if feat.empty or "sma_200" not in feat.columns:
            return pd.DataFrame()
        valid = np.isfinite(feat["sma_200"].astype(float)) & (feat["sma_200"].astype(float) > 0.0)
        return feat[valid].reset_index(drop=True)

    @staticmethod
    def _has_valid_swing_context(df_feat: pd.DataFrame) -> bool:
        """True se a última linha tem sma_200 finita e > 0 (sem fallback 0.0 de séries curtas)."""
        if df_feat is None or df_feat.empty or "sma_200" not in df_feat.columns:
            return False
        v = df_feat["sma_200"].iloc[-1]
        return bool(np.isfinite(v) and float(v) > 0.0)

    async def update_active_positions(self) -> None:
        """Atualiza todas as posições abertas com os preços de mercado mais recentes em paralelo."""
        open_positions = list(self.engine.positions.values())
        if not open_positions:
            return

        symbols_to_update = list({p.symbol for p in open_positions if p.status == "OPEN"})

        async def _update_symbol(sym: str) -> None:
            try:
                df = await self.loader.get_candles(sym, "15m", limit=5, exchange="binance")
                if df.empty:
                    df = await self.loader.get_candles(sym, "15m", limit=5, exchange="bitget")
                if not df.empty:
                    latest = df.iloc[-1]
                    high = float(latest["high"])
                    low = float(latest["low"])
                    close = float(latest["close"])
                    ts = int(latest["timestamp"])

                    triggered = self.engine.update_market_tick(
                        symbol=sym,
                        high=high,
                        low=low,
                        close=close,
                        timestamp=ts,
                    )
                    for t in triggered:
                        logger.info(f"[{sym}] GATILHO SIMULADO: {t.action} @ ${t.price:,.2f} | PnL: ${t.pnl:+.2f}")
            except Exception as e:
                logger.debug(f"Aviso ao atualizar posições para {sym}: {e}")

        await asyncio.gather(*[_update_symbol(s) for s in symbols_to_update], return_exceptions=True)

    # ------------------------------------------------------------------
    # GOLDEN_POT v2 - helpers (ADDENDUM E5)
    # ------------------------------------------------------------------
    @staticmethod
    def _gp_base_asset(symbol: str) -> str:
        s = symbol.upper().replace("-", "").replace("_", "").replace("/", "")
        for q in GP_QUOTE_SUFFIXES:
            if s.endswith(q) and len(s) > len(q):
                return s[: -len(q)]
        return s

    @classmethod
    def is_gp_excluded_symbol(cls, symbol: str) -> bool:
        """True para perps de ações/ETFs tokenizados (lista GP_EXCLUDED_BASE_ASSETS)."""
        return cls._gp_base_asset(symbol) in GP_EXCLUDED_BASE_ASSETS

    @staticmethod
    def _regime_aligned(direction: str, latest: pd.Series) -> Optional[bool]:
        """True se a entrada é a favor da tendência (close vs sma_200); None se indisponível."""
        try:
            close, sma = float(latest["close"]), float(latest["sma_200"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (np.isfinite(close) and np.isfinite(sma)) or sma <= 0.0:
            return None
        return (close >= sma) if direction == "BUY" else (close <= sma)

    def _gp_entered_bar(self, bar_id: int) -> bool:
        """True se já houve uma entrada GP nesta vela de 15m (lê o ledger -> sobrevive a restart)."""
        n = sum(
            1 for t in self.engine.trade_history
            if t.action.startswith("OPEN_GOLDEN_POT_") and int(PaperTradingEngine._ts_ms(t.timestamp)) // GP_BAR_MS == bar_id
        )
        return n >= GP_MAX_ENTRIES_PER_BAR

    def _gp_in_cooldown(self, symbol: str, bar_id: int) -> bool:
        """True se o símbolo teve fechamento (qualquer saída total) há < GP_COOLDOWN_BARS velas de 15m.

        TP1_PARTIAL não é fechamento. Lê trade_history (restaurado do ledger no restart).
        """
        for t in self.engine.trade_history:
            if t.symbol != symbol or t.action.startswith("OPEN_") or t.action == "TP1_PARTIAL":
                continue
            if bar_id - int(PaperTradingEngine._ts_ms(t.timestamp)) // GP_BAR_MS < GP_COOLDOWN_BARS:
                return True
        return False

    @staticmethod
    def gp_falling_knife_blocked(direction: str, df_feat: pd.DataFrame) -> Optional[str]:
        """Filtro de regime / faca-caindo. Retorna motivo do bloqueio ou None.

        Tendência forte contra a entrada = (close vs sma_200) E (RSI extremo OU movimento > GP_KNIFE_ATR_MOVE
        ATRs na última hora). Nesse caso só entra com confirmação de reversão (vela fechada a favor
        e RSI virando); caso contrário bloqueia. Simétrico para SHORT.
        """
        if len(df_feat) < GP_KNIFE_LOOKBACK_BARS + 2:
            return "histórico insuficiente para filtro de regime"
        last, prev = df_feat.iloc[-1], df_feat.iloc[-2]
        close, sma = float(last["close"]), float(last["sma_200"])
        rsi, rsi_prev = float(last.get("rsi_14", 50.0)), float(prev.get("rsi_14", 50.0))
        atr = max(float(last.get("atr_14", 0.0)), 1e-12)
        move_atr = (close - float(df_feat["close"].iloc[-1 - GP_KNIFE_LOOKBACK_BARS])) / atr
        if direction == "BUY":
            strong = close < sma and (rsi < GP_KNIFE_RSI_LOW or move_atr < -GP_KNIFE_ATR_MOVE)
            confirmed = close > float(last["open"]) and rsi > rsi_prev
        else:
            strong = close > sma and (rsi > GP_KNIFE_RSI_HIGH or move_atr > GP_KNIFE_ATR_MOVE)
            confirmed = close < float(last["open"]) and rsi < rsi_prev
        if strong and not confirmed:
            return f"faca caindo: tendência forte contra {direction} (RSI {rsi:.1f}, mov {move_atr:+.1f} ATR) sem reversão"
        return None

    async def _gp_fetch_closed_candles(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        df = await self.loader.get_candles(sym, "15m", limit=GP_CANDLE_LIMIT, exchange="binance")
        if len(df) < GP_MIN_WARMUP_CANDLES:
            df = await self.loader.get_candles(sym, "15m", limit=GP_CANDLE_LIMIT, exchange="bitget")
        if len(df) < GP_MIN_WARMUP_CANDLES:
            return None
        # Somente velas FECHADAS: descarta a última se ainda em formação (timestamp = abertura)
        if int(df["timestamp"].iloc[-1]) + GP_BAR_MS > now_ts:
            df = df.iloc[:-1]
        return df if len(df) >= GP_MIN_WARMUP_CANDLES else None

    async def evaluate_golden_pot_opportunities(self) -> None:
        """
        GOLDEN_POT v2 (ADDENDUM E5). O GoldenPotAgent é só o gatilho de DESCOBERTA; a entrada exige:
        símbolo fora da lista de exclusão, sma_200 válida (>=400 velas 15m fechadas), 1 entrada GP/vela,
        cooldown por símbolo, filtro de faca-caindo e concordância do comitê completo
        (core.evaluate_market_async) na mesma direção, sem veto do RiskSentinel.
        confidence/consensus_score/risk_dollars vêm do TradeSignal real. Cota diária própria (máx 3/dia).
        """
        now_ts = int(time.time() * 1000)
        can_exec, reason = self.engine.can_execute_golden_pot(now_ts)
        if not can_exec:
            return  # Teto diário de Golden Pots atingido

        settings = self.engine.get_simulation_settings()
        op_mode = settings.get("operation_mode", "INSTITUTIONAL")
        vwap_pot_thresh = 1.2 if op_mode == "DEMO" else 2.5
        rsi_pot_os = 42.0 if op_mode == "DEMO" else 38.0
        rsi_pot_ob = 58.0 if op_mode == "DEMO" else 62.0

        scan_res = await self.scanner.scan_market(force_refresh=False)
        candidates = [
            p for p in scan_res.get("top_scalps", [])
            if not self.is_gp_excluded_symbol(str(p.get("symbol", "")))
        ][:GP_MAX_CANDIDATES]
        if not candidates:
            return

        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            return  # Margem livre insuficiente

        for pot in candidates:
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                break

            can_exec, _ = self.engine.can_execute_golden_pot(now_ts)
            if not can_exec:
                break

            sym = pot["symbol"]
            # Se já há posição aberta neste par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                df = await self._gp_fetch_closed_candles(sym, now_ts)
                if df is None:
                    logger.debug(f"[{sym}] GP ignorado: < {GP_MIN_WARMUP_CANDLES} velas 15m (listagem nova/sem warmup).")
                    continue

                df_feat = self._compute_swing_features(df)  # exige sma_200 válida
                if df_feat.empty or not self._has_valid_swing_context(df_feat):
                    logger.debug(f"[{sym}] GP ignorado: sma_200 sem warmup válido.")
                    continue

                latest = df_feat.iloc[-1]
                ts = int(latest["timestamp"])
                bar_id = ts // GP_BAR_MS
                # Limite de 1 entrada GP por vela de 15m (qualquer símbolo) e cooldown por símbolo
                if self._gp_entered_bar(bar_id):
                    break
                if self._gp_in_cooldown(sym, bar_id):
                    logger.debug(f"[{sym}] GP em cooldown ({GP_COOLDOWN_BARS} barras).")
                    continue

                close = float(latest["close"])
                vwap_dist = float(latest.get("dist_vwap_pct", 0.0))
                rsi = float(latest.get("rsi_14", 50.0))
                atr = float(latest.get("atr_14", close * 0.008))

                # Confluências de POI e Zonas de Liquidez
                sweep_bull = int(latest.get("liquidity_sweep_bull", 0))
                sweep_bear = int(latest.get("liquidity_sweep_bear", 0))
                at_bull_poi = int(latest.get("at_bullish_poi", 0))
                at_bear_poi = int(latest.get("at_bearish_poi", 0))

                # Gatilho de descoberta: Desvio expressivo de VWAP OU Liquidity Sweep confirmado em POI
                direction = None
                rationale_extra = []

                if (vwap_dist <= -vwap_pot_thresh and rsi <= rsi_pot_os) or (sweep_bull == 1 and at_bull_poi == 1):
                    direction = "BUY"
                    if sweep_bull == 1: rationale_extra.append("Liquidity Sweep de Fundo (SSL)")
                    if at_bull_poi == 1: rationale_extra.append("Em POI Institucional (OB/FVG)")
                elif (vwap_dist >= vwap_pot_thresh and rsi >= rsi_pot_ob) or (sweep_bear == 1 and at_bear_poi == 1):
                    direction = "SELL"
                    if sweep_bear == 1: rationale_extra.append("Liquidity Sweep de Topo (BSL)")
                    if at_bear_poi == 1: rationale_extra.append("Em POI Institucional (OB/FVG)")

                if not direction:
                    continue

                # Filtro de regime / faca-caindo
                blocked = self.gp_falling_knife_blocked(direction, df_feat)
                if blocked:
                    logger.info(f"[{sym}] GP bloqueado: {blocked}")
                    continue

                # Comitê completo (mesmo caminho de decisão do swing, em 15m) - confirmação obrigatória
                committee = await self.engine.core.evaluate_market_async(
                    context=latest.to_dict(), symbol=sym, current_price=close, atr=atr, timestamp=ts,
                )
                if committee.is_vetoed:
                    logger.info(f"[{sym}] GP bloqueado: veto do RiskSentinel.")
                    continue
                if committee.direction != direction:
                    logger.info(f"[{sym}] GP bloqueado: comitê diverge (gatilho {direction} x comitê {committee.direction}).")
                    continue
                if committee.confidence < GP_MIN_COMMITTEE_CONFIDENCE:
                    logger.info(f"[{sym}] GP bloqueado: confiança do comitê {committee.confidence:.2f} < {GP_MIN_COMMITTEE_CONFIDENCE}.")
                    continue

                # Lote = MARGEM (paper_lot_usd, lido do DB em tempo real); notional = margem x alavancagem
                # decidida pelo Smart Trader com a confiança REAL do comitê. risk_dollars = size * dist. SL.
                lev = decide_leverage(
                    committee.confidence,
                    atr_pct=atr / close if close > 0 else 0.0,
                    regime_alignment=self._regime_aligned(direction, latest),
                    is_vetoed=committee.is_vetoed,
                    low_history_or_liquidity=float(pot.get("volume_24h_usd", 0.0) or 0.0) < GP_LOW_LIQUIDITY_USD,
                )
                margin = min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95)
                notional = margin * lev
                sl_dist = GP_SL_ATR_MULT * atr
                extra_str = f" | {', '.join(rationale_extra)}" if rationale_extra else ""
                sig = replace(
                    committee,
                    entry_price=close,
                    stop_loss=close - sl_dist if direction == "BUY" else close + sl_dist,
                    take_profit=close + GP_TP2_ATR_MULT * atr if direction == "BUY" else close - GP_TP2_ATR_MULT * atr,
                    atr=atr,
                    position_size=notional,
                    risk_dollars=(notional / close) * sl_dist,
                    rationale=f"Golden Pot v2: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f}{extra_str} | {committee.rationale}",
                )
                pos = self.engine.execute_golden_pot_signal(
                    signal=sig,
                    current_price=close,
                    timestamp=ts,
                    tp1_atr_mult=GP_TP1_ATR_MULT,
                    tp2_atr_mult=GP_TP2_ATR_MULT,
                    sl_atr_mult=GP_SL_ATR_MULT,
                    leverage=lev,
                )
                if pos:
                    logger.info(
                        f"[{sym}] 🌟 GOLDEN POT v2 DISPARADO: {direction} @ ${close:,.2f} | "
                        f"conf {sig.confidence:.2f} cons {sig.consensus_score:+.2f} | lev {pos.leverage:g}x | "
                        f"Cota Golden Pots: {self.engine.daily_golden_pots_count}/{self.engine.max_daily_golden_pots}"
                    )
                    break  # no máximo 1 entrada GP por vela
            except Exception as e:
                logger.debug(f"Aviso ao avaliar golden pot para {sym}: {e}")

    async def evaluate_scalp_opportunities(self) -> None:
        """
        Avalia o mercado para scalps padrões em pares CORE (ETH, SOL, BTC).
        Consome a cota diária de scalps (mínimo 2, máximo 5/dia).
        """
        now_ts = int(time.time() * 1000)
        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            return  # Margem livre insuficiente

        can_exec, reason = self.engine.can_execute_scalp(now_ts)
        if not can_exec:
            return  # Teto diário de 5 scalps atingido

        for sym in ["ETHUSDT", "SOLUSDT", "BTCUSDT"]:
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                break

            can_exec, _ = self.engine.can_execute_scalp(now_ts)
            if not can_exec:
                break

            # Se já há scalp aberto no par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                df = await self.loader.get_candles(sym, "15m", limit=100, exchange="binance")
                if len(df) < 50:
                    df = await self.loader.get_candles(sym, "15m", limit=100, exchange="bitget")
                if len(df) < 50:
                    continue

                df_feat = self.fe.compute_all_features(df, drop_na=True)
                if df_feat.empty:
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                vwap_dist = float(latest.get("dist_vwap_pct", 0.0))
                rsi = float(latest.get("rsi_14", 50.0))
                atr = float(latest.get("atr_14", close * 0.005))
                vol_z = float(latest.get("volume_zscore_20", 0.0))

                settings = self.engine.get_simulation_settings()
                vwap_dev = float(settings.get("scalp_vwap_dev_pct", 2.0))
                rsi_os = float(settings.get("scalp_rsi_oversold", 35.0))
                rsi_ob = float(settings.get("scalp_rsi_overbought", 65.0))
                vol_z_min = float(settings.get("scalp_volume_z", 0.2))

                direction = None
                if vwap_dist <= -vwap_dev and rsi <= rsi_os and vol_z >= vol_z_min:
                    direction = "BUY"
                elif vwap_dist >= vwap_dev and rsi >= rsi_ob and vol_z >= vol_z_min:
                    direction = "SELL"

                if direction:
                    sig = TradeSignal(
                        signal_id=f"scalp-{sym}-{int(time.time())}",
                        timestamp=int(latest["timestamp"]),
                        symbol=sym,
                        direction=direction,
                        entry_price=close,
                        stop_loss=close - (0.9 * atr) if direction == "BUY" else close + (0.9 * atr),
                        take_profit=close + (1.8 * atr) if direction == "BUY" else close - (1.8 * atr),
                        atr=atr,
                        confidence=0.85,
                        consensus_score=0.80 if direction == "BUY" else -0.80,
                        position_size=min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95, self.engine.balance * 0.25),
                        risk_dollars=min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95, self.engine.balance * 0.25) / close * 0.9 * atr,
                        rationale=f"Scalp Padrão Core: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f} | Vol Z {vol_z:+.1f}",
                    )
                    pos = self.engine.execute_scalp_signal(
                        signal=sig,
                        current_price=close,
                        timestamp=int(latest["timestamp"]),
                        tp1_atr_mult=1.0,
                        tp2_atr_mult=1.8,
                        sl_atr_mult=0.9,
                    )
                    if pos:
                        logger.info(
                            f"[{sym}] ⚡ SCALP PADRÃO DISPARADO: {direction} @ ${close:,.2f} | "
                            f"Cota Scalps: {self.engine.daily_scalps_count}/5"
                        )
            except Exception as e:
                logger.debug(f"Aviso ao avaliar scalp padrão para {sym}: {e}")

    async def evaluate_swing_opportunities(self) -> None:
        """Avalia oportunidades de Swing Trade (1h/4h) com o Comitê de Decisão."""
        for sym in self.core_symbols:
            if any(p.symbol == sym and p.strategy_type == "DIRECTIONAL" and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                # BIAS: sma_200 exige >=200 velas válidas após o dropna -> busca 400 candles e
                # pula o símbolo se o warmup não for satisfeito (nunca opera com sma_200=0.0).
                df = await self.loader.get_candles(sym, "1h", limit=SWING_CANDLE_LIMIT, exchange="binance")
                if len(df) < SWING_MIN_WARMUP_CANDLES:
                    df = await self.loader.get_candles(sym, "1h", limit=SWING_CANDLE_LIMIT, exchange="bitget")
                if len(df) < SWING_MIN_WARMUP_CANDLES:
                    logger.debug(f"[{sym}] Swing ignorado: {len(df)} velas < warmup {SWING_MIN_WARMUP_CANDLES}.")
                    continue

                df_feat = self._compute_swing_features(df)
                if df_feat.empty or not self._has_valid_swing_context(df_feat):
                    logger.debug(f"[{sym}] Swing ignorado: features inválidas (sma_200 ausente/zero).")
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                ts = int(latest["timestamp"])

                eval_res = self.engine.core.evaluate_multi_strategy(
                    symbol=sym,
                    current_price=close,
                    context=latest.to_dict(),
                    annualized_funding_rate=0.0, # Carry em standby
                    timestamp=ts,
                )

                dir_sig = eval_res.get("directional_signal", {})
                if dir_sig.get("direction") in ["BUY", "SELL"] and not dir_sig.get("is_vetoed", False):
                    # Exige confiança institucional (>= 0.70)
                    if dir_sig.get("confidence", 0.0) >= 0.70:
                        # Contrato: TradeSignal.position_size vem do core em UNIDADES do ativo;
                        # o engine interpreta como NOTIONAL em USD -> converte aqui (senão
                        # BTC 0.55 virava $0.55 e a ordem era rejeitada por margem < $10).
                        # Lote = margem (paper_lot_usd); notional = margem x alavancagem do Smart Trader.
                        sl_dist = abs(close - float(dir_sig.get("stop_loss", close)))
                        lev = decide_leverage(
                            float(dir_sig.get("confidence", 0.0)),
                            atr_pct=float(latest.get("atr_14", 0.0)) / close if close > 0 else 0.0,
                            regime_alignment=self._regime_aligned(dir_sig["direction"], latest),
                            is_vetoed=bool(dir_sig.get("is_vetoed", False)),
                        )
                        notional = min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95) * lev
                        dir_sig = {**dir_sig, "position_size": notional, "risk_dollars": (notional / close) * sl_dist}
                        sig = TradeSignal(**dir_sig)
                        pos = self.engine.execute_directional_signal(sig, current_price=close, timestamp=ts, leverage=lev)
                        if pos:
                            logger.info(f"[{sym}] SWING TRADE DISPARADO: {sig.direction} @ ${close:,.2f}")
            except Exception as e:
                logger.debug(f"Aviso ao avaliar swing para {sym}: {e}")

    async def run_loop(self, max_cycles: int = 0) -> None:
        """Loop principal do daemon em tempo real."""
        self.is_running = True
        logger.info("================================================================================")
        logger.info("🚀 MOTOR DE PAPER TRADING INICIADO EM MODO DEMO / SIMULAÇÃO CONTÍNUA")
        logger.info(f"   Saldo Inicial: ${self.engine.balance:,.2f} USD | Modo: 100% Simulação (Zero Risco)")
        logger.info(f"   Estratégias Ativas: Scalper Cirúrgico (15m, Máx 5/dia) & Swing Trade (1h/4h)")
        logger.info("   Arbitragem de Carry: STANDBY")
        logger.info(f"   Dashboard ao vivo em: http://localhost:5000")
        logger.info("================================================================================")

        cycle = 0
        try:
            while self.is_running:
                cycle += 1
                logger.debug(f"--- Ciclo de Simulação #{cycle} ---")
                self.engine.sync_settings_from_db()

                # 1. Atualiza posições abertas e testa gatilhos de saída (SL, TP1, TP2) em paralelo
                await self.update_active_positions()

                # 2. Avalia novas oportunidades de Golden Pots (Cota separada e independente: máx 3/dia)
                await self.evaluate_golden_pot_opportunities()

                # 3. Avalia novas oportunidades de Scalper Cirúrgico em pares Core (Cota padrão: 2 a 5/dia)
                await self.evaluate_scalp_opportunities()

                # 4. Avalia novas oportunidades de Swing Trade
                await self.evaluate_swing_opportunities()

                # 5. Registra métricas e snapshot no banco
                m = self.engine.get_summary_metrics()
                scalp_stat = self.engine.get_daily_scalp_status()
                logger.info(
                    f"[STATUS DEMO] Equity: ${m['current_equity']:,.2f} | "
                    f"PnL: ${m['net_profit']:+,.2f} ({m['return_pct']:+.2f}%) | "
                    f"Scalps Hoje: {scalp_stat['daily_scalps_count']}/{scalp_stat['max_daily_scalps']} ({scalp_stat['quota_status']}) | "
                    f"Golden Pots: {m['daily_golden_pots_count']}/{m['max_daily_golden_pots']} | "
                    f"Posições Abertas: {m['open_positions']} | Trades Totais: {m['total_trades_count']}"
                )

                if max_cycles > 0 and cycle >= max_cycles:
                    logger.info(f"Limite de {max_cycles} ciclos atingido. Encerrando execução.")
                    break

                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Daemon de Paper Trading recebendo sinal de interrupção...")
        finally:
            self.is_running = False
            await self.loader.close()
            await self.scanner.close()
            logger.info("Daemon de Paper Trading encerrado graciosamente.")


async def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    daemon = PaperTradingDaemon(poll_interval_seconds=poll)

    # Tratamento de sinais POSIX para encerramento gracioso
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: setattr(daemon, "is_running", False))
        except NotImplementedError:
            pass

    await daemon.run_loop(max_cycles=cycles)


if __name__ == "__main__":
    asyncio.run(main())
