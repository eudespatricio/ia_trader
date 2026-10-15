"""
scripts/run_paper_simulation.py - Daemon Autônomo de Simulação em Tempo Real (Paper Trading).
Executa o ciclo contínuo de trading simulado com microestrutura institucional:
1. Carry Arbitrage em STANDBY (conforme diretriz estratégica).
2. Foco Total em:
   - Scalper Cirúrgico Intradiário (15m): candidato VWAP/RSI/volume + APROVAÇÃO OBRIGATÓRIA do comitê,
     execução Maker (0.02%) e cota diária rigorosa (máximo 5 scalps/dia).
   - SCALP, GOLDEN POT (momentum + volatilidade extraordinária) e DAY TRADE 1h passam pelo comitê e
     compartilham a Smart Trade Exit Ladder (TP1/TP2/TP3 + BE + trailing; ADDENDUM E7).
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from agents.golden_pot_agent import GoldenPotAgent
from agents.tape_reading_agent import TapeReadingAgent
from signal_generator import TradeSignal
from smart_trade import (PaperTradingEngine, NON_CLOSING_TRADE_ACTIONS, STRATEGY_PROFILES,
                         FUNDING_INTERVAL_MS, DIRECTIONAL_BAR_MS, INTRADAY_BAR_MS)
from leverage_policy import decide_leverage

logger = logging.getLogger("ia_trader.paper_simulation")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [paper_daemon] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


SWING_BAR_MS = DIRECTIONAL_BAR_MS     # vela de decisão do day trade/swing (1h)
POLL_BAR_MS = INTRADAY_BAR_MS         # vela do poll de posições (15m): só barras FECHADAS alimentam SL/TP/trailing
FUNDING_REFRESH_LIMIT = 10            # nº de registros de funding pedidos quando a liquidação ainda não está no cache
SWING_CANDLE_LIMIT = 400          # velas 1h buscadas (sma_200 + folga após dropna)
SWING_MIN_WARMUP_CANDLES = 250    # mínimo bruto para sma_200 válida em ao menos 50 linhas

# --- GOLDEN_POT v2 (ADDENDUM E5, 2026-10-15) -------------------------------------------------
GP_BAR_MS = 15 * 60 * 1000          # duração da vela 15m (chave de "1 entrada por vela" e cooldown)
GP_CANDLE_LIMIT = 400               # velas 15m buscadas (binance -> fallback bitget)
GP_MIN_WARMUP_CANDLES = 250         # mínimo bruto; além disso exige sma_200 válida (exclui listagens novas)
GP_MAX_CANDIDATES = 4               # candidatos do scanner avaliados por ciclo
GP_COOLDOWN_BARS = 16               # sem reentrada no símbolo por N barras 15m após qualquer fechamento
GP_MAX_ENTRIES_PER_BAR = 1          # no máximo 1 nova entrada GP por vela 15m (qualquer símbolo)
GP_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["GOLDEN_POT"].min_committee_confidence   # 0.70 (perfil)
# Alvos/stop/trailing por estratégia: ÚNICA fonte = smart_trade.STRATEGY_PROFILES (ladder comum, ADDENDUM E7).
GP_TP1_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp1_atr_mult
GP_TP2_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp2_atr_mult
GP_TP3_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp3_atr_mult
GP_SL_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].sl_atr_mult
GP_TRAIL_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].trail_atr_mult
# --- GATILHO GP v3 (ADDENDUM E7): MOMENTUM com VOLATILIDADE EXTRAORDINÁRIA (não é fade/mean-reversion) ---
GP_ATR_MEDIAN_BARS = 96             # janela (24h de 15m) da mediana de ATR, EXCLUINDO a barra atual
GP_VOL_EXPANSION_MIN = 1.8          # ATR atual / mediana(ATR janela) >= limiar
GP_RANGE_ATR_MIN = 1.5              # (high-low) da vela de sinal >= k x ATR
GP_VOLUME_Z_MIN = 2.0               # volume_zscore_20 da vela de sinal >= limiar
GP_CLOSE_EXTREME_MIN = 0.75         # fechamento no topo (BUY) / fundo (SELL) da vela: posição >= 0.75 / <= 0.25
GP_BREAKOUT_BARS = 20               # fecha acima da máxima / abaixo da mínima das N barras ANTERIORES
# --- SCALP via comitê (ADDENDUM E7): candidato = setup VWAP/RSI/volume (simulation_settings); entrada = comitê ---
SCALP_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["SCALP"].min_committee_confidence     # 0.70 (perfil)
SCALP_COOLDOWN_BARS = 4             # sem reentrada do símbolo por N barras 15m (1h) após qualquer fechamento
SCALP_MAX_ENTRIES_PER_BAR = 1       # no máximo 1 nova entrada SCALP por vela 15m (qualquer símbolo)
SCALP_CANDLE_LIMIT = GP_CANDLE_LIMIT
# --- DAY TRADE / DIRECTIONAL 1h ---
SWING_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["DIRECTIONAL"].min_committee_confidence   # 0.70 (perfil)
# Filtro faca-caindo (legado do GP v2 de fade; mantido como utilitário, NÃO aplicado no gatilho de momentum v3)
GP_KNIFE_RSI_LOW = 20.0             # RSI abaixo => "faca caindo" (LONG) se close < sma_200
GP_KNIFE_RSI_HIGH = 80.0            # simétrico (SHORT) se close > sma_200
GP_KNIFE_ATR_MOVE = 3.0             # movimento > X ATR na última hora também caracteriza tendência forte
GP_KNIFE_LOOKBACK_BARS = 4          # 4 barras de 15m = 1h
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
        tape_agent: Optional[TapeReadingAgent] = None,
    ):
        self.poll_interval = poll_interval_seconds
        self.engine = engine or PaperTradingEngine()
        self.loader = loader or DataLoader()
        self.fe = fe or FeatureEngineer()
        self.scanner = golden_pot_agent or GoldenPotAgent(min_volume_usd=20_000_000.0)
        self.tape_agent = tape_agent or TapeReadingAgent()
        self.last_tape: Optional[Dict[str, object]] = None   # último voto do tape (telemetria)
        self.is_running = False
        self._last_tick_bar_ts: Dict[str, int] = {}        # última barra 15m FECHADA já aplicada por símbolo
        self._funding_refresh_done: Dict[str, int] = {}    # liquidação de funding já consultada (evita rede repetida)

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

    async def _refresh_funding_cache(self, symbols: List[str], now_ts: int) -> None:
        """P5: garante uma taxa de funding recente (<= 8h) no cache. Só vai à rede se ela ainda não estiver
        no SQLite (no máx. 1 tentativa por símbolo/liquidação); falha de rede -> o engine usa o fallback documentado."""
        last_settle = (now_ts // FUNDING_INTERVAL_MS) * FUNDING_INTERVAL_MS
        for sym in symbols:
            if self._funding_refresh_done.get(sym) == last_settle or self.engine.has_funding_rate(sym, now_ts):
                continue
            self._funding_refresh_done[sym] = last_settle
            try:
                await self.loader.get_funding_history(sym, limit=FUNDING_REFRESH_LIMIT, exchange="binance")
            except Exception as e:
                logger.debug(f"Aviso ao atualizar funding de {sym}: {e}")

    async def update_active_positions(self) -> None:
        """Atualiza posições abertas.

        Paridade com o backtest: SL/TP/trailing/funding/timeout só avançam com barras 15m FECHADAS (a barra em
        formação é descartada e processada quando fechar, em ordem, sem repetir barras já vistas). O preço da barra
        em formação apenas marca a mercado (`mark_to_market`, UI) - nenhum gatilho. Após reinício, só são
        reprocessadas barras posteriores à entrada (as anteriores são pré-entrada para o engine).
        """
        open_positions = list(self.engine.positions.values())
        if not open_positions:
            return

        symbols_to_update = list({p.symbol for p in open_positions if p.status == "OPEN"})
        now_ts = int(time.time() * 1000)
        await self._refresh_funding_cache(symbols_to_update, now_ts)

        async def _update_symbol(sym: str) -> None:
            try:
                df = await self.loader.get_candles(sym, "15m", limit=5, exchange="binance")
                if df.empty:
                    df = await self.loader.get_candles(sym, "15m", limit=5, exchange="bitget")
                if df.empty:
                    return
                closed = self._drop_forming_candle(df, POLL_BAR_MS, now_ts)
                last_seen = self._last_tick_bar_ts.get(sym, 0)
                for _, bar in closed.iterrows():
                    bts = int(bar["timestamp"])
                    if bts <= last_seen:
                        continue
                    triggered = self.engine.update_market_tick(
                        symbol=sym, high=float(bar["high"]), low=float(bar["low"]), close=float(bar["close"]),
                        timestamp=bts, open_price=float(bar["open"]),
                    )
                    self._last_tick_bar_ts[sym] = last_seen = bts
                    for t in triggered:
                        logger.info(f"[{sym}] GATILHO SIMULADO: {t.action} @ ${t.price:,.2f} | PnL: ${t.pnl:+.2f}")
                # vela em formação: só marca a mercado (UI), sem gatilhos
                if len(closed) < len(df):
                    self.engine.mark_to_market(sym, float(df.iloc[-1]["close"]))
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

    def _entered_bar(self, prefix: str, bar_id: int, max_per_bar: int) -> bool:
        """True se já houve `max_per_bar` entradas com ação `prefix` nesta vela de 15m (lê o ledger -> sobrevive a restart)."""
        # P2: o ledger grava a entrada em ts(t+1) = ts(vela de decisão t) + 15m -> subtrai 1 barra p/ comparar com bar_id(t)
        n = sum(
            1 for t in self.engine.trade_history
            if t.action.startswith(prefix) and int(PaperTradingEngine._ts_ms(t.timestamp)) // GP_BAR_MS - 1 == bar_id
        )
        return n >= max_per_bar

    def _in_cooldown(self, symbol: str, bar_id: int, cooldown_bars: int) -> bool:
        """True se o símbolo teve fechamento (qualquer saída total) há < cooldown_bars velas de 15m.

        Saídas parciais (TP1/TP2_PARTIAL) não são fechamento. Lê trade_history (restaurado do ledger no restart).
        """
        for t in self.engine.trade_history:
            if t.symbol != symbol or t.action.startswith("OPEN_") or t.action in NON_CLOSING_TRADE_ACTIONS:
                continue
            if bar_id - int(PaperTradingEngine._ts_ms(t.timestamp)) // GP_BAR_MS < cooldown_bars:
                return True
        return False

    def _gp_entered_bar(self, bar_id: int) -> bool:
        return self._entered_bar("OPEN_GOLDEN_POT_", bar_id, GP_MAX_ENTRIES_PER_BAR)

    def _gp_in_cooldown(self, symbol: str, bar_id: int) -> bool:
        return self._in_cooldown(symbol, bar_id, GP_COOLDOWN_BARS)

    @staticmethod
    def gp_falling_knife_blocked(direction: str, df_feat: pd.DataFrame) -> Optional[str]:
        """Filtro de regime / faca-caindo (legado do GP v2 de fade; NÃO é mais aplicado no gatilho de momentum v3).

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

    @staticmethod
    def gp_momentum_trigger(df_feat: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """Gatilho GP v3 (ADDENDUM E7): MOMENTUM com VOLATILIDADE EXTRAORDINÁRIA na última vela FECHADA.

        Todas as condições (constantes GP_*), na direção do momentum:
          1. expansão de volatilidade: ATR atual / mediana(ATR das GP_ATR_MEDIAN_BARS barras anteriores) >= GP_VOL_EXPANSION_MIN;
          2. range da vela >= GP_RANGE_ATR_MIN x ATR;
          3. volume_zscore_20 >= GP_VOLUME_Z_MIN;
          4. fechamento perto da extrema na direção (posição na vela >= GP_CLOSE_EXTREME_MIN p/ BUY, <= 1-x p/ SELL)
             e vela na direção (close > open p/ BUY);
          5. rompimento: close > máxima (BUY) / < mínima (SELL) das GP_BREAKOUT_BARS barras ANTERIORES.
        Retorna (direção, justificativa) ou None. Usa só barras fechadas; janelas excluem a barra atual (sem lookahead).
        Mean-reversion (VWAP/RSI extremos contra o preço) NÃO dispara.
        """
        need = max(GP_ATR_MEDIAN_BARS, GP_BREAKOUT_BARS) + 1
        if df_feat is None or len(df_feat) < need or "atr_14" not in df_feat.columns:
            return None
        last = df_feat.iloc[-1]
        try:
            o, h, l, c = (float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"]))
            atr = float(last["atr_14"])
            vol_z = float(last.get("volume_zscore_20", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(np.isfinite(x) for x in (o, h, l, c, atr, vol_z)) or atr <= 0.0 or h <= l:
            return None
        prev_atr = df_feat["atr_14"].astype(float).iloc[-1 - GP_ATR_MEDIAN_BARS:-1]
        med_atr = float(prev_atr[prev_atr > 0].median()) if (prev_atr > 0).any() else 0.0
        if med_atr <= 0.0 or not np.isfinite(med_atr):
            return None
        expansion = atr / med_atr
        rng_atr = (h - l) / atr
        if expansion < GP_VOL_EXPANSION_MIN or rng_atr < GP_RANGE_ATR_MIN or vol_z < GP_VOLUME_Z_MIN:
            return None
        close_pos = (c - l) / (h - l)
        prev = df_feat.iloc[-1 - GP_BREAKOUT_BARS:-1]
        prior_high, prior_low = float(prev["high"].max()), float(prev["low"].min())
        direction: Optional[str] = None
        if c > o and close_pos >= GP_CLOSE_EXTREME_MIN and c > prior_high:
            direction = "BUY"
        elif c < o and close_pos <= 1.0 - GP_CLOSE_EXTREME_MIN and c < prior_low:
            direction = "SELL"
        if direction is None:
            return None
        extra = []
        if direction == "BUY" and int(last.get("liquidity_sweep_bull", 0)) == 1:
            extra.append("confluência: sweep de fundo")
        if direction == "SELL" and int(last.get("liquidity_sweep_bear", 0)) == 1:
            extra.append("confluência: sweep de topo")
        if direction == "BUY" and int(last.get("at_bullish_poi", 0)) == 1:
            extra.append("confluência: POI de alta")
        if direction == "SELL" and int(last.get("at_bearish_poi", 0)) == 1:
            extra.append("confluência: POI de baixa")
        why = (f"momentum {direction}: ATR {expansion:.2f}x mediana | range {rng_atr:.2f} ATR | VolZ {vol_z:+.1f} | "
               f"fecha {close_pos:.0%} da vela | rompe {GP_BREAKOUT_BARS} barras")
        return direction, why + (f" | {', '.join(extra)}" if extra else "")

    @staticmethod
    def _drop_forming_candle(df: pd.DataFrame, bar_ms: int, now_ts: int) -> pd.DataFrame:
        """Descarta a última vela se ainda em formação (timestamp = abertura; fecha em ts + bar_ms)."""
        if df is not None and len(df) > 0 and int(df["timestamp"].iloc[-1]) + bar_ms > now_ts:
            return df.iloc[:-1]
        return df

    async def _fetch_closed_candles_15m(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        df = await self.loader.get_candles(sym, "15m", limit=GP_CANDLE_LIMIT, exchange="binance")
        if len(df) < GP_MIN_WARMUP_CANDLES:
            df = await self.loader.get_candles(sym, "15m", limit=GP_CANDLE_LIMIT, exchange="bitget")
        if len(df) < GP_MIN_WARMUP_CANDLES:
            return None
        # Somente velas FECHADAS: descarta a última se ainda em formação (timestamp = abertura)
        df = self._drop_forming_candle(df, GP_BAR_MS, now_ts)
        return df if len(df) >= GP_MIN_WARMUP_CANDLES else None

    async def _gp_fetch_closed_candles(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        return await self._fetch_closed_candles_15m(sym, now_ts)

    # ---- Funil de sinais (telemetria pura; falha nunca interrompe o trading) ----
    def _fi(self, strategy: str, sym: str, bar_ts: int, stage: str) -> None:
        try:
            self.engine.funnel_inc(strategy, sym, bar_ts, stage)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"funil: falha ao contar {stage}: {e}")

    def _fr(self, strategy: str, sym: str, bar_ts: int, reason: str) -> None:
        try:
            self.engine.funnel_reject(strategy, sym, bar_ts, reason)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"funil: falha ao contar reject:{reason}: {e}")

    @staticmethod
    def _nominal_bar(now_ts: int, bar_ms: int) -> int:
        """Barra nominal (abertura da vela corrente) p/ rejeições anteriores ao fetch de candles."""
        return (int(now_ts) // bar_ms) * bar_ms

    def _fr_open(self, strategy: str, sym: str, bar_ts: int) -> None:
        """Após execute_* retornar None: motivo real registrado pelo engine (quota/margin/breaker/veto/other)."""
        self._fr(strategy, sym, bar_ts, str(getattr(self.engine, "last_open_reject", "other")))

    async def _committee_gate(
        self, label: str, sym: str, direction: str, latest: pd.Series, close: float, atr: float, ts: int,
        min_confidence: float,
    ) -> Optional[TradeSignal]:
        """PORTA OBRIGATÓRIA (diretriz E7): SCALP e GOLDEN_POT só entram se o comitê completo
        (core.evaluate_market_async = agentes + consenso + veto do RiskSentinel + SignalGenerator) concordar
        na mesma direção, sem veto e com confiança >= mínimo do perfil. Retorna o TradeSignal REAL ou None."""
        self._committee_reject = "other"  # funil: motivo da última recusa do comitê
        committee = await self.engine.core.evaluate_market_async(
            context=latest.to_dict(), symbol=sym, current_price=close, atr=atr, timestamp=ts,
        )
        if committee.is_vetoed:
            logger.info(f"[{sym}] {label} bloqueado: veto do RiskSentinel.")
            self._committee_reject = "veto"
            return None
        if committee.direction != direction:
            logger.info(f"[{sym}] {label} bloqueado: comitê diverge (gatilho {direction} x comitê {committee.direction}).")
            self._committee_reject = "committee_disagree"
            return None
        if committee.confidence < min_confidence:
            logger.info(f"[{sym}] {label} bloqueado: confiança do comitê {committee.confidence:.2f} < {min_confidence}.")
            self._committee_reject = "low_confidence"
            return None
        return committee

    def _flow_gate(
        self, strategy_type: str, sym: str, direction: str, sig: TradeSignal, lev: float, close: float, atr: float,
    ) -> bool:
        """GATE DE FLUXO (ADDENDUM E8), aplicado DEPOIS da porta do comitê. O voto do TapeReadingAgent deve ser
        IGUAL à direção do sinal; oposto/HOLD => bloqueia (False); UNAVAILABLE => passa (fail-open) marcado.
        TODO sinal que chega aqui é gravado em flow_gate_log (inclusive bloqueados) para análise contrafactual."""
        now_ms = int(time.time() * 1000)
        try:
            tape = self.tape_agent.evaluate(sym, direction, now_ms=now_ms)
        except Exception as e:  # noqa: BLE001 - falha do agente = indisponível (fail-open registrado)
            tape = {"vote": "HOLD", "status": "UNAVAILABLE", "reason": f"agent_error:{type(e).__name__}", "features": {}}
        self.last_tape = {**tape, "strategy_type": strategy_type, "direction": direction}
        if tape.get("status") == "UNAVAILABLE":
            decision = "UNAVAILABLE_PASSED"
        elif tape.get("vote") == direction:
            decision = "PASSED"
        else:
            decision = "BLOCKED"
        f = tape.get("features") or {}
        prof = STRATEGY_PROFILES[strategy_type]
        sgn = 1.0 if direction == "BUY" else -1.0
        self.engine.log_flow_gate({
            "ts_ms": now_ms, "symbol": sym, "strategy_type": strategy_type, "direction": direction,
            "decision": decision, "tape_vote": tape.get("vote"), "tape_status": tape.get("status"),
            "tape_reason": tape.get("reason"), "taker_imb_5m": f.get("taker_imb_5m"),
            "book_imb_10": f.get("book_imb_10"), "microprice_dev": f.get("microprice_dev"),
            "spread_bps": f.get("spread_bps"), "entry_price": close, "stop_loss": sig.stop_loss,
            "tp1": close + sgn * prof.tp1_atr_mult * atr, "tp2": close + sgn * prof.tp2_atr_mult * atr,
            "tp3": close + sgn * prof.tp3_atr_mult * atr, "atr": atr, "confidence": sig.confidence,
            "consensus_score": sig.consensus_score, "leverage": lev,
        })
        if decision == "BLOCKED":
            logger.info(f"[{sym}] {strategy_type} bloqueado pelo gate de fluxo: tape {tape.get('vote')} "
                        f"({tape.get('reason')}) x sinal {direction}.")
            return False
        return True

    def flow_gate_telemetry(self) -> str:
        c = self.engine.flow_gate_counts()
        lt = self.last_tape
        last = f"{lt.get('vote')}/{lt.get('status')}({lt.get('reason')})" if lt else "n/a"
        return (f"Tape: último {last} | Gate P/B/U: {c['PASSED']}/{c['BLOCKED']}/{c['UNAVAILABLE_PASSED']}")

    def _build_ladder_signal(
        self, committee: TradeSignal, strategy_type: str, direction: str, close: float, atr: float,
        latest: pd.Series, margin_cap: float, rationale: str, low_liquidity: bool = False,
    ) -> Tuple[TradeSignal, float]:
        """Alavancagem via decide_leverage (confiança REAL) + sinal com SL/TP3 do perfil. Lote = margem."""
        prof = STRATEGY_PROFILES[strategy_type]
        lev = decide_leverage(
            committee.confidence,
            atr_pct=atr / close if close > 0 else 0.0,
            regime_alignment=self._regime_aligned(direction, latest),
            is_vetoed=committee.is_vetoed,
            low_history_or_liquidity=low_liquidity,
        )
        margin = min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95, margin_cap)
        notional = margin * lev
        sl_dist = prof.sl_atr_mult * atr
        sgn = 1.0 if direction == "BUY" else -1.0
        sig = replace(
            committee,
            entry_price=close,
            stop_loss=close - sgn * sl_dist,
            take_profit=close + sgn * prof.tp3_atr_mult * atr,
            atr=atr,
            position_size=notional,
            risk_dollars=(notional / close) * sl_dist,
            rationale=f"{rationale} | {committee.rationale}",
        )
        return sig, lev

    async def evaluate_golden_pot_opportunities(self) -> None:
        """
        GOLDEN_POT v3 (ADDENDUM E7). Oportunidade = MOMENTUM com VOLATILIDADE EXTRAORDINÁRIA (gp_momentum_trigger)
        em candidatos do scanner (volume/volatilidade). A entrada exige: símbolo fora da lista de exclusão,
        sma_200 válida (>=400 velas 15m fechadas), 1 entrada GP/vela, cooldown por símbolo, liquidez e
        concordância do comitê completo (core.evaluate_market_async) na MESMA direção do momentum, sem veto
        do RiskSentinel e com confiança >= mínimo do perfil. confidence/consensus_score/risk_dollars vêm do
        TradeSignal real. Cota diária própria (máx 3/dia). Saídas: ladder comum (perfil GOLDEN_POT).
        """
        GPN = "GOLDEN_POT"
        now_ts = int(time.time() * 1000)
        nom = self._nominal_bar(now_ts, GP_BAR_MS)
        can_exec, reason = self.engine.can_execute_golden_pot(now_ts)
        if not can_exec:
            self._fr(GPN, "*", nom, "breaker" if "Circuit Breaker" in reason else "quota")
            return  # Teto diário de Golden Pots atingido

        scan_res = await self.scanner.scan_market(force_refresh=False)
        candidates = []
        for p in scan_res.get("top_scalps", []):
            if self.is_gp_excluded_symbol(str(p.get("symbol", ""))):
                self._fr(GPN, str(p.get("symbol", "")), nom, "excluded_symbol")
            else:
                candidates.append(p)
        candidates = candidates[:GP_MAX_CANDIDATES]
        if not candidates:
            self._fr(GPN, "*", nom, "no_data")
            return

        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            self._fr(GPN, "*", nom, "margin")
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            self._fr(GPN, "*", nom, "margin")
            return  # Margem livre insuficiente

        for pot in candidates:
            sym = pot["symbol"]
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                self._fr(GPN, sym, nom, "margin")
                break

            can_exec, _reason = self.engine.can_execute_golden_pot(now_ts)
            if not can_exec:
                self._fr(GPN, sym, nom, "breaker" if "Circuit Breaker" in _reason else "quota")
                break

            # Se já há posição aberta neste par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                self._fr(GPN, sym, nom, "margin")
                continue

            bar_ref = nom
            try:
                df = await self._gp_fetch_closed_candles(sym, now_ts)
                if df is None:
                    logger.debug(f"[{sym}] GP ignorado: < {GP_MIN_WARMUP_CANDLES} velas 15m (listagem nova/sem warmup).")
                    self._fr(GPN, sym, nom, "no_data")
                    continue

                df_feat = self._compute_swing_features(df)  # exige sma_200 válida
                if df_feat.empty or not self._has_valid_swing_context(df_feat):
                    logger.debug(f"[{sym}] GP ignorado: sma_200 sem warmup válido.")
                    self._fr(GPN, sym, nom, "no_data")
                    continue

                latest = df_feat.iloc[-1]
                ts = int(latest["timestamp"])
                bar_ref = ts
                bar_id = ts // GP_BAR_MS
                self._fi(GPN, sym, ts, "evaluated")
                # Limite de 1 entrada GP por vela de 15m (qualquer símbolo) e cooldown por símbolo
                if self._gp_entered_bar(bar_id):
                    self._fr(GPN, sym, ts, "bar_limit")
                    break
                if self._gp_in_cooldown(sym, bar_id):
                    logger.debug(f"[{sym}] GP em cooldown ({GP_COOLDOWN_BARS} barras).")
                    self._fr(GPN, sym, ts, "cooldown")
                    continue

                trig = self.gp_momentum_trigger(df_feat)
                if trig is None:
                    self._fr(GPN, sym, ts, "no_trigger")
                    continue
                direction, why = trig
                self._fi(GPN, sym, ts, "candidate")

                close = float(latest["close"])
                atr = float(latest.get("atr_14", close * 0.008))
                committee = await self._committee_gate(
                    "GP", sym, direction, latest, close, atr, ts, GP_MIN_COMMITTEE_CONFIDENCE)
                if committee is None:
                    self._fr(GPN, sym, ts, getattr(self, "_committee_reject", "other"))
                    continue
                self._fi(GPN, sym, ts, "committee_agree")
                self._fi(GPN, sym, ts, "confidence_ok")
                self._fi(GPN, sym, ts, "risk_ok")

                sig, lev = self._build_ladder_signal(
                    committee, "GOLDEN_POT", direction, close, atr, latest,
                    margin_cap=self.engine.balance * 0.20,
                    rationale=f"Golden Pot v3: {why}",
                    low_liquidity=float(pot.get("volume_24h_usd", 0.0) or 0.0) < GP_LOW_LIQUIDITY_USD,
                )
                if not self._flow_gate("GOLDEN_POT", sym, direction, sig, lev, close, atr):
                    self._fr(GPN, sym, ts, "gate_blocked")
                    continue
                self._fi(GPN, sym, ts, "gate_passed")
                pos = self.engine.execute_golden_pot_signal(
                    signal=sig, current_price=close, timestamp=ts, leverage=lev,
                )
                if not pos:
                    self._fr_open(GPN, sym, ts)
                if pos:
                    self._fi(GPN, sym, ts, "executed")
                    logger.info(
                        f"[{sym}] 🌟 GOLDEN POT v3 DISPARADO: {direction} @ ${close:,.2f} | "
                        f"conf {sig.confidence:.2f} cons {sig.consensus_score:+.2f} | lev {pos.leverage:g}x | "
                        f"Cota Golden Pots: {self.engine.daily_golden_pots_count}/{self.engine.max_daily_golden_pots}"
                    )
                    break  # no máximo 1 entrada GP por vela
            except Exception as e:
                self._fr(GPN, sym, bar_ref, "other")
                logger.debug(f"Aviso ao avaliar golden pot para {sym}: {e}")

    async def evaluate_scalp_opportunities(self) -> None:
        """
        SCALP (15m, BTC/ETH/SOL) via COMITÊ (ADDENDUM E7). Candidato = setup VWAP/RSI/volume dos
        simulation_settings (scalp_vwap_dev_pct/rsi/volume_z, inalterados). A entrada só ocorre se o comitê
        completo concordar na mesma direção, sem veto, com confiança >= SCALP_MIN_COMMITTEE_CONFIDENCE, usando
        400 velas fechadas com sma_200 válida. confidence/consensus/risk_dollars/rationale são do sinal REAL
        (nenhuma constante); alavancagem via decide_leverage; cooldown e 1 entrada/vela. Cota: máx scalps/dia.
        """
        SCN = "SCALP"
        scalp_syms = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
        now_ts = int(time.time() * 1000)
        nom = self._nominal_bar(now_ts, GP_BAR_MS)
        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            for _s in scalp_syms:
                self._fr(SCN, _s, nom, "margin")
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            for _s in scalp_syms:
                self._fr(SCN, _s, nom, "margin")
            return  # Margem livre insuficiente

        can_exec, reason = self.engine.can_execute_scalp(now_ts)
        if not can_exec:
            for _s in scalp_syms:
                self._fr(SCN, _s, nom, "breaker" if "Circuit Breaker" in reason else "quota")
            return  # Teto diário de scalps atingido

        for sym in scalp_syms:
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                self._fr(SCN, sym, nom, "margin")
                break

            can_exec, _reason = self.engine.can_execute_scalp(now_ts)
            if not can_exec:
                self._fr(SCN, sym, nom, "breaker" if "Circuit Breaker" in _reason else "quota")
                break

            # Se já há posição aberta no par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                self._fr(SCN, sym, nom, "margin")
                continue

            bar_ref = nom
            try:
                df = await self._fetch_closed_candles_15m(sym, now_ts)
                if df is None:
                    self._fr(SCN, sym, nom, "no_data")
                    continue

                df_feat = self._compute_swing_features(df)  # sma_200 válida (warmup)
                if df_feat.empty or not self._has_valid_swing_context(df_feat):
                    logger.debug(f"[{sym}] Scalp ignorado: sma_200 sem warmup válido.")
                    self._fr(SCN, sym, nom, "no_data")
                    continue

                latest = df_feat.iloc[-1]
                ts = int(latest["timestamp"])
                bar_ref = ts
                bar_id = ts // GP_BAR_MS
                self._fi(SCN, sym, ts, "evaluated")
                if self._entered_bar("OPEN_SCALP_", bar_id, SCALP_MAX_ENTRIES_PER_BAR):
                    self._fr(SCN, sym, ts, "bar_limit")
                    break
                if self._in_cooldown(sym, bar_id, SCALP_COOLDOWN_BARS):
                    logger.debug(f"[{sym}] Scalp em cooldown ({SCALP_COOLDOWN_BARS} barras).")
                    self._fr(SCN, sym, ts, "cooldown")
                    continue

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
                if not direction:
                    self._fr(SCN, sym, ts, "no_trigger")
                    continue
                self._fi(SCN, sym, ts, "candidate")

                committee = await self._committee_gate(
                    "Scalp", sym, direction, latest, close, atr, ts, SCALP_MIN_COMMITTEE_CONFIDENCE)
                if committee is None:
                    self._fr(SCN, sym, ts, getattr(self, "_committee_reject", "other"))
                    continue
                self._fi(SCN, sym, ts, "committee_agree")
                self._fi(SCN, sym, ts, "confidence_ok")
                self._fi(SCN, sym, ts, "risk_ok")

                sig, lev = self._build_ladder_signal(
                    committee, "SCALP", direction, close, atr, latest,
                    margin_cap=self.engine.balance * 0.25,
                    rationale=f"Scalp Padrão Core: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f} | Vol Z {vol_z:+.1f}",
                )
                if not self._flow_gate("SCALP", sym, direction, sig, lev, close, atr):
                    self._fr(SCN, sym, ts, "gate_blocked")
                    continue
                self._fi(SCN, sym, ts, "gate_passed")
                pos = self.engine.execute_scalp_signal(
                    signal=sig, current_price=close, timestamp=ts, leverage=lev,
                )
                if not pos:
                    self._fr_open(SCN, sym, ts)
                if pos:
                    self._fi(SCN, sym, ts, "executed")
                    logger.info(
                        f"[{sym}] ⚡ SCALP (comitê) DISPARADO: {direction} @ ${close:,.2f} | "
                        f"conf {sig.confidence:.2f} cons {sig.consensus_score:+.2f} | lev {pos.leverage:g}x | "
                        f"Cota Scalps: {self.engine.daily_scalps_count}/{self.engine.max_daily_scalps}"
                    )
                    break  # no máximo 1 entrada SCALP por vela
            except Exception as e:
                self._fr(SCN, sym, bar_ref, "other")
                logger.debug(f"Aviso ao avaliar scalp padrão para {sym}: {e}")

    async def evaluate_swing_opportunities(self) -> None:
        """Avalia oportunidades de Swing Trade (1h/4h) com o Comitê de Decisão."""
        DTN = "DAY_TRADE"
        for sym in self.core_symbols:
            nom = self._nominal_bar(int(time.time() * 1000), SWING_BAR_MS)
            bar_ref = nom
            if any(p.symbol == sym and p.strategy_type == "DIRECTIONAL" and p.status == "OPEN" for p in self.engine.positions.values()):
                self._fr(DTN, sym, nom, "margin")
                continue

            try:
                # BIAS: sma_200 exige >=200 velas válidas após o dropna -> busca 400 candles e
                # pula o símbolo se o warmup não for satisfeito (nunca opera com sma_200=0.0).
                df = await self.loader.get_candles(sym, "1h", limit=SWING_CANDLE_LIMIT, exchange="binance")
                if len(df) < SWING_MIN_WARMUP_CANDLES:
                    df = await self.loader.get_candles(sym, "1h", limit=SWING_CANDLE_LIMIT, exchange="bitget")
                # P3: decisão SOMENTE em vela FECHADA (descarta a vela 1h em formação, como GP/scalp e o backtest)
                df = self._drop_forming_candle(df, SWING_BAR_MS, int(time.time() * 1000))
                if len(df) < SWING_MIN_WARMUP_CANDLES:
                    logger.debug(f"[{sym}] Swing ignorado: {len(df)} velas < warmup {SWING_MIN_WARMUP_CANDLES}.")
                    self._fr(DTN, sym, nom, "no_data")
                    continue

                df_feat = self._compute_swing_features(df)
                if df_feat.empty or not self._has_valid_swing_context(df_feat):
                    logger.debug(f"[{sym}] Swing ignorado: features inválidas (sma_200 ausente/zero).")
                    self._fr(DTN, sym, nom, "no_data")
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                ts = int(latest["timestamp"])
                bar_ref = ts
                self._fi(DTN, sym, ts, "evaluated")
                self._fi(DTN, sym, ts, "candidate")  # day trade: símbolo/barra elegível ao comitê

                eval_res = self.engine.core.evaluate_multi_strategy(
                    symbol=sym,
                    current_price=close,
                    context=latest.to_dict(),
                    annualized_funding_rate=0.0, # Carry em standby
                    timestamp=ts,
                )

                dir_sig = eval_res.get("directional_signal", {})
                if dir_sig.get("direction") in ["BUY", "SELL"] and not dir_sig.get("is_vetoed", False):
                    # Exige confiança institucional (>= SWING_MIN_COMMITTEE_CONFIDENCE, perfil DIRECTIONAL)
                    if dir_sig.get("confidence", 0.0) >= SWING_MIN_COMMITTEE_CONFIDENCE:
                        self._fi(DTN, sym, ts, "committee_agree")
                        self._fi(DTN, sym, ts, "confidence_ok")
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
                        self._fi(DTN, sym, ts, "risk_ok")
                        if not self._flow_gate("DIRECTIONAL", sym, sig.direction, sig, lev, close,
                                               float(latest.get("atr_14", 0.0) or 0.0)):
                            self._fr(DTN, sym, ts, "gate_blocked")
                            continue
                        self._fi(DTN, sym, ts, "gate_passed")
                        pos = self.engine.execute_directional_signal(sig, current_price=close, timestamp=ts, leverage=lev)
                        if not pos:
                            self._fr_open(DTN, sym, ts)
                        if pos:
                            self._fi(DTN, sym, ts, "executed")
                            logger.info(f"[{sym}] SWING TRADE DISPARADO: {sig.direction} @ ${close:,.2f}")
                    else:
                        self._fr(DTN, sym, ts, "low_confidence")
                else:
                    self._fr(DTN, sym, ts,
                             "veto" if dir_sig.get("direction") in ["BUY", "SELL"] else "committee_disagree")
            except Exception as e:
                self._fr(DTN, sym, bar_ref, "other")
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
                    f"Posições Abertas: {m['open_positions']} | Trades Totais: {m['total_trades_count']} | "
                    f"{self.flow_gate_telemetry()}"
                )

                if max_cycles > 0 and cycle >= max_cycles:
                    logger.info(f"Limite de {max_cycles} ciclos atingido. Encerrando execução.")
                    break

                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Daemon de Paper Trading recebendo sinal de interrupção...")
        finally:
            self.is_running = False
            try:
                self.engine.funnel_flush()  # funil de sinais: grava o buffer pendente no shutdown
            except Exception as e:  # noqa: BLE001
                logger.warning(f"signal_funnel: falha no flush final: {e}")
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
