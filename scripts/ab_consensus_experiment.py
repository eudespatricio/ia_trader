"""
Experimento A/B OFFLINE das propostas de ajuste de consenso da auditoria (P1..P6).

NÃO altera nenhum arquivo de produção: todas as variantes são subclasses/wrappers definidos AQUI.
Leitura do SQLite do cache é estritamente somente-leitura (`mode=ro`).

Desenho (cínico por construção)
-------------------------------
* Baseline = réplica do que o piloto roda hoje em `evaluate_swing_opportunities`:
  contexto = última linha de `FeatureEngineer.compute_all_features(df_1h, drop_na=False)` (sem
  fear_and_greed, sem dxy/vix/sp500, sem stablecoins, SEM `volume_sma` e SEM `atr_median`),
  TechnicalAgent sem modelo (fallback de regras), `ConsensusJudge(0.30)`, gate `confidence >= 0.70`.
* Dois "mundos":
    - PROD : exatamente o contexto acima (Sentiment/Macro/OnChain sempre HOLD "sem dados").
    - FULL : contexto enriquecido com F&G, DXY, VIX, S&P500 e stablecoins do cache diário
             (defasados em +1 dia, sem lookahead) para que P3/P6 tenham algo a modificar.
             TVL não existe no cache -> a metade "TVL" de P3 é inobservável em backtest.
* Sem lookahead: decisão no FECHAMENTO da barra t (candle fechado; a última barra do cache é
  descartada), entrada na ABERTURA de t+1, SL/TP de `SignalGenerator` (1.5/1.0 ATR), SL vence TP
  na mesma barra, timeout de 48 barras, funding REAL (tabela funding_rates) debitado/creditado.
* Custos de estresse: taker 0,06% + slippage 0,02% por perna (0,16% round-trip) em TODOS os fills.
* Split: TRAIN = primeiros 60% do tempo; HOLDOUT = após embargo de 10 dias (purga: trades que
  cruzam a fronteira são descartados). Nenhum parâmetro é ajustado -> não há "treino" no sentido
  de fit; o split serve para checar consistência da melhora e impedir seleção pós-hoc no holdout.
* Rigor: DSR (N = nº de variantes testadas), bootstrap por blocos da diferença pareada diária vs
  baseline com correção de Bonferroni, consistência em 6 blocos temporais.

Uso:  .venv/bin/python scripts/ab_consensus_experiment.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base_agent import AgentSignal  # noqa: E402
from agents.macro_agent import MacroAgent  # noqa: E402
from agents.onchain_agent import OnChainAgent  # noqa: E402
from agents.orderflow_agent import OrderFlowAgent  # noqa: E402
from agents.risk_sentinel_agent import RiskSentinelAgent  # noqa: E402
from agents.sentiment_agent import SentimentAgent  # noqa: E402
from agents.technical_agent import TechnicalAgent  # noqa: E402
from consensus_judge import ConsensusDecision, ConsensusJudge  # noqa: E402
from signal_generator import SignalGenerator  # noqa: E402

DB_PATH = os.path.join(ROOT, "datasets", "market_cache.db")
TRIAL_LOG = os.path.join(ROOT, "datasets", "reports", "trial_log.csv")
REPORT_MD = os.path.join(ROOT, "docs", "AB_CONSENSO_RESULTADOS.md")
REPORT_JSON = os.path.join(ROOT, "docs", "ab_consenso_resultados.json")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HOUR_MS = 3_600_000
DAY_MS = 86_400_000

# ---- Parâmetros de execução/custos (fixos a priori; NÃO são otimizados) ----
TAKER_FEE = 0.0006
SLIPPAGE = 0.0002
MAX_HOLD_BARS = 48
CONF_GATE = 0.70            # gate do loop de produção (run_paper_simulation.py L365)
CAPITAL = 10_000.0
TRAIN_FRAC = 0.60
EMBARGO_DAYS = 10
N_BLOCKS = 6
FULL_START_MS = int(pd.Timestamp("2021-12-01", tz="UTC").timestamp() * 1000)
ATR_MEDIAN_WINDOW = 100     # P2: janela fixada a priori (sem sweep)
VOL_CONFIRM_RATIO = 1.2     # P4: mesmo limiar já usado no próprio agente (prioridade 4)
BOOT_ITERS = 5000
BLOCK_LEN = 7


# =====================================================================
# Variantes (monkeypatch por subclasse — produção intocada)
# =====================================================================

class TechnicalAgentP5(TechnicalAgent):
    """P5: momentum BUY só com RSI<=70; simetria: momentum SELL só com RSI>=30.

    Implementado neutralizando o RSI passado ao fallback original para uma zona que não dispara
    nenhum ramo de momentum (47 em tendência de alta, 53 em baixa) — o restante da lógica
    (confluência POI/sweep, gatilho estrutural) permanece idêntico.
    """

    RSI_CAP = 70.0
    RSI_FLOOR = 30.0

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        ctx = context
        try:
            rsi = float(context.get("rsi_14"))
            has_ml = (context.get("proba_buy") or 0) or (context.get("proba_sell") or 0)
            if not has_ml and rsi == rsi:
                close = float(context.get("close"))
                sma = float(context.get("sma_200"))
                bull = close >= sma
                if bull and rsi > self.RSI_CAP:
                    ctx = {**context, "rsi_14": 47.0}
                elif (not bull) and rsi < self.RSI_FLOOR:
                    ctx = {**context, "rsi_14": 53.0}
        except (TypeError, ValueError):
            ctx = context
        sig = super().evaluate(ctx)
        if ctx is not context:
            sig.metrics["rsi_14"] = float(context["rsi_14"])
        return sig


class OrderFlowAgentP4(OrderFlowAgent):
    """P4: 'Neurônio de Fluxo Ativo' (prioridade 5) exige |close_position|>=0.5 OU vol_ratio>=1.2."""

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        sig = super().evaluate(context)
        if sig.direction != "HOLD" and sig.rationale.startswith("Neurônio de Fluxo Ativo"):
            cp = float(sig.metrics.get("close_position", 0.0))
            vr = float(sig.metrics.get("vol_ratio", 1.0))
            if not (abs(cp) >= 0.5 or vr >= VOL_CONFIRM_RATIO):
                return AgentSignal(
                    agent_name=self.name, direction="HOLD", confidence=0.50, weight=self.weight,
                    rationale="P4: fluxo fraco sem confirmação (|pos|<0.5 e sem volume).",
                    metrics=sig.metrics, timestamp=sig.timestamp,
                )
        return sig


class SentimentAgentP6(SentimentAgent):
    """P6: confiança monotônica na fronteira F&G=25 (e, por simetria, 75).

    Original: extremo (<=25) tem conf 0.60 em f=25 e salta para ~0.64 em f=25.01 (moderado).
    P6: extremo = 0.65 + 0.20*(25-f)/25 (>= limite do ramo moderado, monotônico decrescente em f).
    O mesmo defeito existe em 75 (SELL); corrigimos simetricamente para não criar viés direcional.
    """

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        sig = super().evaluate(context)
        fng = context.get("fear_and_greed")
        if fng is None:
            return sig
        f = float(fng)
        if f <= self.extreme_fear_threshold:
            sig.confidence = min(0.85, 0.65 + ((self.extreme_fear_threshold - f) / self.extreme_fear_threshold) * 0.20)
        elif f >= self.extreme_greed_threshold:
            sig.confidence = min(0.85, 0.65 + ((f - self.extreme_greed_threshold) / (100.0 - self.extreme_greed_threshold)) * 0.20)
        return sig


class MacroAgentP3(MacroAgent):
    """P3 (parte DXY): contribuição total do DXY (ret_5d ±1.0 + sma20 ±0.5) limitada a ±1.0."""

    DXY_CLAMP = 1.0

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        dxy_ret = context.get("dxy_ret_5d")
        dxy_above = context.get("dxy_above_sma20")
        dxy_score = 0.0
        if dxy_ret is not None:
            if float(dxy_ret) < -0.005:
                dxy_score += 1.0
            elif float(dxy_ret) > 0.005:
                dxy_score -= 1.0
        if dxy_above is not None:
            dxy_score += -0.5 if dxy_above else 0.5
        excess = dxy_score - max(-self.DXY_CLAMP, min(self.DXY_CLAMP, dxy_score))
        if excess == 0.0:
            return super().evaluate(context)
        # Remove o excesso reconstruindo o score sem alterar o resto da lógica original:
        # reavalia com o termo sma20 neutralizado (None) quando é ele que gera o excesso.
        ctx = {**context, "dxy_above_sma20": None}
        sig = super().evaluate(ctx)
        # score_without_sma = dxy_ret term ±1.0 (já dentro do clamp) -> equivale ao clamp ±1.0
        return sig


class OnChainAgentP3(OnChainAgent):
    """P3 (parte TVL): se a stablecoin é DERIVADA do TVL (proxy 0.75x), conta a contribuição uma vez (max)."""

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        stbl = context.get("stablecoin_growth_pct_30d")
        tvl = context.get("tvl_growth_pct_7d")
        if tvl is None:
            tvl = context.get("tvl_change_24h")
        if stbl is None and tvl is not None:
            # contribuições: stbl_proxy (±1.2 se |0.75*tvl|>0.005) e tvl (±0.8 se |tvl|>0.01).
            # Como |tvl|>0.01 => |0.75*tvl|>0.0075>0.005, max == contribuição da stablecoin proxy.
            ctx = {k: v for k, v in context.items() if k not in ("tvl_growth_pct_7d", "tvl_change_24h")}
            ctx["stablecoin_growth_pct_30d"] = float(tvl) * 0.75
            sig = super().evaluate(ctx)
            sig.metrics["tvl_growth_pct_7d"] = float(tvl)
            return sig
        return super().evaluate(context)


class JudgeNoDataExcluded(ConsensusJudge):
    """P1a: exclui do denominador agentes SEM DADOS (Sentiment/Macro/OnChain sem contexto; Sentinel sem veto)."""

    @staticmethod
    def has_no_data(sig: AgentSignal) -> bool:
        if sig.is_veto:
            return False
        m = sig.metrics or {}
        if sig.agent_name == "RiskSentinelAgent":
            return True  # só chega aqui sem veto
        if sig.agent_name == "SentimentAgent":
            return "fear_and_greed" not in m
        if sig.agent_name == "MacroAgent":
            return all(m.get(k) is None for k in ("dxy_ret_5d", "vix", "sp500_ret_5d"))
        if sig.agent_name == "OnChainAgent":
            return m.get("stablecoin_growth_pct_30d") is None and m.get("tvl_growth_pct_7d") is None
        return False

    def judge_deterministic(self, signals, weights_override=None):  # type: ignore[override]
        kept = [s for s in signals if not self.has_no_data(s)]
        return super().judge_deterministic(kept, weights_override)


def map_p2_context(ctx: Dict[str, Any], atr_median: float) -> Dict[str, Any]:
    """P2: volume_sma <- vol_sma_20 ; atr_median <- rolling median(atr_14)."""
    return {**ctx, "volume_sma": ctx.get("vol_sma_20"), "atr_median": atr_median}


@dataclass(frozen=True)
class Variant:
    name: str
    world: str = "PROD"           # PROD | FULL
    p1a: bool = False
    p1b: bool = False
    p2: bool = False
    p3: bool = False
    p4: bool = False
    p5: bool = False
    p6: bool = False
    desc: str = ""

    def is_baseline(self) -> bool:
        return self.name.startswith("BASE")


def build_variants() -> List[Variant]:
    v = [
        Variant("BASE", desc="Réplica do piloto congelado"),
        Variant("P1a", p1a=True, desc="Juiz exclui agentes sem dados do denominador"),
        Variant("P1b", p1b=True, desc="min_consensus_threshold 0.30→0.22"),
        Variant("P2", p2=True, desc="volume_sma/atr_median mapeados (vetos + vol_ratio reais)"),
        Variant("P3", p3=True, desc="OnChain TVL 1x + DXY ±1.0 (inerte em PROD)"),
        Variant("P4", p4=True, desc="OrderFlow prio 5: |pos|>=0.5 ou volume"),
        Variant("P5", p5=True, desc="Technical momentum RSI 30..70"),
        Variant("P6", p6=True, desc="Sentiment monotônico F&G=25 (inerte em PROD)"),
        Variant("P2+P4", p2=True, p4=True, desc="P4 com vol_ratio real (requer P2)"),
        Variant("ALL_a", p1a=True, p2=True, p3=True, p4=True, p5=True, p6=True, desc="Tudo, com P1a"),
        Variant("ALL_b", p1b=True, p2=True, p3=True, p4=True, p5=True, p6=True, desc="Tudo, com P1b"),
        Variant("BASE_F", world="FULL", desc="Baseline com contexto enriquecido (F&G/macro/stable)"),
        Variant("P1a_F", world="FULL", p1a=True, desc="P1a com contexto enriquecido"),
        Variant("P1b_F", world="FULL", p1b=True, desc="P1b com contexto enriquecido"),
        Variant("P3_F", world="FULL", p3=True, desc="DXY ±1.0 (TVL inobservável)"),
        Variant("P6_F", world="FULL", p6=True, desc="Sentiment monotônico"),
        Variant("ALL_a_F", world="FULL", p1a=True, p2=True, p3=True, p4=True, p5=True, p6=True, desc="Tudo, com P1a"),
    ]
    return v


# =====================================================================
# Dados (somente leitura)
# =====================================================================

def _ro_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)


def load_candles(symbol: str, db_path: str = DB_PATH) -> pd.DataFrame:
    with _ro_conn(db_path) as c:
        df = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM candles "
            "WHERE exchange='binance' AND symbol=? AND timeframe='1h' ORDER BY timestamp",
            c, params=(symbol,),
        )
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    # Candle fechado apenas: descarta a última barra (pode estar em formação).
    return df.iloc[:-1].reset_index(drop=True)


def load_funding(symbol: str, db_path: str = DB_PATH) -> Tuple[np.ndarray, np.ndarray]:
    with _ro_conn(db_path) as c:
        f = pd.read_sql(
            "SELECT timestamp, funding_rate FROM funding_rates WHERE exchange='binance' AND symbol=? ORDER BY timestamp",
            c, params=(symbol,),
        )
    ts = f["timestamp"].to_numpy(dtype=np.int64)
    cum = np.concatenate([[0.0], np.cumsum(f["funding_rate"].to_numpy(dtype=float))])
    return ts, cum


def funding_between(ts: np.ndarray, cum: np.ndarray, a_ms: int, b_ms: int) -> float:
    """Soma de funding rates liquidados em (a, b]."""
    if len(ts) == 0:
        return 0.0
    i = int(np.searchsorted(ts, a_ms, side="right"))
    j = int(np.searchsorted(ts, b_ms, side="right"))
    return float(cum[j] - cum[i])


def load_daily_context(db_path: str = DB_PATH) -> pd.DataFrame:
    """Série diária de contexto com disponibilidade = data + 24h (conservador, sem lookahead)."""
    with _ro_conn(db_path) as c:
        m = pd.read_sql("SELECT series_name, date_str, value FROM macro_daily_history", c)
    out = {}
    for name, g in m.groupby("series_name"):
        s = g.drop_duplicates("date_str").set_index(pd.to_datetime(g.drop_duplicates("date_str")["date_str"]))["value"].sort_index()
        out[name] = s
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in out.values()])))
    d = pd.DataFrame(index=idx)
    for name, s in out.items():
        d[name] = s
    d = d.ffill()  # só para alinhar calendários; cada derivado é calculado sobre a série própria abaixo
    res = pd.DataFrame(index=idx)
    res["fear_and_greed"] = out["FEAR_AND_GREED"].reindex(idx).ffill()
    dxy = out["DXY"]
    res["dxy_ret_5d"] = (dxy / dxy.shift(5) - 1.0).reindex(idx).ffill()
    res["dxy_above_sma20"] = (dxy > dxy.rolling(20).mean()).astype(float).where(dxy.rolling(20).mean().notna()).reindex(idx).ffill()
    res["vix_value"] = out["VIX"].reindex(idx).ffill()
    sp = out["SP500"]
    res["sp500_ret_5d"] = (sp / sp.shift(5) - 1.0).reindex(idx).ffill()
    st = out["STABLECOINS_CIRCULATING_USD"]
    res["stablecoin_growth_pct_30d"] = (st / st.shift(30) - 1.0).reindex(idx).ffill()
    res["avail_ms"] = (idx.view("int64") // 1_000_000) + DAY_MS
    return res.reset_index(drop=True)


def align_daily_context(bar_ts_ms: np.ndarray, daily: pd.DataFrame) -> pd.DataFrame:
    """asof backward: barra fechando em t+1h só enxerga dados com avail_ms <= t+1h."""
    left = pd.DataFrame({"close_ms": bar_ts_ms + HOUR_MS})
    d = daily.sort_values("avail_ms")
    merged = pd.merge_asof(left, d, left_on="close_ms", right_on="avail_ms", direction="backward")
    return merged.drop(columns=["close_ms", "avail_ms"])


CTX_COLS = [
    "timestamp", "open", "high", "low", "close", "volume", "sma_200", "dist_sma_200", "rsi_14",
    "dist_vwap_pct", "atr_14", "natr_14", "vol_sma_20", "volume_zscore_20", "funding_rate",
    "funding_zscore", "at_bullish_poi", "at_bearish_poi", "liquidity_sweep_bull",
    "liquidity_sweep_bear", "dist_support_pct", "dist_resistance_pct",
]
FULL_COLS = ["fear_and_greed", "dxy_ret_5d", "dxy_above_sma20", "vix_value", "sp500_ret_5d", "stablecoin_growth_pct_30d"]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    from data.feature_engineering import FeatureEngineer
    return FeatureEngineer.compute_all_features(df, drop_na=False)


# =====================================================================
# Geração de decisões (réplica de core.evaluate_market_deterministic)
# =====================================================================

class DecisionEngine:
    """Gera decisões por barra para uma variante, com cache de sinais por (agente, config)."""

    def __init__(self, ctxs: List[Dict[str, Any]], atr_median: np.ndarray, world: str):
        self.ctxs = ctxs
        self.atr_median = atr_median
        self.world = world
        self._cache: Dict[Tuple, List[AgentSignal]] = {}

    def _ctx(self, i: int, p2: bool) -> Dict[str, Any]:
        c = self.ctxs[i]
        return map_p2_context(c, float(self.atr_median[i])) if p2 else c

    def _signals(self, key: Tuple, agent, p2: bool, subset: Optional[str] = None) -> List[AgentSignal]:
        if key not in self._cache:
            self._cache[key] = [agent.evaluate(self._ctx(i, p2)) for i in range(len(self.ctxs))]
        return self._cache[key]

    def decisions(self, v: Variant) -> Dict[str, np.ndarray]:
        n = len(self.ctxs)
        sentinel = self._signals(("sentinel", v.p2), RiskSentinelAgent(), v.p2)
        tech = self._signals(("tech", v.p5), TechnicalAgentP5() if v.p5 else TechnicalAgent(), False)
        flow = self._signals(("flow", v.p4, v.p2), OrderFlowAgentP4() if v.p4 else OrderFlowAgent(), v.p2)
        sent = self._signals(("sent", v.p6), SentimentAgentP6() if v.p6 else SentimentAgent(), False)
        macro = self._signals(("macro", v.p3), MacroAgentP3() if v.p3 else MacroAgent(), False)
        onch = self._signals(("onch", v.p3), OnChainAgentP3() if v.p3 else OnChainAgent(), False)
        if v.p1a:
            judge: ConsensusJudge = JudgeNoDataExcluded(min_consensus_threshold=0.30)
        elif v.p1b:
            judge = ConsensusJudge(min_consensus_threshold=0.22)
        else:
            judge = ConsensusJudge(min_consensus_threshold=0.30)

        direction = np.zeros(n, dtype=np.int8)
        conf = np.zeros(n, dtype=float)
        score = np.zeros(n, dtype=float)
        vetoed = np.zeros(n, dtype=bool)
        raw = np.zeros(n, dtype=bool)
        for i in range(n):
            s0 = sentinel[i]
            if s0.is_veto:
                vetoed[i] = True
                continue
            d = judge.judge_deterministic([s0, tech[i], flow[i], sent[i], macro[i], onch[i]])
            score[i] = d.aggregated_score
            if d.direction in ("BUY", "SELL"):
                raw[i] = True
                if d.confidence >= CONF_GATE:
                    direction[i] = 1 if d.direction == "BUY" else -1
                    conf[i] = d.confidence
        return {"direction": direction, "conf": conf, "score": score, "vetoed": vetoed, "raw": raw}


def symbol_worker(args: Tuple[str, List[Variant], bool, str]) -> Dict[str, Any]:
    """Processo isolado por símbolo: features -> decisões de todas as variantes."""
    symbol, variants, quick, db_path = args
    candles = load_candles(symbol, db_path)
    if quick:
        candles = candles.iloc[-12000:].reset_index(drop=True)
    feat = compute_features(candles)
    valid = np.isfinite(feat["sma_200"].astype(float)) & (feat["sma_200"].astype(float) > 0.0)
    feat = feat[valid].reset_index(drop=True)
    atr_med = feat["atr_14"].rolling(ATR_MEDIAN_WINDOW, min_periods=ATR_MEDIAN_WINDOW).median().to_numpy()
    atr_med = np.where(np.isfinite(atr_med), atr_med, np.nan)

    base_ctxs = feat[CTX_COLS].to_dict("records")
    out: Dict[str, Any] = {
        "symbol": symbol,
        "ts": feat["timestamp"].to_numpy(dtype=np.int64),
        "open": feat["open"].to_numpy(float), "high": feat["high"].to_numpy(float),
        "low": feat["low"].to_numpy(float), "close": feat["close"].to_numpy(float),
        "atr": feat["atr_14"].to_numpy(float),
        "dec": {},
    }
    # NaN em atr_median -> P2 não veta (baseline_atr NaN: `float(nan) > 0` é False)
    eng_prod = DecisionEngine(base_ctxs, atr_med, "PROD")
    for v in [x for x in variants if x.world == "PROD"]:
        out["dec"][v.name] = eng_prod.decisions(v)
    del eng_prod

    full_variants = [x for x in variants if x.world == "FULL"]
    if full_variants:
        daily = load_daily_context(db_path)
        extra = align_daily_context(out["ts"], daily)
        full_ctxs = []
        for i, c in enumerate(base_ctxs):
            if c["timestamp"] < FULL_START_MS:
                full_ctxs.append(c)
                continue
            e = {k: (None if pd.isna(extra.at[i, k]) else float(extra.at[i, k])) for k in FULL_COLS}
            if e["dxy_above_sma20"] is not None:
                e["dxy_above_sma20"] = bool(e["dxy_above_sma20"])
            full_ctxs.append({**c, **e})
        eng_full = DecisionEngine(full_ctxs, atr_med, "FULL")
        for v in full_variants:
            d = eng_full.decisions(v)
            # FULL só vale a partir de FULL_START_MS (antes disso faltam DXY/VIX/SP500)
            mask = out["ts"] < FULL_START_MS
            d["direction"][mask] = 0
            d["raw"][mask] = False
            d["conf"][mask] = 0.0
            out["dec"][v.name] = d
    return out


# =====================================================================
# Simulação de trades
# =====================================================================

def simulate_trades(sym: Dict[str, Any], dec: Dict[str, np.ndarray], fund: Tuple[np.ndarray, np.ndarray],
                    fee: float = TAKER_FEE, slip: float = SLIPPAGE, max_hold: int = MAX_HOLD_BARS) -> pd.DataFrame:
    """Uma posição por símbolo. Decisão no fechamento de t, entrada na abertura de t+1."""
    ts, o, h, l, c, atr = sym["ts"], sym["open"], sym["high"], sym["low"], sym["close"], sym["atr"]
    direction = dec["direction"]
    n = len(ts)
    gen = SignalGenerator(capital=CAPITAL, risk_per_trade=0.01)
    rows = []
    t = 0
    while t < n - 2:
        d = int(direction[t])
        if d == 0 or not np.isfinite(atr[t]) or atr[t] <= 0:
            t += 1
            continue
        # Níveis/tamanho via SignalGenerator (produção), a partir do fechamento de t.

        cd = ConsensusDecision(direction="BUY" if d > 0 else "SELL", confidence=float(dec["conf"][t]))
        ts_sig = gen.generate(cd, "X", float(c[t]), float(atr[t]), int(ts[t]))
        sl, tp, size = ts_sig.stop_loss, ts_sig.take_profit, ts_sig.position_size
        e_idx = t + 1
        entry_raw = float(o[e_idx])
        entry_px = entry_raw * (1.0 + slip) if d > 0 else entry_raw * (1.0 - slip)
        last = min(n - 1, e_idx + max_hold - 1)
        hs, ls = h[e_idx:last + 1], l[e_idx:last + 1]
        if d > 0:
            sl_hit = ls <= sl
            tp_hit = hs >= tp
        else:
            sl_hit = hs >= sl
            tp_hit = ls <= tp
        i_sl = int(np.argmax(sl_hit)) if sl_hit.any() else 10 ** 9
        i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else 10 ** 9
        if i_sl == 10 ** 9 and i_tp == 10 ** 9:
            x_idx, reason, lvl = last, "TIMEOUT", float(c[last])
        elif i_sl <= i_tp:   # SL vence empate intra-barra
            x_idx, reason = e_idx + i_sl, "SL"
            lvl = min(sl, float(o[x_idx])) if (d > 0 and i_sl > 0) else (max(sl, float(o[x_idx])) if (d < 0 and i_sl > 0) else sl)
        else:
            x_idx, reason, lvl = e_idx + i_tp, "TP", tp
        exit_px = lvl * (1.0 - slip) if d > 0 else lvl * (1.0 + slip)
        notional_in = entry_px * size
        gross = d * (exit_px - entry_px) * size
        fees = fee * (notional_in + exit_px * size)
        entry_ts = int(ts[e_idx])
        exit_ts = int(ts[x_idx]) + HOUR_MS   # saída ao fim da barra (conservador p/ alinhamento diário)
        fnd = funding_between(fund[0], fund[1], entry_ts, exit_ts) * notional_in
        fund_cost = fnd if d > 0 else -fnd   # long paga funding>0; short recebe
        net = gross - fees - fund_cost
        rows.append({
            "symbol": sym["symbol"], "dir": d, "entry_ts": entry_ts, "exit_ts": exit_ts, "reason": reason,
            "ret_notional": net / notional_in if notional_in > 0 else 0.0,
            "gross_ret_notional": gross / notional_in if notional_in > 0 else 0.0,
            "pnl_pct_capital": net / CAPITAL,
            "bars": x_idx - e_idx + 1,
        })
        t = x_idx + 1   # só reavalia após fechar (como o piloto: 1 posição DIRECTIONAL/símbolo)
    return pd.DataFrame(rows)


# =====================================================================
# Métricas e estatística
# =====================================================================

def daily_series(trades: pd.DataFrame, t0_ms: int, t1_ms: int) -> np.ndarray:
    d0, d1 = int(t0_ms // DAY_MS), int(t1_ms // DAY_MS)
    arr = np.zeros(max(0, d1 - d0))
    if trades is None or len(trades) == 0 or len(arr) == 0:
        return arr
    days = (trades["exit_ts"].to_numpy() // DAY_MS) - d0
    ok = (days >= 0) & (days < len(arr))
    np.add.at(arr, days[ok].astype(int), trades["pnl_pct_capital"].to_numpy()[ok])
    return arr


def sharpe_ann(daily: np.ndarray) -> float:
    if len(daily) < 2 or np.std(daily, ddof=1) < 1e-14:
        return 0.0
    return float(np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(365.0))


def max_drawdown(daily: np.ndarray) -> float:
    """MaxDD aditivo (capital fixo), em fração do capital."""
    if len(daily) == 0:
        return 0.0
    eq = np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.min(eq - peak))


def bootstrap_mean_ci(x: np.ndarray, iters: int = BOOT_ITERS, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def block_bootstrap_diff(diff: np.ndarray, iters: int = BOOT_ITERS, block: int = BLOCK_LEN, seed: int = 11,
                         alpha_two_sided: float = 0.05) -> Dict[str, float]:
    """Bootstrap por blocos móveis da média da diferença diária pareada (preserva autocorrelação)."""
    n = len(diff)
    if n < 2 * block or np.allclose(diff, 0.0):
        return {"mean": float(np.mean(diff)) if n else 0.0, "lo": 0.0, "hi": 0.0, "p_le0": 1.0, "lo_adj": 0.0}
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(iters, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(iters, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    # Centraliza sob H0 para p-valor unilateral (mean<=0) — p = P(mean* - mean >= mean_obs)
    centered = means - diff.mean()
    p_le0 = float(np.mean(centered >= diff.mean()))
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, alpha_two_sided / 2)),
        "hi": float(np.quantile(means, 1 - alpha_two_sided / 2)),
        "p_le0": p_le0,
        "lo_adj": float(np.quantile(means, alpha_two_sided / 2)),  # preenchido pelo chamador com alpha ajustado
    }


def trial_log_rows() -> int:
    try:
        return int(len(pd.read_csv(TRIAL_LOG)))
    except Exception:
        return 0


def segment_metrics(trades: pd.DataFrame, t0: int, t1: int, strict_exit: bool = True) -> Dict[str, Any]:
    if len(trades):
        if strict_exit:
            sel = trades[(trades["entry_ts"] >= t0) & (trades["exit_ts"] <= t1)]
        else:
            sel = trades[(trades["entry_ts"] >= t0) & (trades["entry_ts"] < t1)]
    else:
        sel = trades
    daily = daily_series(sel, t0, t1)
    n = len(sel)
    if n:
        hit = float((sel["ret_notional"] > 0).mean())
        ghit = float((sel["gross_ret_notional"] > 0).mean())
        exp_net = float(sel["ret_notional"].mean())
        exp_gross = float(sel["gross_ret_notional"].mean())
        lo, hi = bootstrap_mean_ci(sel["ret_notional"].to_numpy(float))
    else:
        hit = ghit = exp_net = exp_gross = 0.0
        lo = hi = float("nan")
    return {
        "n_trades": int(n), "hit_rate": hit, "hit_rate_gross": ghit,
        "expectancy_net": exp_net, "expectancy_gross": exp_gross, "exp_net_ci": (lo, hi),
        "sharpe": sharpe_ann(daily), "max_dd": max_drawdown(daily),
        "total_return": float(daily.sum()), "daily": daily,
        "by_reason": sel["reason"].value_counts().to_dict() if n else {},
        "rets": sel["ret_notional"].to_numpy(float) if n else np.array([]),
    }


def analyze(per_symbol: List[Dict[str, Any]], variants: List[Variant], funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
            n_trials_total: int) -> Dict[str, Any]:
    from statistical_rigor import deflated_sharpe_ratio

    results: Dict[str, Any] = {}
    trades_by_var: Dict[str, pd.DataFrame] = {}
    for v in variants:
        parts = [simulate_trades(s, s["dec"][v.name], funding[s["symbol"]]) for s in per_symbol]
        parts = [p for p in parts if len(p)]
        trades_by_var[v.name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["symbol", "dir", "entry_ts", "exit_ts", "reason", "ret_notional", "gross_ret_notional", "pnl_pct_capital", "bars"])

    worlds = {"PROD": [v for v in variants if v.world == "PROD"], "FULL": [v for v in variants if v.world == "FULL"]}
    n_log = trial_log_rows()
    k_comparisons = max(1, len([v for v in variants if not v.is_baseline()]))
    alpha_adj = 0.05 / k_comparisons

    for world, vs in worlds.items():
        if not vs:
            continue
        start = min(int(s["ts"][0]) for s in per_symbol)
        if world == "FULL":
            start = max(start, FULL_START_MS)
        end = max(int(s["ts"][-1]) for s in per_symbol) + HOUR_MS
        split = int(start + (end - start) * TRAIN_FRAC)
        hold0 = split + EMBARGO_DAYS * DAY_MS
        edges = np.linspace(start, end, N_BLOCKS + 1).astype(np.int64)
        base_name = "BASE" if world == "PROD" else "BASE_F"

        # Métricas por segmento
        seg: Dict[str, Dict[str, Any]] = {}
        for v in vs:
            tr = trades_by_var[v.name]
            seg[v.name] = {
                "ALL": segment_metrics(tr, start, end),
                "TRAIN": segment_metrics(tr, start, split),
                "HOLDOUT": segment_metrics(tr, hold0, end),
            }
        # Variância dos Sharpes entre as variantes do mundo (holdout) -> DSR
        sh_hold = np.array([seg[v.name]["HOLDOUT"]["sharpe"] for v in vs])
        var_sh = float(np.var(sh_hold, ddof=1)) if len(sh_hold) > 1 else 0.0

        base_hold = seg[base_name]["HOLDOUT"]["daily"]
        r_base_hold = seg[base_name]["HOLDOUT"]
        base_train = seg[base_name]["TRAIN"]["daily"]
        base_all = seg[base_name]["ALL"]["daily"]
        base_dec = next(s for s in per_symbol)["dec"][base_name]  # só p/ referência de forma
        for v in vs:
            r = seg[v.name]
            # Inerte? decisões idênticas ao baseline em todos os símbolos.
            identical = all(np.array_equal(s["dec"][v.name]["direction"], s["dec"][base_name]["direction"]) for s in per_symbol)
            hold = r["HOLDOUT"]
            dsr = deflated_sharpe_ratio(hold["sharpe"], hold["daily"], n_trials_total, var_sh, periods_per_year=365.0) if len(hold["daily"]) > 5 else {"dsr": 0.0, "expected_max_sr": 0.0}
            dsr_log = deflated_sharpe_ratio(hold["sharpe"], hold["daily"], n_trials_total + n_log, var_sh, periods_per_year=365.0) if len(hold["daily"]) > 5 else {"dsr": 0.0}
            res: Dict[str, Any] = {"world": world, "desc": v.desc, "identical_to_baseline": bool(identical),
                                   "dsr": float(dsr["dsr"]), "dsr_expected_max_sr": float(dsr.get("expected_max_sr", 0.0)),
                                   "dsr_with_log": float(dsr_log["dsr"]), "n_trials_dsr": n_trials_total,
                                   "n_trial_log": n_log}
            for segname in ("ALL", "TRAIN", "HOLDOUT"):
                m = {k: val for k, val in r[segname].items() if k not in ("daily", "rets")}
                res[segname] = m
            raw_total = int(sum(s["dec"][v.name]["raw"].sum() for s in per_symbol))
            veto_total = int(sum(s["dec"][v.name]["vetoed"].sum() for s in per_symbol))
            res["n_raw_decisions"] = raw_total
            res["net_alpha"] = bool(hold["n_trades"] >= 30 and hold["exp_net_ci"][0] > 0.0)
            res["n_vetoed_bars"] = veto_total
            # Diferença pareada vs baseline
            if v.name != base_name:
                dh = hold["daily"] - base_hold
                bb = block_bootstrap_diff(dh)
                # IC ajustado Bonferroni (bilateral com alpha/K)
                if len(dh) >= 2 * BLOCK_LEN and not np.allclose(dh, 0.0):
                    rng = np.random.default_rng(11)
                    n = len(dh)
                    nb = int(math.ceil(n / BLOCK_LEN))
                    starts = rng.integers(0, n - BLOCK_LEN + 1, size=(BOOT_ITERS, nb))
                    idx = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]).reshape(BOOT_ITERS, -1)[:, :n]
                    means = dh[idx].mean(axis=1)
                    bb["lo_adj"] = float(np.quantile(means, alpha_adj / 2))
                res["paired_holdout"] = {k: float(val) for k, val in bb.items()}
                res["train_diff_mean_daily"] = float((r["TRAIN"]["daily"] - base_train).mean()) if len(base_train) else 0.0
                # Δ expectância por trade (bootstrap não-pareado) e decomposição do Δ de retorno total:
                # Δ = (n_v - n_b)*m_b  [efeito "menos trades"]  +  n_v*(m_v - m_b)  [efeito "melhor trade"]
                rv, rb = hold["rets"], r_base_hold["rets"]
                if len(rv) > 5 and len(rb) > 5:
                    rng2 = np.random.default_rng(13)
                    mv = rng2.choice(rv, size=(BOOT_ITERS, len(rv))).mean(axis=1)
                    mb = rng2.choice(rb, size=(BOOT_ITERS, len(rb))).mean(axis=1)
                    dd = mv - mb
                    res["exp_diff_per_trade"] = {
                        "mean": float(rv.mean() - rb.mean()),
                        "lo": float(np.quantile(dd, 0.025)), "hi": float(np.quantile(dd, 0.975)),
                        "lo_adj": float(np.quantile(dd, alpha_adj / 2)),
                    }
                n_v, n_b = hold["n_trades"], r_base_hold["n_trades"]
                m_v = hold["total_return"] / n_v if n_v else 0.0
                m_b = r_base_hold["total_return"] / n_b if n_b else 0.0
                eff_fewer, eff_better = (n_v - n_b) * m_b, n_v * (m_v - m_b)
                res["delta_total_return"] = float(hold["total_return"] - r_base_hold["total_return"])
                res["delta_from_fewer_trades"] = float(eff_fewer)
                res["delta_from_better_trades"] = float(eff_better)
                # Consistência em blocos temporais (ALL)
                pos = 0
                blk = []
                for b in range(N_BLOCKS):
                    a, z = int(edges[b]), int(edges[b + 1])
                    dv = daily_series(trades_by_var[v.name], a, z) - daily_series(trades_by_var[base_name], a, z)
                    blk.append(float(dv.mean()) if len(dv) else 0.0)
                    pos += 1 if blk[-1] > 0 else 0
                res["block_diff_means"] = blk
                res["blocks_positive"] = pos
                # Veredito
                reasons = []
                if identical:
                    reasons.append("inerte (decisões idênticas ao baseline)")
                else:
                    if bb["mean"] <= 0:
                        reasons.append("holdout não melhora")
                    if bb["lo_adj"] <= 0:
                        reasons.append("IC Bonferroni inclui 0")
                    if res["dsr"] < 0.95:
                        reasons.append(f"DSR {res['dsr']:.2f}<0.95")
                    if res["train_diff_mean_daily"] <= 0:
                        reasons.append("treino não melhora")
                    if pos < 4:
                        reasons.append(f"só {pos}/6 blocos melhoram")
                    if hold["n_trades"] < 100:
                        reasons.append(f"n_holdout={hold['n_trades']}<100")
                res["defensible"] = bool(not reasons)
                res["verdict_reasons"] = reasons
            results[v.name] = res
        results.setdefault("_meta", {})[world] = {
            "start": start, "split": split, "holdout_start": hold0, "end": end,
            "var_sharpe_holdout": var_sh, "alpha_bonferroni": alpha_adj,
        }
    return results


# =====================================================================
# Relatório
# =====================================================================

def _pct(x: float, d: int = 2) -> str:
    return "nan" if x != x else f"{x * 100:.{d}f}%"


def _ts(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def render_report(results: Dict[str, Any], variants: List[Variant], elapsed: float, quick: bool) -> str:
    meta = results["_meta"]
    L: List[str] = []
    L.append("# A/B de Ajustes de Consenso (P1–P6) — Resultados OFFLINE\n")
    L.append("> Gerado por `scripts/ab_consensus_experiment.py` (produção intocada; cache SQLite em modo somente-leitura). "
             "Piloto congelado desde 2026-10-15 00:47 UTC — **nada aqui altera parâmetros do piloto**.\n")
    if quick:
        L.append("> ⚠️ Execução `--quick` (amostra reduzida) — NÃO usar para decisão.\n")
    L.append("## 1. Metodologia\n")
    L.append("- **Dados:** Binance 1h do cache (BTC/ETH/SOL), candles fechados (última barra descartada), features `FeatureEngineer.compute_all_features(drop_na=False)` como no piloto.")
    L.append("- **Baseline (BASE)** = réplica do `evaluate_swing_opportunities`: contexto só com features 1h (sem F&G/macro/on-chain, **sem `volume_sma` nem `atr_median`**), TechnicalAgent sem modelo, `ConsensusJudge(0.30)`, gate `confidence ≥ 0.70`.")
    L.append("- **Mundo PROD** = contexto idêntico ao piloto. **Mundo FULL** = + F&G, DXY, VIX, S&P, stablecoins diários defasados +24h (somente ≥ 2021-12-01). TVL não existe no cache → metade TVL de P3 só é coberta por teste unitário.")
    L.append("- **Execução:** decisão no fechamento de *t*, entrada na abertura de *t+1*; SL 1.0×ATR / TP 1.5×ATR (`SignalGenerator`, sizing 1% risco, 1× alavancagem, capital $10k); SL vence TP na mesma barra; timeout 48 barras; 1 posição por símbolo; funding real do cache.")
    L.append("- **Custos:** taker 0,06% + slippage 0,02% por perna em *todos* os fills (0,16% round-trip) — SL tratado como taker.")
    for w, m in meta.items():
        L.append(f"- **Split {w}:** {_ts(m['start'])} → TREINO até {_ts(m['split'])}; embargo {EMBARGO_DAYS}d; HOLDOUT {_ts(m['holdout_start'])} → {_ts(m['end'])} (trades que cruzam a fronteira são descartados).")
    L.append(f"- **Estatística:** DSR com N = {len(variants)} variantes (e N+{results[variants[0].name]['n_trial_log']} do trial_log como cenário mais severo); variância dos Sharpes = entre variantes do mundo no holdout; bootstrap por blocos móveis (bloco {BLOCK_LEN}d, {BOOT_ITERS} reamostras) da diferença diária pareada vs baseline; IC com correção de Bonferroni (K = {len(variants) - 2}; α = {meta[next(iter(meta))]['alpha_bonferroni']:.4f}); consistência em {N_BLOCKS} blocos temporais.")
    L.append("- **Critério de 'melhora defensável' (TODOS):** diferença pareada média > 0 no holdout, limite inferior Bonferroni > 0, DSR ≥ 0,95, melhora também no treino, ≥ 4/6 blocos positivos, ≥ 100 trades no holdout, e variante não-inerte.\n")

    for world in ("PROD", "FULL"):
        vs = [v for v in variants if v.world == world]
        if not vs:
            continue
        L.append(f"## 2.{'1' if world == 'PROD' else '2'} Mundo {world}\n")
        for seg in ("HOLDOUT", "ALL"):
            L.append(f"### {world} — {seg}\n")
            L.append("| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
            for v in vs:
                r = results[v.name]
                m = r[seg]
                lo, hi = m["exp_net_ci"]
                L.append(f"| {v.name} | {m['n_trades']} | {r['n_raw_decisions'] if seg == 'ALL' else '—'} | {r['n_vetoed_bars'] if seg == 'ALL' else '—'} | {_pct(m['hit_rate'], 1)} | {_pct(m['hit_rate_gross'], 1)} | "
                         f"{_pct(m['expectancy_gross'], 3)} | {_pct(m['expectancy_net'], 3)} | [{_pct(lo, 3)}, {_pct(hi, 3)}] | {m['sharpe']:+.2f} | {_pct(m['max_dd'])} | {_pct(m['total_return'])} |")
            L.append("")
        L.append(f"### {world} — Defensabilidade estatística vs baseline (HOLDOUT)\n")
        L.append("| Variante | Δ média diária (cap.) | IC95 bloco | IC Bonferroni inf. | p(Δ≤0) | DSR (N) | DSR (N+log) | Δ treino | Blocos + | Veredito |")
        L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for v in vs:
            if v.is_baseline():
                r = results[v.name]
                L.append(f"| {v.name} | — | — | — | — | {r['dsr']:.2f} | {r['dsr_with_log']:.2f} | — | — | referência |")
                continue
            r = results[v.name]
            pr = r["paired_holdout"]
            verdict = "✅ defensável" if r["defensible"] else "❌ " + "; ".join(r["verdict_reasons"])
            L.append(f"| {v.name} | {pr['mean'] * 100:+.4f}% | [{pr['lo'] * 100:+.4f}%, {pr['hi'] * 100:+.4f}%] | {pr['lo_adj'] * 100:+.4f}% | {pr['p_le0']:.3f} | {r['dsr']:.2f} | {r['dsr_with_log']:.2f} | "
                     f"{r['train_diff_mean_daily'] * 100:+.4f}% | {r['blocks_positive']}/{N_BLOCKS} | {verdict} |")
        L.append("")
        L.append(f"### {world} — O que explica o Δ? (HOLDOUT; retorno total em % do capital, não composto)\n")
        L.append("| Variante | Δ retorno total | do qual: menos trades | do qual: trade melhor | Δ exp./trade | IC95 | IC Bonferroni inf. | Alfa líquido (IC95 inf.>0)? |")
        L.append("|---|---:|---:|---:|---:|---|---:|---|")
        for v in vs:
            r = results[v.name]
            if v.is_baseline():
                L.append(f"| {v.name} | — | — | — | — | — | — | {'sim' if r['net_alpha'] else 'não'} |")
                continue
            ed = r.get("exp_diff_per_trade")
            if ed:
                edc = f"{ed['mean'] * 100:+.3f}% | [{ed['lo'] * 100:+.3f}%, {ed['hi'] * 100:+.3f}%] | {ed['lo_adj'] * 100:+.3f}%"
            else:
                edc = "— | — | —"
            L.append(f"| {v.name} | {r['delta_total_return'] * 100:+.1f}% | {r['delta_from_fewer_trades'] * 100:+.1f}% | {r['delta_from_better_trades'] * 100:+.1f}% | "
                     f"{edc} | {'sim' if r['net_alpha'] else 'não'} |")
        L.append("")
    L.append("## 3. Legenda das variantes\n")
    for v in variants:
        L.append(f"- **{v.name}** ({v.world}): {v.desc}")
    L.append("")
    L.append(f"_Tempo de execução: {elapsed:.0f}s._\n")
    return "\n".join(L)


def json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o != o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="amostra reduzida (smoke test)")
    ap.add_argument("--no-write", action="store_true", help="não grava docs/")
    args = ap.parse_args(argv)
    t0 = time.time()
    variants = build_variants()
    with ProcessPoolExecutor(max_workers=len(SYMBOLS)) as ex:
        per_symbol = list(ex.map(symbol_worker, [(s, variants, args.quick, DB_PATH) for s in SYMBOLS]))
    funding = {s: load_funding(s) for s in SYMBOLS}
    results = analyze(per_symbol, variants, funding, n_trials_total=len(variants))
    elapsed = time.time() - t0
    md = render_report(results, variants, elapsed, args.quick)
    print(md)
    if not args.no_write:
        with open(REPORT_JSON, "w", encoding="utf-8") as fh:
            json.dump(json_safe(results), fh, indent=1)
        name = "_ab_consenso_tabelas_quick.md" if args.quick else "_ab_consenso_tabelas.md"
        with open(os.path.join(ROOT, "docs", name), "w", encoding="utf-8") as fh:
            fh.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
