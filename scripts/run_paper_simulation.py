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
import warnings
from logging.handlers import RotatingFileHandler

_pw = os.environ.get("PYTHONWARNINGS", "")
if "sklearn.utils.parallel" not in _pw:
    os.environ["PYTHONWARNINGS"] = (
        f"{_pw},ignore::UserWarning:sklearn.utils.parallel"
        if _pw
        else "ignore::UserWarning:sklearn.utils.parallel"
    )
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel.*", category=UserWarning)
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from agents.golden_pot_agent import GoldenPotAgent
from agents.tape_reading_agent import TapeReadingAgent
from agents.technical_agent import _disable_estimator_parallelism
from signal_generator import TradeSignal
from chatgpt_assistant import ChatGPTAssistant
from smart_trade import (PaperTradingEngine, NON_CLOSING_TRADE_ACTIONS, STRATEGY_PROFILES,
                         FUNDING_INTERVAL_MS, DIRECTIONAL_BAR_MS, INTRADAY_BAR_MS,
                         SCALP_BAR_MS, GOLDEN_POT_BAR_MS)
from leverage_policy import decide_leverage, MAX_LEVERAGE

logger = logging.getLogger("ia_trader.paper_simulation")
if not logger.handlers:
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [paper_daemon] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    log_file = os.getenv("PAPER_DAEMON_LOG_FILE")
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


SWING_BAR_MS = DIRECTIONAL_BAR_MS     # vela de decisão do day trade/swing (1h)
POLL_BAR_MS = SCALP_BAR_MS            # vela do poll de posições (3m): só barras FECHADAS alimentam SL/TP/trailing
FUNDING_REFRESH_LIMIT = 10            # nº de registros de funding pedidos quando a liquidação ainda não está no cache
SWING_CANDLE_LIMIT = 400          # velas 1h buscadas (sma_200 + folga após dropna)
SWING_MIN_WARMUP_CANDLES = 250    # mínimo bruto para sma_200 válida em ao menos 50 linhas

# --- GOLDEN_POT v3 (5m Momentum Radar de Altcoins) -------------------------------------------
GP_TIMEFRAME = "5m"
GP_BAR_MS = GOLDEN_POT_BAR_MS        # duração da vela 5m (chave de "1 entrada por vela" e cooldown)
GP_CANDLE_LIMIT = 400               # velas 5m buscadas (binance -> fallback bitget)
GP_MIN_WARMUP_CANDLES = 200         # mínimo bruto; além disso exige sma_200 válida (exclui listagens novas)
GP_MAX_CANDIDATES = 6               # candidatos do scanner avaliados por ciclo (swings + scalps)
GP_COOLDOWN_BARS = 12               # sem reentrada no símbolo por N barras 5m (1h) após qualquer fechamento
GP_MAX_ENTRIES_PER_BAR = 1          # no máximo 1 nova entrada GP por vela 5m (qualquer símbolo)
GP_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["GOLDEN_POT"].min_committee_confidence   # 0.70 (perfil)
# Alvos/stop/trailing por estratégia: ÚNICA fonte = smart_trade.STRATEGY_PROFILES (ladder comum, ADDENDUM E7).
GP_TP1_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp1_atr_mult
GP_TP2_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp2_atr_mult
GP_TP3_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].tp3_atr_mult
GP_SL_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].sl_atr_mult
GP_TRAIL_ATR_MULT = STRATEGY_PROFILES["GOLDEN_POT"].trail_atr_mult
# --- GATILHO GP v3: MOMENTUM com VOLATILIDADE EXTRAORDINÁRIA (5m Real-Time) ---
GP_ATR_MEDIAN_BARS = 48             # janela (4h de 5m) da mediana de ATR, EXCLUINDO a barra atual
GP_VOL_EXPANSION_MIN = 1.3          # ATR atual / mediana(ATR janela) >= limiar (calibrado)
GP_RANGE_ATR_MIN = 1.1              # (high-low) da vela de sinal >= k x ATR
GP_VOLUME_Z_MIN = 1.0               # volume_zscore_20 da vela de sinal >= limiar
GP_CLOSE_EXTREME_MIN = 0.70         # fechamento no topo (BUY) / fundo (SELL) da vela: posição >= 0.70 / <= 0.30
GP_BREAKOUT_BARS = 12               # fecha acima da máxima / abaixo da mínima das N barras ANTERIORES (1h)
# --- PROTEÇÕES FORENSES GP (Audit Forense & ADDENDUM E9) ---
GP_STALE_BAR_MAX_AGE_MS = 90_000        # descarta sinal se vela fechou há > 90s (ajustado para ciclo de scan)
GP_CLIMAX_MAX_ATR_MOVE = 2.0            # bloqueia se variação em 3-4 barras > 2.0x ATR (exaustão de clímax)
GP_CLIMAX_RSI_OVERSOLD = 25.0           # bloqueia SELL se RSI < 25.0
GP_CLIMAX_RSI_OVERBOUGHT = 75.0         # bloqueia BUY se RSI > 75.0
GP_MAX_LEVERAGE = 2.0                   # teto de alavancagem 2x para altcoins voláteis 5m
GP_ADAPTIVE_SL_MARGIN = 1.15            # multiplicador sobre a maior amplitude das últimas 3 velas
# --- SCALP via comitê (15m Momentum + Setup Fallback): Majors BTC/ETH/SOL (Migrado na S2) ---
SCALP_TIMEFRAME = "15m"
SCALP_BAR_MS = SCALP_BAR_MS         # duração da vela 15m
SCALP_STALE_BAR_MAX_AGE_MS = 180_000 # descarta sinal se vela 15m fechou há > 3min (proporcional ao timeframe)
SCALP_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["SCALP"].min_committee_confidence     # 0.70 (perfil)
SCALP_COOLDOWN_BARS = 4             # sem reentrada do símbolo por N barras 15m (60m) após qualquer fechamento
SCALP_STREAK_STOP_COOLDOWN_BARS = 8 # cooldown estendido (120m) após 2 STOP_LOSS consecutivos no símbolo
SCALP_MIN_ATR_PCT = 0.0025          # piso de ATR (25 bps) no modo S1 para evitar SL < taxas round-trip
SCALP_MAX_LEVERAGE = 3.0            # teto de alavancagem 3x no modo S1 (elimina faixa 4x de PF negativo)
SCALP_MAX_ENTRIES_PER_BAR = 1       # no máximo 1 nova entrada SCALP por vela 15m (qualquer símbolo)
SCALP_BREAKOUT_BARS = 20            # rompimento de 20 barras 15m (5h)
SCALP_VOLUME_Z_MIN = 0.8            # volume_zscore_20 no 15m >= 0.8
SCALP_RANGE_ATR_MIN = 1.0           # range >= 1.0 ATR
SCALP_CLOSE_EXTREME_MIN = 0.70      # fechamento no terço extremo
SCALP_CANDLE_LIMIT = 400
SCALP_MIN_WARMUP_CANDLES = 200
# --- DAY TRADE / DIRECTIONAL 1h ---
SWING_MIN_COMMITTEE_CONFIDENCE = STRATEGY_PROFILES["DIRECTIONAL"].min_committee_confidence   # 0.70 (perfil)
SWING_STREAK_STOP_COOLDOWN_BARS = 2   # cooldown estendido (2h) após 2 STOP_LOSS consecutivos no símbolo
SWING_MAX_LEVERAGE = 3.0              # teto de alavancagem 3x no modo S1 para day trade
# --- PATCH S4.2 (v1.0.7): Blindagem Multi-Estratégia & SCALP Squeeze+ML15m Universal ---
MAX_SAME_DIR_MAJORS = 2               # trava de correlação direcional nos majors (BTC/ETH/SOL)
GP_BTC_MACRO_MAX_COUNTER_DIST = 0.015 # bloqueia GP contra tendência macro do BTC (> ±1.5% SMA200 1h)
GP_UNAVAILABLE_TAPE_RISK_MULT = 0.50  # sizing defensivo 0.50x no GP quando altcoin está sem cobertura L2
SCALP_SQUEEZE_VOL_Z_MIN = 1.0         # volume_zscore_20 mínimo no rompimento de Squeeze BB/Keltner 15m
SCALP_SQUEEZE_NATR_BPS_MIN = 35.0     # ATR normalizado mínimo (bps) no Squeeze 15m
SCALP_ML15_MIN_PROB = 0.45            # probabilidade mínima do ensemble dedicado 15m (LGBM+RF)
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
    "SNXX", "SNDK", "UNITREE", "OPENAI", "SPACEX", "ANTHROPIC", "STRIPE", "FIGMA",
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
        risk_per_trade_pct: Optional[float] = None,
        require_strict_tape_majors: Optional[bool] = None,
        use_scalp_squeeze_ml15: Optional[bool] = None,
        enforce_gp_btc_macro: Optional[bool] = None,
    ):
        self.poll_interval = poll_interval_seconds
        self.engine = engine or PaperTradingEngine()
        if risk_per_trade_pct is not None:
            self.engine.risk_per_trade_pct = risk_per_trade_pct
        if hasattr(self.engine, "core") and self.engine.core:
            # Desempate Semântico via Gemini Flash ou GPU Local (Ollama / RTX 3060)
            if hasattr(self.engine.core, "llm_assistant") and not getattr(self.engine.core.llm_assistant, "enabled", False):
                gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if gemini_key:
                    self.engine.core.llm_assistant = ChatGPTAssistant(
                        enabled=True,
                        provider="gemini",
                        model="gemini-3.8-flash",
                        api_key=gemini_key,
                        timeout=5.0,
                    )
                else:
                    self.engine.core.llm_assistant = ChatGPTAssistant(
                        enabled=True,
                        provider="ollama",
                        model="llama3.1:latest",
                        timeout=8.5,
                    )
        self.loader = loader or DataLoader()
        self.fe = fe or FeatureEngineer()
        self.scanner = golden_pot_agent or GoldenPotAgent(min_volume_usd=20_000_000.0)
        self.tape_agent = tape_agent or TapeReadingAgent()
        self.last_tape: Optional[Dict[str, object]] = None   # último voto do tape (telemetria)
        self.is_running = False
        self._last_tick_bar_ts: Dict[str, int] = {}        # última barra FECHADA já aplicada por símbolo
        self._flow_gate_bar_cache: Dict[Tuple[str, str, int], bool] = {}  # dedupe do _flow_gate por (strategy, sym, bar_ts)
        self._funding_refresh_done: Dict[str, int] = {}    # liquidação de funding já consultada (evita rede repetida)
        self.stale_bar_max_age_ms: int = 0 if type(self.loader).__name__ == "FakeLoader" else GP_STALE_BAR_MAX_AGE_MS
        self._swing_entered_bars: Set[Tuple[str, int]] = set()
        self._last_breaker_log_ts: Dict[str, float] = {}
        is_fake_loader = (type(self.loader).__name__ == "FakeLoader")
        self.require_strict_tape_majors: bool = (not is_fake_loader) if require_strict_tape_majors is None else bool(require_strict_tape_majors)
        self.use_scalp_squeeze_ml15: bool = (not is_fake_loader) if use_scalp_squeeze_ml15 is None else bool(use_scalp_squeeze_ml15)
        self.enforce_gp_btc_macro: bool = (not is_fake_loader) if enforce_gp_btc_macro is None else bool(enforce_gp_btc_macro)
        self._h1_dist_cache: Dict[str, Tuple[float, int, Optional[float]]] = {}
        self._scalp_ml15_models: List[Dict[str, Any]] = self._load_scalp_ml15_models()

        # Pares fixos monitorados continuamente
        self.core_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    @staticmethod
    def _load_scalp_ml15_models() -> List[Dict[str, Any]]:
        """Carrega os modelos dedicados de 15m (lgbm_15m_v1.joblib e rf_15m_v1.joblib) se disponíveis."""
        bundles: List[Dict[str, Any]] = []
        for fname in ("lgbm_15m_v1.joblib", "rf_15m_v1.joblib"):
            fpath = os.path.join(BASE_DIR, "models", fname)
            if not os.path.exists(fpath):
                continue
            try:
                bundle = joblib.load(fpath)
                if isinstance(bundle, dict) and bundle.get("model") is not None and bundle.get("feature_names"):
                    _disable_estimator_parallelism(bundle["model"])
                    bundles.append(bundle)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Falha ao carregar modelo 15m {fname}: {exc}")
        if bundles:
            logger.info(f"Ensemble 15m SCALP carregado: {len(bundles)} modelo(s) ativo(s).")
        return bundles

    def _predict_scalp_ml15_probs(self, latest: pd.Series) -> Optional[Tuple[float, float]]:
        """Retorna (p_buy, p_sell) médios do ensemble dedicado 15m; None se nenhum modelo estiver carregado."""
        if not self._scalp_ml15_models:
            return None
        p_buys: List[float] = []
        p_sells: List[float] = []
        for bundle in self._scalp_ml15_models:
            try:
                model = bundle["model"]
                fnames = list(bundle["feature_names"])
                row = {}
                for col in fnames:
                    val = latest.get(col, 0.0)
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        fval = 0.0
                    row[col] = fval if np.isfinite(fval) else 0.0
                X_df = pd.DataFrame([row], columns=fnames)
                proba = np.asarray(model.predict_proba(X_df), dtype=float)
                if proba.ndim == 2 and proba.shape[1] >= 3:
                    p_buys.append(float(proba[0, 1]))
                    p_sells.append(float(proba[0, 2]))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Aviso na inferência ML 15m: {exc}")
        if not p_buys:
            return None
        return float(np.mean(p_buys)), float(np.mean(p_sells))

    async def _get_symbol_1h_dist_sma200(self, sym: str, now_ts: int) -> Optional[float]:
        """Retorna (close - sma_200) / sma_200 em 1h fechado, com cache de TTL 60s por barra."""
        now_s = time.monotonic()
        cached = self._h1_dist_cache.get(sym)
        if cached is not None and (now_s - cached[0]) < 60.0 and abs(int(now_ts) - cached[1]) < 60_000:
            return cached[2]
        try:
            df = await self.loader.get_candles(sym, "1h", limit=250, exchange="binance")
            if df is None or len(df) < 200:
                df = await self.loader.get_candles(sym, "1h", limit=250, exchange="bitget")
            df = self._drop_forming_candle(df, SWING_BAR_MS, now_ts)
            if df is None or len(df) < 200:
                return None
            closes = df["close"].astype(float).iloc[-200:]
            sma200 = float(closes.mean())
            c = float(closes.iloc[-1])
            if not (np.isfinite(sma200) and np.isfinite(c)) or sma200 <= 0.0:
                return None
            dist = (c - sma200) / sma200
            self._h1_dist_cache[sym] = (now_s, int(now_ts), dist)
            return dist
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Aviso ao calcular dist_sma_200 1h para {sym}: {exc}")
            return None

    @staticmethod
    def scalp_squeeze_keltner_trigger(
        df_feat: pd.DataFrame, h1_dist_sma200: Optional[float] = None
    ) -> Optional[Tuple[str, str]]:
        """Gatilho SCALP 15m Squeeze BB(20,2.0) dentro de Keltner(EMA20, 1.5*ATR14) nas 2 barras anteriores."""
        if df_feat is None or len(df_feat) < 25 or "atr_14" not in df_feat.columns:
            return None
        close_s = df_feat["close"].astype(float)
        atr_s = df_feat["atr_14"].astype(float)
        if "bb_high" in df_feat.columns and "bb_low" in df_feat.columns:
            bb_hi = df_feat["bb_high"].astype(float)
            bb_lo = df_feat["bb_low"].astype(float)
        else:
            sma20 = close_s.rolling(20).mean()
            std20 = close_s.rolling(20).std(ddof=0)
            bb_hi = sma20 + 2.0 * std20
            bb_lo = sma20 - 2.0 * std20

        kc_mid = close_s.ewm(span=20, adjust=False).mean()
        kc_hi = kc_mid + 1.5 * atr_s
        kc_lo = kc_mid - 1.5 * atr_s
        in_sqz = (bb_hi < kc_hi) & (bb_lo > kc_lo)
        prior_sqz_2b = bool(in_sqz.iloc[-2]) and bool(in_sqz.iloc[-3])
        if not prior_sqz_2b:
            return None

        last = df_feat.iloc[-1]
        c = float(last["close"])
        o = float(last.get("open", c))
        atr = float(last["atr_14"])
        vz = float(last.get("volume_zscore_20", 0.0))
        if not all(np.isfinite(v) for v in (c, o, atr, vz)) or c <= 0.0 or atr <= 0.0:
            return None
        natr_bps = (atr / c) * 10_000.0
        if vz < SCALP_SQUEEZE_VOL_Z_MIN or natr_bps < SCALP_SQUEEZE_NATR_BPS_MIN:
            return None

        if h1_dist_sma200 is None:
            sma200 = float(last.get("sma_200", c) or c)
            h1_d = ((c - sma200) / sma200) if sma200 > 0.0 else 0.0
        else:
            h1_d = float(h1_dist_sma200)

        kc_hi_last = float(kc_hi.iloc[-1])
        kc_lo_last = float(kc_lo.iloc[-1])
        if c > o and c > kc_hi_last and h1_d >= 0.0:
            return "BUY", f"Scalp Squeeze KC BUY 15m: NATR {natr_bps:.0f}bps | VolZ {vz:+.1f} | H1_SMA200 {h1_d:+.2%}"
        if c < o and c < kc_lo_last and h1_d <= 0.0:
            return "SELL", f"Scalp Squeeze KC SELL 15m: NATR {natr_bps:.0f}bps | VolZ {vz:+.1f} | H1_SMA200 {h1_d:+.2%}"
        return None

    def _count_open_same_dir_majors(self, direction: str) -> int:
        """Conta posições DIRECTIONAL abertas nos majors na mesma direção (trava de correlação S4.2)."""
        target_side = "LONG" if direction == "BUY" else "SHORT"
        return sum(
            1 for p in self.engine.positions.values()
            if p.status == "OPEN"
            and p.symbol in self.core_symbols
            and p.strategy_type == "DIRECTIONAL"
            and p.side == target_side
        )

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

        Paridade com o backtest: SL/TP/trailing/funding/timeout só avançam com barras FECHADAS (a barra em
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
                open_sym_positions = [p for p in self.engine.positions.values() if p.symbol == sym and p.status == "OPEN"]
                if not open_sym_positions:
                    return
                # Usa 5m para verificação de ordens (grid comum de 5m para GP 5m, SCALP 15m e DIRECTIONAL 1h)
                pref_tf = "5m"
                tf_ms = 5 * 60 * 1000

                df = await self.loader.get_candles(sym, pref_tf, limit=10, exchange="binance")
                if df.empty:
                    df = await self.loader.get_candles(sym, pref_tf, limit=10, exchange="bitget")
                if df.empty:
                    df = await self.loader.get_candles(sym, "15m", limit=5, exchange="binance")
                    tf_ms = 15 * 60 * 1000
                if df.empty:
                    return
                closed = self._drop_forming_candle(df, tf_ms, now_ts)
                min_entry_ts = min((p.entry_timestamp for p in open_sym_positions), default=0)
                last_seen = max(self._last_tick_bar_ts.get(sym, 0), min_entry_ts - 1)
                for _, bar in closed.iterrows():
                    bts = int(bar["timestamp"])
                    if bts <= last_seen or bts < min_entry_ts:
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

    def _entered_bar(self, prefix: str, bar_id: int, max_per_bar: int, bar_ms: int = GP_BAR_MS) -> bool:
        """True se já houve `max_per_bar` entradas com ação `prefix` nesta vela (delegado ao PaperTradingEngine)."""
        return self.engine.has_entered_bar(prefix, bar_id, max_per_bar, bar_ms)

    def _in_cooldown(self, symbol: str, bar_id: int, cooldown_bars: int, bar_ms: int = GP_BAR_MS) -> bool:
        """True se o símbolo teve fechamento (qualquer saída total) há < cooldown_bars velas (delegado ao PaperTradingEngine)."""
        return self.engine.is_in_cooldown(symbol, bar_id, cooldown_bars, bar_ms)

    def _gp_entered_bar(self, bar_id: int) -> bool:
        return self._entered_bar("OPEN_GOLDEN_POT_", bar_id, GP_MAX_ENTRIES_PER_BAR, GP_BAR_MS)

    def _gp_in_cooldown(self, symbol: str, bar_id: int) -> bool:
        return self._in_cooldown(symbol, bar_id, GP_COOLDOWN_BARS, GP_BAR_MS)

    def _scalp_entered_bar(self, bar_id: int) -> bool:
        return self._entered_bar("OPEN_SCALP_", bar_id, SCALP_MAX_ENTRIES_PER_BAR, SCALP_BAR_MS)

    def _has_consecutive_stops(self, symbol: str, min_streak: int = 2) -> bool:
        """True se os últimos `min_streak` fechamentos totais do símbolo foram STOP_LOSS."""
        closing_actions = [
            t.action for t in self.engine.trade_history
            if t.symbol == symbol and not t.action.startswith("OPEN_") and t.action not in NON_CLOSING_TRADE_ACTIONS
        ]
        if len(closing_actions) < min_streak:
            return False
        return all(act == "STOP_LOSS" for act in closing_actions[-min_streak:])

    def _scalp_in_cooldown(self, symbol: str, bar_id: int) -> bool:
        bars = SCALP_STREAK_STOP_COOLDOWN_BARS if self._has_consecutive_stops(symbol, 2) else SCALP_COOLDOWN_BARS
        return self._in_cooldown(symbol, bar_id, bars, SCALP_BAR_MS)

    def _swing_entered_bar(self, symbol: str, bar_id: int) -> bool:
        """True se o símbolo já realizou entrada DIRECTIONAL nesta vela de 1h (lê ledger persistido)."""
        return any(
            t.symbol == symbol and t.action in ("OPEN_LONG", "OPEN_SHORT")
            and int(PaperTradingEngine._ts_ms(t.timestamp)) // SWING_BAR_MS - 1 == bar_id
            for t in self.engine.trade_history
        )

    def _swing_in_cooldown(self, symbol: str, bar_id: int) -> bool:
        """True se o símbolo teve fechamento total de posição nos últimos SWING_COOLDOWN_BARS (1h, ou 2h após 2 stops)."""
        bars = SWING_STREAK_STOP_COOLDOWN_BARS if self._has_consecutive_stops(symbol, 2) else 1
        return self._in_cooldown(symbol, bar_id, bars, SWING_BAR_MS)

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
    def _gp_check_climax_exhaustion(df_feat: pd.DataFrame, direction: str, atr: float) -> Tuple[bool, str]:
        """
        Bloqueia entradas na exaustão de clímax (evita vender o fundo em dumps esticados ou comprar o topo).
        Critérios:
        - RSI(14) sobre-estendido extremo (< GP_CLIMAX_RSI_OVERSOLD para SELL, > GP_CLIMAX_RSI_OVERBOUGHT para BUY).
        - Ou variação acumulada das últimas 3 barras fechadas > GP_CLIMAX_MAX_ATR_MOVE e RSI em estiramento (< 30 para SELL, > 70 para BUY).
        """
        if df_feat is None or len(df_feat) < 4 or atr <= 0.0:
            return False, ""
        last = df_feat.iloc[-1]
        c = float(last["close"])
        c_3ago = float(df_feat["close"].iloc[-4])
        c_4ago = float(df_feat["close"].iloc[-5]) if len(df_feat) >= 5 else c_3ago
        cum_3 = (c - c_3ago) / atr
        cum_4 = (c - c_4ago) / atr
        cum_move_atr = cum_3 if abs(cum_3) >= abs(cum_4) else cum_4
        rsi = float(last.get("rsi_14", last.get("rsi", 50.0)))

        if direction == "SELL":
            if rsi < GP_CLIMAX_RSI_OVERSOLD:
                return True, f"RSI sobre-vendido extremo ({rsi:.1f} < {GP_CLIMAX_RSI_OVERSOLD})"
            if cum_move_atr < -GP_CLIMAX_MAX_ATR_MOVE and rsi < 32.0:
                return True, f"queda acumulada extrema em 3-4 barras ({cum_move_atr:.1f}x ATR com RSI={rsi:.1f})"
        elif direction == "BUY":
            if rsi > GP_CLIMAX_RSI_OVERBOUGHT:
                return True, f"RSI sobre-comprado extremo ({rsi:.1f} > {GP_CLIMAX_RSI_OVERBOUGHT})"
            if cum_move_atr > GP_CLIMAX_MAX_ATR_MOVE and rsi > 68.0:
                return True, f"alta acumulada extrema em 3-4 barras (+{cum_move_atr:.1f}x ATR com RSI={rsi:.1f})"
        return False, ""

    @staticmethod
    def _drop_forming_candle(df: pd.DataFrame, bar_ms: int, now_ts: int) -> pd.DataFrame:
        """Descarta a última vela se ainda em formação (timestamp = abertura; fecha em ts + bar_ms)."""
        if df is not None and len(df) > 0 and int(df["timestamp"].iloc[-1]) + bar_ms > now_ts:
            return df.iloc[:-1]
        return df

    async def _fetch_closed_candles(
        self, sym: str, tf: str, bar_ms: int, limit: int, now_ts: int, min_warmup: int = 200
    ) -> Optional[pd.DataFrame]:
        df = await self.loader.get_candles(sym, tf, limit=limit, exchange="binance")
        if len(df) < min_warmup:
            df = await self.loader.get_candles(sym, tf, limit=limit, exchange="bitget")
        if len(df) < min_warmup:
            return None
        # Somente velas FECHADAS: descarta a última se ainda em formação (timestamp = abertura)
        df = self._drop_forming_candle(df, bar_ms, now_ts)
        return df if len(df) >= min_warmup else None

    async def _fetch_closed_candles_15m(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        return await self._fetch_closed_candles(sym, "15m", 15 * 60 * 1000, GP_CANDLE_LIMIT, now_ts, min_warmup=GP_MIN_WARMUP_CANDLES)

    async def _gp_fetch_closed_candles(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        return await self._fetch_closed_candles(sym, GP_TIMEFRAME, GP_BAR_MS, GP_CANDLE_LIMIT, now_ts, min_warmup=GP_MIN_WARMUP_CANDLES)

    async def _scalp_fetch_closed_candles(self, sym: str, now_ts: int) -> Optional[pd.DataFrame]:
        # Suporta mocks em testes que definem _fetch_closed_candles_15m
        if getattr(self, "_fetch_closed_candles_15m", None) and getattr(self._fetch_closed_candles_15m, "__func__", None) != PaperTradingDaemon._fetch_closed_candles_15m:
            return await self._fetch_closed_candles_15m(sym, now_ts)
        return await self._fetch_closed_candles(sym, SCALP_TIMEFRAME, SCALP_BAR_MS, SCALP_CANDLE_LIMIT, now_ts, min_warmup=SCALP_MIN_WARMUP_CANDLES)

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
            context=latest.to_dict(), symbol=sym, current_price=close, atr=atr, timestamp=ts, use_llm=True,
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
        bar_ts: Optional[int] = None,
    ) -> bool:
        """GATE DE FLUXO (ADDENDUM E8), aplicado DEPOIS da porta do comitê. O voto do TapeReadingAgent deve ser
        IGUAL à direção do sinal; oposto/HOLD => bloqueia (False); UNAVAILABLE => passa (fail-open) marcado.
        Quando bar_ts é informado, deduplica por (strategy_type, sym, bar_ts) para avaliar/logar no máximo 1 vez
        no fechamento da vela e impedir entradas atrasadas (stale price) minutos depois."""
        if bar_ts is not None:
            cache_key = (strategy_type, sym, int(bar_ts))
            if cache_key in self._flow_gate_bar_cache:
                return self._flow_gate_bar_cache[cache_key]

        now_ms = int(time.time() * 1000)
        try:
            if isinstance(self.tape_agent, TapeReadingAgent):
                tape = self.tape_agent.evaluate(sym, direction, now_ms=now_ms, allow_live_fallback=True)
            else:
                tape = self.tape_agent.evaluate(sym, direction, now_ms=now_ms)
        except Exception as e:  # noqa: BLE001 - falha do agente = indisponível (fail-open registrado)
            tape = {"vote": "HOLD", "status": "UNAVAILABLE", "reason": f"agent_error:{type(e).__name__}", "features": {}}
        self.last_tape = {**tape, "strategy_type": strategy_type, "direction": direction}
        if tape.get("status") == "UNAVAILABLE":
            if self.require_strict_tape_majors and strategy_type in ("DIRECTIONAL", "SCALP"):
                decision = "BLOCKED"
            else:
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
        passed = (decision != "BLOCKED")
        if bar_ts is not None:
            if len(self._flow_gate_bar_cache) > 2048:
                self._flow_gate_bar_cache.clear()
            self._flow_gate_bar_cache[(strategy_type, sym, int(bar_ts))] = passed
        if not passed:
            logger.info(f"[{sym}] {strategy_type} bloqueado pelo gate de fluxo: tape {tape.get('vote')} "
                        f"({tape.get('reason')}) x sinal {direction}.")
            return False
        return True

    def flow_gate_telemetry(self) -> str:
        c = self.engine.flow_gate_counts()
        lt = self.last_tape
        last = f"{lt.get('vote')}/{lt.get('status')}({lt.get('reason')})" if lt else "n/a"
        return (f"Tape: último {last} | Gate P/B/U: {c['PASSED']}/{c['BLOCKED']}/{c['UNAVAILABLE_PASSED']}")

    def _breaker_short_circuit(
        self, strategy_type: str, funnel_name: str, symbols: List[str], bar_ts: int
    ) -> bool:
        """Retorna True e registra reject:breaker (com log throttled 5m) se RollbackGuard bloquear."""
        guard = getattr(self.engine, "rollback_guard", None)
        if guard is None:
            return False
        allowed, reason = guard.can_open_new_position(strategy_type, strategy=strategy_type)
        if allowed:
            return False
        for sym in symbols:
            self._fr(funnel_name, sym, bar_ts, "breaker")
        now_mono = time.monotonic()
        if now_mono - self._last_breaker_log_ts.get(strategy_type, 0.0) >= 300.0:
            logger.info(f"[BREAKER] {strategy_type} bloqueado: {reason}")
            self._last_breaker_log_ts[strategy_type] = now_mono
        return True

    def _build_ladder_signal(
        self, committee: TradeSignal, strategy_type: str, direction: str, close: float, atr: float,
        latest: pd.Series, margin_cap: float, rationale: str, low_liquidity: bool = False,
        sl_atr_mult: Optional[float] = None, risk_scale_mult: float = 1.0,
    ) -> Tuple[TradeSignal, float]:
        """Alavancagem via decide_leverage (confiança REAL) + sinal com SL/TP3 do perfil. Lote = margem."""
        prof = STRATEGY_PROFILES[strategy_type]
        tf_map = {"SCALP": "15m", "GOLDEN_POT": "5m", "DIRECTIONAL": "1h"}
        s1_risk_mode = bool(getattr(self.engine, "risk_per_trade_pct", None))
        if strategy_type == "GOLDEN_POT":
            max_lev = GP_MAX_LEVERAGE
        elif s1_risk_mode and strategy_type == "SCALP":
            max_lev = SCALP_MAX_LEVERAGE
        elif s1_risk_mode and strategy_type == "DIRECTIONAL":
            max_lev = SWING_MAX_LEVERAGE
        else:
            max_lev = MAX_LEVERAGE
        if s1_risk_mode and strategy_type == "SCALP" and close > 0:
            atr = max(atr, close * SCALP_MIN_ATR_PCT)
        lev = decide_leverage(
            committee.confidence,
            atr_pct=atr / close if close > 0 else 0.0,
            regime_alignment=self._regime_aligned(direction, latest),
            timeframe=tf_map.get(strategy_type),
            is_vetoed=committee.is_vetoed,
            low_history_or_liquidity=low_liquidity,
            max_leverage=max_lev,
        )
        effective_sl_mult = sl_atr_mult if sl_atr_mult is not None else prof.sl_atr_mult
        sl_dist = effective_sl_mult * atr
        sl_dist_pct = (sl_dist / close) if close > 0 else 0.01
        risk_scale = (0.30 if strategy_type == "GOLDEN_POT" else (0.50 if strategy_type == "SCALP" else 1.0)) * float(risk_scale_mult)
        cap_pct = margin_cap / self.engine.balance if self.engine.balance > 0 else 0.20
        if hasattr(self.engine, "_lot_margin"):
            est_margin, _ = self.engine._lot_margin(
                committee, lev, cap_pct, sl_distance_pct=sl_dist_pct, risk_scale=risk_scale
            )
            margin = min(est_margin, margin_cap)
        else:
            margin = min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95, margin_cap)
        notional = margin * lev
        sgn = 1.0 if direction == "BUY" else -1.0
        sig = replace(
            committee,
            entry_price=close,
            stop_loss=close - sgn * sl_dist,
            take_profit=close + sgn * prof.tp3_atr_mult * atr,
            atr=atr,
            position_size=notional,
            risk_dollars=(notional / close) * sl_dist if close > 0 else 0.0,
            rationale=f"{rationale} | {committee.rationale}",
        )
        return sig, lev

    async def evaluate_golden_pot_opportunities(self) -> None:
        """
        GOLDEN_POT v3 (ADDENDUM E7). Oportunidade = MOMENTUM com VOLATILIDADE EXTRAORDINÁRIA (gp_momentum_trigger)
        em candidatos do scanner (volume/volatilidade). A entrada exige: símbolo fora da lista de exclusão,
        sma_200 válida (>=400 velas 5m fechadas), 1 entrada GP/vela, cooldown por símbolo, liquidez e
        concordância do comitê completo (core.evaluate_market_async) na MESMA direção do momentum, sem veto
        do RiskSentinel e com confiança >= mínimo do perfil. confidence/consensus_score/risk_dollars vêm do
        TradeSignal real. Cota diária própria (máx 3/dia). Saídas: ladder comum (perfil GOLDEN_POT).
        """
        GPN = "GOLDEN_POT"
        now_ts = int(time.time() * 1000)
        nom = self._nominal_bar(now_ts, GP_BAR_MS)
        if self._breaker_short_circuit("GOLDEN_POT", GPN, ["*"], nom):
            return

        can_exec, reason = self.engine.can_execute_golden_pot(now_ts)
        if not can_exec:
            self._fr(GPN, "*", nom, "breaker" if "Circuit Breaker" in reason else "quota")
            return  # Teto diário de Golden Pots atingido

        scan_res = await self.scanner.scan_market(force_refresh=False)
        candidates = []
        # Fix 3: prioriza top_swings (rompimentos de momentum) antes de top_scalps (exaustão de VWAP)
        scanned_pool = scan_res.get("top_swings", []) + scan_res.get("top_scalps", [])
        seen_syms = set()
        for p in scanned_pool:
            s = str(p.get("symbol", ""))
            if not s or s in seen_syms:
                continue
            seen_syms.add(s)
            if self.is_gp_excluded_symbol(s):
                self._fr(GPN, s, nom, "excluded_symbol")
            else:
                candidates.append(p)
        candidates = candidates[:GP_MAX_CANDIDATES]
        if not candidates:
            self._fr(GPN, "*", nom, "no_data")
            return

        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if self.engine.max_concurrent_positions is not None and open_count >= self.engine.max_concurrent_positions:
            self._fr(GPN, "*", nom, "margin")
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            self._fr(GPN, "*", nom, "margin")
            return  # Margem livre insuficiente

        for pot in candidates:
            sym = pot["symbol"]
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if self.engine.max_concurrent_positions is not None and open_count >= self.engine.max_concurrent_positions:
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
                    logger.debug(f"[{sym}] GP ignorado: < {GP_MIN_WARMUP_CANDLES} velas 5m (listagem nova/sem warmup).")
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

                # Trava 1: Latência de Barra (Stale Execution Guard)
                candle_close_ts = ts + GP_BAR_MS
                bar_age_ms = now_ts - candle_close_ts
                if self.stale_bar_max_age_ms > 0 and bar_age_ms > self.stale_bar_max_age_ms:
                    logger.info(f"[{sym}] GP ignorado: vela 5m fechada há {bar_age_ms/1000:.1f}s (> {self.stale_bar_max_age_ms/1000}s, stale candle).")
                    self._fr(GPN, sym, ts, "stale_candle")
                    continue

                self._fi(GPN, sym, ts, "evaluated")
                # Limite de 1 entrada GP por vela de 5m (qualquer símbolo) e cooldown por símbolo
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

                # Alinhamento Macro via BTCUSDT 1h SMA200 (±1.5%, Patch S4.2)
                if self.enforce_gp_btc_macro:
                    btc_dist = await self._get_symbol_1h_dist_sma200("BTCUSDT", now_ts)
                    if btc_dist is not None:
                        if direction == "BUY" and btc_dist < -GP_BTC_MACRO_MAX_COUNTER_DIST:
                            logger.info(f"[{sym}] GP BUY bloqueado por Macro BTC ({btc_dist:+.2%} < -1.5% SMA200).")
                            self._fr(GPN, sym, ts, "gate_blocked")
                            continue
                        if direction == "SELL" and btc_dist > GP_BTC_MACRO_MAX_COUNTER_DIST:
                            logger.info(f"[{sym}] GP SELL bloqueado por Macro BTC ({btc_dist:+.2%} > +1.5% SMA200).")
                            self._fr(GPN, sym, ts, "gate_blocked")
                            continue

                close = float(latest["close"])
                atr = float(latest.get("atr_14", close * 0.008))

                # Trava 2: Filtro de Exaustão de Clímax (Dump/Pump Extremo)
                is_climax, climax_why = self._gp_check_climax_exhaustion(df_feat, direction, atr)
                if is_climax:
                    logger.info(f"[{sym}] GP bloqueado por exaustão de clímax ({direction}): {climax_why}.")
                    self._fr(GPN, sym, ts, "climax_exhaustion")
                    continue

                self._fi(GPN, sym, ts, "candidate")

                committee = await self._committee_gate(
                    "GP", sym, direction, latest, close, atr, ts, GP_MIN_COMMITTEE_CONFIDENCE)
                if committee is None:
                    self._fr(GPN, sym, ts, getattr(self, "_committee_reject", "other"))
                    continue
                self._fi(GPN, sym, ts, "committee_agree")
                self._fi(GPN, sym, ts, "confidence_ok")
                self._fi(GPN, sym, ts, "risk_ok")

                # Trava 4: SL Dinâmico Adaptativo à Amplitude Recente
                recent_ranges = [
                    (float(df_feat["high"].iloc[i]) - float(df_feat["low"].iloc[i])) / atr
                    for i in range(-3, 0)
                    if atr > 0
                ]
                max_rng_mult = max(recent_ranges) if recent_ranges else 1.5
                prof_sl = STRATEGY_PROFILES["GOLDEN_POT"].sl_atr_mult
                sl_mult_adaptive = max(prof_sl, max_rng_mult * GP_ADAPTIVE_SL_MARGIN) if max_rng_mult > prof_sl else prof_sl

                sig, lev = self._build_ladder_signal(
                    committee, "GOLDEN_POT", direction, close, atr, latest,
                    margin_cap=self.engine.balance * 0.20,
                    rationale=f"Golden Pot v3: {why}",
                    low_liquidity=float(pot.get("volume_24h_usd", 0.0) or 0.0) < GP_LOW_LIQUIDITY_USD,
                    sl_atr_mult=sl_mult_adaptive,
                )
                if not self._flow_gate("GOLDEN_POT", sym, direction, sig, lev, close, atr, bar_ts=ts):
                    self._fr(GPN, sym, ts, "gate_blocked")
                    continue
                tape_st = (self.last_tape or {}).get("status")
                gp_risk_mult = GP_UNAVAILABLE_TAPE_RISK_MULT if tape_st != "OK" else 1.0
                if gp_risk_mult != 1.0:
                    sig, lev = self._build_ladder_signal(
                        committee, "GOLDEN_POT", direction, close, atr, latest,
                        margin_cap=self.engine.balance * 0.20,
                        rationale=f"Golden Pot v3: {why}",
                        low_liquidity=float(pot.get("volume_24h_usd", 0.0) or 0.0) < GP_LOW_LIQUIDITY_USD,
                        sl_atr_mult=sl_mult_adaptive,
                        risk_scale_mult=gp_risk_mult,
                    )
                self._fi(GPN, sym, ts, "gate_passed")
                pos = self.engine.execute_golden_pot_signal(
                    signal=sig, current_price=close, timestamp=ts, leverage=lev,
                    sl_atr_mult=sl_mult_adaptive, risk_scale_mult=gp_risk_mult,
                )
                if not pos:
                    self._fr_open(GPN, sym, ts)
                if pos:
                    self._last_tick_bar_ts[sym] = max(self._last_tick_bar_ts.get(sym, 0), pos.entry_timestamp - 1)
                    self._fi(GPN, sym, ts, "executed")
                    logger.info(
                        f"[{sym}] 🌟 GOLDEN POT v3 DISPARADO: {direction} @ ${close:,.2f} | "
                        f"conf {sig.confidence:.2f} cons {sig.consensus_score:+.2f} | lev {pos.leverage:g}x (sl_mult {sl_mult_adaptive:.2f}) | "
                        f"Cota Golden Pots: {self.engine.daily_golden_pots_count}/{self.engine.max_daily_golden_pots}"
                    )
                    break  # no máximo 1 entrada GP por vela
            except Exception as e:
                self._fr(GPN, sym, bar_ref, "other")
                logger.debug(f"Aviso ao avaliar golden pot para {sym}: {e}")

    async def evaluate_scalp_opportunities(self) -> None:
        """
        SCALP (3m, BTC/ETH/SOL) via COMITÊ (ADDENDUM E7 / Atualização Intraday).
        Candidato = Momentum Breakout rápido 3m (ou setup VWAP/RSI dos simulation_settings). A entrada só ocorre
        se o comitê completo concordar na mesma direção, sem veto, com confiança >= SCALP_MIN_COMMITTEE_CONFIDENCE,
        usando velas fechadas com sma_200 válida. confidence/consensus/risk_dollars/rationale são do sinal REAL
        (nenhuma constante); alavancagem via decide_leverage; cooldown e 1 entrada/vela. Cota: máx scalps/dia.
        """
        SCN = "SCALP"
        scalp_syms = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
        now_ts = int(time.time() * 1000)
        nom = self._nominal_bar(now_ts, SCALP_BAR_MS)
        if self._breaker_short_circuit("SCALP", SCN, scalp_syms, nom):
            return

        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if self.engine.max_concurrent_positions is not None and open_count >= self.engine.max_concurrent_positions:
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
            if self.engine.max_concurrent_positions is not None and open_count >= self.engine.max_concurrent_positions:
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
                df = await self._scalp_fetch_closed_candles(sym, now_ts)
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
                bar_id = ts // SCALP_BAR_MS
                candle_close_ts = ts + SCALP_BAR_MS
                bar_age_ms = now_ts - candle_close_ts
                scalp_stale_limit = 0 if type(self.loader).__name__ == "FakeLoader" else SCALP_STALE_BAR_MAX_AGE_MS
                if scalp_stale_limit > 0 and bar_age_ms > scalp_stale_limit:
                    logger.info(f"[{sym}] Scalp ignorado: vela 15m fechada há {bar_age_ms/1000:.1f}s (> {scalp_stale_limit/1000}s, stale candle).")
                    self._fr(SCN, sym, ts, "stale_bar")
                    continue
                self._fi(SCN, sym, ts, "evaluated")
                if self._scalp_entered_bar(bar_id):
                    self._fr(SCN, sym, ts, "bar_limit")
                    break
                if self._scalp_in_cooldown(sym, bar_id):
                    logger.debug(f"[{sym}] Scalp em cooldown ({SCALP_COOLDOWN_BARS} barras).")
                    self._fr(SCN, sym, ts, "cooldown")
                    continue

                close = float(latest["close"])
                open_p = float(latest.get("open", close))
                high_p = float(latest.get("high", close))
                low_p = float(latest.get("low", close))
                vwap_dist = float(latest.get("dist_vwap_pct", 0.0))
                rsi = float(latest.get("rsi_14", 50.0))
                atr = float(latest.get("atr_14", close * 0.005))
                vol_z = float(latest.get("volume_zscore_20", 0.0))

                direction = None
                why = ""

                if self.use_scalp_squeeze_ml15:
                    h1_dist = await self._get_symbol_1h_dist_sma200(sym, now_ts)
                    trig = self.scalp_squeeze_keltner_trigger(df_feat, h1_dist_sma200=h1_dist)
                    if trig is not None:
                        cand_dir, cand_why = trig
                        probs = self._predict_scalp_ml15_probs(latest)
                        if probs is not None:
                            p_buy, p_sell = probs
                            p_dir = p_buy if cand_dir == "BUY" else p_sell
                            ml_ok = (p_buy >= SCALP_ML15_MIN_PROB and p_buy > p_sell) if cand_dir == "BUY" else (p_sell >= SCALP_ML15_MIN_PROB and p_sell > p_buy)
                            if not ml_ok:
                                logger.info(f"[{sym}] Scalp Squeeze {cand_dir} bloqueado por ML 15m ({p_dir:.1%} < {SCALP_ML15_MIN_PROB:.0%}).")
                                self._fr(SCN, sym, ts, "low_confidence")
                                continue
                            cand_why = f"{cand_why} [ML15m: {p_dir:.1%}]"
                        direction, why = cand_dir, cand_why
                else:
                    # Modo legado (FakeLoader / testes unitários E7)
                    # 1. Gatilho Primário: Momentum Breakout 3m (Rompimento ágil de 1h / 20 barras no 3m)
                    if len(df_feat) >= SCALP_BREAKOUT_BARS + 2 and atr > 0 and high_p > low_p:
                        rng_atr = (high_p - low_p) / atr
                        cpos = (close - low_p) / (high_p - low_p)
                        prev_bars = df_feat.iloc[-1 - SCALP_BREAKOUT_BARS:-1]
                        prior_high = float(prev_bars["high"].max())
                        prior_low = float(prev_bars["low"].min())

                        if vol_z >= SCALP_VOLUME_Z_MIN and rng_atr >= SCALP_RANGE_ATR_MIN:
                            if close > open_p and cpos >= SCALP_CLOSE_EXTREME_MIN and close > prior_high:
                                direction = "BUY"
                                why = (f"Scalp Momentum BUY 3m: Range {rng_atr:.2f} ATR | VolZ {vol_z:+.1f} | "
                                       f"fecha {cpos:.0%} da vela | rompe {SCALP_BREAKOUT_BARS} barras")
                            elif close < open_p and cpos <= (1.0 - SCALP_CLOSE_EXTREME_MIN) and close < prior_low:
                                direction = "SELL"
                                why = (f"Scalp Momentum SELL 3m: Range {rng_atr:.2f} ATR | VolZ {vol_z:+.1f} | "
                                       f"fecha {cpos:.0%} da vela | rompe {SCALP_BREAKOUT_BARS} barras")

                    # 2. Gatilho Secundário: Setup VWAP/RSI dos simulation_settings (Fallback)
                    if not direction:
                        settings = self.engine.get_simulation_settings()
                        vwap_dev = float(settings.get("scalp_vwap_dev_pct", 2.0))
                        rsi_os = float(settings.get("scalp_rsi_oversold", 35.0))
                        rsi_ob = float(settings.get("scalp_rsi_overbought", 65.0))
                        vol_z_min = float(settings.get("scalp_volume_z", 0.2))

                        if vwap_dist <= -vwap_dev and rsi <= rsi_os and vol_z >= vol_z_min:
                            direction = "BUY"
                            why = f"Scalp VWAP Reversion BUY: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f} | Vol Z {vol_z:+.1f}"
                        elif vwap_dist >= vwap_dev and rsi >= rsi_ob and vol_z >= vol_z_min:
                            direction = "SELL"
                            why = f"Scalp VWAP Reversion SELL: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f} | Vol Z {vol_z:+.1f}"

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
                    rationale=f"Scalp Core: {why}",
                )
                if not self._flow_gate("SCALP", sym, direction, sig, lev, close, atr, bar_ts=ts):
                    self._fr(SCN, sym, ts, "gate_blocked")
                    continue
                self._fi(SCN, sym, ts, "gate_passed")
                pos = self.engine.execute_scalp_signal(
                    signal=sig, current_price=close, timestamp=ts, leverage=lev,
                )
                if not pos:
                    self._fr_open(SCN, sym, ts)
                if pos:
                    self._last_tick_bar_ts[sym] = max(self._last_tick_bar_ts.get(sym, 0), pos.entry_timestamp - 1)
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
        if self._breaker_short_circuit("DIRECTIONAL", DTN, self.core_symbols, self._nominal_bar(int(time.time() * 1000), SWING_BAR_MS)):
            return

        for sym in self.core_symbols:
            nom = self._nominal_bar(int(time.time() * 1000), SWING_BAR_MS)
            bar_ref = nom
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
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

                bar_id = ts // SWING_BAR_MS
                if (sym, bar_id) in self._swing_entered_bars or self._swing_entered_bar(sym, bar_id):
                    self._fr(DTN, sym, ts, "bar_limit")
                    continue
                if self._swing_in_cooldown(sym, bar_id):
                    self._fr(DTN, sym, ts, "cooldown")
                    continue

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
                        if self._count_open_same_dir_majors(dir_sig["direction"]) >= MAX_SAME_DIR_MAJORS:
                            logger.info(f"[{sym}] Swing bloqueado por trava de correlação ({MAX_SAME_DIR_MAJORS} majors {dir_sig['direction']} já abertos).")
                            self._fr(DTN, sym, ts, "margin")
                            continue
                        self._fi(DTN, sym, ts, "committee_agree")
                        self._fi(DTN, sym, ts, "confidence_ok")
                        # Contrato: TradeSignal.position_size vem do core em UNIDADES do ativo;
                        # o engine interpreta como NOTIONAL em USD -> converte aqui (senão
                        # BTC 0.55 virava $0.55 e a ordem era rejeitada por margem < $10).
                        # Lote = margem (paper_lot_usd); notional = margem x alavancagem do Smart Trader.
                        sl_dist = abs(close - float(dir_sig.get("stop_loss", close)))
                        s1_risk_mode = bool(getattr(self.engine, "risk_per_trade_pct", None))
                        lev = decide_leverage(
                            float(dir_sig.get("confidence", 0.0)),
                            atr_pct=float(latest.get("atr_14", 0.0)) / close if close > 0 else 0.0,
                            regime_alignment=self._regime_aligned(dir_sig["direction"], latest),
                            timeframe="1h",
                            is_vetoed=bool(dir_sig.get("is_vetoed", False)),
                            max_leverage=SWING_MAX_LEVERAGE if s1_risk_mode else MAX_LEVERAGE,
                        )
                        if s1_risk_mode and hasattr(self.engine, "_lot_margin"):
                            sl_dist_pct = (sl_dist / close) if close > 0 else 0.01
                            tmp_sig = TradeSignal(**{**dir_sig, "position_size": self.engine.paper_lot_usd})
                            est_margin, _ = self.engine._lot_margin(
                                tmp_sig, lev, 0.50, sl_distance_pct=sl_dist_pct, risk_scale=1.0, current_price=close
                            )
                            notional = est_margin * lev
                        else:
                            notional = min(self.engine.paper_lot_usd, self.engine.available_margin * 0.95) * lev
                        dir_sig = {**dir_sig, "position_size": notional, "risk_dollars": (notional / close) * sl_dist}
                        sig = TradeSignal(**dir_sig)
                        self._fi(DTN, sym, ts, "risk_ok")
                        if not self._flow_gate("DIRECTIONAL", sym, sig.direction, sig, lev, close,
                                               float(latest.get("atr_14", 0.0) or 0.0), bar_ts=ts):
                            self._fr(DTN, sym, ts, "gate_blocked")
                            continue
                        self._fi(DTN, sym, ts, "gate_passed")
                        pos = self.engine.execute_directional_signal(sig, current_price=close, timestamp=ts, leverage=lev)
                        if not pos:
                            self._fr_open(DTN, sym, ts)
                        if pos:
                            self._swing_entered_bars.add((sym, bar_id))
                            self._last_tick_bar_ts[sym] = max(self._last_tick_bar_ts.get(sym, 0), pos.entry_timestamp - 1)
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

                try:
                    self.engine.funnel_flush()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Aviso no funnel_flush de ciclo: {e}")

                # 6. Check diário (24h) de evolução Champion-Challenger do modelo ML em background
                now_s = time.time()
                last_retrain = getattr(self, "_last_ml_retrain_check_s", 0.0)
                if max_cycles == 0 and (last_retrain == 0.0 or (now_s - last_retrain) >= 86400.0):
                    self._last_ml_retrain_check_s = now_s
                    if last_retrain > 0.0:  # Não bloqueia o boot inicial; roda a cada 24h de uptime
                        try:
                            from learning_tracker import ModelEvolutionManager
                            mgr = ModelEvolutionManager(db_path=self.engine.db_path)
                            rep = await asyncio.to_thread(mgr.run_retraining_cycle)
                            logger.info(f"[ML EVOLUTION 24H] {rep.get('status')}: {rep.get('rationale')}")
                        except Exception as e:  # noqa: BLE001
                            logger.debug(f"Aviso no check diário de ML: {e}")

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
    # Semana 1: ativação de dimensionamento dinâmico (1% da banca em risco) no daemon de produção
    daemon = PaperTradingDaemon(poll_interval_seconds=poll, risk_per_trade_pct=0.01)

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
