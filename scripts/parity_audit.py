"""
scripts/parity_audit.py - HARNESS DE PARIDADE BACKTEST x PAPER (piloto).

Diretriz "laranjas com laranjas": o piloto só testa o que o backtest validou se, DADAS AS MESMAS VELAS
FECHADAS e o MESMO CONJUNTO DE SINAIS, ambos os motores produzirem os mesmos trades. Este script:

  1. lê (SOMENTE LEITURA, SQLite mode=ro) velas 1h fechadas de BTC/ETH/SOL (binance) num período curto;
  2. gera um conjunto de sinais determinístico (comitê PROD-world = o que o piloto swing roda; ou "forced");
  3. passa o MESMO vetor de sinais pelo `BacktestEngine` (exec. open t+1, sem rede) e pelo
     `PaperTradingEngine` (DB temporário, sem rede) reproduzindo a sequência do daemon
     (tick da vela -> avaliar sinal -> `execute_directional_signal` ao CLOSE da vela t);
  4. reconcilia trade a trade (chave = barra do sinal) e classifica cada diferença em CATEGORIAS.

Nenhum arquivo de produção é alterado. Uso:
    .venv/bin/python scripts/parity_audit.py [--bars 1500] [--scenario committee|forced] [--json out.json]

Convenções
----------
* "bt"  = BacktestEngine configurado como nos experimentos validados (A/B + phase4): fee taker 0.06%,
          slippage 0.02%, funding 0.01%/8h, risco 1%, 1x, TP 1.5 ATR / SL 1.0 ATR, timeout 48 barras.
* "pp"  = PaperTradingEngine (smart_trade.py) chamado como `evaluate_swing_opportunities` faz
          (leverage=1.0 aqui para isolar a mecânica; a alavancagem do Smart Trader é outro item da matriz).
* As categorias de divergência estão em `CATEGORIES`; as aceitas POR DESENHO em `ACCEPTED_BY_DESIGN`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "datasets", "market_cache.db")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HOUR_MS = 3_600_000

# Parâmetros "validados" (A/B + phase4): fixos a priori
BT_PARAMS = dict(
    initial_capital=10_000.0, fee_rate=0.0006, slippage=0.0002, funding_rate_8h=0.0001, leverage=1.0,
    tp_atr_multiple=1.5, sl_atr_multiple=1.0, trailing_stop=False, risk_per_trade=0.01,
    allow_reversal=False, timeframe="1h", max_holding_bars=48,
)
CONF_GATE = 0.70
WARMUP_BARS = 700          # velas antes da janela de teste (sma_200 válida + folga)

# ----------------------------------------------------------------------------------------------
# Taxonomia de divergências
# ----------------------------------------------------------------------------------------------
CATEGORIES: Dict[str, str] = {
    "ENTRY_TIMING": "Backtest entra no OPEN de t+1; piloto entra no CLOSE de t (timestamp da vela t) - A3.",
    "ENTRY_PRICE": "Preço de entrada difere (open[t+1] vs close[t]; slippage spot/perp vs único).",
    "EXIT_REASON": "Motivo de saída difere (ex.: HORIZON_TIMEOUT inexistente no piloto; ladder; liquidação).",
    "EXIT_TIMING": "Barra de saída difere.",
    "EXIT_FEE_MODEL": "Taxa de saída/entrada por unidade de notional difere (maker/taker, slippage perp).",
    "FUNDING": "Backtest aplica funding 8h; piloto direcional não aplica funding.",
    "SIZING": "Notional absoluto difere (1% de risco vs lote fixo em margem). Comparar % do notional.",
    "PNL_PCT": "Retorno líquido % do notional difere além da tolerância.",
    "UNMATCHED_BT": "Trade existe só no backtest (piloto não entrou/ já estava ocupado/ etc.).",
    "UNMATCHED_PAPER": "Trade existe só no piloto.",
}

# Divergências aceitas POR DESENHO (não reprovam o teste de paridade; documentadas em docs/PARIDADE_BACKTEST_PILOTO.md)
ACCEPTED_BY_DESIGN: Dict[str, str] = {
    "SIZING": "Piloto usa lote fixo em MARGEM (paper_lot_usd) x alavancagem do Smart Trader; backtest usa 1% de "
              "risco. Taxas/slippage/PnL são proporcionais ao notional -> compara-se retorno % do notional.",
}


# ----------------------------------------------------------------------------------------------
# Dados (somente leitura)
# ----------------------------------------------------------------------------------------------
def _ro_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)


def load_closed_candles(symbol: str, n_last: int, db_path: str = DB_PATH, timeframe: str = "1h",
                        exchange: str = "binance") -> pd.DataFrame:
    """Últimas `n_last` velas; a ÚLTIMA barra do cache é descartada (pode estar em formação)."""
    with _ro_conn(db_path) as c:
        df = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM candles "
            "WHERE exchange=? AND symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
            c, params=(exchange, symbol, timeframe, int(n_last) + 1),
        )
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


def load_funding(symbol: str, db_path: str = DB_PATH) -> pd.DataFrame:
    with _ro_conn(db_path) as c:
        return pd.read_sql(
            "SELECT timestamp, funding_rate FROM funding_rates WHERE exchange='binance' AND symbol=? "
            "ORDER BY timestamp", c, params=(symbol,))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mesmo caminho do piloto (`_compute_swing_features`): drop_na=False + filtro sma_200 válida."""
    from data.feature_engineering import FeatureEngineer
    feat = FeatureEngineer.compute_all_features(df, drop_na=False)
    sma = feat["sma_200"].astype(float)
    return feat[np.isfinite(sma) & (sma > 0.0)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Conjunto de sinais (idêntico para os dois motores)
# ----------------------------------------------------------------------------------------------
def committee_signals(feat: pd.DataFrame, test_start: int) -> np.ndarray:
    """Sinais do comitê no 'mundo PROD' (contexto = linha de features, sem macro/sentiment/onchain,
    sem volume_sma/atr_median, TechnicalAgent sem modelo) com gate de confiança 0.70."""
    from core import TradingCore
    core = TradingCore()
    sig = np.zeros(len(feat), dtype=int)
    for i in range(test_start, len(feat)):
        row = feat.iloc[i]
        ts = core.evaluate_market_deterministic(
            context=row.to_dict(), symbol="X", current_price=float(row["close"]),
            atr=float(row["atr_14"]), timestamp=int(row["timestamp"]))
        if ts.direction in ("BUY", "SELL") and not ts.is_vetoed and ts.confidence >= CONF_GATE:
            sig[i] = 1 if ts.direction == "BUY" else 2
    return sig


def forced_signals(feat: pd.DataFrame, test_start: int, every: int = 18) -> np.ndarray:
    """Sinais sintéticos determinísticos (garantem amostra mesmo sem sinais do comitê):
    a cada `every` barras, direção = sinal do retorno de 24 barras (sem lookahead)."""
    close = feat["close"].to_numpy(float)
    sig = np.zeros(len(feat), dtype=int)
    for i in range(max(test_start, 24), len(feat), every):
        sig[i] = 1 if close[i] >= close[i - 24] else 2
    return sig


# ----------------------------------------------------------------------------------------------
# Execução nos dois motores
# ----------------------------------------------------------------------------------------------
@dataclass
class TradeView:
    """Visão normalizada de um trade (bt ou pp)."""
    sig_idx: int
    direction: str
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    notional: float
    fees: float
    funding: float
    pnl_net: float
    reasons: List[str] = field(default_factory=list)

    @property
    def fee_pct(self) -> float:
        return self.fees / self.notional if self.notional > 0 else 0.0

    @property
    def ret_pct(self) -> float:
        return self.pnl_net / self.notional if self.notional > 0 else 0.0


def run_backtest(feat: pd.DataFrame, signals: np.ndarray, test_start: int) -> List[TradeView]:
    from backtesting import BacktestEngine
    df = feat.iloc[test_start:].reset_index(drop=True)
    sig = signals[test_start:]
    res = BacktestEngine(**BT_PARAMS).run(df, sig, atr_col="atr_14")
    out: List[TradeView] = []
    for t in res.trades:
        out.append(TradeView(
            sig_idx=test_start + t.entry_idx - 1, direction=t.direction, entry_ts=int(t.entry_timestamp),
            entry_price=t.entry_price, exit_ts=int(t.exit_timestamp), exit_price=t.exit_price,
            notional=t.entry_price * t.size, fees=t.fee_total, funding=t.funding_total,
            pnl_net=t.pnl_net, reasons=[t.exit_reason]))
    return out


def run_paper(symbol: str, feat: pd.DataFrame, signals: np.ndarray, test_start: int,
              leverage: float = 1.0) -> Tuple[List[TradeView], Dict[str, Any]]:
    """Reproduz a sequência do daemon: [tick da vela] -> [avaliar sinal -> execute_directional_signal ao CLOSE]."""
    from smart_trade import PaperTradingEngine
    from signal_generator import SignalGenerator
    from consensus_judge import ConsensusDecision
    from dataclasses import replace

    meta: Dict[str, Any] = {"rejected": 0}
    tmpdir = tempfile.mkdtemp(prefix="parity_")
    eng = PaperTradingEngine(initial_capital=BT_PARAMS["initial_capital"],
                             db_path=os.path.join(tmpdir, "parity.db"), restore=False)
    gen = SignalGenerator(capital=BT_PARAMS["initial_capital"], risk_per_trade=0.01)
    pos_sig_idx: Dict[str, int] = {}

    for i in range(test_start, len(feat)):
        row = feat.iloc[i]
        ts = int(row["timestamp"])
        # 1) update_active_positions (daemon): tick com a vela
        eng.update_market_tick(symbol=symbol, high=float(row["high"]), low=float(row["low"]),
                               close=float(row["close"]), timestamp=ts)
        # 2) evaluate_swing_opportunities
        d = int(signals[i])
        if d in (1, 2):
            cd = ConsensusDecision(direction="BUY" if d == 1 else "SELL", confidence=0.8)
            close, atr = float(row["close"]), float(row["atr_14"])
            sig = gen.generate(cd, symbol, close, atr, ts)
            sl_dist = abs(close - sig.stop_loss)
            notional = min(eng.paper_lot_usd, eng.available_margin * 0.95) * leverage
            sig = replace(sig, position_size=notional, risk_dollars=(notional / close) * sl_dist)
            pos = eng.execute_directional_signal(sig, current_price=close, timestamp=ts, leverage=leverage)
            if pos:
                pos_sig_idx[pos.position_id] = i
            else:
                meta["rejected"] += 1

    # reconstrói trades por posição a partir do ledger em memória
    by_pos: Dict[str, List[Any]] = {}
    for t in eng.trade_history:
        by_pos.setdefault(t.position_id, []).append(t)
    out: List[TradeView] = []
    open_at_end = 0
    for pid, trs in by_pos.items():
        opens = [t for t in trs if t.action.startswith("OPEN_")]
        exits = [t for t in trs if not t.action.startswith("OPEN_")]
        if not opens:
            continue
        if pid in eng.positions:   # ainda aberta: não comparável (backtest fecha em END_OF_DATA)
            open_at_end += 1
            continue
        o = opens[0]
        qty_exit = sum(t.quantity for t in exits) or 1.0
        exit_px = sum(t.price * t.quantity for t in exits) / qty_exit
        fees = o.fee + sum(t.fee for t in exits)
        net = sum(t.pnl for t in exits) - o.fee
        out.append(TradeView(
            sig_idx=pos_sig_idx.get(pid, -1), direction="LONG" if "LONG" in o.action else "SHORT",
            entry_ts=int(o.timestamp), entry_price=o.price, exit_ts=int(exits[-1].timestamp),
            exit_price=exit_px, notional=o.price * o.quantity, fees=fees, funding=0.0, pnl_net=net,
            reasons=[t.action for t in exits]))
    meta["open_at_end"] = open_at_end
    out.sort(key=lambda t: t.sig_idx)
    return out, meta


# ----------------------------------------------------------------------------------------------
# Reconciliação
# ----------------------------------------------------------------------------------------------
_REASON_MAP = {"SL": "STOP_LOSS", "TP": "TAKE_PROFIT", "TRAILING_SL": "TRAILING_STOP"}


@dataclass
class Diff:
    symbol: str
    sig_idx: int
    category: str
    detail: str


def reconcile(symbol: str, bt: List[TradeView], pp: List[TradeView], *, tol_price: float = 1e-4,
              tol_ret: float = 5e-4, tol_fee: float = 1e-5, tol_size: float = 0.5) -> List[Diff]:
    diffs: List[Diff] = []
    bt_m = {t.sig_idx: t for t in bt if t.reasons != ["END_OF_DATA"]}
    pp_m = {t.sig_idx: t for t in pp}
    for k in sorted(set(bt_m) | set(pp_m)):
        b, p = bt_m.get(k), pp_m.get(k)
        if b is None:
            diffs.append(Diff(symbol, k, "UNMATCHED_PAPER", f"só piloto: {p.direction} entry {p.entry_price:.4f}"))
            continue
        if p is None:
            diffs.append(Diff(symbol, k, "UNMATCHED_BT", f"só backtest: {b.direction} entry {b.entry_price:.4f}"))
            continue
        if b.entry_ts != p.entry_ts:
            diffs.append(Diff(symbol, k, "ENTRY_TIMING",
                              f"bt entra ts={b.entry_ts} (open t+1) vs pp ts={p.entry_ts} (close t); "
                              f"delta={(b.entry_ts - p.entry_ts) / HOUR_MS:+.1f}h"))
        if abs(b.entry_price / p.entry_price - 1.0) > tol_price:
            diffs.append(Diff(symbol, k, "ENTRY_PRICE",
                              f"bt {b.entry_price:.4f} vs pp {p.entry_price:.4f} ({(b.entry_price / p.entry_price - 1) * 1e4:+.1f} bps)"))
        b_reason = _REASON_MAP.get(b.reasons[-1], b.reasons[-1])
        if b_reason != p.reasons[-1]:
            diffs.append(Diff(symbol, k, "EXIT_REASON", f"bt {b.reasons[-1]} vs pp {p.reasons}"))
        if b.exit_ts != p.exit_ts and b.reasons[-1] != "HORIZON_TIMEOUT":
            diffs.append(Diff(symbol, k, "EXIT_TIMING", f"bt exit ts={b.exit_ts} vs pp exit ts={p.exit_ts}"))
        if abs(b.fee_pct - p.fee_pct) > tol_fee:
            diffs.append(Diff(symbol, k, "EXIT_FEE_MODEL", f"fee/notional bt {b.fee_pct:.5%} vs pp {p.fee_pct:.5%}"))
        if abs(b.funding) > 1e-9:
            diffs.append(Diff(symbol, k, "FUNDING", f"bt funding {b.funding:+.4f} USD ({b.funding / b.notional:+.5%}) vs pp 0"))
        if p.notional > 0 and abs(np.log(b.notional / p.notional)) > np.log(1.0 + tol_size):
            diffs.append(Diff(symbol, k, "SIZING", f"notional bt {b.notional:,.0f} vs pp {p.notional:,.0f}"))
        if abs(b.ret_pct - p.ret_pct) > tol_ret:
            diffs.append(Diff(symbol, k, "PNL_PCT", f"ret%notional bt {b.ret_pct:+.4%} vs pp {p.ret_pct:+.4%}"))
    return diffs


@dataclass
class ParityReport:
    scenario: str
    bars: int
    per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diffs: List[Diff] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for d in self.diffs:
            c[d.category] = c.get(d.category, 0) + 1
        return c

    def relevant_counts(self) -> Dict[str, int]:
        return {k: v for k, v in self.counts().items() if k not in ACCEPTED_BY_DESIGN}


def run_parity(symbols: Optional[List[str]] = None, bars: int = 1500, scenario: str = "forced",
               db_path: str = DB_PATH) -> ParityReport:
    import logging
    symbols = symbols or SYMBOLS
    rep = ParityReport(scenario=scenario, bars=bars)
    prev = logging.root.manager.disable
    logging.disable(logging.WARNING)   # silencia INFO de smart_trade/learning_tracker durante o replay
    try:
        return _run_parity_inner(symbols, bars, scenario, db_path, rep)
    finally:
        logging.disable(prev)


def _run_parity_inner(symbols: List[str], bars: int, scenario: str, db_path: str,
                      rep: ParityReport) -> ParityReport:
    for sym in symbols:
        candles = load_closed_candles(sym, bars + WARMUP_BARS, db_path)
        feat = build_features(candles)
        test_start = max(0, len(feat) - bars)
        signals = (committee_signals(feat, test_start) if scenario == "committee"
                   else forced_signals(feat, test_start))
        bt = run_backtest(feat, signals, test_start)
        pp, meta = run_paper(sym, feat, signals, test_start)
        diffs = reconcile(sym, bt, pp)
        rep.diffs.extend(diffs)
        rep.per_symbol[sym] = {
            "n_signals": int((signals[test_start:] > 0).sum()), "n_bt_trades": len(bt), "n_pp_trades": len(pp),
            "bt_net_pct_notional_sum": float(sum(t.ret_pct for t in bt)),
            "pp_net_pct_notional_sum": float(sum(t.ret_pct for t in pp)),
            "bt_reasons": _count([t.reasons[-1] for t in bt]),
            "pp_reasons": _count([t.reasons[-1] for t in pp]), **meta,
        }
    return rep


def _count(xs: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def format_report(rep: ParityReport, max_examples: int = 3) -> str:
    lines = [f"=== PARITY AUDIT | scenario={rep.scenario} | bars={rep.bars} ==="]
    for s, m in rep.per_symbol.items():
        lines.append(f"[{s}] sinais={m['n_signals']} bt_trades={m['n_bt_trades']} pp_trades={m['n_pp_trades']} "
                     f"| Σret%notional bt={m['bt_net_pct_notional_sum']:+.2%} pp={m['pp_net_pct_notional_sum']:+.2%} "
                     f"| bt_exits={m['bt_reasons']} pp_exits={m['pp_reasons']}")
    lines.append("--- Divergências por categoria (A=aceita por desenho) ---")
    for cat, n in sorted(rep.counts().items(), key=lambda kv: -kv[1]):
        tag = "A" if cat in ACCEPTED_BY_DESIGN else "!"
        ex = [d.detail for d in rep.diffs if d.category == cat][:max_examples]
        lines.append(f"[{tag}] {cat}: {n}  - {CATEGORIES.get(cat, '')}")
        for e in ex:
            lines.append(f"      ex: {e}")
    if not rep.diffs:
        lines.append("Nenhuma divergência: PARIDADE.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bars", type=int, default=1500)
    ap.add_argument("--scenario", choices=["committee", "forced"], default="forced")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = run_parity(a.symbols, a.bars, a.scenario)
    print(format_report(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"scenario": rep.scenario, "bars": rep.bars, "per_symbol": rep.per_symbol,
                       "counts": rep.counts(), "diffs": [asdict(d) for d in rep.diffs]}, f, indent=2)
    return 1 if rep.relevant_counts() else 0


if __name__ == "__main__":
    sys.exit(main())
