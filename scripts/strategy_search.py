"""
scripts/strategy_search.py — Busca controlada e honesta de uma estratégia com edge (BTC/ETH/SOL, diário).

Regras de higiene (alinhadas a docs/PROTOCOLO_EXPERIMENTAL.md):
- Família PRÉ-DEFINIDA de hipóteses com fundamento econômico (sem grade de otimização).
- Período de desenvolvimento: 2020-09-14 .. 2025-10-13. HOLDOUT (2025-10-14 .. 2026-10-14) fica TRANCADO:
  só é computado com --holdout, e UMA vez, para a(s) hipótese(s) que passarem no desenvolvimento.
- Custos: taker 0,06% + slippage 0,02% por unidade de giro (por perna) + funding real (long paga quando > 0).
- Execução: sinal no fechamento de t, posição ganha o retorno de t+1 (sem lookahead).
- DSR com N = tentativas históricas do trial_log + configs desta busca; variância dos Sharpes entre as configs.
- Somente leitura do SQLite (mode=ro).

Uso:  .venv/bin/python scripts/strategy_search.py [--holdout H_ID ...]
"""
import argparse
import json
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statistical_rigor import calculate_annualized_sharpe, deflated_sharpe_ratio  # noqa: E402

DB = "file:" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "market_cache.db") + "?mode=ro"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEV_START, DEV_END = "2020-09-14", "2025-10-13"
HOLD_START, HOLD_END = "2025-10-14", "2026-10-13"
FEE = 0.0006 + 0.0002          # por unidade de giro
PRIOR_TRIALS = 84              # linhas do trial_log.csv (histórico) antes desta busca
BAND = 0.10


def load():
    c = sqlite3.connect(DB, uri=True)
    px = {}
    for s in ASSETS:
        d = pd.read_sql_query(
            "select timestamp, close from candles where exchange='binance' and symbol=? and timeframe='1d' order by timestamp",
            c, params=(s,))
        d["date"] = pd.to_datetime(d["timestamp"], unit="ms").dt.normalize()
        px[s] = d.drop_duplicates("date").set_index("date")["close"]
    close = pd.DataFrame(px).dropna()
    f = pd.read_sql_query("select symbol, timestamp, funding_rate from funding_rates where exchange='binance' and symbol in (?,?,?)",
                          c, params=tuple(ASSETS))
    f["date"] = pd.to_datetime(f["timestamp"], unit="ms").dt.normalize()
    fund = f.groupby(["date", "symbol"])["funding_rate"].sum().unstack().reindex(close.index).fillna(0.0)
    fg = pd.read_sql_query("select date_str, value from macro_daily_history where series_name='FEAR_AND_GREED' order by timestamp_ms", c)
    fg["date"] = pd.to_datetime(fg["date_str"])
    fgs = fg.drop_duplicates("date").set_index("date")["value"].reindex(close.index).ffill()
    return close, fund, fgs


def band_weights(target: pd.DataFrame, band=BAND) -> pd.DataFrame:
    """Rebalanceia somente se |alvo - atual| > band por ativo (controle de giro)."""
    cur = np.zeros(target.shape[1])
    out = np.zeros(target.shape)
    for i, row in enumerate(target.values):
        for j in range(len(cur)):
            if abs(row[j] - cur[j]) > band or (row[j] == 0.0 and cur[j] != 0.0):
                cur[j] = row[j]
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def simulate(w: pd.DataFrame, close: pd.DataFrame, fund: pd.DataFrame):
    """w[t] decidido no fechamento de t; ganha retorno de t+1. Retorna série diária líquida, bruta e giro."""
    ret = close.pct_change().shift(-1)           # retorno de t -> t+1
    f_next = fund.shift(-1)                      # funding pago durante t+1
    gross = (w * ret).sum(axis=1)
    fpnl = -(w * f_next).sum(axis=1)
    turn = w.diff().abs().fillna(w.abs()).sum(axis=1)
    cost = turn * FEE
    net = (gross + fpnl - cost).iloc[:-1]
    return net, gross.iloc[:-1], turn.iloc[:-1], fpnl.iloc[:-1], cost.iloc[:-1]


def vol(x, n=30):
    return x.pct_change().rolling(n).std() * math.sqrt(365)


# ---------------- Hipóteses ----------------
def h_A(close, fund, fg):
    """A: beta com volatility-management (long-only EW, alvo 20% a.a., teto 1.5x). Sem sinal direcional."""
    rp = close.pct_change().mean(axis=1)
    v = (rp.rolling(30).std() * math.sqrt(365)).shift(0)
    scale = (0.20 / v.clip(lower=0.05)).clip(upper=1.5)
    t = pd.DataFrame({a: scale / len(ASSETS) for a in ASSETS})
    return band_weights(t.fillna(0.0))


def h_B(z_thr):
    def f(close, fund, fg):
        """B: funding extremo contrarian (posicionamento crowded). z do funding 7d vs 90d; segura 3 dias."""
        f7 = fund.rolling(7).mean()
        z = (f7 - f7.rolling(90).mean()) / f7.rolling(90).std()
        sig = pd.DataFrame(0.0, index=close.index, columns=ASSETS)
        sig[z > z_thr] = -1.0
        sig[z < -z_thr] = 1.0
        held = sig.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
        scale = (0.20 / vol(close).clip(lower=0.05)).clip(upper=1.0)
        t = held * scale / len(ASSETS)
        return band_weights(t.fillna(0.0))
    return f


def h_C(lb):
    def f(close, fund, fg):
        """C: momentum cross-sectional market-neutral (long top / short bottom), rebalance semanal."""
        r = close.pct_change(lb)
        t = pd.DataFrame(0.0, index=close.index, columns=ASSETS)
        for i in range(len(close)):
            if i % 7 or r.iloc[i].isna().any():
                continue
            rk = r.iloc[i].rank()
            t.iloc[i, :] = np.where(rk == rk.max(), 0.5, np.where(rk == rk.min(), -0.5, 0.0))
        # mantém a posição entre rebalanceamentos
        held = t.copy()
        mask = pd.Series([i % 7 == 0 for i in range(len(close))], index=close.index)
        held[~mask] = np.nan
        return held.ffill().fillna(0.0)
    return f


def h_D(lb):
    def f(close, fund, fg):
        """D: reversão cross-sectional de curtíssimo prazo (long loser / short winner), rebalance diário."""
        r = close.pct_change(lb)
        t = pd.DataFrame(0.0, index=close.index, columns=ASSETS)
        for i in range(len(close)):
            if r.iloc[i].isna().any():
                continue
            rk = r.iloc[i].rank()
            t.iloc[i, :] = np.where(rk == rk.min(), 0.5, np.where(rk == rk.max(), -0.5, 0.0))
        return t
    return f


def h_F(close, fund, fg):
    """F: Fear & Greed contrarian (BTC long quando F&G <= 25 até >= 50)."""
    pos = pd.Series(0.0, index=close.index)
    on = False
    for d, v in fg.items():
        if np.isnan(v):
            pos[d] = 1.0 if on else 0.0
            continue
        if not on and v <= 25:
            on = True
        elif on and v >= 50:
            on = False
        pos[d] = 1.0 if on else 0.0
    t = pd.DataFrame(0.0, index=close.index, columns=ASSETS)
    t["BTCUSDT"] = pos
    return t


HYP = {
    "A_volmanaged_beta": h_A,
    "B_funding_contrarian_z2.0": h_B(2.0),
    "B_funding_contrarian_z1.5": h_B(1.5),
    "C_xs_momentum_28d": h_C(28),
    "C_xs_momentum_14d": h_C(14),
    "D_xs_reversal_3d": h_D(3),
    "F_fear_greed_contrarian": h_F,
}


def stats_for(net, gross, turn, fpnl, cost, bh):
    n = len(net)
    ann_ret = (1 + net).prod() ** (365.0 / n) - 1 if n else float("nan")
    vol_a = net.std(ddof=1) * math.sqrt(365)
    sr = calculate_annualized_sharpe(net.values)
    eq = (1 + net).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    corr = float(net.corr(bh.reindex(net.index))) if net.std() > 0 else float("nan")
    gsr = calculate_annualized_sharpe(gross.values)
    return dict(n=n, ann_ret=ann_ret, vol=vol_a, sharpe=sr, gross_sharpe=gsr, mdd=mdd,
                turnover_yr=float(turn.mean() * 365), cost_yr=float(cost.mean() * 365), funding_yr=float(fpnl.mean() * 365),
                corr_bh=corr)


def block_bootstrap_sharpe_ci(net, block=20, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    x = net.values
    T = len(x)
    if T < 2 * block:
        return (float("nan"), float("nan"))
    out = []
    nb = int(math.ceil(T / block))
    for _ in range(n):
        starts = rng.integers(0, T - block, size=nb)
        s = np.concatenate([x[a:a + block] for a in starts])[:T]
        out.append(calculate_annualized_sharpe(s))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(window, ids, close, fund, fg):
    lo, hi = window
    rows = {}
    bh_w = pd.DataFrame(1.0 / len(ASSETS), index=close.index, columns=ASSETS)
    bh_net, bh_g, bh_t, bh_f, bh_c = simulate(bh_w, close, fund)
    bh_net = bh_net.loc[lo:hi]
    for hid in ids:
        w = HYP[hid](close, fund, fg)
        net, g, t, fp, cs = simulate(w, close, fund)
        sl = slice(lo, hi)
        st = stats_for(net.loc[sl], g.loc[sl], t.loc[sl], fp.loc[sl], cs.loc[sl], bh_net)
        yearly = net.loc[sl].groupby(net.loc[sl].index.year).apply(lambda s: calculate_annualized_sharpe(s.values) if len(s) > 30 else float("nan"))
        st["yearly_sharpe"] = {int(k): round(float(v), 2) for k, v in yearly.items()}
        st["sharpe_ci95"] = block_bootstrap_sharpe_ci(net.loc[sl])
        rows[hid] = (st, net.loc[sl])
    bh_stats = stats_for(bh_net, bh_g.loc[lo:hi], bh_t.loc[lo:hi], bh_f.loc[lo:hi], bh_c.loc[lo:hi], bh_net)
    return rows, bh_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", nargs="*", default=None, help="IDs de hipóteses a avaliar no HOLDOUT (uso único).")
    a = ap.parse_args()
    close, fund, fg = load()
    print(f"dados: {close.index.min().date()} .. {close.index.max().date()}  ({len(close)} dias, ativos={ASSETS})")

    ids = list(HYP.keys())
    if a.holdout is not None:
        ids = a.holdout
        window = (HOLD_START, HOLD_END)
        print("*** HOLDOUT — USO ÚNICO. Resultados não podem ser re-otimizados. ***")
    else:
        window = (DEV_START, DEV_END)
        print(f"desenvolvimento: {DEV_START} .. {DEV_END} (holdout trancado)")

    rows, bh = run(window, ids, close, fund, fg)
    n_trials = PRIOR_TRIALS + len(HYP)
    sharpes = [r[0]["sharpe"] for r in rows.values()]
    var_sr = max(float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.25, 0.25)

    def line(name, st, dsr=None):
        ci = st.get("sharpe_ci95")
        cis = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "   -   "
        d = f"DSR={dsr:.3f}" if dsr is not None else ""
        print(f"{name:30s} ret/a={st['ann_ret']*100:+7.1f}%  vol={st['vol']*100:5.1f}%  SR={st['sharpe']:+5.2f} (bruto {st['gross_sharpe']:+5.2f}) "
              f"CI95{cis}  MDD={st['mdd']*100:6.1f}%  giro/a={st['turnover_yr']:6.1f}  custo/a={st['cost_yr']*100:5.1f}%  fund/a={st['funding_yr']*100:+5.1f}%  corrBH={st['corr_bh']:+.2f} {d}")

    print(f"\nN de tentativas para DSR: {n_trials}  (histórico {PRIOR_TRIALS} + {len(HYP)} desta busca); var(SR)={var_sr:.2f}")
    print("-" * 200)
    line("BUY&HOLD EW (BTC/ETH/SOL)", bh)
    out = {"window": window, "buy_hold": {k: v for k, v in bh.items() if k != "yearly_sharpe"}, "hyp": {}}
    for hid, (st, net) in rows.items():
        d = deflated_sharpe_ratio(st["sharpe"], net.values, n_trials, var_sr)
        line(hid, st, d["dsr"])
        print(f"{'':30s} Sharpe por ano: {st['yearly_sharpe']}")
        out["hyp"][hid] = {**{k: v for k, v in st.items()}, "dsr": d["dsr"], "expected_max_sr_null": d["expected_max_sr"]}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "reports",
                     "strategy_search_%s.json" % ("holdout" if a.holdout is not None else "dev"))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nsalvo em", p)


if __name__ == "__main__":
    main()
