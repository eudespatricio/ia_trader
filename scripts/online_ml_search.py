"""
scripts/online_ml_search.py — Aprendizado contínuo (walk-forward mensal) sobre indicadores técnicos, com custos reais.

Pergunta honesta: um modelo que reaprende todo mês com os indicadores (o que o usuário quer ver "refinado
ao longo do tempo") consegue prever, FORA DA AMOSTRA, trades com expectância líquida positiva depois de custos?

Protocolo (pré-definido, sem tuning):
- Dados: Binance 1h BTC/ETH/SOL, features causais do FeatureEngineer (sem macro). Dev: até 2025-10-13. HOLDOUT trancado.
- Rótulo (por barra, cada lado): resultado realizado de um trade com TP=1.5xATR, SL=1.0xATR, horizonte H barras,
  SL antes do TP na mesma barra; saída no close do horizonte se nenhuma barreira. Retorno líquido = bruto - 0.16% (taker+slip, ida e volta).
- Modelo: LightGBM binário (P[retorno líquido > 0]) por lado, parâmetros FIXOS e conservadores, pool dos 3 ativos.
- Reaprendizado: janela EXPANDIDA, retreino a cada mês, embargo de H+10 barras, previsões somente do mês seguinte (OOS puro).
- Regra de entrada (fixa): EV_R = p*1.5 - (1-p)*1.0 - custo/ATR%; entra se EV_R > 0.10 e é o melhor dos dois lados;
  uma posição por ativo por vez.
- Avaliação: expectância líquida por trade, IC por bootstrap, Sharpe diário, DSR (N tentativas), teste de permutação
  (mesmo nº de trades em barras aleatórias), calibração por decil e estabilidade anual.
Uso: .venv/bin/python scripts/online_ml_search.py [--H 24] [--holdout]
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.feature_engineering import FeatureEngineer  # noqa: E402
from statistical_rigor import calculate_annualized_sharpe, deflated_sharpe_ratio  # noqa: E402
import lightgbm as lgb  # noqa: E402

DB = "file:" + os.path.join(ROOT, "datasets", "market_cache.db") + "?mode=ro"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEV_END = pd.Timestamp("2025-10-13 23:00")
HOLD_START, HOLD_END = pd.Timestamp("2025-10-14 00:00"), pd.Timestamp("2026-10-13 23:00")
TP_M, SL_M, COST = 1.5, 1.0, 0.0016  # sobrescritos por --tp/--sl
EV_MIN = 0.10
PRIOR_TRIALS = 95
WARMUP = 300
FIRST_PRED = pd.Timestamp("2022-01-01")

LGB_PARAMS = dict(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=300,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=8,
                  random_state=42)


def load_candles(sym):
    c = sqlite3.connect(DB, uri=True)
    d = pd.read_sql_query("select timestamp,open,high,low,close,volume from candles where exchange='binance' and symbol=? and timeframe='1h' order by timestamp",
                          c, params=(sym,))
    d = d.drop_duplicates("timestamp").reset_index(drop=True)
    return d


EXT = False
_BTC_CACHE = {}


def add_ext_features(feat, sym):
    """Features estendidas (todas causais): funding (merge backward), Fear&Greed (defasado 1 dia),
    fator de mercado BTC (retornos/vol 6h e 24h) e hora/dia da semana."""
    c = sqlite3.connect(DB, uri=True)
    ts = feat["timestamp"].values
    # funding: taxa vigente + z de 7d vs 90d (amostras de 8h)
    f = pd.read_sql_query("select timestamp, funding_rate from funding_rates where exchange='binance' and symbol=? order by timestamp",
                          c, params=(sym,)).drop_duplicates("timestamp")
    f["f7"] = f["funding_rate"].rolling(21).mean()
    f["f_z"] = (f["f7"] - f["f7"].rolling(270).mean()) / f["f7"].rolling(270).std()
    base = pd.DataFrame({"timestamp": ts})
    m = pd.merge_asof(base, f[["timestamp", "funding_rate", "f_z"]].rename(columns={"funding_rate": "x_funding"}),
                      on="timestamp", direction="backward")
    feat["x_funding"] = m["x_funding"].values
    feat["x_funding_z"] = m["f_z"].values
    # Fear & Greed (dia anterior)
    fg = pd.read_sql_query("select date_str, value from macro_daily_history where series_name='FEAR_AND_GREED'", c)
    fg["d"] = pd.to_datetime(fg["date_str"]).dt.normalize() + pd.Timedelta(days=1)   # só vale a partir do dia seguinte
    fg = fg.drop_duplicates("d").set_index("d")["value"]
    day = pd.to_datetime(feat["timestamp"], unit="ms").dt.normalize()
    feat["x_fng"] = day.map(fg).values
    # fator de mercado BTC
    if "btc" not in _BTC_CACHE:
        b = load_candles("BTCUSDT").set_index("timestamp")["close"]
        _BTC_CACHE["btc"] = pd.DataFrame({"b_r6": b.pct_change(6), "b_r24": b.pct_change(24),
                                          "b_vol24": b.pct_change().rolling(24).std()})
    btc = _BTC_CACHE["btc"].reindex(ts)
    for k in ("b_r6", "b_r24", "b_vol24"):
        feat["x_" + k] = btc[k].values
    dtm = pd.to_datetime(feat["timestamp"], unit="ms")
    feat["x_hour_sin"] = np.sin(2 * np.pi * dtm.dt.hour / 24)
    feat["x_hour_cos"] = np.cos(2 * np.pi * dtm.dt.hour / 24)
    feat["x_dow"] = dtm.dt.dayofweek
    return feat


def barrier_outcomes(df, atr, H):
    """Retorno bruto realizado (fração) de LONG e SHORT com TP/SL em ATR, por barra de entrada (decisão no close)."""
    n = len(df)
    c, hi, lo = df["close"].values, df["high"].values, df["low"].values
    out = {}
    for side in ("long", "short"):
        res = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
        if side == "long":
            tp, sl = c + TP_M * atr, c - SL_M * atr
        else:
            tp, sl = c - TP_M * atr, c + SL_M * atr
        for k in range(1, H + 1):
            hk = np.full(n, np.nan); lk = np.full(n, np.nan)
            hk[:n - k] = hi[k:]; lk[:n - k] = lo[k:]
            valid = ~done & ~np.isnan(hk)
            if side == "long":
                sl_hit = valid & (lk <= sl); tp_hit = valid & (hk >= tp) & ~sl_hit
                res[sl_hit] = (sl[sl_hit] / c[sl_hit] - 1); res[tp_hit] = (tp[tp_hit] / c[tp_hit] - 1)
            else:
                sl_hit = valid & (hk >= sl); tp_hit = valid & (lk <= tp) & ~sl_hit
                res[sl_hit] = (c[sl_hit] - sl[sl_hit]) / c[sl_hit]; res[tp_hit] = (c[tp_hit] - tp[tp_hit]) / c[tp_hit]
            done |= sl_hit | tp_hit
        # timeout: close em t+H
        ce = np.full(n, np.nan); ce[:n - H] = c[H:]
        t_mask = ~done & ~np.isnan(ce)
        res[t_mask] = (ce[t_mask] / c[t_mask] - 1) if side == "long" else (c[t_mask] - ce[t_mask]) / c[t_mask]
        out[side] = res
    return out


def build(H):
    fe = FeatureEngineer()
    frames = []
    for s in ASSETS:
        raw = load_candles(s)
        feat = fe.compute_all_features(raw, drop_na=False)
        feat = feat.reset_index(drop=True)
        feat["symbol"] = s
        feat["dt"] = pd.to_datetime(feat["timestamp"], unit="ms")
        atr = feat["atr_14"].values.astype(float)
        bo = barrier_outcomes(feat, atr, H)
        feat["r_long"] = bo["long"] - COST
        feat["r_short"] = bo["short"] - COST
        feat["atr_pct"] = atr / feat["close"].values
        if EXT:
            feat = add_ext_features(feat, s)
        feat = feat.iloc[WARMUP:].copy()
        frames.append(feat)
    df = pd.concat(frames, ignore_index=True).sort_values(["dt", "symbol"]).reset_index(drop=True)
    drop_cols = {"timestamp", "dt", "symbol", "open", "high", "low", "close", "volume", "r_long", "r_short", "atr_pct"}
    cand = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    # remove colunas em escala de preço (corr alta com close) e constantes — critério sem rótulo
    keep = []
    for c in cand:
        x = df[c].replace([np.inf, -np.inf], np.nan)
        if x.isna().mean() > 0.2 or x.nunique() < 5:
            continue
        sub = pd.concat([x, df["close"]], axis=1).dropna()
        if len(sub) > 100 and abs(sub.corr().iloc[0, 1]) > 0.8:
            continue
        keep.append(c)
    X = df[keep].replace([np.inf, -np.inf], np.nan)
    return df, X, keep


def walk_forward(df, X, H, until):
    months = pd.period_range(FIRST_PRED, until, freq="M")
    p_long = np.full(len(df), np.nan)
    p_short = np.full(len(df), np.nan)
    dt = df["dt"]
    emb = pd.Timedelta(hours=H + 10)
    for m in months:
        t0, t1 = m.start_time, m.end_time
        test = (dt >= t0) & (dt <= t1)
        if not test.any():
            continue
        train = (dt < t0 - emb)
        for side, tgt in (("long", "r_long"), ("short", "r_short")):
            tr = train & df[tgt].notna()
            y = (df.loc[tr, tgt] > 0).astype(int)
            mdl = lgb.LGBMClassifier(**LGB_PARAMS)
            mdl.fit(X.loc[tr], y)
            pr = mdl.predict_proba(X.loc[test])[:, 1]
            (p_long if side == "long" else p_short)[test.values] = pr
    return p_long, p_short


def simulate(df, p_long, p_short, H):
    """Regra fixa de entrada; uma posição por ativo por vez; retorna lista de trades (pnl líquido já inclui custo)."""
    df = df.assign(pl=p_long, ps=p_short)
    df = df[df["pl"].notna()]
    trades = []
    for s, g in df.groupby("symbol"):
        g = g.sort_values("dt")
        busy_until = pd.Timestamp.min
        for row in g.itertuples(index=False):
            if row.dt < busy_until:
                continue
            ev_l = row.pl * TP_M - (1 - row.pl) * SL_M - COST / row.atr_pct
            ev_s = row.ps * TP_M - (1 - row.ps) * SL_M - COST / row.atr_pct
            side = None
            if ev_l > EV_MIN and ev_l >= ev_s:
                side = "long"
            elif ev_s > EV_MIN:
                side = "short"
            if side is None:
                continue
            r = row.r_long if side == "long" else row.r_short
            if np.isnan(r):
                continue
            trades.append((s, row.dt, side, float(r), float(row.pl if side == "long" else row.ps)))
            busy_until = row.dt + pd.Timedelta(hours=H)
    return pd.DataFrame(trades, columns=["symbol", "dt", "side", "r", "p"])


def evaluate(df, trades, H, label):
    n = len(trades)
    print(f"\n=== {label} ===")
    if n == 0:
        print("sem trades"); return {}
    exp = trades["r"].mean()
    wr = (trades["r"] > 0).mean()
    # bootstrap IC
    rng = np.random.default_rng(1)
    bs = [rng.choice(trades["r"].values, n).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # Sharpe diário (pnl atribuído ao dia de entrada, peso 1/3 por ativo)
    daily = (trades.assign(d=trades["dt"].dt.normalize()).groupby("d")["r"].sum() / len(ASSETS))
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx).fillna(0.0)
    sr = calculate_annualized_sharpe(daily.values)
    # permutação: mesmo nº de trades em barras OOS aleatórias, lado aleatório
    pool = df[df["pl"].notna()] if "pl" in df else df
    pool_r = np.concatenate([pool["r_long"].dropna().values, pool["r_short"].dropna().values])
    null = np.array([rng.choice(pool_r, n).mean() for _ in range(2000)])
    pval = float((null >= exp).mean())
    yr = trades.groupby(trades["dt"].dt.year)["r"].agg(["count", "mean"]).round(5)
    print(f"trades={n}  win_rate={wr*100:.1f}%  expectância líquida/trade={exp*100:+.3f}%  IC95=[{lo*100:+.3f},{hi*100:+.3f}]%  "
          f"média nula(aleatório)={null.mean()*100:+.3f}%  p(perm)={pval:.3f}")
    print(f"Sharpe diário anualizado={sr:+.2f}   trades/ano≈{n / max(1, (trades['dt'].max() - trades['dt'].min()).days / 365):.0f}")
    print("por ano (n, expectância):", {int(k): (int(v['count']), round(float(v['mean']) * 100, 3)) for k, v in yr.iterrows()})
    by_side = trades.groupby("side")["r"].agg(["count", "mean"])
    print("por lado:", {k: (int(v['count']), round(float(v['mean']) * 100, 3)) for k, v in by_side.iterrows()})
    return dict(n=n, exp=exp, ci=(lo, hi), sr=sr, pval=pval, wr=wr, daily=daily)


def calibration(df, p, tgt, name):
    d = pd.DataFrame({"p": p, "win": (df[tgt] > 0).astype(float).where(df[tgt].notna())}).dropna()
    if len(d) < 1000:
        return
    d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
    t = d.groupby("bin").agg(p_med=("p", "mean"), win=("win", "mean"), n=("p", "size"))
    print(f"calibração {name}: (prev -> realizado) " + "  ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(t['p_med'], t['win'])))
    auc_num = d.sort_values("p")
    from sklearn.metrics import roc_auc_score
    print(f"   AUC OOS {name}: {roc_auc_score(d['win'], d['p']):.4f} (0.5 = sem informação)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--ext", action="store_true")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.0)
    a = ap.parse_args()
    H = a.H
    global TP_M, SL_M, EXT
    EXT = a.ext
    TP_M, SL_M = a.tp, a.sl
    df, X, keep = build(H)
    print(f"features usadas: {len(keep)}  | barras: {len(df)}  | H={H}  | custo ida/volta={COST*100:.2f}%")
    until = pd.Timestamp("2026-10-01") if a.holdout else pd.Timestamp("2025-09-01")
    cut = HOLD_END if a.holdout else DEV_END
    mask_all = df["dt"] <= cut
    df, X = df[mask_all].reset_index(drop=True), X[mask_all].reset_index(drop=True)
    pl, ps = walk_forward(df, X, H, until)
    df["pl"], df["ps"] = pl, ps
    oos = df[df["pl"].notna()]
    print(f"previsões OOS: {len(oos)} barras ({oos['dt'].min().date()} .. {oos['dt'].max().date()})")
    calibration(oos, oos["pl"].values, "r_long", "LONG")
    calibration(oos, oos["ps"].values, "r_short", "SHORT")
    tr = simulate(df, pl, ps, H)
    if a.holdout:
        tr = tr[tr["dt"] >= HOLD_START]
    res = evaluate(df, tr, H, f"{'HOLDOUT' if a.holdout else 'DEV'} OOS H={H}")
    if res:
        n_trials = PRIOR_TRIALS + 2
        d = deflated_sharpe_ratio(res["sr"], res["daily"].values, n_trials, 0.5)
        print(f"DSR (N={n_trials}) = {d['dsr']:.3f}   E[max SR] sob nulo = {d['expected_max_sr']:.2f}")
    out = os.path.join(ROOT, "datasets", "reports", f"online_ml_{'holdout' if a.holdout else 'dev'}_H{H}_tp{TP_M}_sl{SL_M}{'_ext' if EXT else ''}.json")
    if res:
        with open(out, "w") as fh:
            json.dump({k: (v if k != "daily" else None) for k, v in res.items()}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
