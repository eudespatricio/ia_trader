#!/usr/bin/env python3
"""
scripts/pilot_report.py - Veredito objetivo do piloto de paper trading.

Implementa docs/CRITERIOS_APROVACAO_PILOTO.md. Lê SOMENTE dados reais
(datasets/market_cache.db, datasets/supervisor_state.json, datasets/supervisor.log,
datasets/reports/trial_log.csv) e emite APROVADO / REPROVADO / INCONCLUSIVO.

Regras de ouro:
  * Custos recalculados em modelo ESTRESSE (taker 0,06% + slippage 0,02% por lado),
    ignorando as taxas maker que o simulador debita nas entradas/saídas de GOLDEN_POT/SCALP.
  * Reprovação é assimétrica: basta UMA condição de refutação. Aprovação exige TODAS.
  * Ausência de dado = falha do critério (nunca "passa por omissão").

Uso:  .venv/bin/python scripts/pilot_report.py [--start ISO_UTC] [--json] [--ref docs/pilot_backtest_reference.json]
Exit code: 0 APROVADO | 1 REPROVADO | 2 INCONCLUSIVO
"""
import argparse
import json
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from statistical_rigor import (  # noqa: E402
    calculate_annualized_sharpe, deflated_sharpe_ratio, get_total_trials_count)

DB = os.path.join(BASE, "datasets", "market_cache.db")
SUP_STATE = os.path.join(BASE, "datasets", "supervisor_state.json")
SUP_LOG = os.path.join(BASE, "datasets", "supervisor.log")
TRIAL_LOG = os.path.join(BASE, "datasets", "reports", "trial_log.csv")
HYP_DOC = os.path.join(BASE, "docs", "HIPOTESES_PRE_REGISTRADAS.md")
OUT_JSON = os.path.join(BASE, "datasets", "reports", "pilot_report.json")

# ----------------------------- LIMIARES (congelados) -----------------------------
TAKER, SLIP = 0.0006, 0.0002
COST_SIDE = TAKER + SLIP                 # 0,08% por perna sobre o notional de cada fill
INITIAL_CAPITAL = 10_000.0
MIN_DAYS, MIN_TRADES = 56, 100
MIN_SHARPE, MIN_DSR = 1.0, 0.95
MAX_DD_APPROVE, MAX_DD_REFUTE = 15.0, 20.0
FUTILITY_N = 30                          # N mínimo p/ refutação precoce
MIN_COVERAGE, MAX_GAP_MIN, SNAP_MIN = 0.98, 60, 15
MAX_DAEMON_RESTARTS_30D = 2
MAX_COOLDOWNS_REFUTE = 3
MIN_WEEKS_POSITIVE = 0.60
DIVERGENCE_MIN_RATIO, FREQ_TOL = 0.50, 0.30
WEEK_MIN_FOR_TEST = 8
PREREG_ALLOW = {"DELTA_NEUTRAL_CARRY", "TSMOM", "CMOM", "TREND_FOLLOWING"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_mean_ci(x, n=5000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))  # one-sided 95%


def load():
    c = sqlite3.connect(DB)
    d = {
        "trades": pd.read_sql("select rowid as _rowid, * from paper_trades order by timestamp", c),
        "pos": pd.read_sql("select * from paper_positions", c),
        "pf": pd.read_sql("select * from paper_portfolio order by timestamp", c),
        "cb": pd.read_sql("select * from circuit_breaker_events order by timestamp", c),
        "set": pd.read_sql("select * from simulation_settings", c),
    }
    c.close()
    return d


def build_positions(d):
    t = d["trades"].copy()
    t["notional"] = t.price * t.quantity
    t["is_open"] = t.action.str.startswith("OPEN")
    t["cost"] = t.notional * COST_SIDE
    # pnl nas linhas de fechamento já é líquido da taxa de saída BOOKED -> reconstitui bruto
    t["gross"] = np.where(t.is_open, 0.0, t.pnl + t.fee)
    closed_ids = set(d["pos"].loc[d["pos"].status == "CLOSED", "position_id"])
    rows = []
    for pid, g in t.groupby("position_id"):
        if pid not in closed_ids:
            continue
        o = g[g.is_open]
        x = g[~g.is_open]
        if o.empty or x.empty:
            continue
        entry_notional = float(o.notional.sum())
        gross = float(x.gross.sum())
        cost = float(g.cost.sum())
        booked_fees = float(g.fee.sum())
        rows.append(dict(pid=pid, symbol=g.symbol.iloc[0], t_open=int(o.timestamp.min()),
                         t_close=int(x.timestamp.max()), entry_notional=entry_notional,
                         gross=gross, cost_stress=cost, booked_fees=booked_fees,
                         net_stress=gross - cost, net_booked=gross - booked_fees))
    return t, pd.DataFrame(rows)


def stress_equity(d, t):
    """Curva de equity em custos ESTRESSE: capital + bruto acumulado - custos estresse + MtM do snapshot."""
    pf = d["pf"]
    pf = pf[pf.equity > 0]
    eq = [(int(pf.timestamp.min()) if len(pf) else 0, INITIAL_CAPITAL)]
    tt = t.sort_values("timestamp")
    cum = (tt.gross - tt.cost).cumsum().values
    ts = tt.timestamp.values
    for _, r in pf.iterrows():
        k = np.searchsorted(ts, r.timestamp, side="right")
        realized = cum[k - 1] if k > 0 else 0.0
        eq.append((int(r.timestamp), INITIAL_CAPITAL + realized + float(r.unrealized_pnl)))
    s = pd.Series([e for _, e in eq], index=[a for a, _ in eq])
    dd = (s / s.cummax() - 1) * 100
    return s, float(-dd.min())


EPOCH_KEYS = {"ledger_epoch_trade_rowid", "ledger_epoch_pos_rowid", "quota_epoch_trade_rowid", "MANUAL_RESET"}
RESET_TOLERANCE_MS = 120_000  # o evento/flag do próprio reset que define o início não conta como "reset na janela"


def reset_epoch(d):
    """Marco do último reset intencional (gravado por PaperTradingEngine.manual_reset/restore_state).

    Retorna (epoch_start_ms, epoch_trade_rowid) ou (None, 0) se nunca houve reset.
    """
    st = d["set"]
    row = st[st.key == "ledger_epoch_trade_rowid"]
    if row.empty:
        return None, 0
    ms = int((pd.to_datetime(row.updated_at.iloc[0], utc=True) - pd.Timestamp(0, tz="UTC"))
             // pd.Timedelta(milliseconds=1))
    try:
        rid = int(float(row.value.iloc[0]))
    except (TypeError, ValueError):
        rid = 0
    return ms, rid


def last_reset_and_setting_change(d, start_ms, reset_tol_ms=0):
    ev = d["cb"]
    resets = ev[(ev.event_type == "MANUAL_RESET") & (ev.timestamp >= start_ms + reset_tol_ms)]
    sets = d["set"].copy()
    sets = sets[~sets.key.isin(EPOCH_KEYS)] if reset_tol_ms else sets
    sets["ms"] = ((pd.to_datetime(sets.updated_at, utc=True) - pd.Timestamp(0, tz="UTC"))
                  // pd.Timedelta(milliseconds=1))
    changed = sets[sets.ms >= start_ms]
    marks = list(resets.timestamp) + list(changed.ms)
    return resets, changed, (max(marks) if marks else start_ms)


def ops_health(d, start_ms, now_ms):
    out = {}
    pf = d["pf"]
    pf = pf[pf.timestamp >= start_ms]
    elapsed_min = max(1.0, (now_ms - start_ms) / 60000)
    expected = elapsed_min / SNAP_MIN + 1
    out["snapshots"] = int(len(pf))
    out["coverage"] = float(min(1.0, len(pf) / expected))
    gaps = np.diff(pf.timestamp.values) / 60000 if len(pf) > 1 else np.array([0.0])
    out["max_gap_min"] = float(gaps.max()) if len(gaps) else 0.0
    out["last_snapshot_age_min"] = float((now_ms - pf.timestamp.max()) / 60000) if len(pf) else float("inf")
    sup = {}
    if os.path.exists(SUP_STATE):
        sup = json.load(open(SUP_STATE))
    svc = {s["name"]: s for s in sup.get("services", [])}
    pd_ = svc.get("paper_trading_daemon", {})
    out["supervisor_status"] = sup.get("overall_status", "MISSING")
    out["daemon_status"] = pd_.get("status", "MISSING")
    out["daemon_restarts"] = pd_.get("restart_count", None)
    out["daemon_uptime_h"] = round(pd_.get("uptime_seconds", 0) / 3600, 2)
    out["db_status"] = sup.get("database", {}).get("status", "MISSING")
    fails = errs = 0
    if os.path.exists(SUP_LOG):
        for ln in open(SUP_LOG, errors="ignore"):
            if "Falha detectada" in ln:
                fails += 1
            if "[ERROR]" in ln or "[CRITICAL]" in ln:
                errs += 1
    out["supervisor_failures"], out["supervisor_errors"] = fails, errs
    # Reconciliação ledger: soma pnl dos trades vs soma realized das posições
    led = float(d["trades"].pnl.sum())
    pos = float(d["pos"].realized_pnl.sum())
    out["ledger_mismatch_usd"] = abs(led - pos)
    out["restarts_applicable"] = pd_.get("restart_count", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="início oficial do piloto (ISO UTC). Default: 1º trade do ledger")
    ap.add_argument("--ref", default=os.path.join(BASE, "docs", "pilot_backtest_reference.json"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    d = load()
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    if d["trades"].empty:
        print("INCONCLUSIVO: ledger vazio")
        return 2
    epoch_ms, epoch_rowid = reset_epoch(d)
    if a.start:
        start_ms = int(pd.Timestamp(a.start, tz="UTC").timestamp() * 1000)
    elif epoch_ms is not None:
        start_ms = epoch_ms          # janela do piloto começa no último reset intencional
    else:
        start_ms = int(d["trades"].timestamp.min())
    t, P_all = build_positions(d)
    # Só trades FECHADOS depois do marco contam; o resto é histórico pré-reset (viesado).
    P = P_all[P_all.t_close >= start_ms] if len(P_all) else P_all
    P_pre = P_all.drop(P.index) if len(P_all) else P_all
    t = t[t.timestamp >= start_ms] if len(t) else t
    n = len(P)
    resets, changed, clean_start = last_reset_and_setting_change(
        d, start_ms, RESET_TOLERANCE_MS if (epoch_ms is not None and not a.start) else 0)
    total_days = (now_ms - start_ms) / 86_400_000
    clean_days = (now_ms - clean_start) / 86_400_000

    R = {"start_utc": datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat(),
         "total_days": round(total_days, 3), "clean_days": round(clean_days, 3), "closed_trades": n,
         "reset_epoch_utc": (datetime.fromtimestamp(epoch_ms / 1000, timezone.utc).isoformat()
                             if epoch_ms is not None else None),
         "ledger_epoch_trade_rowid": epoch_rowid,
         "pre_reset_history": {
             "label": "histórico pré-reset (viesado, não conta)",
             "closed_trades": int(len(P_pre)),
             "net_pnl_stress_usd": float(P_pre.net_stress.sum()) if len(P_pre) else 0.0,
             "expectancy_net_stress_pct": (float((P_pre.net_stress / P_pre.entry_notional).mean() * 100)
                                           if len(P_pre) else float("nan"))}}
    ref = json.load(open(a.ref)) if os.path.exists(a.ref) else None

    # ----- métricas de trade -----
    if n:
        ret = (P.net_stress / P.entry_notional).values
        gross_ret = (P.gross / P.entry_notional).values
        R["expectancy_net_stress_pct"] = float(ret.mean() * 100)
        R["expectancy_gross_pct"] = float(gross_ret.mean() * 100)
        R["expectancy_booked_pct"] = float((P.net_booked / P.entry_notional).mean() * 100)
        R["cost_stress_per_trade_pct"] = float((P.cost_stress / P.entry_notional).mean() * 100)
        lo, hi = boot_mean_ci(ret)
        R["exp_ci95_one_sided_pct"] = [lo * 100, hi * 100]
        R["net_pnl_stress_usd"] = float(P.net_stress.sum())
        R["net_pnl_booked_usd"] = float(P.net_booked.sum())
        wins = int((P.gross > 0).sum())
        R["win_rate_gross"] = wins / n
        wl, wh = wilson(wins, n)
        W = P.gross[P.gross > 0].mean() if wins else 0.0
        L = -P.gross[P.gross <= 0].mean() if wins < n else 0.0
        C = P.cost_stress.mean()
        R["breakeven_wr"] = float((L + C) / (W + L)) if (W + L) > 0 else float("nan")
        R["wr_wilson95"] = [wl, wh]
    else:
        lo = hi = wl = wh = float("nan")
        R.update(breakeven_wr=float("nan"), win_rate_gross=float("nan"), wr_wilson95=[wl, wh],
                 expectancy_net_stress_pct=float("nan"), net_pnl_stress_usd=0.0)

    # ----- diário / Sharpe / DSR -----
    sharpe = dsr_p = float("nan")
    weeks_pos = float("nan")
    if n:
        day = pd.to_datetime(P.t_close, unit="ms", utc=True).dt.floor("D")
        s = (P.net_stress.groupby(day.values).sum() / INITIAL_CAPITAL)
        idx = pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC").floor("D"),
                            pd.Timestamp(now_ms, unit="ms", tz="UTC").floor("D"), freq="D")
        s = s.reindex(idx, fill_value=0.0)
        R["daily_obs"] = int(len(s))
        if len(s) >= 30:
            sharpe = calculate_annualized_sharpe(s.values, 365.0)
            tl = pd.read_csv(TRIAL_LOG)
            var_tr = float(np.nanvar(pd.to_numeric(tl.sharpe_ratio, errors="coerce"), ddof=1))
            dsr = deflated_sharpe_ratio(sharpe, s.values, get_total_trials_count() + 1, var_tr, 365.0)
            dsr_p = dsr["dsr"]
        if len(s) >= WEEK_MIN_FOR_TEST * 7:
            w = s.resample("7D").sum()
            weeks_pos = float((w > 0).mean())
    R["sharpe_net_ann"], R["dsr"], R["weeks_positive_frac"] = sharpe, dsr_p, weeks_pos

    eq, mdd = stress_equity(d, t)
    R["max_dd_stress_pct"] = mdd
    R["equity_stress_last"] = float(eq.iloc[-1])

    # ----- integridade -----
    R["manual_resets_in_window"] = int(len(resets))
    R["setting_changes_in_window"] = int(len(changed))
    cd = d["cb"][(d["cb"].event_type == "CONSECUTIVE_LOSS_COOLDOWN") & (d["cb"].timestamp >= start_ms)]
    R["cooldown_events"] = int(len(cd))
    R["rollback_events"] = int(((d["cb"].event_type == "DEGRADATION_ROLLBACK_ACTIVATED")
                                & (d["cb"].timestamp >= start_ms)).sum())
    strat = sorted(set(d["pos"].strategy_type))
    hyp = open(HYP_DOC, errors="ignore").read().upper() if os.path.exists(HYP_DOC) else ""
    unreg = [s_ for s_ in strat if s_.upper() not in PREREG_ALLOW
             and s_.upper() not in hyp and s_.upper().replace("_", " ") not in hyp]
    R["strategies_traded"], R["unregistered_strategies"] = strat, unreg
    ops = ops_health(d, start_ms, now_ms)
    R["ops"] = ops

    # ----- divergência paper vs backtest -----
    div = {"reference_present": ref is not None}
    div_ok = False
    if ref and n:
        e_ok = R["expectancy_net_stress_pct"] >= DIVERGENCE_MIN_RATIO * ref.get("expectancy_net_pct", float("inf"))
        s_ok = (not math.isnan(sharpe)) and sharpe >= DIVERGENCE_MIN_RATIO * ref.get("sharpe_net", float("inf"))
        tpd = n / max(total_days, 1e-9)
        f_ok = abs(tpd / ref.get("trades_per_day", float("inf")) - 1) <= FREQ_TOL
        d_ok = mdd <= 1.5 * ref.get("max_dd_pct", 0.0)
        div.update(expectancy_ok=e_ok, sharpe_ok=s_ok, freq_ok=f_ok, dd_ok=d_ok, trades_per_day=tpd)
        div_ok = e_ok and s_ok and f_ok and d_ok
    R["divergence"] = div

    # ============================ CRITÉRIOS DE REFUTAÇÃO ============================
    ref_hits = []
    if mdd >= MAX_DD_REFUTE:
        ref_hits.append(f"R1 MaxDD estresse {mdd:.1f}% >= {MAX_DD_REFUTE}%")
    if n >= FUTILITY_N and not math.isnan(hi) and hi < 0:
        ref_hits.append(f"R2 N={n}>={FUTILITY_N} e limite superior IC95 unilateral da expectância líquida "
                        f"({hi*100:+.3f}%/trade) < 0")
    if n >= FUTILITY_N and wh < R["breakeven_wr"]:
        ref_hits.append(f"R3 Wilson sup. do hit rate {wh:.1%} < breakeven {R['breakeven_wr']:.1%}")
    if R["cooldown_events"] >= MAX_COOLDOWNS_REFUTE:
        ref_hits.append(f"R4 {R['cooldown_events']} disparos de cooldown por perdas consecutivas (>= {MAX_COOLDOWNS_REFUTE})")
    if clean_days >= 28 and R["net_pnl_stress_usd"] < 0:
        ref_hits.append("R5 >= 28 dias limpos com P&L líquido estresse negativo")
    if n and R["expectancy_gross_pct"] <= 0 and n >= FUTILITY_N:
        ref_hits.append(f"R6 expectância BRUTA {R['expectancy_gross_pct']:+.3f}% <= 0: sem alfa antes de custos")
    if total_days >= MIN_DAYS and ((not math.isnan(sharpe) and sharpe < MIN_SHARPE)
                                   or (not math.isnan(dsr_p) and dsr_p < MIN_DSR)):
        ref_hits.append("R7 prazo final atingido com Sharpe<1,0 ou DSR<0,95")

    # ============================ CRITÉRIOS DE APROVAÇÃO ============================
    def nz(x):
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    A = [
        ("A1 Duração limpa >= 56d", clean_days >= MIN_DAYS, f"{clean_days:.2f}d (total {total_days:.2f}d)"),
        ("A2 Trades fechados >= 100", n >= MIN_TRADES, f"{n}"),
        ("A3 Estratégia pré-registrada", not unreg, f"não registradas: {unreg}" if unreg else "ok"),
        ("A4 Expectância líquida estresse > 0 e IC95 inf > 0", nz(lo) and lo > 0,
         f"{R['expectancy_net_stress_pct']:+.3f}%/trade; IC inf {lo*100 if nz(lo) else float('nan'):+.3f}%"),
        ("A5 Sharpe líquido anual >= 1,0", nz(sharpe) and sharpe >= MIN_SHARPE, f"{sharpe:.2f}"),
        ("A6 DSR >= 0,95", nz(dsr_p) and dsr_p >= MIN_DSR, f"{dsr_p:.3f}"),
        ("A7 MaxDD estresse <= 15%", mdd <= MAX_DD_APPROVE, f"{mdd:.2f}%"),
        ("A8 Hit rate (Wilson inf) > breakeven", nz(wl) and wl > R["breakeven_wr"],
         f"WR {R['win_rate_gross']:.1%} [{wl:.1%},{wh:.1%}] vs BE {R['breakeven_wr']:.1%}"),
        ("A9 Divergência paper vs backtest dentro da tolerância", div_ok,
         json.dumps(div, default=str)),
        ("A10 Semanas positivas >= 60% (>=8 sem.)", nz(weeks_pos) and weeks_pos >= MIN_WEEKS_POSITIVE,
         f"{weeks_pos:.0%}" if nz(weeks_pos) else "dados insuficientes"),
        ("A11 Integridade: 0 resets/alterações de parâmetro na janela",
         R["manual_resets_in_window"] == 0 and R["setting_changes_in_window"] == 0,
         f"resets={R['manual_resets_in_window']} settings={R['setting_changes_in_window']}"),
        ("A12 Cobertura de snapshots >= 98% e gap máx <= 60min",
         ops["coverage"] >= MIN_COVERAGE and ops["max_gap_min"] <= MAX_GAP_MIN,
         f"cov {ops['coverage']:.0%} gap {ops['max_gap_min']:.0f}min"),
        ("A13 Daemon saudável, reinícios <= 2, DB ok, snapshot fresco (<=30min)",
         ops["daemon_status"] == "HEALTHY" and ops["db_status"] == "HEALTHY"
         and (ops["daemon_restarts"] or 0) <= MAX_DAEMON_RESTARTS_30D and ops["last_snapshot_age_min"] <= 30,
         f"daemon={ops['daemon_status']} restarts={ops['daemon_restarts']} snap_age={ops['last_snapshot_age_min']:.0f}min"),
        ("A14 Ledger reconcilia (|Δ| < $0,01)", ops["ledger_mismatch_usd"] < 0.01,
         f"Δ=${ops['ledger_mismatch_usd']:.4f}"),
    ]
    if n < FUTILITY_N:
        ref_hits = []  # N insuficiente: refutação precoce proibida (R1-R7)
    R["criteria"] = [{"id": k, "pass": bool(v), "detail": det} for k, v, det in A]
    R["refutations"] = ref_hits

    if ref_hits:
        verdict = "REPROVADO"
    elif all(v for _, v, _ in A):
        verdict = "APROVADO"
    else:
        verdict = "INCONCLUSIVO"
    R["verdict"] = verdict
    json.dump(R, open(OUT_JSON, "w"), indent=2, default=str)

    if a.json:
        print(json.dumps(R, indent=2, default=str))
    else:
        print(f"=== PILOT REPORT | {now.isoformat(timespec='seconds')} | início {R['start_utc']} ===")
        print(f"Trades fechados: {n} | dias totais {total_days:.2f} | dias limpos {clean_days:.2f}")
        if n:
            print(f"Expectância/trade: bruta {R['expectancy_gross_pct']:+.3f}% | líquida booked "
                  f"{R['expectancy_booked_pct']:+.3f}% | líquida ESTRESSE {R['expectancy_net_stress_pct']:+.3f}% "
                  f"(custo estresse {R['cost_stress_per_trade_pct']:.3f}%/trade)")
            print(f"P&L USD: booked {R['net_pnl_booked_usd']:+.2f} | estresse {R['net_pnl_stress_usd']:+.2f} | "
                  f"MaxDD estresse {mdd:.2f}%")
        pre = R["pre_reset_history"]
        print(f"Reset intencional (início da janela): {R['reset_epoch_utc']} | rowid {epoch_rowid}")
        print(f"[{pre['label']}] trades fechados: {pre['closed_trades']} | "
              f"P&L estresse {pre['net_pnl_stress_usd']:+.2f} USD")
        if n < FUTILITY_N:
            print(f"N pós-reset={n} < {FUTILITY_N}: refutações R1-R7 desativadas (INCONCLUSIVO)")
        print("\nCRITÉRIOS DE APROVAÇÃO (todos obrigatórios):")
        for k, v, det in A:
            print(f"  [{'OK' if v else 'XX'}] {k}: {det}")
        print("\nCONDIÇÕES DE REFUTAÇÃO ATINGIDAS (qualquer uma = REPROVADO):")
        for h in ref_hits or ["(nenhuma)"]:
            print(f"  - {h}")
        print(f"\nVEREDITO: {verdict}")
    return {"APROVADO": 0, "REPROVADO": 1, "INCONCLUSIVO": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
