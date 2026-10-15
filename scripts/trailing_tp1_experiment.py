"""Experimento OFFLINE: trailing ligando já no TP1 ('tp1') vs padrão do piloto ('tp2'), baseline PROD (variante BASE).
Usa só dados <= DEV_END (2025-10-13 23:00 UTC): o holdout do projeto (2025-10-14..2026-10-14) permanece trancado.
Uso: .venv/bin/python scripts/trailing_tp1_experiment.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.ab_consensus_experiment as ab  # noqa: E402

DEV_END_MS = int(pd.Timestamp("2025-10-13 23:00", tz="UTC").timestamp() * 1000)


def main() -> None:
    variants = [v for v in ab.build_variants() if v.name == "BASE"]
    ps = [ab.symbol_worker((s, variants, False, ab.DB_PATH)) for s in ab.SYMBOLS]
    for sy in ps:                                   # trunca em DEV_END
        k = int(np.searchsorted(sy["ts"], DEV_END_MS, side="right"))
        for key in ("ts", "open", "high", "low", "close", "atr", "sma"):
            sy[key] = sy[key][:k]
        sy["dec"]["BASE"] = {kk: vv[:k] for kk, vv in sy["dec"]["BASE"].items()}
    funding = {s: ab.load_funding(s) for s in ab.SYMBOLS}
    start = min(int(s["ts"][0]) for s in ps)
    end = DEV_END_MS + ab.HOUR_MS
    res = {}
    for mode in ("tp2", "tp1"):
        tr = ab.simulate_trades_pilot(ps, "BASE", funding, trailing_activation=mode)
        m = ab.segment_metrics(tr, start, end)
        res[mode] = m
        print(f"\n== trailing_activation={mode} ==")
        print(f"n={m['n_trades']} hit={m['hit_rate']*100:.1f}% exp_net={m['expectancy_net']*100:+.4f}% "
              f"IC95=[{m['exp_net_ci'][0]*100:+.4f},{m['exp_net_ci'][1]*100:+.4f}]% exp_gross={m['expectancy_gross']*100:+.4f}% "
              f"sharpe={m['sharpe']:+.2f} maxDD={m['max_dd']*100:.1f}% ret_total_capital={m['total_return']*100:+.2f}%")
        print("motivos:", m["by_reason"])
        yr = tr.assign(y=pd.to_datetime(tr["entry_ts"], unit="ms").dt.year).groupby("y")["ret_notional"].agg(["count", "mean"])
        print("por ano:", {int(k): (int(v["count"]), round(float(v["mean"]) * 100, 3)) for k, v in yr.iterrows()})
    d = res["tp1"]["daily"] - res["tp2"]["daily"]
    b = ab.block_bootstrap_diff(d)
    print(f"\nDiferença diária pareada (tp1 - tp2): média={b['mean']*1e4:+.3f} bps/dia(capital) "
          f"IC95 bloco=[{b['lo']*1e4:+.3f},{b['hi']*1e4:+.3f}]  p(<=0)={b['p_le0']:.3f}")


if __name__ == "__main__":
    main()
