#!/usr/bin/env python3
"""
scripts/backtest_market_open_strategies.py - Backtest Institucional (OOS + Pilot Parity).

Refatorado (S4 Audit, 2026-11-13):
- Elimina vazamento ML in-sample dos arquivos .joblib de produção e não mistura modelos 1h no 15m.
- Executa via `BacktestEngine(pilot_parity=True)` (Smart Trade 40/30/30, BE +0.04%, trailing pós-TP2, funding real 8h).
- Gera probabilidades ML estritamente out-of-sample (4-fold PurgedTimeSeriesSplit com embargo) por ativo e timeframe.
- Reporta tabela consolidada de portfólio (`run_portfolio`) e breakdown isolado por ativo (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.feature_engineering import FeatureEngineer  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from scripts.eval_s4_calibration_and_open import (  # noqa: E402
    MAJORS,
    _eval_strategies_oos_pilot,
    _generate_block2_signals_15m,
    _generate_block2_signals_1h,
    load_candles_clean,
    load_funding_map,
    select_stationary_features,
)
from statistical_rigor import get_prior_trials_count  # noqa: E402


def _build_oos_ml_dataset(timeframe: str, limit: int = 25000) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, np.ndarray]]]:
    data_oos: Dict[str, pd.DataFrame] = {}
    sigs_oos: Dict[str, Dict[str, np.ndarray]] = {}

    for sym in MAJORS:
        df = load_candles_clean(sym, timeframe, limit=limit)
        if df.empty or len(df) < 1000:
            continue
        feat = FeatureEngineer.compute_all_features(df, drop_na=False)
        fcols = select_stationary_features(feat)
        probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for mtype in ("lgbm", "rf"):
            mt = ModelTrainer(
                model_type=mtype,
                horizon=5,
                multiple_tp=1.2,
                multiple_sl=1.2,
                confidence_threshold=0.50,
                n_jobs=8,
            )
            X, y = mt.prepare_dataset(feat, feature_columns=fcols)
            res = mt.train_walk_forward(X, y, n_splits=4)
            idx_df = X.index[np.asarray(res["test_indices"], dtype=int)].to_numpy()
            probs[mtype] = (idx_df, np.asarray(res["y_proba"], dtype=np.float64))

        idx_l, p_l = probs["lgbm"]
        idx_r, p_r = probs["rf"]
        common = np.intersect1d(idx_l, idx_r)
        p_buy = np.zeros(len(feat), dtype=np.float64)
        p_sell = np.zeros(len(feat), dtype=np.float64)
        map_l = {int(i): k for k, i in enumerate(idx_l)}
        map_r = {int(i): k for k, i in enumerate(idx_r)}
        for i in common:
            kl, kr = map_l[int(i)], map_r[int(i)]
            p_buy[int(i)] = 0.5 * (p_l[kl, 1] + p_r[kr, 1])
            p_sell[int(i)] = 0.5 * (p_l[kl, 2] + p_r[kr, 2])

        oos_start = int(common.min())
        if timeframe == "1h":
            sigs = _generate_block2_signals_1h(feat, p_buy, p_sell)
            df_slice = feat.iloc[oos_start:].copy().reset_index(drop=True)
        else:
            sigs = _generate_block2_signals_15m(feat, p_buy, p_sell)
            df_slice = feat.iloc[oos_start:].copy().reset_index(drop=True)
            df_slice["atr_14"] = np.maximum(
                df_slice["atr_14"].to_numpy(float),
                df_slice["close"].to_numpy(float) * 0.0025,
            )

        data_oos[sym] = df_slice
        sigs_oos[sym] = {k: v[oos_start:] for k, v in sigs.items()}

    return data_oos, sigs_oos


def _print_table(title: str, res_block: Dict[str, Any]) -> None:
    print(f"\n{'=' * 126}")
    print(f"{title} ({res_block['oos_start']} .. {res_block['oos_end']})")
    print(f"{'=' * 126}")
    fmt = "{:<44} | {:>6} | {:>6} | {:>6} | {:>8} | {:>10} | {:>7} | {:>19} | {:>19} | {:>19}"
    print(fmt.format("Estratégia", "Trades", "WR%", "PF", "E[R]", "PnL ($10k)", "MaxDD%", "BTC WR/PF/PnL", "ETH WR/PF/PnL", "SOL WR/PF/PnL"))
    print("-" * 126)
    for name, pm in res_block["consolidated_portfolio"].items():
        pa = res_block["per_asset"][name]
        def _a(sym: str) -> str:
            d = pa[sym]
            return f"{d['win_rate_pct']:.1f}%/{d['profit_factor']:.2f}/${d['net_pnl_usd']:+.0f}"
        print(
            fmt.format(
                name[:44],
                pm["trades"],
                f"{pm['win_rate_pct']:.1f}%",
                f"{pm['profit_factor']:.2f}",
                f"{pm['expectancy_r']:+.3f}R",
                f"${pm['net_pnl_usd']:+,.2f}",
                f"{pm['max_dd_pct']:.1f}%",
                _a("BTCUSDT"),
                _a("ETHUSDT"),
                _a("SOLUSDT"),
            )
        )


def main() -> int:
    t0 = time.time()
    n_prior = get_prior_trials_count()
    funding = load_funding_map(MAJORS)

    data_1h, sigs_1h = _build_oos_ml_dataset("1h", limit=25000)
    res_1h = _eval_strategies_oos_pilot(
        data_1h, sigs_1h, funding, "DIRECTIONAL", "1h",
        daily_quota=None, cooldown_bars=0, n_prior_trials=n_prior,
    )
    _print_table("CONSOLIDADO 1H OOS (DIRECTIONAL Pilot Parity + Walk-Forward Purgado)", res_1h)

    data_15m, sigs_15m = _build_oos_ml_dataset("15m", limit=25000)
    res_15m = _eval_strategies_oos_pilot(
        data_15m, sigs_15m, funding, "SCALP", "15m",
        daily_quota=15, cooldown_bars=4, n_prior_trials=n_prior + 7,
    )
    _print_table("CONSOLIDADO 15M OOS (SCALP Pilot Parity + Walk-Forward Purgado)", res_15m)
    print(f"\n[OK] Finalizado em {time.time() - t0:.1f}s (zero lookahead / zero vazamento in-sample).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
