"""
Script de Geração de Relatórios Completos e Avaliação em Holdout Congelado da Fase 4.
Executa:
1. Exportação completa de todas as 36 configurações de ablação para CSV (/datasets/reports/phase4_ablation_complete.csv)
   com correção de Bonferroni estrita (36 testes) e rotulagem mandante:
   - Se N < 100: 'AMOSTRA INSUFICIENTE (N < 100)'
   - Maker rotulado como 'Sensibilidade Hipotética (sem modelagem de fill rate/seleção adversa)'
2. Congelamento e Avaliação Única em Holdout Puro:
   - Treino e calibração de regras até a data de corte: 2026-04-14 (2 anos de histórico).
   - Avaliação ÚNICA e intocada nos 6 meses mais recentes: 2026-04-14 a 2026-10-14.
   - Sem refinamento ou seleção a posteriori.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Adiciona raiz ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from agents.base_agent import AgentSignal
from agents.macro_agent import MacroAgent
from agents.onchain_agent import OnChainAgent
from agents.orderflow_agent import OrderFlowAgent
from agents.risk_sentinel_agent import RiskSentinelAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from backtesting import BacktestEngine, BacktestResult, TradeRecord
from consensus_judge import ConsensusDecision, ConsensusJudge
from data.feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer, PurgedTimeSeriesSplit

DB_PATH = "/home/eudespatricio/ia_trader/datasets/market_cache.db"
REPORTS_DIR = "/home/eudespatricio/ia_trader/datasets/reports"
CSV_PATH = os.path.join(REPORTS_DIR, "phase4_ablation_complete.csv")

TARGETS = [
    ("BTCUSDT", "1h"),
    ("BTCUSDT", "4h"),
    ("ETHUSDT", "1h"),
    ("ETHUSDT", "4h"),
    ("SOLUSDT", "1h"),
    ("SOLUSDT", "4h"),
]

TIMEFRAME_CONFIGS = {
    "1h": {"horizon": 6, "multiple_tp": 1.5, "multiple_sl": 1.0, "bar_hours": 1.0},
    "4h": {"horizon": 6, "multiple_tp": 2.0, "multiple_sl": 1.2, "bar_hours": 4.0},
}

ABLATION_STEPS = [
    ("1_TECH", "Técnico Sozinho", ["TechnicalAgent"]),
    ("2_TECH_FLOW", "+ Fluxo", ["TechnicalAgent", "OrderFlowAgent"]),
    ("3_TECH_FLOW_MACRO", "+ Macro", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent"]),
    ("4_TECH_FLOW_MACRO_SENT", "+ Sentimento", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent"]),
    ("5_FULL_NO_VETO", "+ On-Chain", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent", "OnChainAgent"]),
    ("6_COMMITTEE_VETO", "+ Sentinela Veto (Completo)", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent", "OnChainAgent", "RiskSentinelAgent"]),
]


def load_auxiliary_data() -> Tuple[Dict[Tuple[str, str], float], set]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT series_name, date_str, value FROM macro_daily_history")
    series_dict = {(row[0], row[1]): float(row[2]) for row in cursor.fetchall()}

    cursor.execute("SELECT date_str FROM macro_events_calendar")
    event_dates = {row[0] for row in cursor.fetchall()}
    conn.close()
    return series_dict, event_dates


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper(), timeframe.lower()))
    conn.close()
    return df


def calculate_bootstrap_ci(trades: List[TradeRecord], n_iterations: int = 1000) -> Tuple[float, float]:
    if not trades:
        return 0.0, 0.0
    wins = np.array([1 if t.pnl_net > 0 else 0 for t in trades])
    boot_means = []
    np.random.seed(42)
    for _ in range(n_iterations):
        sample = np.random.choice(wins, size=len(wins), replace=True)
        boot_means.append(np.mean(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def run_full_ablation_grid() -> List[Dict[str, Any]]:
    aux_series, cpi_fomc_dates = load_auxiliary_data()
    all_rows = []

    tech_agent = TechnicalAgent(confidence_threshold=0.55)
    flow_agent = OrderFlowAgent()
    sent_agent = SentimentAgent()
    macro_agent = MacroAgent()
    onchain_agent = OnChainAgent()
    sentinel_agent = RiskSentinelAgent(max_atr_volatility_multiplier=2.5)

    all_agents_map = {
        "TechnicalAgent": tech_agent,
        "OrderFlowAgent": flow_agent,
        "MacroAgent": macro_agent,
        "SentimentAgent": sent_agent,
        "OnChainAgent": onchain_agent,
        "RiskSentinelAgent": sentinel_agent,
    }

    for symbol, timeframe in TARGETS:
        cfg = TIMEFRAME_CONFIGS[timeframe]
        horizon = cfg["horizon"]
        multiple_tp = cfg["multiple_tp"]
        multiple_sl = cfg["multiple_sl"]

        raw_df = load_candles(symbol, timeframe)
        if len(raw_df) < 300:
            continue

        features_df = FeatureEngineer.compute_all_features(raw_df, drop_na=True)
        trainer = ModelTrainer(
            model_type="lgbm",
            horizon=horizon,
            multiple_tp=multiple_tp,
            multiple_sl=multiple_sl,
            confidence_threshold=0.55,
            n_jobs=8,
        )
        X, y = trainer.prepare_dataset(features_df)

        splitter = PurgedTimeSeriesSplit(n_splits=4, horizon_embargo=horizon)
        splits = splitter.split(X, y)

        oos_records = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]

            model = trainer._build_estimator(random_state=42 + fold)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)
            classes = model.classes_

            buy_col = np.where(classes == 1)[0]
            sell_col = np.where(classes == 2)[0]
            p_buy = y_proba[:, buy_col[0]] if len(buy_col) > 0 else np.zeros(len(X_test))
            p_sell = y_proba[:, sell_col[0]] if len(sell_col) > 0 else np.zeros(len(X_test))

            for local_idx, orig_idx in enumerate(test_idx):
                oos_records.append({
                    "orig_idx": orig_idx,
                    "proba_buy": float(p_buy[local_idx]),
                    "proba_sell": float(p_sell[local_idx]),
                })

        oos_df = pd.DataFrame(oos_records)
        df_oos = features_df.iloc[oos_df["orig_idx"].values].copy().reset_index(drop=True)
        df_oos["proba_buy"] = oos_df["proba_buy"].values
        df_oos["proba_sell"] = oos_df["proba_sell"].values

        taker_engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0006,
            slippage=0.0002,
            funding_rate_8h=0.0001,
            leverage=1.0,
            tp_atr_multiple=multiple_tp,
            sl_atr_multiple=multiple_sl,
            trailing_stop=False,
            risk_per_trade=0.01,
            allow_reversal=False,
            max_holding_bars=horizon,
        )

        maker_engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0002,
            slippage=0.0,
            funding_rate_8h=0.0001,
            leverage=1.0,
            tp_atr_multiple=multiple_tp,
            sl_atr_multiple=multiple_sl,
            trailing_stop=False,
            risk_per_trade=0.01,
            allow_reversal=False,
            max_holding_bars=horizon,
        )

        mean_atr = df_oos["atr_14"].mean()
        mean_price = df_oos["close"].mean()
        atr_pct = mean_atr / mean_price
        taker_friction_atr = 0.0016 / max(1e-4, atr_pct)
        null_wr_taker = (multiple_sl + taker_friction_atr) / (multiple_tp + multiple_sl)
        null_wr_taker = min(0.60, max(0.40, null_wr_taker))

        for step_id, step_name, active_agent_names in ABLATION_STEPS:
            step_weights = {name: 1.0 for name in active_agent_names}
            judge = ConsensusJudge(min_consensus_threshold=0.30, default_weights=step_weights)

            signals_oos = np.zeros(len(df_oos), dtype=int)
            for i in range(len(df_oos)):
                row = df_oos.iloc[i]
                ts = int(row["timestamp"])
                prev_dt = datetime.fromtimestamp((ts - 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                curr_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

                fng_val = aux_series.get(("FEAR_AND_GREED", prev_dt), 50.0)
                stbl_val = aux_series.get(("STABLECOINS_CIRCULATING_USD", prev_dt))
                stbl_prev30 = aux_series.get(("STABLECOINS_CIRCULATING_USD", datetime.fromtimestamp((ts - 30 * 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
                stbl_growth = ((stbl_val - stbl_prev30) / stbl_prev30) if (stbl_val and stbl_prev30 and stbl_prev30 > 0) else 0.0

                dxy_val = aux_series.get(("DXY", prev_dt))
                dxy_prev5 = aux_series.get(("DXY", datetime.fromtimestamp((ts - 5 * 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
                dxy_ret_5d = ((dxy_val - dxy_prev5) / dxy_prev5) if (dxy_val and dxy_prev5) else 0.0
                vix_val = aux_series.get(("VIX", prev_dt), 18.0)
                is_macro_event = (curr_dt in cpi_fomc_dates)

                context = {
                    "timestamp": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "volume_sma": row.get("volume_sma_20", row["volume"]),
                    "atr_14": row["atr_14"],
                    "atr_median": row.get("atr_14", row["close"] * 0.01),
                    "proba_buy": row["proba_buy"],
                    "proba_sell": row["proba_sell"],
                    "fear_and_greed": fng_val,
                    "stablecoin_growth_pct_30d": stbl_growth,
                    "dxy_ret_5d": dxy_ret_5d,
                    "dxy_above_sma20": (dxy_ret_5d > 0),
                    "vix_value": vix_val,
                    "is_cpi_fomc_window": is_macro_event,
                }

                step_signals = [all_agents_map[name].evaluate(context) for name in active_agent_names]
                decision = judge.judge_deterministic(step_signals, weights_override=step_weights)

                if decision.direction == "BUY":
                    signals_oos[i] = 1
                elif decision.direction == "SELL":
                    signals_oos[i] = 2

            res_taker = taker_engine.run(df_oos, signals_oos)
            res_maker = maker_engine.run(df_oos, signals_oos)

            trades_n = res_taker.trades_count
            wins = res_taker.winning_trades
            wr_taker = res_taker.win_rate
            ci_low, ci_high = calculate_bootstrap_ci(res_taker.trades)

            if trades_n > 0:
                b_test = binomtest(k=wins, n=trades_n, p=null_wr_taker, alternative="greater")
                p_val = float(b_test.pvalue)
            else:
                p_val = 1.0

            coverage_pct = float(trades_n / len(df_oos)) if len(df_oos) > 0 else 0.0
            exits = res_taker.exit_reasons or {}

            # Regra estrita de tamanho amostral (N < 100)
            sample_valid = (trades_n >= 100)
            status_label = "AMOSTRA SUFICIENTE" if sample_valid else "AMOSTRA INSUFICIENTE (N < 100)"

            row_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "step_id": step_id,
                "step_name": step_name,
                "active_agents": "+".join(active_agent_names),
                "oos_bars": len(df_oos),
                "trades": trades_n,
                "coverage_pct": round(coverage_pct * 100.0, 2),
                "win_rate": round(wr_taker * 100.0, 2),
                "ci_low": round(ci_low * 100.0, 2),
                "ci_high": round(ci_high * 100.0, 2),
                "null_wr": round(null_wr_taker * 100.0, 2),
                "p_value": round(p_val, 6),
                "sample_size_valid": sample_valid,
                "status_label": status_label,
                "gross_profit": round(res_taker.gross_profit, 2),
                "fees_taker": round(res_taker.total_fees, 2),
                "net_pnl_taker": round(res_taker.net_profit, 2),
                "net_pct_taker": round(res_taker.net_profit_pct, 2),
                "profit_factor_taker": round(res_taker.profit_factor, 2),
                "net_pnl_maker_hypothetical": round(res_maker.net_profit, 2),
                "net_pct_maker_hypothetical": round(res_maker.net_profit_pct, 2),
                "profit_factor_maker_hypothetical": round(res_maker.profit_factor, 2),
                "delta_economy_maker": round(res_maker.net_profit - res_taker.net_profit, 2),
                "tp_count": exits.get("TP", 0),
                "sl_count": exits.get("SL", 0),
                "timeout_count": exits.get("HORIZON_TIMEOUT", 0),
            }
            all_rows.append(row_data)

    # Aplicação estrita de Bonferroni sobre o total real de testes (m = 36)
    m = len(all_rows)
    bonf_alpha = 0.05 / m
    print(f"\n[AUDITORIA ESTATÍSTICA] Total de configurações testadas na matriz: {m}")
    print(f"[AUDITORIA ESTATÍSTICA] Limiar de Bonferroni Estrito: alpha_adj = {bonf_alpha:.6f}")

    for r in all_rows:
        # Só pode ser considerado estatisticamente significativo se N >= 100 E p < bonf_alpha
        r["bonferroni_sig"] = bool(r["sample_size_valid"] and r["p_value"] < bonf_alpha)

    # Exportação para CSV
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[EXPORTAÇÃO CONCLUÍDA] Tabela completa de ablação salva em: {CSV_PATH}")
    return all_rows


def run_frozen_holdout_evaluation():
    """
    Executa a avaliação confirmatória estrita em período de HOLDOUT CONGELADO.
    - Data de corte: 2026-04-14 00:00:00 UTC (1776124800000).
    - Período de Desenvolvimento: Tudo anterior a 2026-04-14 (modelo treinado exclusivamente aqui).
    - Período de Holdout: 2026-04-14 a 2026-10-14 (6 meses / ~4.400 velas de 1h).
    - Configuração candidata CONGELADA a priori: BTCUSDT 1h, Comitê Completo + Veto, Risco 1%, min_consensus_threshold = 0.30.
    """
    print(f"\n{'='*100}")
    print(f"AVALIAÇÃO EM HOLDOUT CONGELADO (INTOCADO): BTCUSDT [1h]")
    print(f"Período de Holdout: 2026-04-14 a 2026-10-14 (6 meses)")
    print(f"{'='*100}")

    cutoff_dt = datetime(2026, 4, 14, 0, 0, 0, tzinfo=timezone.utc)
    cutoff_ts = int(cutoff_dt.timestamp() * 1000)

    raw_df = load_candles("BTCUSDT", "1h")
    features_df = FeatureEngineer.compute_all_features(raw_df, drop_na=True)

    # Separação estrita por data de corte
    train_df = features_df[features_df["timestamp"] < cutoff_ts].copy().reset_index(drop=True)
    holdout_df = features_df[features_df["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)

    print(f"Amostras de Desenvolvimento (Treino): {len(train_df)} velas ({pd.to_datetime(train_df['timestamp'].iloc[0], unit='ms')} a {pd.to_datetime(train_df['timestamp'].iloc[-1], unit='ms')})")
    print(f"Amostras de Holdout (Intocado): {len(holdout_df)} velas ({pd.to_datetime(holdout_df['timestamp'].iloc[0], unit='ms')} a {pd.to_datetime(holdout_df['timestamp'].iloc[-1], unit='ms')})")

    horizon = 6
    multiple_tp = 1.5
    multiple_sl = 1.0

    # 1. Treinamento do Modelo Candidato EXCLUSIVAMENTE nos dados de desenvolvimento
    trainer = ModelTrainer(
        model_type="lgbm",
        horizon=horizon,
        multiple_tp=multiple_tp,
        multiple_sl=multiple_sl,
        confidence_threshold=0.55,
        n_jobs=8,
    )
    X_train, y_train = trainer.prepare_dataset(train_df)
    model = trainer._build_estimator(random_state=42)
    model.fit(X_train, y_train)

    # 2. Inferência no Holdout (O modelo NUNCA viu nenhuma linha deste período)
    # Remove as colunas não-preditoras
    feature_cols = [c for c in X_train.columns]
    X_holdout = holdout_df[feature_cols].copy()

    y_proba = model.predict_proba(X_holdout)
    classes = model.classes_
    buy_col = np.where(classes == 1)[0]
    sell_col = np.where(classes == 2)[0]
    p_buy = y_proba[:, buy_col[0]] if len(buy_col) > 0 else np.zeros(len(X_holdout))
    p_sell = y_proba[:, sell_col[0]] if len(sell_col) > 0 else np.zeros(len(X_holdout))

    holdout_df["proba_buy"] = p_buy
    holdout_df["proba_sell"] = p_sell

    # 3. Montagem do Comitê Completo Congelado
    aux_series, cpi_fomc_dates = load_auxiliary_data()
    tech_agent = TechnicalAgent(confidence_threshold=0.55)
    flow_agent = OrderFlowAgent()
    sent_agent = SentimentAgent()
    macro_agent = MacroAgent()
    onchain_agent = OnChainAgent()
    sentinel_agent = RiskSentinelAgent(max_atr_volatility_multiplier=2.5)

    all_agents = [tech_agent, flow_agent, macro_agent, sent_agent, onchain_agent, sentinel_agent]
    judge = ConsensusJudge(min_consensus_threshold=0.30)

    signals_holdout = np.zeros(len(holdout_df), dtype=int)

    for i in range(len(holdout_df)):
        row = holdout_df.iloc[i]
        ts = int(row["timestamp"])
        prev_dt = datetime.fromtimestamp((ts - 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        curr_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        fng_val = aux_series.get(("FEAR_AND_GREED", prev_dt), 50.0)
        stbl_val = aux_series.get(("STABLECOINS_CIRCULATING_USD", prev_dt))
        stbl_prev30 = aux_series.get(("STABLECOINS_CIRCULATING_USD", datetime.fromtimestamp((ts - 30 * 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
        stbl_growth = ((stbl_val - stbl_prev30) / stbl_prev30) if (stbl_val and stbl_prev30 and stbl_prev30 > 0) else 0.0

        dxy_val = aux_series.get(("DXY", prev_dt))
        dxy_prev5 = aux_series.get(("DXY", datetime.fromtimestamp((ts - 5 * 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
        dxy_ret_5d = ((dxy_val - dxy_prev5) / dxy_prev5) if (dxy_val and dxy_prev5) else 0.0
        vix_val = aux_series.get(("VIX", prev_dt), 18.0)
        is_macro_event = (curr_dt in cpi_fomc_dates)

        context = {
            "timestamp": ts,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "volume_sma": row.get("volume_sma_20", row["volume"]),
            "atr_14": row["atr_14"],
            "atr_median": row.get("atr_14", row["close"] * 0.01),
            "proba_buy": row["proba_buy"],
            "proba_sell": row["proba_sell"],
            "fear_and_greed": fng_val,
            "stablecoin_growth_pct_30d": stbl_growth,
            "dxy_ret_5d": dxy_ret_5d,
            "dxy_above_sma20": (dxy_ret_5d > 0),
            "vix_value": vix_val,
            "is_cpi_fomc_window": is_macro_event,
        }

        agent_sigs = [a.evaluate(context) for a in all_agents]
        decision = judge.judge_deterministic(agent_sigs)

        if decision.direction == "BUY":
            signals_holdout[i] = 1
        elif decision.direction == "SELL":
            signals_holdout[i] = 2

    # 4. Execução ÚNICA do Backtest Institucional no Holdout
    taker_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,
        slippage=0.0002,
        funding_rate_8h=0.0001,
        leverage=1.0,
        tp_atr_multiple=multiple_tp,
        sl_atr_multiple=multiple_sl,
        trailing_stop=False,
        risk_per_trade=0.01,
        allow_reversal=False,
        max_holding_bars=horizon,
    )
    res_holdout = taker_engine.run(holdout_df, signals_holdout)

    # Baselines no mesmo Holdout
    bh_signals = np.zeros(len(holdout_df), dtype=int)
    bh_signals[0] = 1
    bh_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,
        slippage=0.0002,
        funding_rate_8h=0.0001,
        tp_atr_multiple=999.0,
        sl_atr_multiple=999.0,
        risk_per_trade=None,
    )
    res_bh = bh_engine.run(holdout_df, bh_signals)

    np.random.seed(42)
    rand_signals = np.random.choice([0, 1, 2], size=len(holdout_df), p=[0.7, 0.15, 0.15])
    res_rand = taker_engine.run(holdout_df, rand_signals)

    # Métricas Estatísticas do Holdout
    ci_low, ci_high = calculate_bootstrap_ci(res_holdout.trades)
    mean_atr = holdout_df["atr_14"].mean()
    mean_price = holdout_df["close"].mean()
    atr_pct = mean_atr / mean_price
    taker_friction_atr = 0.0016 / max(1e-4, atr_pct)
    null_wr_taker = (multiple_sl + taker_friction_atr) / (multiple_tp + multiple_sl)
    null_wr_taker = min(0.60, max(0.40, null_wr_taker))

    trades_n = res_holdout.trades_count
    wins = res_holdout.winning_trades
    wr = res_holdout.win_rate

    if trades_n > 0:
        b_test = binomtest(k=wins, n=trades_n, p=null_wr_taker, alternative="greater")
        p_val = float(b_test.pvalue)
    else:
        p_val = 1.0

    print(f"\n[RESULTADO OFICIAL DO HOLDOUT CONGELADO (CONFIRMAÇÃO ÚNICA)]:")
    print(f"  Velas no Holdout: {len(holdout_df)} (~6 meses)")
    print(f"  Trades Executados: {trades_n} (Cobertura: {trades_n / len(holdout_df):.2%})")
    print(f"  Win Rate Observado: {wr:.2%} (IC 95%: [{ci_low:.1%}, {ci_high:.1%}])")
    print(f"  Win Rate Nulo de Equilíbrio (Null WR_0): {null_wr_taker:.2%}")
    print(f"  P-Valor do Teste Binomial: {p_val:.4f} -> {'REJEITA H0 (Alfa Significativo)' if p_val < 0.05 and trades_n >= 100 else 'NÃO REJEITA H0 (Inconclusivo / Amostra Insuficiente)'}")
    print(f"  Lucro Bruto: ${res_holdout.gross_profit:,.2f}")
    print(f"  Taxas Taker + Slippage: ${res_holdout.total_fees:,.2f}")
    print(f"  Lucro Líquido Taker: ${res_holdout.net_profit:,.2f} ({res_holdout.net_profit_pct:.2f}%)")
    print(f"  Profit Factor: {res_holdout.profit_factor:.2f}")
    print(f"  Distribuição de Saídas: {json.dumps(res_holdout.exit_reasons)}")
    print(f"  Baselines no Holdout -> Buy & Hold: {res_bh.net_profit_pct:.2f}% | Sinais Aleatórios: {res_rand.net_profit_pct:.2f}% (WR={res_rand.win_rate:.2%})")

    return {
        "holdout_candles": len(holdout_df),
        "trades": trades_n,
        "win_rate": wr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "null_wr": null_wr_taker,
        "p_value": p_val,
        "gross_profit": res_holdout.gross_profit,
        "total_fees": res_holdout.total_fees,
        "net_profit": res_holdout.net_profit,
        "net_profit_pct": res_holdout.net_profit_pct,
        "profit_factor": res_holdout.profit_factor,
        "exit_reasons": res_holdout.exit_reasons,
        "bh_return_pct": res_bh.net_profit_pct,
        "rand_return_pct": res_rand.net_profit_pct,
    }


def main():
    print("=" * 100)
    print("INICIANDO RE-AUDITORIA RIGOROSA DA FASE 4: CSV COMPLETO + HOLDOUT CONGELADO")
    print("=" * 100)

    # 1. Gera tabela completa das 36 combinações em CSV
    run_full_ablation_grid()

    # 2. Executa a avaliação confirmatória única em Holdout Congelado
    run_frozen_holdout_evaluation()


if __name__ == "__main__":
    main()
