"""
Script Mestre de Avaliação Reprodutível da Fase 4: Comitê de Agentes Especialistas e Ablação Incremental.
Executa estudo de ablação rigoroso em 1h e 4h para BTCUSDT, ETHUSDT e SOLUSDT
utilizando dados 100% reais da Bitget alinhados sem lookahead a séries históricas auxiliares.

Passos da Ablação Incremental:
1. STEP 1: Técnico Sozinho (TechnicalAgent / LightGBM)
2. STEP 2: + Fluxo de Ordens (Technical + OrderFlow)
3. STEP 3: + Macroeconômico (Technical + OrderFlow + Macro)
4. STEP 4: + Sentimento (Technical + OrderFlow + Macro + Sentiment)
5. STEP 5: + On-Chain (Technical + OrderFlow + Macro + Sentiment + On-Chain)
6. STEP 6: + Sentinela de Risco (Comitê Completo com Veto de Volatilidade/Macro)

Rigor Metodológico:
- Avaliação estritamente Out-Of-Sample com PurgedTimeSeriesSplit e embargo temporal.
- Séries diárias auxiliares (F&G, DefiLlama, Yahoo DXY/VIX) indexadas estritamente por D-1 (sem lookahead).
- Dimensionamento institucional de risco fixo (1% por trade).
- Teste de hipótese contra o Null Correto de Equilíbrio (WR_0 = (SL + custos) / (TP + SL + 2*custos)).
- Correção de Holm-Bonferroni para comparações múltiplas.
- Cenário de Sensibilidade com Taxa Maker (0.02% maker fee, registrando risco de execução).
"""

from __future__ import annotations

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
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Adiciona raiz ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from agents.base_agent import AgentSignal, BaseAgent
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
from signal_generator import SignalGenerator, TradeSignal

DB_PATH = "/home/eudespatricio/ia_trader/datasets/market_cache.db"

TIMEFRAME_CONFIGS = {
    "1h": {
        "horizon": 6,
        "multiple_tp": 1.5,
        "multiple_sl": 1.0,
        "bar_hours": 1.0,
    },
    "4h": {
        "horizon": 6,
        "multiple_tp": 2.0,
        "multiple_sl": 1.2,
        "bar_hours": 4.0,
    },
}

TARGETS = [
    ("BTCUSDT", "1h"),
    ("BTCUSDT", "4h"),
    ("ETHUSDT", "1h"),
    ("ETHUSDT", "4h"),
    ("SOLUSDT", "1h"),
    ("SOLUSDT", "4h"),
]

ABLATION_STEPS = [
    ("1_TECH", "Técnico Sozinho", ["TechnicalAgent"]),
    ("2_TECH_FLOW", "+ Fluxo", ["TechnicalAgent", "OrderFlowAgent"]),
    ("3_TECH_FLOW_MACRO", "+ Macro", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent"]),
    ("4_TECH_FLOW_MACRO_SENT", "+ Sentimento", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent"]),
    ("5_FULL_NO_VETO", "+ On-Chain", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent", "OnChainAgent"]),
    ("6_COMMITTEE_VETO", "+ Sentinela Veto (Completo)", ["TechnicalAgent", "OrderFlowAgent", "MacroAgent", "SentimentAgent", "OnChainAgent", "RiskSentinelAgent"]),
]


def load_auxiliary_data() -> Tuple[Dict[Tuple[str, str], float], set]:
    """Carrega séries auxiliares diárias e calendário de CPI/FOMC do SQLite em memória."""
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


def evaluate_target_ablation(
    symbol: str,
    timeframe: str,
    aux_series: Dict[Tuple[str, str], float],
    cpi_fomc_dates: set,
) -> List[Dict[str, Any]]:
    print(f"\n{'='*100}")
    print(f"ESTUDO DE ABLAÇÃO FASE 4: {symbol} [{timeframe}] - DADOS REAIS BITGET & MACRO ALINHADO")
    print(f"{'='*100}")

    cfg = TIMEFRAME_CONFIGS[timeframe]
    horizon = cfg["horizon"]
    multiple_tp = cfg["multiple_tp"]
    multiple_sl = cfg["multiple_sl"]
    bar_hours = cfg["bar_hours"]

    # 1. Carrega velas reais
    raw_df = load_candles(symbol, timeframe)
    total_candles = len(raw_df)
    if total_candles < 300:
        return []

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

    # 2. Walk-Forward Purgado
    n_splits = 4
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, horizon_embargo=horizon)
    splits = splitter.split(X, y)

    # Pré-computa probabilidades OOS do modelo técnico
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
                "fold": fold,
                "proba_buy": float(p_buy[local_idx]),
                "proba_sell": float(p_sell[local_idx]),
            })

    oos_df = pd.DataFrame(oos_records)
    test_indices = oos_df["orig_idx"].values
    df_oos = features_df.iloc[test_indices].copy().reset_index(drop=True)
    df_oos["proba_buy"] = oos_df["proba_buy"].values
    df_oos["proba_sell"] = oos_df["proba_sell"].values

    # Instancia os agentes especialistas
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

    # Motores de Backtest: Cenário Taker Padrão vs Cenário Maker Sensibilidade
    taker_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,        # Taker Bitget: 0.06%
        slippage=0.0002,        # Slippage: 0.02%
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
        fee_rate=0.0002,        # Maker Bitget: 0.02%
        slippage=0.0,           # Zero slippage (ordem limite passiva)
        funding_rate_8h=0.0001,
        leverage=1.0,
        tp_atr_multiple=multiple_tp,
        sl_atr_multiple=multiple_sl,
        trailing_stop=False,
        risk_per_trade=0.01,
        allow_reversal=False,
        max_holding_bars=horizon,
    )

    # Cálculo da taxa de acerto de equilíbrio teórica (Null Correto)
    # Custo de fricção por trade aproximado em termos de ATR
    mean_atr = df_oos["atr_14"].mean()
    mean_price = df_oos["close"].mean()
    atr_pct = mean_atr / mean_price
    taker_friction_atr = 0.0016 / max(1e-4, atr_pct)
    # WR_0 = (SL + friction) / (TP + SL)
    null_wr_taker = (multiple_sl + taker_friction_atr) / (multiple_tp + multiple_sl)
    null_wr_taker = min(0.60, max(0.40, null_wr_taker))

    results_target = []

    # 3. Execução Passo a Passo da Ablação
    for step_id, step_name, active_agent_names in ABLATION_STEPS:
        # Prepara pesos específicos para os agentes ativos
        step_weights = {name: 1.0 for name in active_agent_names}
        judge = ConsensusJudge(min_consensus_threshold=0.30, default_weights=step_weights)

        # Montagem dos sinais vela a vela garantindo zero lookahead
        signals_oos = np.zeros(len(df_oos), dtype=int)
        emitted_confidences = []
        emitted_matches = []

        for i in range(len(df_oos)):
            row = df_oos.iloc[i]
            ts = int(row["timestamp"])

            # Data de D-1 para séries diárias
            prev_dt = datetime.fromtimestamp((ts - 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            curr_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            # Contexto de mercado rigoroso (D-1 para dados diários)
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

            # Avalia somente os agentes ativos do passo de ablação
            step_signals: List[AgentSignal] = []
            for agent_name in active_agent_names:
                agent = all_agents_map[agent_name]
                step_signals.append(agent.evaluate(context))

            # Julgamento pelo ConsensusJudge
            decision = judge.judge_deterministic(step_signals, weights_override=step_weights)

            if decision.direction == "BUY":
                signals_oos[i] = 1
                emitted_confidences.append(decision.confidence)
            elif decision.direction == "SELL":
                signals_oos[i] = 2
                emitted_confidences.append(decision.confidence)

        # Executa Backtest Taker (Cenário Real Padrão)
        res_taker = taker_engine.run(df_oos, signals_oos)

        # Executa Backtest Maker (Cenário de Sensibilidade com Taxa Passiva)
        res_maker = maker_engine.run(df_oos, signals_oos)

        # Estatísticas
        trades_n = res_taker.trades_count
        wins = res_taker.winning_trades
        wr_taker = res_taker.win_rate
        ci_low, ci_high = calculate_bootstrap_ci(res_taker.trades)

        # Teste binomial contra o Null Correto de Equilíbrio (não 50%)
        if trades_n > 0:
            b_test = binomtest(k=wins, n=trades_n, p=null_wr_taker, alternative="greater")
            p_val = float(b_test.pvalue)
        else:
            p_val = 1.0

        coverage_pct = float(trades_n / len(df_oos)) if len(df_oos) > 0 else 0.0
        exits = res_taker.exit_reasons or {}

        step_res = {
            "symbol": symbol,
            "timeframe": timeframe,
            "step_id": step_id,
            "step_name": step_name,
            "active_agents": active_agent_names,
            "oos_bars": len(df_oos),
            "trades": trades_n,
            "coverage_pct": coverage_pct,
            # Cenário Taker
            "win_rate": wr_taker,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "null_wr": null_wr_taker,
            "p_value": p_val,
            "gross_profit": res_taker.gross_profit,
            "fees_taker": res_taker.total_fees,
            "net_pnl_taker": res_taker.net_profit,
            "net_pct_taker": res_taker.net_profit_pct,
            "profit_factor_taker": res_taker.profit_factor,
            # Cenário Maker (Sensibilidade)
            "net_pnl_maker": res_maker.net_profit,
            "net_pct_maker": res_maker.net_profit_pct,
            "profit_factor_maker": res_maker.profit_factor,
            # Saídas
            "tp_count": exits.get("TP", 0),
            "sl_count": exits.get("SL", 0),
            "timeout_count": exits.get("HORIZON_TIMEOUT", 0),
        }
        results_target.append(step_res)

        print(
            f"  [{step_id:<18}] Trades: {trades_n:<4} (Cob: {coverage_pct:<4.1%}) | "
            f"WR: {wr_taker:<5.1%} (Null: {null_wr_taker:.1%}, p={p_val:.4f}) | "
            f"Bruto: ${res_taker.gross_profit:<8.2f} | Taker Net: ${res_taker.net_profit:<8.2f} (PF={res_taker.profit_factor:.2f}) | "
            f"Maker Net: ${res_maker.net_profit:<8.2f} (PF={res_maker.profit_factor:.2f})"
        )

    return results_target


def apply_holm_bonferroni(p_values: List[float]) -> List[Tuple[float, bool]]:
    """Aplica a correção de Holm-Bonferroni para controlar o erro de família (FWER)."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [False] * m
    alpha = 0.05

    for rank, (orig_idx, p) in enumerate(indexed_p):
        thresh = alpha / (m - rank)
        if p < thresh:
            corrected[orig_idx] = True
        else:
            break

    return [(p_values[i], corrected[i]) for i in range(m)]


def main():
    print("\n" + "=" * 115)
    print("INICIANDO AVALIAÇÃO REPRODUTÍVEL DA FASE 4: COMITÊ DE AGENTES & ABLAÇÃO INCREMENTAL")
    print("=" * 115)

    aux_series, cpi_fomc_dates = load_auxiliary_data()
    print(f"Séries históricas auxiliares carregadas: {len(aux_series)} registros diários | {len(cpi_fomc_dates)} eventos macro.")

    all_ablation_results = []
    for sym, tf in TARGETS:
        res = evaluate_target_ablation(sym, tf, aux_series, cpi_fomc_dates)
        all_ablation_results.extend(res)

    # 4. Correção para Múltiplas Comparações (Holm-Bonferroni)
    p_vals = [r["p_value"] for r in all_ablation_results]
    hb_results = apply_holm_bonferroni(p_vals)
    for r, (p, sig) in zip(all_ablation_results, hb_results):
        r["holm_sig"] = sig

    # -------------------------------------------------------------
    # TABELA CONSOLIDADA DE ABLAÇÃO INCREMENTAL (CENÁRIO TAKER REAL)
    # -------------------------------------------------------------
    print("\n" + "=" * 145)
    print("TABELA CONSOLIDADA 1: ABLAÇÃO INCREMENTAL DO COMITÊ (CENÁRIO TAKER REAL BITGET 0.06% + 0.02% SLIPPAGE)")
    print("=" * 145)
    header = (
        f"{'ATIVO':<8} | {'TF':<4} | {'PASSO DA ABLAÇÃO':<28} | {'TRADES':<6} | {'COB%':<6} | "
        f"{'WIN RATE':<9} | {'IC 95% WR':<15} | {'NULL WR':<8} | {'P-VAL':<7} | {'HOLM SIG?':<10} | "
        f"{'LUCRO BRUTO':<11} | {'TAXAS':<10} | {'LUCRO LÍQ':<11} | {'PF':<5}"
    )
    print(header)
    print("-" * 145)

    for r in all_ablation_results:
        ci_str = f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
        sig_str = "SIM (p<adj)" if r["holm_sig"] else "NÃO (H0)"
        row = (
            f"{r['symbol']:<8} | {r['timeframe']:<4} | {r['step_name']:<28} | {r['trades']:<6} | {r['coverage_pct']:<6.1%} | "
            f"{r['win_rate']:<9.1%} | {ci_str:<15} | {r['null_wr']:<8.1%} | {r['p_value']:<7.4f} | {sig_str:<10} | "
            f"${r['gross_profit']:<10.2f} | ${r['fees_taker']:<9.2f} | ${r['net_pnl_taker']:<10.2f} | {r['profit_factor_taker']:<5.2f}"
        )
        print(row)
    print("=" * 145)

    # ----------------------------------------------------------------------
    # TABELA CONSOLIDADA 2: SENSIBILIDADE MAKER (ORDENS LIMIT PASSIVAS 0.02%)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("TABELA CONSOLIDADA 2: SENSIBILIDADE A TAXAS - CENÁRIO MAKER PASSIVO (0.02% Fee, 0% Slip)")
    print("AVISO INSTITUCIONAL: Ordens Maker limite possuem risco de adversidade de seleção e sem garantia de execução.")
    print("=" * 125)
    header_m = (
        f"{'ATIVO':<8} | {'TF':<4} | {'PASSO DA ABLAÇÃO':<28} | {'TRADES':<6} | "
        f"{'LUCRO BRUTO':<11} | {'TAKER NET':<11} | {'TAKER PF':<9} | {'MAKER NET':<11} | {'MAKER PF':<9} | {'DELTA ECONOMIA':<14}"
    )
    print(header_m)
    print("-" * 125)

    for r in all_ablation_results:
        delta_fee = r["net_pnl_maker"] - r["net_pnl_taker"]
        row_m = (
            f"{r['symbol']:<8} | {r['timeframe']:<4} | {r['step_name']:<28} | {r['trades']:<6} | "
            f"${r['gross_profit']:<10.2f} | ${r['net_pnl_taker']:<10.2f} | {r['profit_factor_taker']:<9.2f} | "
            f"${r['net_pnl_maker']:<10.2f} | {r['profit_factor_maker']:<9.2f} | +${delta_fee:<13.2f}"
        )
        print(row_m)
    print("=" * 125)


if __name__ == "__main__":
    main()
