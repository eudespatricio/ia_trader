"""
Script Mestre de Re-Auditoria e Avaliação Reprodutível da Fase 3: ML + Backtesting Institucional.
Executa avaliação multi-timeframe e multi-ativo (BTCUSDT, ETHUSDT, SOLUSDT em 15m, 1h, 4h, 1d)
utilizando dados 100% reais da Bitget persistidos no cache SQLite.

Princípios Inflexíveis de Auditoria:
1. Backtest ESTRITAMENTE OUT-OF-SAMPLE: Avaliação estrita fold-a-fold com PurgedTimeSeriesSplit
   e embargo temporal de N velas (zero contaminação cruzada).
2. Política Anti-Overtrading:
   - Dimensionamento de risco fixo institucional (1% do capital arriscado no SL).
   - Sem reversão compulsória de sinais (allow_reversal=False).
   - Saída por tempo máximo de holding (max_holding_bars = horizon -> HORIZON_TIMEOUT).
   - Filtro de alta convicção (confidence_threshold = 0.60 e 0.70).
3. Tratamento Intra-Barra Conservador: a própria vela de entrada (i+1) checa SL/TP,
   e em colisões intra-barra o Stop Loss tem precedência absoluta.
4. Custos Reais Bitget: Taker 0.06%, Slippage 0.02%, Funding Rate 8h 0.01%, Alavancagem 1.0x.
5. Decomposição Transparente: Lucro Bruto vs Custos Totais vs Lucro Líquido, motivos de saída,
   duração média e trades/dia.
6. Métricas de Classificação OOS: Global Acc, Emitted Acc, Coverage %, F1-Macro, F1-Weighted e Precisão por Classe.
7. Significância Estatística: Intervalo de Confiança de 95% (Bootstrap) e Teste Binomial (p-valor vs 50%).
8. Honestidade Absoluta: Reporta os números reais exatamente como calculados.
"""

from __future__ import annotations

import json
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

# Adiciona o diretório raiz ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from backtesting import BacktestEngine, BacktestResult, TradeRecord
from data.feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer, PurgedTimeSeriesSplit

# =====================================================================
# CONFIGURAÇÃO A PRIORI DE PARÂMETROS POR TIMEFRAME
# =====================================================================
TIMEFRAME_CONFIGS = {
    "15m": {
        "horizon": 12,           # 12 velas de 15m = 3 horas
        "multiple_tp": 1.5,
        "multiple_sl": 1.0,
        "bar_hours": 0.25,
        "rationale": "Captura expansão intradiária de 3h com RR 1.5:1",
    },
    "1h": {
        "horizon": 6,            # 6 velas de 1h = 6 horas (uma sessão de mercado)
        "multiple_tp": 1.5,
        "multiple_sl": 1.0,
        "bar_hours": 1.0,
        "rationale": "Captura tendência de sessão intradiária de 6h com RR 1.5:1",
    },
    "4h": {
        "horizon": 6,            # 6 velas de 4h = 24 horas (1 dia completo)
        "multiple_tp": 2.0,
        "multiple_sl": 1.2,
        "bar_hours": 4.0,
        "rationale": "Swing trading diário de 24h buscando 2x ATR com stop em 1.2x ATR",
    },
    "1d": {
        "horizon": 5,            # 5 velas de 1d = 5 dias (1 semana útil)
        "multiple_tp": 2.5,
        "multiple_sl": 1.5,
        "bar_hours": 24.0,
        "rationale": "Position trading semanal buscando 2.5x ATR com stop em 1.5x ATR",
    },
}

EVALUATION_TARGETS = [
    ("BTCUSDT", "15m"),
    ("BTCUSDT", "1h"),
    ("BTCUSDT", "4h"),
    ("BTCUSDT", "1d"),
    ("ETHUSDT", "1h"),
    ("ETHUSDT", "4h"),
    ("SOLUSDT", "1h"),
    ("SOLUSDT", "4h"),
]


def load_candles_from_sqlite(symbol: str, timeframe: str) -> pd.DataFrame:
    """Carrega todo o histórico real do SQLite garantindo ordenação cronológica estrita."""
    db_path = "/home/eudespatricio/ia_trader/datasets/market_cache.db"
    conn = sqlite3.connect(db_path)
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
    """Calcula o Intervalo de Confiança de 95% para o Win Rate via Bootstrap."""
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


def evaluate_target(symbol: str, timeframe: str, confidence_thresh: float = 0.60) -> Optional[Dict[str, Any]]:
    print(f"\n{'='*95}")
    print(f"AVALIANDO: {symbol} [{timeframe}] | Limiar Convicção={confidence_thresh:.2f} | DADOS REAIS BITGET")
    print(f"{'='*95}")

    cfg = TIMEFRAME_CONFIGS[timeframe]
    horizon = cfg["horizon"]
    multiple_tp = cfg["multiple_tp"]
    multiple_sl = cfg["multiple_sl"]
    bar_hours = cfg["bar_hours"]

    # 1. Carrega velas reais
    raw_df = load_candles_from_sqlite(symbol, timeframe)
    total_candles = len(raw_df)
    if total_candles < 300:
        print(f"Dados insuficientes para {symbol} [{timeframe}]: {total_candles} velas. Pulando.")
        return None

    start_dt = datetime.fromtimestamp(raw_df["timestamp"].iloc[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt = datetime.fromtimestamp(raw_df["timestamp"].iloc[-1] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Histórico carregado: {total_candles} velas ({start_dt} a {end_dt})")

    # 2. Engenharia de Features Local Vetorizada
    t0 = time.time()
    features_df = FeatureEngineer.compute_all_features(raw_df, drop_na=True)
    t_feat = time.time() - t0
    print(f"Features calculadas em {t_feat:.2f}s: {features_df.shape[0]} amostras pós-aquecimento.")

    # 3. Preparação do Dataset e Rotulagem Triple-Barrier
    trainer = ModelTrainer(
        model_type="lgbm",
        horizon=horizon,
        multiple_tp=multiple_tp,
        multiple_sl=multiple_sl,
        confidence_threshold=confidence_thresh,
        n_jobs=8,
    )
    X, y = trainer.prepare_dataset(features_df)

    # 4. Divisão Walk-Forward Purgada com Embargo (4 splits)
    n_splits = 4
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, horizon_embargo=horizon)
    splits = splitter.split(X, y)
    if not splits:
        print(f"Falha ao gerar splits para {symbol} [{timeframe}].")
        return None

    # Arrays para métricas de classificação OOS
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_y_proba: List[np.ndarray] = []
    all_test_indices: List[int] = []

    # Listas de trades do backtest (estritamente fold-a-fold)
    strat_trades: List[TradeRecord] = []
    bh_trades: List[TradeRecord] = []
    rand_trades: List[TradeRecord] = []

    # Configuração dos motores de backtest institucional
    strat_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,        # Taker Bitget: 0.06%
        slippage=0.0002,        # Slippage: 0.02%
        funding_rate_8h=0.0001, # Funding Rate 8h
        leverage=1.0,           # 1.0x não alavancado
        tp_atr_multiple=multiple_tp,
        sl_atr_multiple=multiple_sl,
        trailing_stop=False,
        risk_per_trade=0.01,    # Risco fixo de 1% do capital
        allow_reversal=False,   # Sem reversão forçada por sinal contrário
        max_holding_bars=horizon, # Saída forçada ao atingir o horizonte
    )

    bh_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,
        slippage=0.0002,
        funding_rate_8h=0.0001,
        leverage=1.0,
        tp_atr_multiple=999.0,
        sl_atr_multiple=999.0,
        risk_per_trade=None,
        allow_reversal=False,
    )

    rand_engine = BacktestEngine(
        initial_capital=10000.0,
        fee_rate=0.0006,
        slippage=0.0002,
        funding_rate_8h=0.0001,
        leverage=1.0,
        tp_atr_multiple=multiple_tp,
        sl_atr_multiple=multiple_sl,
        risk_per_trade=0.01,
        allow_reversal=False,
        max_holding_bars=horizon,
    )

    total_test_bars = 0
    np.random.seed(42)

    for fold, (train_idx, test_idx) in enumerate(splits):
        # Assert matemático anti-vazamento
        assert len(set(train_idx).intersection(set(test_idx))) == 0, f"Vazamento no fold {fold+1}!"
        assert min(test_idx) >= max(train_idx) + horizon, f"Gap violado no fold {fold+1}!"

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = trainer._build_estimator(random_state=42 + fold)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.append(y_proba)
        all_test_indices.extend(test_idx.tolist())

        # Geração de sinais operacionais para o fold
        classes = getattr(model, "classes_", np.array([0, 1, 2]))
        buy_col = np.where(classes == 1)[0]
        sell_col = np.where(classes == 2)[0]

        p_buy = y_proba[:, buy_col[0]] if len(buy_col) > 0 else np.zeros(len(X_test))
        p_sell = y_proba[:, sell_col[0]] if len(sell_col) > 0 else np.zeros(len(X_test))

        fold_signals = np.zeros(len(X_test), dtype=int)
        for i in range(len(X_test)):
            if p_buy[i] >= confidence_thresh and p_buy[i] > p_sell[i]:
                fold_signals[i] = 1  # BUY
            elif p_sell[i] >= confidence_thresh and p_sell[i] > p_buy[i]:
                fold_signals[i] = 2  # SELL

        # Execução Backtest Fold OOS
        df_fold = features_df.iloc[test_idx].copy().reset_index(drop=True)
        total_test_bars += len(df_fold)

        res_fold_strat = strat_engine.run(df_fold, fold_signals)
        strat_trades.extend(res_fold_strat.trades)

        # Baseline Buy & Hold Fold
        bh_signals = np.zeros(len(df_fold), dtype=int)
        bh_signals[0] = 1
        res_fold_bh = bh_engine.run(df_fold, bh_signals)
        bh_trades.extend(res_fold_bh.trades)

        # Baseline Sinais Aleatórios Fold
        rand_signals = np.random.choice([0, 1, 2], size=len(df_fold), p=[0.7, 0.15, 0.15])
        res_fold_rand = rand_engine.run(df_fold, rand_signals)
        rand_trades.extend(res_fold_rand.trades)

    # 5. Consolidação de Métricas de Classificação OOS
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_proba_arr = np.vstack(all_y_proba)

    classes_all = np.array([0, 1, 2])
    global_acc = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    precisions = precision_score(y_true_arr, y_pred_arr, average=None, labels=[0, 1, 2], zero_division=0)
    prec_hold, prec_buy, prec_sell = float(precisions[0]), float(precisions[1]), float(precisions[2])

    buy_col = np.where(classes_all == 1)[0]
    sell_col = np.where(classes_all == 2)[0]
    p_buy_all = y_proba_arr[:, buy_col[0]] if len(buy_col) > 0 else np.zeros(len(y_true_arr))
    p_sell_all = y_proba_arr[:, sell_col[0]] if len(sell_col) > 0 else np.zeros(len(y_true_arr))

    active_mask = (p_buy_all >= confidence_thresh) | (p_sell_all >= confidence_thresh)
    total_active = int(np.sum(active_mask))
    coverage_pct = float(total_active / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0

    if total_active > 0:
        active_true = y_true_arr[active_mask]
        active_emitted = np.where(p_buy_all[active_mask] >= p_sell_all[active_mask], 1, 2)
        emitted_acc = float(np.mean(active_true == active_emitted))
    else:
        emitted_acc = 0.0

    # 6. Consolidação e Decomposição de Performance do Backtest
    trades_count = len(strat_trades)
    winning_trades = sum(1 for t in strat_trades if t.pnl_net > 0)
    losing_trades = sum(1 for t in strat_trades if t.pnl_net <= 0)
    win_rate = (winning_trades / trades_count) if trades_count > 0 else 0.0

    gross_profit = float(sum(t.pnl_gross for t in strat_trades))
    total_fees = float(sum(t.fee_total + t.funding_total for t in strat_trades))
    net_profit = float(sum(t.pnl_net for t in strat_trades))
    net_profit_pct = (net_profit / 10000.0) * 100.0

    pos_profits = sum(t.pnl_net for t in strat_trades if t.pnl_net > 0)
    neg_losses = abs(sum(t.pnl_net for t in strat_trades if t.pnl_net < 0))
    profit_factor = (pos_profits / neg_losses) if neg_losses > 0 else (99.0 if pos_profits > 0 else 0.0)

    # Decomposição dos Motivos de Saída
    exit_reasons: Dict[str, int] = {}
    durations_bars: List[int] = []
    for t in strat_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        durations_bars.append(max(1, t.exit_idx - t.entry_idx))

    avg_duration_bars = float(np.mean(durations_bars)) if durations_bars else 0.0
    avg_duration_hours = avg_duration_bars * bar_hours

    total_test_hours = total_test_bars * bar_hours
    total_test_days = max(1.0, total_test_hours / 24.0)
    trades_per_day = trades_count / total_test_days

    # Baselines
    bh_net_profit = float(sum(t.pnl_net for t in bh_trades))
    bh_net_profit_pct = (bh_net_profit / 10000.0) * 100.0

    rand_count = len(rand_trades)
    rand_wins = sum(1 for t in rand_trades if t.pnl_net > 0)
    rand_wr = (rand_wins / rand_count) if rand_count > 0 else 0.0
    rand_net_profit = float(sum(t.pnl_net for t in rand_trades))
    rand_net_profit_pct = (rand_net_profit / 10000.0) * 100.0

    # 7. Testes Estatísticos de Significância
    ci_low, ci_high = calculate_bootstrap_ci(strat_trades)
    if trades_count > 0:
        b_test = binomtest(k=winning_trades, n=trades_count, p=0.5, alternative="greater")
        p_val = float(b_test.pvalue)
    else:
        p_val = 1.0

    stat_sig = "SIM (p<0.05)" if p_val < 0.05 else ("NÃO (p>=0.05)" if trades_count >= 30 else "INCONCL. (N<30)")

    # Exibição Detalhada no Terminal
    print(f"\n[CLASSIFICAÇÃO OOS - {symbol} {timeframe}]:")
    print(f"  Amostras OOS: {len(y_true_arr)} | Cobertura: {coverage_pct:.1%} ({total_active} sinais)")
    print(f"  Acurácia Global: {global_acc:.2%} | Acurácia Sinais Emitidos: {emitted_acc:.2%}")
    print(f"  F1-Macro: {f1_mac:.4f} | F1-Weighted: {f1_weighted:.4f}")
    print(f"  Precisões: BUY={prec_buy:.2%} | SELL={prec_sell:.2%} | HOLD={prec_hold:.2%}")

    print(f"\n[BACKTEST & DECOMPOSIÇÃO FINANCEIRA]:")
    print(f"  Trades Executados: {trades_count} ({trades_per_day:.2f} trades/dia | Duração Média: {avg_duration_bars:.1f} barras / {avg_duration_hours:.1f}h)")
    print(f"  Win Rate Líquido: {win_rate:.2%} (IC 95%: [{ci_low:.1%}, {ci_high:.1%}], p-val={p_val:.4f} -> {stat_sig})")
    print(f"  Lucro Bruto: ${gross_profit:,.2f} | Custos Totais: ${total_fees:,.2f} | Lucro Líquido: ${net_profit:,.2f} ({net_profit_pct:.2f}%)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Distribuição de Saídas: {json.dumps(exit_reasons)}")
    print(f"  Baselines -> Buy & Hold: {bh_net_profit_pct:.2f}% | Sinais Aleatórios: WR={rand_wr:.2%}, PnL={rand_net_profit_pct:.2f}%")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "confidence_thresh": confidence_thresh,
        "total_candles": total_candles,
        "oos_samples": len(y_true_arr),
        # Classificação
        "global_acc": global_acc,
        "emitted_acc": emitted_acc,
        "coverage_pct": coverage_pct,
        "f1_macro": f1_mac,
        "f1_weighted": f1_weighted,
        "prec_buy": prec_buy,
        "prec_sell": prec_sell,
        "prec_hold": prec_hold,
        # Backtest & Decomposição
        "trades_count": trades_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "stat_sig": stat_sig,
        "gross_profit": gross_profit,
        "total_fees": total_fees,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "profit_factor": profit_factor,
        "avg_duration_bars": avg_duration_bars,
        "avg_duration_hours": avg_duration_hours,
        "trades_per_day": trades_per_day,
        "exit_reasons": exit_reasons,
        # Baselines
        "bh_net_profit_pct": bh_net_profit_pct,
        "rand_wr": rand_wr,
        "rand_net_profit_pct": rand_net_profit_pct,
    }


def main():
    print("\n" + "=" * 110)
    print("INICIANDO RE-AUDITORIA INSTITUCIONAL FASE 3: MACHINE LEARNING + BACKTESTING MULTI-TIMEFRAME")
    print("=" * 110)

    for thresh in [0.60, 0.70]:
        print(f"\n\n##############################################################################################")
        print(f"BATERIA DE AVALIAÇÃO: LIMIAR DE CONVICÇÃO = {thresh:.2f} (POLÍTICA ANTI-OVERTRADING 1% RISCO)")
        print(f"##############################################################################################")

        results = []
        for sym, tf in EVALUATION_TARGETS:
            res = evaluate_target(sym, tf, confidence_thresh=thresh)
            if res:
                results.append(res)

        # -------------------------------------------------------------
        # TABELA 1: MÉTRICAS DE CLASSIFICAÇÃO OUT-OF-SAMPLE
        # -------------------------------------------------------------
        print("\n" + "=" * 115)
        print(f"TABELA 1: MÉTRICAS DE CLASSIFICAÇÃO OUT-OF-SAMPLE (Limiar = {thresh:.2f})")
        print("=" * 115)
        print(f"{'ATIVO':<8} | {'TF':<4} | {'AMOSTRAS':<8} | {'COBERTURA':<9} | {'ACC GLOB':<8} | {'ACC EMIT':<8} | {'F1-MACRO':<8} | {'F1-POND':<8} | {'PREC BUY':<8} | {'PREC SELL':<9} | {'PREC HOLD':<9}")
        print("-" * 115)
        for r in results:
            print(
                f"{r['symbol']:<8} | {r['timeframe']:<4} | {r['oos_samples']:<8} | {r['coverage_pct']:<9.1%} | "
                f"{r['global_acc']:<8.1%} | {r['emitted_acc']:<8.1%} | {r['f1_macro']:<8.4f} | {r['f1_weighted']:<8.4f} | "
                f"{r['prec_buy']:<8.1%} | {r['prec_sell']:<9.1%} | {r['prec_hold']:<9.1%}"
            )
        print("=" * 115)

        # -------------------------------------------------------------
        # TABELA 2: PERFORMANCE FINANCEIRA & DECOMPOSIÇÃO INSTITUCIONAL
        # -------------------------------------------------------------
        print("\n" + "=" * 135)
        print(f"TABELA 2: PERFORMANCE FINANCEIRA & DECOMPOSIÇÃO INSTITUCIONAL (Risco Fixo 1%, Sem Reversão)")
        print("=" * 135)
        print(f"{'ATIVO':<8} | {'TF':<4} | {'TRADES':<6} | {'TR/DIA':<6} | {'DUR(h)':<6} | {'WIN RATE':<9} | {'IC 95% WR':<15} | {'P-VALOR':<8} | {'LUCRO BRUTO':<11} | {'TAXAS':<10} | {'LUCRO LÍQ':<11} | {'PF':<5} | {'B&H%':<8} | {'SIG?':<10}")
        print("-" * 135)
        for r in results:
            ci_str = f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
            print(
                f"{r['symbol']:<8} | {r['timeframe']:<4} | {r['trades_count']:<6} | {r['trades_per_day']:<6.2f} | {r['avg_duration_hours']:<6.1f} | "
                f"{r['win_rate']:<9.1%} | {ci_str:<15} | {r['p_value']:<8.4f} | "
                f"${r['gross_profit']:<10.2f} | ${r['total_fees']:<9.2f} | ${r['net_profit']:<10.2f} | "
                f"{r['profit_factor']:<5.2f} | {r['bh_net_profit_pct']:<7.1f}% | {r['stat_sig']:<10}"
            )
        print("=" * 135)

        # -------------------------------------------------------------
        # TABELA 3: DECOMPOSIÇÃO DE MOTIVOS DE SAÍDA
        # -------------------------------------------------------------
        print("\n" + "=" * 90)
        print(f"TABELA 3: DISTRIBUIÇÃO DOS MOTIVOS DE SAÍDA DOS TRADES (Limiar = {thresh:.2f})")
        print("=" * 90)
        print(f"{'ATIVO':<8} | {'TF':<4} | {'TRADES':<6} | {'TP':<6} | {'SL':<6} | {'TIMEOUT':<8} | {'END_DATA':<8}")
        print("-" * 90)
        for r in results:
            ex = r["exit_reasons"]
            tp_c = ex.get("TP", 0)
            sl_c = ex.get("SL", 0)
            to_c = ex.get("HORIZON_TIMEOUT", 0)
            end_c = ex.get("END_OF_DATA", 0)
            print(f"{r['symbol']:<8} | {r['timeframe']:<4} | {r['trades_count']:<6} | {tp_c:<6} | {sl_c:<6} | {to_c:<8} | {end_c:<8}")
        print("=" * 90)


if __name__ == "__main__":
    main()
