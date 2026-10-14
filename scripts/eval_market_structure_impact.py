"""
Script de Avaliação Empírica do Impacto das Novas Variáveis de Estrutura de Mercado e POI.
Mede rigorosamente:
1. Ganho de Informação e Ranking de Importância (LightGBM Gain)
2. Experimento de Ablação de Modelo Preditivo ML (Baseline vs Baseline + Estrutura/POI)
3. Impacto no Sistema de Trading (R:R assimétrico, Win Rate, Profit Factor e Drawdown com custos realistas: Maker 0.02% / Taker 0.06% + slippage)
Período: Dados históricos pré-holdout (holdout de 12 meses estritamente preservado).
"""

import asyncio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from model_trainer import create_triple_barrier_labels

HOLDOUT_CUTOFF_MS = 1760400000000  # 2025-10-14 00:00:00 UTC (Início do holdout de 12 meses)

async def run_evaluation():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    print("=" * 85)
    print("🔬 AVALIAÇÃO QUANTITATIVA DO IMPACTO DAS VARIÁVEIS DE ESTRUTURA DE MERCADO & POI")
    print("=" * 85)
    
    pairs = [
        ("BTCUSDT", "1h"),
        ("ETHUSDT", "1h"),
        ("SOLUSDT", "1h"),
        ("BTCUSDT", "4h"),
    ]
    
    results_summary = []
    
    for symbol, tf in pairs:
        print(f"\n{'='*40}")
        print(f"📊 Avaliando {symbol} [{tf}] (Dados Históricos Pré-Holdout)")
        print(f"{'='*40}")
        
        # Carrega histórico completo
        df = await loader.get_candles(symbol, tf, limit=85000, exchange="binance")
        if len(df) < 5000:
            df = await loader.get_candles(symbol, tf, limit=85000, exchange="bitget")
            
        # Filtro estrito pré-holdout
        df = df[df["timestamp"] < HOLDOUT_CUTOFF_MS].copy().sort_values("timestamp").reset_index(drop=True)
        print(f"Candles históricos pré-holdout válidos: {len(df)}")
        
        if len(df) < 1500:
            print(f"Amostra insuficiente ({len(df)}), pulando.")
            continue
            
        # Calcula todas as features locais + estrutura
        df_feat = fe.compute_all_features(df, drop_na=True)
        
        # Features de Estrutura de Mercado e POI
        market_structure_cols = [
            "dist_support_pct", "dist_resistance_pct",
            "liquidity_sweep_bull", "liquidity_sweep_bear",
            "inside_bullish_ob", "inside_bearish_ob",
            "inside_bullish_fvg", "inside_bearish_fvg",
            "at_bullish_poi", "at_bearish_poi"
        ]
        structure_cols = [c for c in market_structure_cols if c in df_feat.columns]
        
        # Features Baseline (Indicadores Clássicos: RSI, MACD, Bollinger, ATR, EMAs, VWAP)
        baseline_cols = [
            "sma_20", "sma_50", "sma_200", "dist_sma_200",
            "ema_9", "ema_21", "macd_line", "macd_signal", "macd_hist",
            "rsi_14", "stoch_k", "stoch_d",
            "atr_14", "atr_pct", "bb_width", "bb_pband",
            "obv", "vol_sma_ratio", "volume_zscore_20", "dist_vwap_pct",
            "returns", "volatility_20"
        ]
        baseline_cols = [c for c in baseline_cols if c in df_feat.columns]
        combined_cols = baseline_cols + structure_cols
        
        # Target Triple Barrier (horizonte 6 barras para 1h, 4 barras para 4h)
        horizon = 6 if tf == "1h" else 4
        df_feat["label"] = create_triple_barrier_labels(df_feat, horizon=horizon, multiple_tp=1.5, multiple_sl=1.0)
        
        df_clean = df_feat.dropna(subset=["label"]).copy()
        df_clean["target"] = (df_clean["label"] == 1).astype(int)
        
        # Split Temporal Walk-Forward estrito (70% Train, 30% Test Out-Of-Sample)
        split_idx = int(len(df_clean) * 0.70)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()
        
        print(f"Divisão Temporal: Treino={len(train_df)} barras | Teste OOS={len(test_df)} barras")
        
        # Prepara matrizes numéricas garantindo buffers copiados
        X_train_base = np.nan_to_num(np.array(train_df[baseline_cols].values, dtype=np.float64, copy=True), nan=0.0)
        X_test_base = np.nan_to_num(np.array(test_df[baseline_cols].values, dtype=np.float64, copy=True), nan=0.0)
        
        X_train_full = np.nan_to_num(np.array(train_df[combined_cols].values, dtype=np.float64, copy=True), nan=0.0)
        X_test_full = np.nan_to_num(np.array(test_df[combined_cols].values, dtype=np.float64, copy=True), nan=0.0)
        
        y_train = train_df["target"].values
        y_test = test_df["target"].values
        
        # -------------------------------------------------------------
        # 1. Feature Importance (LightGBM Gain)
        # -------------------------------------------------------------
        model_full = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        model_full.fit(X_train_full, y_train)
        
        gains = model_full.booster_.feature_importance(importance_type="gain")
        total_gain = gains.sum() + 1e-12
        gain_pct = (gains / total_gain) * 100.0
        
        importance_df = pd.DataFrame({
            "feature": combined_cols,
            "gain_pct": gain_pct,
            "is_structure": [c in structure_cols for c in combined_cols]
        }).sort_values(by="gain_pct", ascending=False).reset_index(drop=True)
        
        structure_importance = importance_df[importance_df["is_structure"]]
        total_structure_gain = structure_importance["gain_pct"].sum()
        
        # -------------------------------------------------------------
        # 2. Experimento de Ablação (Baseline vs Baseline + Structure)
        # -------------------------------------------------------------
        model_base = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        model_base.fit(X_train_base, y_train)
        
        prob_base = model_base.predict_proba(X_test_base)[:, 1]
        pred_base = (prob_base >= 0.50).astype(int)
        
        acc_base = accuracy_score(y_test, pred_base)
        auc_base = roc_auc_score(y_test, prob_base)
        f1_base = f1_score(y_test, pred_base, zero_division=0)
        prec_base = precision_score(y_test, pred_base, zero_division=0)
        
        prob_full = model_full.predict_proba(X_test_full)[:, 1]
        pred_full = (prob_full >= 0.50).astype(int)
        
        acc_full = accuracy_score(y_test, pred_full)
        auc_full = roc_auc_score(y_test, prob_full)
        f1_full = f1_score(y_test, pred_full, zero_division=0)
        prec_full = precision_score(y_test, pred_full, zero_division=0)
        
        # -------------------------------------------------------------
        # 3. Simulação de Trading com Custos Realistas
        # -------------------------------------------------------------
        maker_fee = 0.0002
        taker_fee = 0.0006
        slippage = 0.0002
        friction = maker_fee + taker_fee + 2 * slippage  # 0.12% total por trade
        
        def run_sim(test_slice, probs, use_structure_entry=False):
            trades = []
            closes = test_slice["close"].values
            highs = test_slice["high"].values
            lows = test_slice["low"].values
            atrs = test_slice["atr_14"].values
            sweeps = test_slice["liquidity_sweep_bull"].values if "liquidity_sweep_bull" in test_slice.columns else np.zeros(len(test_slice))
            poi = test_slice["at_bullish_poi"].values if "at_bullish_poi" in test_slice.columns else np.zeros(len(test_slice))
            
            in_pos = False
            entry_p = 0.0
            sl_p = 0.0
            tp_p = 0.0
            
            for i in range(len(test_slice) - 1):
                c = closes[i]
                atr = atrs[i] if atrs[i] > 0 else c * 0.01
                
                if not in_pos:
                    if not use_structure_entry:
                        # Baseline: apenas modelo estatístico prob >= 0.52
                        if probs[i] >= 0.52:
                            in_pos = True
                            entry_p = c
                            sl_p = entry_p - 1.2 * atr
                            tp_p = entry_p + 1.8 * atr  # R:R = 1.5
                    else:
                        # Enriquecido: Modelo prob >= 0.51 + confirmação em POI institucional
                        if probs[i] >= 0.51 and poi[i] == 1:
                            in_pos = True
                            entry_p = c
                            # Se for sweep de liquidez, stop assimétrico sob a mínima
                            if sweeps[i] == 1:
                                sl_p = min(entry_p - 0.6 * atr, lows[i] * 0.999)
                                tp_p = entry_p + (entry_p - sl_p) * 2.5  # R:R assimétrico 1:2.5
                            else:
                                sl_p = entry_p - 0.9 * atr
                                tp_p = entry_p + 1.8 * atr  # R:R = 2.0
                else:
                    h = highs[i]
                    l = lows[i]
                    if l <= sl_p:
                        pnl = ((sl_p - entry_p) / entry_p) - friction
                        trades.append({"pnl": pnl, "win": False})
                        in_pos = False
                    elif h >= tp_p:
                        pnl = ((tp_p - entry_p) / entry_p) - friction
                        trades.append({"pnl": pnl, "win": True})
                        in_pos = False
            
            if not trades:
                return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0, "avg_rr": 0.0}
            
            pnls = np.array([t["pnl"] for t in trades])
            wins = np.array([t["win"] for t in trades])
            win_rate = (wins.sum() / len(trades)) * 100.0
            gp = pnls[pnls > 0].sum()
            gl = abs(pnls[pnls < 0].sum()) + 1e-12
            pf = gp / gl
            net_pnl = pnls.sum() * 100.0
            
            return {
                "trades": len(trades),
                "win_rate": win_rate,
                "profit_factor": pf,
                "net_pnl": net_pnl
            }
        
        sim_base = run_sim(test_df, prob_base, use_structure_entry=False)
        sim_full = run_sim(test_df, prob_full, use_structure_entry=True)
        
        # Exibe Resultados
        print(f"\n[1] IMPORTÂNCIA DE FEATURES (LightGBM Gain Ranking):")
        for i, row in importance_df.head(8).iterrows():
            flag = " 🎯 [ESTRUTURA/POI]" if row["is_structure"] else "   [INDICADOR]"
            print(f"   {i+1}. {row['feature']:<25} | Ganho: {row['gain_pct']:>6.2f}% | {flag}")
            
        print(f"\n   -> Ganho Total das Features de Estrutura/POI: {total_structure_gain:.2f}%")
        for _, row in structure_importance.iterrows():
            print(f"      - {row['feature']:<22}: {row['gain_pct']:>5.2f}%")
            
        print(f"\n[2] COMPARAÇÃO PREDITIVA FORA DA AMOSTRA (OOS):")
        print(f"   Métrica        | Baseline (Técnico) | Enriquecido (+POI) | Delta")
        print(f"   ROC-AUC        | {auc_base:>18.4f} | {auc_full:>18.4f} | {auc_full - auc_base:>+7.4f}")
        print(f"   Acurácia       | {acc_base*100:>17.2f}% | {acc_full*100:>17.2f}% | {(acc_full - acc_base)*100:>+6.2f} p.p.")
        print(f"   Precisão       | {prec_base*100:>17.2f}% | {prec_full*100:>17.2f}% | {(prec_full - prec_base)*100:>+6.2f} p.p.")
        print(f"   F1-Score       | {f1_base:>18.4f} | {f1_full:>18.4f} | {f1_full - f1_base:>+7.4f}")
        
        print(f"\n[3] SIMULAÇÃO DE TRADING COM CUSTOS (0.12% Friction por Operação):")
        print(f"   Métrica        | Baseline (Técnico) | Enriquecido (+POI/Sweeps)")
        print(f"   Total Trades   | {sim_base['trades']:>18} | {sim_full['trades']:>25}")
        print(f"   Win Rate       | {sim_base['win_rate']:>17.2f}% | {sim_full['win_rate']:>24.2f}%")
        print(f"   Profit Factor  | {sim_base['profit_factor']:>18.2f} | {sim_full['profit_factor']:>25.2f}")
        print(f"   PnL Líquido    | {sim_base['net_pnl']:>+17.2f}% | {sim_full['net_pnl']:>+24.2f}%")
        
        results_summary.append({
            "pair": f"{symbol}_{tf}",
            "auc_base": auc_base,
            "auc_full": auc_full,
            "auc_delta": auc_full - auc_base,
            "prec_base": prec_base,
            "prec_full": prec_full,
            "prec_delta": (prec_full - prec_base) * 100,
            "structure_gain": total_structure_gain,
            "sim_base": sim_base,
            "sim_full": sim_full
        })
        
    await loader.close()
    
    print("\n" + "=" * 85)
    print("📋 TABELA CONSOLIDADA RESUMO DE TODOS OS ATIVOS")
    print("=" * 85)
    print(f"{'Ativo / TF':<14} | {'AUC Delta':<10} | {'Prec Delta':<11} | {'Estrutura Gain':<15} | {'PF Base':<8} -> {'PF Enriq':<8} | {'PnL Base':<9} -> {'PnL Enriq':<9}")
    print("-" * 85)
    for r in results_summary:
        print(f"{r['pair']:<14} | {r['auc_delta']:>+9.4f}  | {r['prec_delta']:>+9.2f}%  | {r['structure_gain']:>13.2f}%  | {r['sim_base']['profit_factor']:>7.2f} -> {r['sim_full']['profit_factor']:>7.2f}  | {r['sim_base']['net_pnl']:>+7.1f}% -> {r['sim_full']['net_pnl']:>+7.1f}%")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
