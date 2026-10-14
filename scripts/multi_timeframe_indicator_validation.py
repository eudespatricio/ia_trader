"""
Multi-Timeframe Indicator Validation Script
Valida cientificamente os melhores indicadores em múltiplos timeframes (15m, 1h, 4h, 1d)
usando Information Coefficient (IC), Information Ratio (IC_IR), Informação Mútua (MI)
e Ganho de Informação de Árvores de Decisão (LightGBM).
"""

import asyncio
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from model_trainer import create_triple_barrier_labels

async def validate_timeframe(loader, fe, symbol="BTCUSDT", tf="1h", exchange="binance", max_candles=30000):
    print(f"\n=================================================================")
    print(f"📊 AVALIANDO TIMEFRAME: {tf} | Símbolo: {symbol} | Exchange: {exchange}")
    print(f"=================================================================")
    
    # 1. Carrega velas
    df = await loader.get_candles(symbol, tf, limit=max_candles, exchange=exchange)
    if len(df) < 500 and exchange != "bitget":
        df = await loader.get_candles(symbol, tf, limit=max_candles, exchange="bitget")
        
    print(f"Total de candles carregados: {len(df):,}")
    if len(df) < 300:
        print(f"⚠️ Amostra insuficiente ({len(df)} candles) para {tf}. Pulando.")
        return None
        
    # 2. Computa features
    df_feat = fe.compute_all_features(df, drop_na=True)
    feature_cols = [c for c in fe.get_feature_names() if c in df_feat.columns]
    
    # 3. Calcula forward returns para múltiplos horizontes
    # h1 (1 barra à frente), h4 (4 barras), h12 (12 barras)
    df_feat["fwd_ret_1"] = df_feat["close"].shift(-1) / df_feat["close"] - 1.0
    df_feat["fwd_ret_4"] = df_feat["close"].shift(-4) / df_feat["close"] - 1.0
    df_feat["fwd_ret_12"] = df_feat["close"].shift(-12) / df_feat["close"] - 1.0
    
    # Labels Triple-Barrier para classificação direcional (horizonte de 6 barras)
    df_feat["tb_label"] = create_triple_barrier_labels(df_feat, horizon=6, multiple_tp=1.5, multiple_sl=1.0)
    
    # Limpa linhas com NaN nos horizontes
    df_clean = df_feat.dropna(subset=["fwd_ret_1", "fwd_ret_4", "fwd_ret_12", "tb_label"]).copy()
    print(f"Candles válidos para análise pós-limpeza: {len(df_clean):,}")
    
    # 4. Cálculo do Information Coefficient (IC) via Spearman Correlation
    ic_results = []
    
    for feat in feature_cols:
        series_x = df_clean[feat].values
        # Checa variância não-nula
        if np.std(series_x) < 1e-12:
            continue
            
        corr_1, p_1 = spearmanr(series_x, df_clean["fwd_ret_1"])
        corr_4, p_4 = spearmanr(series_x, df_clean["fwd_ret_4"])
        corr_12, p_12 = spearmanr(series_x, df_clean["fwd_ret_12"])
        
        # Rolling IC em janelas de 200 barras para calcular IC Information Ratio (IC_IR = mean / std)
        rolling_ics = []
        step = max(50, len(df_clean) // 30)
        window_size = min(300, len(df_clean) // 3)
        for i in range(0, len(df_clean) - window_size, step):
            sub_x = series_x[i:i+window_size]
            sub_y = df_clean["fwd_ret_4"].iloc[i:i+window_size].values
            if np.std(sub_x) > 1e-12 and np.std(sub_y) > 1e-12:
                r, _ = spearmanr(sub_x, sub_y)
                if not np.isnan(r):
                    rolling_ics.append(r)
                    
        ic_mean = np.mean(rolling_ics) if rolling_ics else corr_4
        ic_std = np.std(rolling_ics) if rolling_ics and np.std(rolling_ics) > 1e-6 else 1.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        ic_results.append({
            "feature": feat,
            "ic_1bar": corr_1 if not np.isnan(corr_1) else 0.0,
            "ic_4bar": corr_4 if not np.isnan(corr_4) else 0.0,
            "ic_12bar": corr_12 if not np.isnan(corr_12) else 0.0,
            "abs_ic_4bar": abs(corr_4) if not np.isnan(corr_4) else 0.0,
            "ic_ir": ic_ir
        })
        
    df_ic = pd.DataFrame(ic_results)
    
    # 5. Informação Mútua (MI) com Triple-Barrier
    X = np.nan_to_num(df_clean[df_ic["feature"].tolist()].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_tb = df_clean["tb_label"].astype(int).values
    
    mi_scores = mutual_info_classif(X, y_tb, random_state=42, n_neighbors=5)
    df_ic["mutual_info"] = mi_scores
    
    # 6. LightGBM Gain Importance
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1, verbosity=-1)
    clf_lgb.fit(X, y_tb)
    gains = clf_lgb.booster_.feature_importance(importance_type="gain")
    df_ic["lgb_gain_pct"] = (gains / (gains.sum() + 1e-12)) * 100
    
    # 7. Score Composto do Timeframe
    df_ic["rank_ic"] = df_ic["abs_ic_4bar"].rank(pct=True)
    df_ic["rank_mi"] = df_ic["mutual_info"].rank(pct=True)
    df_ic["rank_lgb"] = df_ic["lgb_gain_pct"].rank(pct=True)
    df_ic["score_composite"] = (df_ic["rank_ic"] * 0.35 + df_ic["rank_mi"] * 0.35 + df_ic["rank_lgb"] * 0.30) * 100
    df_ic["timeframe"] = tf
    
    df_sorted = df_ic.sort_values(by="score_composite", ascending=False).reset_index(drop=True)
    
    print(f"\nTop 8 Melhores Indicadores para {tf}:")
    print(df_sorted[["feature", "score_composite", "abs_ic_4bar", "mutual_info", "lgb_gain_pct"]].head(8).to_string())
    
    return df_sorted

async def main():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    timeframes = [
        ("15m", "bitget", 30000),
        ("1h", "binance", 30000),
        ("4h", "bitget", 8000),
        ("1d", "binance", 2400)
    ]
    
    all_results = []
    
    for tf, ex, max_c in timeframes:
        res = await validate_timeframe(loader, fe, symbol="BTCUSDT", tf=tf, exchange=ex, max_candles=max_c)
        if res is not None:
            all_results.append(res)
            
    await loader.close()
    
    if not all_results:
        print("Erro: Nenhum resultado gerado.")
        return
        
    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("datasets/reports/multi_timeframe_indicator_analysis.csv", index=False)
    
    # Consolidação Geral: Pontuação Média através de TODOS os timeframes
    pivot_scores = full_df.pivot(index="feature", columns="timeframe", values="score_composite")
    pivot_scores["mean_composite_score"] = pivot_scores.mean(axis=1)
    pivot_scores = pivot_scores.sort_values(by="mean_composite_score", ascending=False)
    
    pivot_ic = full_df.pivot(index="feature", columns="timeframe", values="abs_ic_4bar")
    pivot_ic["mean_abs_ic"] = pivot_ic.mean(axis=1)
    
    summary_df = pd.DataFrame({
        "mean_score": pivot_scores["mean_composite_score"],
        "score_15m": pivot_scores.get("15m", np.nan),
        "score_1h": pivot_scores.get("1h", np.nan),
        "score_4h": pivot_scores.get("4h", np.nan),
        "score_1d": pivot_scores.get("1d", np.nan),
        "mean_abs_ic": pivot_ic["mean_abs_ic"]
    }).sort_values(by="mean_score", ascending=False)
    
    summary_df.to_csv("datasets/reports/indicator_cross_timeframe_summary.csv")
    
    print("\n" + "="*80)
    print("🏆 CONSOLIDAÇÃO GERAL MULTI-TIMEFRAME (15m, 1h, 4h, 1d):")
    print("="*80)
    print(summary_df.head(20).to_string())
    print("\n" + "="*80)
    print("🗑️ OS 10 PIORES INDICADORES (RUÍDO E REDUNDÂNCIA EM TODOS OS TIMEFRAMES):")
    print("="*80)
    print(summary_df.tail(10).to_string())

if __name__ == "__main__":
    asyncio.run(main())
