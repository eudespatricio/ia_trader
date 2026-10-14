import asyncio
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from model_trainer import create_triple_barrier_labels

async def run_feature_importance_audit():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    print("=================================================================")
    print("🔍 AUDITORIA QUANTITATIVA DE IMPORTÂNCIA DE FEATURES (BTC 1H)")
    print("=================================================================")
    
    # Busca velas reais do cache SQLite
    df = await loader.get_candles("BTCUSDT", "1h", limit=30000, exchange="binance")
    if len(df) < 2000:
        df = await loader.get_candles("BTCUSDT", "1h", limit=30000, exchange="bitget")
        
    print(f"Total de candles carregados: {len(df)}")
    
    # Extrai macro contexto do SQLite
    macro_df = None
    df_feat = fe.compute_all_features(df, macro_df=macro_df, drop_na=True)
    
    feature_cols = fe.get_feature_names()
    available_cols = [c for c in feature_cols if c in df_feat.columns]
    print(f"Features calculadas: {len(available_cols)}")
    
    # Cria labels Triple-Barrier com horizonte 6 barras (6h), TP=1.5x ATR, SL=1.0x ATR
    df_feat["label"] = create_triple_barrier_labels(df_feat, horizon=6, multiple_tp=1.5, multiple_sl=1.0)
    
    # Descarta últimas 6 barras que têm label NaN
    df_clean = df_feat.dropna(subset=["label"]).copy()
    df_clean["label"] = df_clean["label"].astype(int)
    
    X = df_clean[available_cols].values
    y = df_clean["label"].values
    
    # Substitui qualquer resíduo inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Informação Mútua (Mutual Information - Não linear e independente de modelo)
    print("\nCalculando Informação Mútua (Mutual Information)...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    
    # 2. Feature Importance via LightGBM (Gini / Split & Gain)
    print("Treinando LightGBM para medição de Ganho de Informação (Gain)...")
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1, verbosity=-1)
    clf_lgb.fit(X, y)
    lgb_gain_importance = clf_lgb.booster_.feature_importance(importance_type="gain")
    lgb_gain_importance = lgb_gain_importance / (lgb_gain_importance.sum() + 1e-12)
    
    # 3. Feature Importance via Random Forest (MDI)
    print("Treinando Random Forest para Validação Cruzada de MDI...")
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf_rf.fit(X, y)
    rf_importance = clf_rf.feature_importances_
    
    # Consolida Ranking
    res_df = pd.DataFrame({
        "feature": available_cols,
        "mutual_info": mi_scores,
        "lgb_gain_pct": lgb_gain_importance * 100,
        "rf_importance_pct": rf_importance * 100
    })
    
    # Score Composto Normalizado (0 a 100)
    res_df["rank_mi"] = res_df["mutual_info"].rank(pct=True)
    res_df["rank_lgb"] = res_df["lgb_gain_pct"].rank(pct=True)
    res_df["rank_rf"] = res_df["rf_importance_pct"].rank(pct=True)
    res_df["composite_score"] = (res_df["rank_mi"] * 0.4 + res_df["rank_lgb"] * 0.4 + res_df["rank_rf"] * 0.2) * 100
    
    res_df = res_df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    
    # Salva relatório em datasets/reports
    res_df.to_csv("datasets/reports/feature_importance_ranking.csv", index=False)
    
    print("\n🏆 RANKING DOS 15 MELHORES INDICADORES (PREDITORES REAIS):")
    print(res_df[["feature", "composite_score", "lgb_gain_pct", "mutual_info", "rf_importance_pct"]].head(15).to_string())
    
    print("\n🗑️ OS 10 INDICADORES MAIS FRACOS (RUÍDO PURO / DESCARTÁVEIS):")
    print(res_df[["feature", "composite_score", "lgb_gain_pct", "mutual_info"]].tail(10).to_string())
    
    await loader.close()

if __name__ == "__main__":
    asyncio.run(run_feature_importance_audit())
