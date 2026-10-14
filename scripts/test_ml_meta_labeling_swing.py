"""
Test ML Meta-Labeling for Swing Trade Setups.
Avalia a precisão out-of-sample do LightGBM ao filtrar os setups de Swing Trade em 4h.
"""

import asyncio
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer

async def run_ml_meta_labeling():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    print("==========================================================================")
    print("🤖 AVALIANDO LIGHTGBM COMO FILTRO DE PRECISÃO (META-LABELING EM SWING 4H)")
    print("==========================================================================")
    
    df = await loader.get_candles("BTCUSDT", "4h", limit=8000, exchange="bitget")
    await loader.close()
    
    df_feat = fe.compute_all_features(df, drop_na=True)
    inst_features = [c for c in fe.get_institutional_feature_names() if c in df_feat.columns]
    
    # 1. Identifica os setups candidatos
    setups = []
    horizon = 12 # 12 barras de 4h = 48 horas (2 dias)
    
    for i in range(len(df_feat) - horizon):
        row = df_feat.iloc[i]
        close = row["close"]
        sma_200 = row["sma_200"]
        rsi = row["rsi_14"]
        vwap_dist = row["dist_vwap_pct"]
        vol_z = row["volume_zscore_20"]
        atr = row["atr_14"]
        
        is_long = close > sma_200 and rsi <= 44.0 and vwap_dist <= 0.8 and vol_z >= 0.5
        is_short = close < sma_200 and rsi >= 56.0 and vwap_dist >= -0.8 and vol_z >= 0.5
        
        if is_long or is_short:
            # Avalia se atinge TP1 (1.5x ATR) antes de SL (1.0x ATR) nas próximas N barras
            sub_df = df_feat.iloc[i+1 : i+1+horizon]
            entry_p = sub_df.iloc[0]["open"]
            
            if is_long:
                tp1 = entry_p + 1.5 * atr
                sl = entry_p - 1.0 * atr
                hit_tp1 = False
                hit_sl = False
                for _, bar in sub_df.iterrows():
                    if bar["low"] <= sl:
                        hit_sl = True
                        break
                    if bar["high"] >= tp1:
                        hit_tp1 = True
                        break
                label = 1 if hit_tp1 else 0
            else:
                tp1 = entry_p - 1.5 * atr
                sl = entry_p + 1.0 * atr
                hit_tp1 = False
                hit_sl = False
                for _, bar in sub_df.iterrows():
                    if bar["high"] >= sl:
                        hit_sl = True
                        break
                    if bar["low"] <= tp1:
                        hit_tp1 = True
                        break
                label = 1 if hit_tp1 else 0
                
            setups.append({
                "idx": i,
                "label": label,
                **{f: row[f] for f in inst_features}
            })
            
    df_setups = pd.DataFrame(setups)
    print(f"Total de setups gerados pelas regras primárias: {len(df_setups)}")
    print(f"Taxa de acerto bruta sem ML: {df_setups['label'].mean() * 100:.1f}%")
    
    # 2. Treina LightGBM com TimeSeriesSplit para testar Out-of-Sample
    X = df_setups[inst_features].values
    y = df_setups["label"].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    all_oos_preds = []
    all_oos_y = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        clf = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=42,
            verbosity=-1
        )
        clf.fit(X_train, y_train)
        
        probas = clf.predict_proba(X_test)[:, 1]
        all_oos_preds.extend(probas)
        all_oos_y.extend(y_test)
        
    all_oos_preds = np.array(all_oos_preds)
    all_oos_y = np.array(all_oos_y)
    
    print("\n--- IMPACTO DO FILTRO LIGHTGBM (OUT-OF-SAMPLE) ---")
    thresholds = [0.50, 0.55, 0.60, 0.65]
    for th in thresholds:
        selected = all_oos_preds >= th
        count = selected.sum()
        if count > 0:
            win_rate = all_oos_y[selected].mean() * 100.0
            print(f"Limiar de Convicção >= {th:.2f} | Trades Filtrados: {count}/{len(all_oos_y)} ({count/len(all_oos_y)*100:.1f}%) | Win Rate Real: {win_rate:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_ml_meta_labeling())
