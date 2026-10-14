"""
Test Candidate Institutional Features
Avalia 6 features institucionais de alto sinal para verificar se agregam valor
sem trazer ruído ou overfitting.
"""

import asyncio
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
def check_stationarity_df(s):
    """Teste Dickey-Fuller simplificado via OLS com numpy"""
    s = s[~np.isnan(s)]
    if len(s) < 50:
        return False, 1.0
    y = np.diff(s)
    x = s[:-1]
    # OLS: y = beta * x + alpha
    X_mat = np.column_stack([x, np.ones(len(x))])
    params, residuals, rank, s_vals = np.linalg.lstsq(X_mat, y, rcond=None)
    beta = params[0]
    # Standard error of beta
    res = y - X_mat @ params
    sigma2 = np.sum(res**2) / (len(y) - 2)
    var_beta = sigma2 * np.linalg.inv(X_mat.T @ X_mat)[0, 0]
    t_stat = beta / (np.sqrt(max(var_beta, 1e-12)))
    # T-stat crítico a 5% é aproximadamente -2.86
    is_stat = t_stat < -2.86
    return is_stat, float(t_stat)

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from model_trainer import create_triple_barrier_labels

def get_weights_ffd(d, thres=1e-4):
    """Calcula pesos para Fixed-Width Window Fractional Differentiation (López de Prado)"""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres or k > 100:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])

def frac_diff_ffd(series, d=0.4, thres=1e-4):
    """Aplica FFD a uma série de preços"""
    weights = get_weights_ffd(d, thres)
    width = len(weights)
    res = np.full(len(series), np.nan)
    vals = series.values
    for i in range(width - 1, len(series)):
        res[i] = np.dot(weights, vals[i - width + 1 : i + 1])
    return pd.Series(res, index=series.index)

async def test_candidates():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    print("=================================================================")
    print("🧪 TESTANDO CANDIDATAS INSTITUCIONAIS DE ALTO SINAL (BTCUSDT)")
    print("=================================================================")
    
    # 1. Carrega dados de 1h e 4h
    df = await loader.get_candles("BTCUSDT", "1h", limit=15000, exchange="binance")
    
    # 2. Carrega funding rates do SQLite e faz merge temporal
    con = sqlite3.connect("datasets/market_cache.db")
    funding_df = pd.read_sql_query(
        "SELECT timestamp, funding_rate FROM funding_rates WHERE symbol='BTCUSDT' AND exchange='binance' ORDER BY timestamp ASC",
        con
    )
    con.close()
    
    if not funding_df.empty:
        funding_df["funding_rate"] = funding_df["funding_rate"].astype(float)
        # Forward fill funding rate no timeframe de 1h
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            funding_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    else:
        df["funding_rate"] = 0.0
        
    print(f"Candles carregados: {len(df):,} com funding associado.")
    
    # 3. Calcula as novas candidatas:
    # C1: natr_14 (Normalized ATR % = ATR / close * 100)
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["natr_14"] = (atr_14 / close) * 100.0
    
    # C2: Garman-Klass Volatility (5x mais eficiente que close-to-close)
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df["garman_klass_vol"] = np.sqrt(gk.rolling(24).mean()) * np.sqrt(365 * 24) * 100.0
    
    # C3: Volume Z-Score (20 períodos)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std() + 1e-8
    df["volume_zscore_20"] = (df["volume"] - vol_mean) / vol_std
    
    # C4: Rolling VWAP Distance % (24 períodos)
    pv = (df["close"] * df["volume"]).rolling(24).sum()
    v_sum = df["volume"].rolling(24).sum() + 1e-8
    vwap_24 = pv / v_sum
    df["dist_vwap_pct"] = ((df["close"] - vwap_24) / vwap_24) * 100.0
    
    # C5: Funding Rate Z-score (30 dias = 90 períodos de 8h)
    f_mean = df["funding_rate"].rolling(24 * 30).mean()
    f_std = df["funding_rate"].rolling(24 * 30).std() + 1e-12
    df["funding_zscore"] = (df["funding_rate"] - f_mean) / f_std
    df["funding_rate_annualized"] = df["funding_rate"] * 3 * 365 * 100.0
    
    # C6: Diferenciação Fracionária (FFD d=0.4)
    print("Calculando Diferenciação Fracionária FFD (d=0.4)...")
    df["frac_diff_close"] = frac_diff_ffd(df["close"], d=0.4)
    
    # Indicadores padrão para comparação
    df["sma_200"] = df["close"].rolling(200).mean()
    df["rsi_14"] = fe.add_oscillator_indicators(df.copy())["rsi_14"]
    
    # Target: forward return 4 barras e Triple Barrier
    df["fwd_ret_4"] = df["close"].shift(-4) / df["close"] - 1.0
    df["tb_label"] = create_triple_barrier_labels(df, horizon=6, multiple_tp=1.5, multiple_sl=1.0)
    
    df_clean = df.dropna().copy()
    print(f"Amostra limpa: {len(df_clean):,} candles.")
    
    candidates = [
        "natr_14",
        "garman_klass_vol",
        "volume_zscore_20",
        "dist_vwap_pct",
        "funding_zscore",
        "funding_rate_annualized",
        "frac_diff_close",
        # Baselines para comparação
        "rsi_14",
        "volume"
    ]
    
    results = []
    
    for c in candidates:
        s = df_clean[c].values
        # 1. Teste de Estacionariedade Dickey-Fuller
        is_stat, t_stat = check_stationarity_df(s[:3000])
            
        # 2. IC de Spearman com forward returns 4 barras
        ic_4, _ = spearmanr(s, df_clean["fwd_ret_4"])
        
        # 3. Informação Mútua com Triple Barrier
        mi = mutual_info_classif(s.reshape(-1, 1), df_clean["tb_label"].astype(int).values, random_state=42)[0]
        
        results.append({
            "feature": c,
            "estacionario": "SIM" if is_stat else "NÃO",
            "t_stat_df": round(t_stat, 2),
            "abs_ic_4bar": round(abs(ic_4), 4),
            "mutual_info": round(mi, 5)
        })
        
    res_df = pd.DataFrame(results).sort_values(by="abs_ic_4bar", ascending=False)
    print("\n📊 RESULTADOS DAS FEATURES CANDIDATAS:")
    print(res_df.to_string(index=False))
    
    await loader.close()

if __name__ == "__main__":
    asyncio.run(test_candidates())
