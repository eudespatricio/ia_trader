"""
Avaliação Quantitativa de Desempenho do Swing Trade em 4h e 1D.
Mede Win-Rate, Payoff Ratio, Profit Factor e Lucro Líquido Real.
"""

import asyncio
import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer

async def evaluate_swing_strategy(symbol="BTCUSDT", tf="4h", exchange="bitget"):
    loader = DataLoader()
    fe = FeatureEngineer()
    
    # 1. Carrega dados
    df = await loader.get_candles(symbol, tf, limit=10000, exchange=exchange)
    if len(df) < 500 and exchange != "binance":
        df = await loader.get_candles(symbol, tf, limit=10000, exchange="binance")
        
    await loader.close()
    
    if len(df) < 300:
        return None
        
    # 2. Computa features institucionais
    df_feat = fe.compute_all_features(df, drop_na=True)
    
    # 3. Regras de Swing Trade Institucional:
    # Filtro de Regime: Preço > SMA200 (Alta) ou Preço < SMA200 (Baixa)
    # Gatilho de Pullback: RSI14 <= 42 em alta ou RSI14 >= 58 em baixa
    # Confirmação de VWAP: Distância <= 0.5% em alta ou >= -0.5% em baixa
    # Saída: TP em 2.5x ATR, SL em 1.5x ATR (Payoff Ratio assimétrico = 1.67)
    
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    tp = 0.0
    entry_idx = 0
    
    taker_fee = 0.0006
    slippage = 0.0003
    cost_per_trade = (taker_fee + slippage) * 2.0  # ~0.18% ida e volta
    
    for i in range(len(df_feat) - 1):
        row = df_feat.iloc[i]
        next_row = df_feat.iloc[i + 1]
        
        close = row["close"]
        high = next_row["high"]
        low = next_row["low"]
        sma_200 = row["sma_200"]
        rsi = row["rsi_14"]
        vwap_dist = row["dist_vwap_pct"]
        atr = row["atr_14"]
        
        if not in_pos:
            # Setup Long: Tendência macro de alta + Pullback de RSI + Suporte de VWAP
            if close > sma_200 and rsi <= 40.0 and vwap_dist <= 0.5:
                in_pos = True
                side = "LONG"
                entry_p = next_row["open"] * (1.0 + slippage)
                sl = entry_p - 1.5 * atr
                tp = entry_p + 2.5 * atr
                entry_idx = i + 1
            # Setup Short: Tendência macro de baixa + Repique de RSI + Resistência de VWAP
            elif close < sma_200 and rsi >= 60.0 and vwap_dist >= -0.5:
                in_pos = True
                side = "SHORT"
                entry_p = next_row["open"] * (1.0 - slippage)
                sl = entry_p + 1.5 * atr
                tp = entry_p - 2.5 * atr
                entry_idx = i + 1
        else:
            # Avalia saída na barra seguinte
            if side == "LONG":
                if low <= sl:
                    exit_p = sl * (1.0 - slippage)
                    pnl_pct = (exit_p - entry_p) / entry_p - cost_per_trade
                    trades.append({"side": side, "pnl_pct": pnl_pct, "bars": i - entry_idx, "win": pnl_pct > 0})
                    in_pos = False
                elif high >= tp:
                    exit_p = tp * (1.0 - slippage)
                    pnl_pct = (exit_p - entry_p) / entry_p - cost_per_trade
                    trades.append({"side": side, "pnl_pct": pnl_pct, "bars": i - entry_idx, "win": pnl_pct > 0})
                    in_pos = False
            elif side == "SHORT":
                if high >= sl:
                    exit_p = sl * (1.0 + slippage)
                    pnl_pct = (entry_p - exit_p) / entry_p - cost_per_trade
                    trades.append({"side": side, "pnl_pct": pnl_pct, "bars": i - entry_idx, "win": pnl_pct > 0})
                    in_pos = False
                elif low <= tp:
                    exit_p = tp * (1.0 + slippage)
                    pnl_pct = (entry_p - exit_p) / entry_p - cost_per_trade
                    trades.append({"side": side, "pnl_pct": pnl_pct, "bars": i - entry_idx, "win": pnl_pct > 0})
                    in_pos = False
                    
    if not trades:
        return None
        
    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    wins = df_t[df_t["win"]]
    losses = df_t[~df_t["win"]]
    win_rate = (len(wins) / total_trades) * 100.0
    
    avg_win = wins["pnl_pct"].mean() * 100.0 if len(wins) > 0 else 0.0
    avg_loss = abs(losses["pnl_pct"].mean()) * 100.0 if len(losses) > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    
    gross_gains = wins["pnl_pct"].sum()
    gross_losses = abs(losses["pnl_pct"].sum())
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else 999.0
    
    cum_ret = (1.0 + df_t["pnl_pct"]).prod() - 1.0
    cum_series = (1.0 + df_t["pnl_pct"]).cumprod()
    max_dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100.0
    
    return {
        "symbol": symbol,
        "timeframe": tf,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "payoff_ratio": round(payoff_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "net_profit_pct": round(cum_ret * 100.0, 1),
        "max_drawdown_pct": round(abs(max_dd), 1),
        "avg_bars_held": round(df_t["bars"].mean(), 1),
    }

async def main():
    print("=========================================================================")
    print("📊 DESEMPENHO REAL DO SWING TRADE (TIMEFRAMES 4H E 1D COM DADOS REAIS)")
    print("=========================================================================")
    
    cases = [
        ("BTCUSDT", "4h", "bitget"),
        ("ETHUSDT", "4h", "bitget"),
        ("SOLUSDT", "4h", "bitget"),
        ("BTCUSDT", "1d", "binance"),
        ("ETHUSDT", "1d", "binance"),
        ("SOLUSDT", "1d", "binance"),
    ]
    
    results = []
    for sym, tf, ex in cases:
        r = await evaluate_swing_strategy(sym, tf, ex)
        if r:
            results.append(r)
            
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
