"""
Simulação Calibrada de Staged Exits:
1. TP1 em 1.5x ATR (fecha 25% e move SL para Breakeven).
2. TP2 em 3.0x ATR (fecha 25% e move SL para Trailing).
3. TP3 (50% restante da mão!) surfa a tendência inteira com Trailing Stop de 2.5x ATR.
4. SL inicial curto em 1.0x ATR com filtro de volume institucional (volume_zscore >= 1.0).
"""

import asyncio
import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer

async def simulate_calibrated_staged_swing(symbol="BTCUSDT", tf="4h", exchange="bitget"):
    loader = DataLoader()
    fe = FeatureEngineer()
    
    df = await loader.get_candles(symbol, tf, limit=10000, exchange=exchange)
    if len(df) < 500 and exchange != "binance":
        df = await loader.get_candles(symbol, tf, limit=10000, exchange="binance")
    await loader.close()
    
    if len(df) < 300: return None
    df_feat = fe.compute_all_features(df, drop_na=True)
    
    taker_fee = 0.0006
    slippage = 0.0003
    entry_cost = taker_fee + slippage
    exit_cost = taker_fee + slippage
    
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp1_hit = False
    tp2_hit = False
    pos_qty = 0.0
    initial_qty = 1.0
    highest_p = 0.0
    lowest_p = 0.0
    atr_at_entry = 0.0
    entry_idx = 0
    realized_cash_pnl = 0.0
    
    for i in range(len(df_feat) - 1):
        row = df_feat.iloc[i]
        next_row = df_feat.iloc[i + 1]
        
        close = row["close"]
        high = next_row["high"]
        low = next_row["low"]
        sma_200 = row["sma_200"]
        rsi = row["rsi_14"]
        vwap_dist = row["dist_vwap_pct"]
        vol_z = row.get("volume_zscore_20", 0.0)
        atr = row["atr_14"]
        
        if not in_pos:
            # Entrada seletiva com confirmação de volume
            if close > sma_200 and rsi <= 44.0 and vwap_dist <= 0.8 and vol_z >= 0.5:
                in_pos = True
                side = "LONG"
                entry_p = next_row["open"] * (1.0 + slippage)
                atr_at_entry = atr
                sl = entry_p - 1.0 * atr  # SL Inicial enxuto
                tp1 = entry_p + 1.5 * atr # TP1 busca 1.5 ATR
                tp2 = entry_p + 3.0 * atr # TP2 busca 3.0 ATR
                tp1_hit = False
                tp2_hit = False
                pos_qty = initial_qty
                highest_p = entry_p
                realized_cash_pnl = - (initial_qty * entry_cost)
                entry_idx = i + 1
            elif close < sma_200 and rsi >= 56.0 and vwap_dist >= -0.8 and vol_z >= 0.5:
                in_pos = True
                side = "SHORT"
                entry_p = next_row["open"] * (1.0 - slippage)
                atr_at_entry = atr
                sl = entry_p + 1.0 * atr
                tp1 = entry_p - 1.5 * atr
                tp2 = entry_p - 3.0 * atr
                tp1_hit = False
                tp2_hit = False
                pos_qty = initial_qty
                lowest_p = entry_p
                realized_cash_pnl = - (initial_qty * entry_cost)
                entry_idx = i + 1
        else:
            if side == "LONG":
                highest_p = max(highest_p, high)
                
                # TP1: fecha apenas 25% e protege breakeven
                if not tp1_hit and high >= tp1:
                    tp1_hit = True
                    q1 = initial_qty * 0.25
                    pos_qty -= q1
                    realized_cash_pnl += q1 * ((tp1 - entry_p) / entry_p - exit_cost)
                    sl = entry_p * (1.0 + entry_cost + exit_cost)
                    
                # TP2: fecha mais 25% e sobe stop
                if tp1_hit and not tp2_hit and high >= tp2:
                    tp2_hit = True
                    q2 = initial_qty * 0.25
                    pos_qty -= q2
                    realized_cash_pnl += q2 * ((tp2 - entry_p) / entry_p - exit_cost)
                    sl = tp1
                    
                # TP3: 50% restante surfa com Trailing Stop Chandelier
                if tp1_hit:
                    trailing_sl = highest_p - 2.5 * atr_at_entry
                    sl = max(sl, trailing_sl)
                    
                if low <= sl:
                    exit_p = sl * (1.0 - slippage)
                    realized_cash_pnl += pos_qty * ((exit_p - entry_p) / entry_p - exit_cost)
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": tp1_hit})
                    
            elif side == "SHORT":
                lowest_p = min(lowest_p, low)
                
                if not tp1_hit and low <= tp1:
                    tp1_hit = True
                    q1 = initial_qty * 0.25
                    pos_qty -= q1
                    realized_cash_pnl += q1 * ((entry_p - tp1) / entry_p - exit_cost)
                    sl = entry_p * (1.0 - entry_cost - exit_cost)
                    
                if tp1_hit and not tp2_hit and low <= tp2:
                    tp2_hit = True
                    q2 = initial_qty * 0.25
                    pos_qty -= q2
                    realized_cash_pnl += q2 * ((entry_p - tp2) / entry_p - exit_cost)
                    sl = tp1
                    
                if tp1_hit:
                    trailing_sl = lowest_p + 2.5 * atr_at_entry
                    sl = min(sl, trailing_sl)
                    
                if high >= sl:
                    exit_p = sl * (1.0 + slippage)
                    realized_cash_pnl += pos_qty * ((entry_p - exit_p) / entry_p - exit_cost)
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": tp1_hit})
                    
    if not trades: return None
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["win"]]
    losses = df_t[~df_t["win"]]
    win_rate = (len(wins) / len(df_t)) * 100.0
    avg_win = wins["pnl_pct"].mean() * 100.0 if len(wins) > 0 else 0.0
    avg_loss = abs(losses["pnl_pct"].mean()) * 100.0 if len(losses) > 0 else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    profit_factor = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())) if len(losses) > 0 else 999.0
    cum_series = (1.0 + df_t["pnl_pct"]).cumprod()
    net_ret = cum_series.iloc[-1] - 1.0 if not cum_series.empty else 0.0
    max_dd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100.0
    
    return {
        "symbol": symbol, "timeframe": tf, "total_trades": len(df_t),
        "win_rate_pct": round(win_rate, 1),
        "payoff_ratio": round(payoff, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "net_profit_pct": round(net_ret * 100.0, 1),
        "max_drawdown_pct": round(abs(max_dd), 1),
    }

async def main():
    print("==========================================================================================")
    print("💎 SWING TRADE OTIMIZADO: TP1 (25% + BE), TP2 (25%), TP3 (50% TRAILING LIVRE)")
    print("==========================================================================================")
    cases = [
        ("BTCUSDT", "4h", "bitget"),
        ("ETHUSDT", "4h", "bitget"),
        ("SOLUSDT", "4h", "bitget"),
        ("BTCUSDT", "1d", "binance"),
        ("ETHUSDT", "1d", "binance"),
        ("SOLUSDT", "1d", "binance"),
    ]
    res = []
    for s, tf, ex in cases:
        r = await simulate_calibrated_staged_swing(s, tf, ex)
        if r: res.append(r)
    df = pd.DataFrame(res)
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
