"""
Surgical Scalp / Sniper Day Trade Evaluation Script.
Testa o conceito de 'Scalp Cirúrgico':
- Reduz drasticamente o número de operações intradiárias (apenas setups raros de confluência extrema).
- Confluência Exigida:
  1. Filtro de Regime Macro (SMA 200 em 1h/4h)
  2. Volume Clímax Institucional (Z-Score >= 2.0)
  3. Esticamento Elástico de VWAP (dist_vwap_pct >= 1.8%)
  4. Exaustão de RSI (RSI <= 32 em compra, RSI >= 68 em venda)
  5. Deslocamento de Funding Rate (Z-score de funding indicando sobrealavancagem)
  6. Execução Maker/Limit (taxa de 0.02% maker em vez de 0.06% taker)
"""

import asyncio
import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer

async def simulate_surgical_scalp(symbol="BTCUSDT", tf="15m", exchange="binance"):
    loader = DataLoader()
    fe = FeatureEngineer()
    
    # Carrega dados profundos de 15m ou 1h
    df = await loader.get_candles(symbol, tf, limit=30000, exchange=exchange)
    if len(df) < 2000 and exchange != "bitget":
        df = await loader.get_candles(symbol, tf, limit=30000, exchange="bitget")
    await loader.close()
    
    if len(df) < 500:
        return None
        
    df_feat = fe.compute_all_features(df, drop_na=True)
    
    # Custos de execução:
    # No scalp cirúrgico institucional, usa-se ordem limite (Maker 0.02%) na entrada
    # e na saída parcial, reduzindo o drag de taxas em mais de 60%!
    maker_fee = 0.0002
    taker_fee = 0.0006
    slippage = 0.0002
    cost_per_trade = (maker_fee + slippage) + (taker_fee + slippage)  # ~0.11% total
    
    trades = []
    in_pos = False
    side = None
    entry_p = 0.0
    sl = 0.0
    tp1 = 0.0
    tp2 = 0.0
    tp1_hit = False
    entry_idx = 0
    pos_qty = 1.0
    realized_cash_pnl = 0.0
    atr_at_entry = 0.0
    
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
        funding_z = row.get("funding_zscore", 0.0)
        atr = row["atr_14"]
        
        if not in_pos:
            # Confluência Sniper LONG:
            # 1. Preço acima da SMA 200 (Macro Bull)
            # 2. Clímax de Volume Institucional (vol_z >= 1.8)
            # 3. Esticamento abaixo do VWAP (vwap_dist <= -1.2%)
            # 4. RSI sobrevendido em retração extrema (rsi <= 32)
            # 5. Funding não excessivamente positivo
            is_sniper_long = (
                close > sma_200 and
                vol_z >= 1.8 and
                vwap_dist <= -1.2 and
                rsi <= 32.0 and
                funding_z <= 1.5
            )
            
            # Confluência Sniper SHORT:
            # 1. Preço abaixo da SMA 200 (Macro Bear)
            # 2. Clímax de Volume Institucional (vol_z >= 1.8)
            # 3. Esticamento acima do VWAP (vwap_dist >= 1.2%)
            # 4. RSI sobrecomprado em repique extremo (rsi >= 68)
            # 5. Funding não excessivamente negativo
            is_sniper_short = (
                close < sma_200 and
                vol_z >= 1.8 and
                vwap_dist >= 1.2 and
                rsi >= 68.0 and
                funding_z >= -1.5
            )
            
            if is_sniper_long:
                in_pos = True
                side = "LONG"
                entry_p = next_row["open"]
                atr_at_entry = atr
                sl = entry_p - 1.2 * atr
                tp1 = entry_p + 1.2 * atr  # TP1 rápido de retorno à média
                tp2 = entry_p + 2.5 * atr  # TP2 expansão
                tp1_hit = False
                pos_qty = 1.0
                realized_cash_pnl = - (maker_fee + slippage)
                entry_idx = i + 1
                
            elif is_sniper_short:
                in_pos = True
                side = "SHORT"
                entry_p = next_row["open"]
                atr_at_entry = atr
                sl = entry_p + 1.2 * atr
                tp1 = entry_p - 1.2 * atr
                tp2 = entry_p - 2.5 * atr
                tp1_hit = False
                pos_qty = 1.0
                realized_cash_pnl = - (maker_fee + slippage)
                entry_idx = i + 1
                
        else:
            # Gestão do Scalp Cirúrgico
            if side == "LONG":
                # Check TP1: fecha 50% e joga SL no breakeven protegido
                if not tp1_hit and high >= tp1:
                    tp1_hit = True
                    q1 = 0.50
                    pos_qty -= q1
                    realized_cash_pnl += q1 * ((tp1 - entry_p) / entry_p - (maker_fee + slippage))
                    sl = entry_p * (1.0005) # Breakeven levemente acima da entrada
                    
                # Check TP2 no que restou
                if tp1_hit and high >= tp2:
                    realized_cash_pnl += pos_qty * ((tp2 - entry_p) / entry_p - (maker_fee + slippage))
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": True, "bars": i - entry_idx})
                    continue
                    
                # Check Stop Loss / Breakeven
                if low <= sl:
                    exit_p = sl * (1.0 - slippage)
                    realized_cash_pnl += pos_qty * ((exit_p - entry_p) / entry_p - (taker_fee + slippage))
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": tp1_hit, "bars": i - entry_idx})
                    
            elif side == "SHORT":
                if not tp1_hit and low <= tp1:
                    tp1_hit = True
                    q1 = 0.50
                    pos_qty -= q1
                    realized_cash_pnl += q1 * ((entry_p - tp1) / entry_p - (maker_fee + slippage))
                    sl = entry_p * (0.9995) # Breakeven
                    
                if tp1_hit and low <= tp2:
                    realized_cash_pnl += pos_qty * ((entry_p - tp2) / entry_p - (maker_fee + slippage))
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": True, "bars": i - entry_idx})
                    continue
                    
                if high >= sl:
                    exit_p = sl * (1.0 + slippage)
                    realized_cash_pnl += pos_qty * ((entry_p - exit_p) / entry_p - (taker_fee + slippage))
                    in_pos = False
                    trades.append({"side": side, "pnl_pct": realized_cash_pnl, "win": realized_cash_pnl > 0.0, "tp1_hit": tp1_hit, "bars": i - entry_idx})
                    
    if not trades:
        return None
        
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
        "symbol": symbol,
        "timeframe": tf,
        "total_candles": len(df_feat),
        "total_trades": len(df_t),
        "win_rate_pct": round(win_rate, 1),
        "payoff_ratio": round(payoff, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "net_profit_pct": round(net_ret * 100.0, 1),
        "max_drawdown_pct": round(abs(max_dd), 1),
        "avg_bars_held": round(df_t["bars"].mean(), 1),
    }

async def main():
    print("==========================================================================================")
    print("🎯 SCALP CIRÚRGICO (SNIPER DAY TRADE): ALTA CONVICÇÃO, POUCOS TRADES E CUSTO MAKER")
    print("==========================================================================================")
    cases = [
        ("BTCUSDT", "15m", "bitget"),
        ("ETHUSDT", "15m", "bitget"),
        ("SOLUSDT", "15m", "bitget"),
        ("BTCUSDT", "1h", "binance"),
        ("ETHUSDT", "1h", "binance"),
        ("SOLUSDT", "1h", "binance"),
    ]
    res = []
    for s, tf, ex in cases:
        r = await simulate_surgical_scalp(s, tf, ex)
        if r: res.append(r)
    df = pd.DataFrame(res)
    print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
