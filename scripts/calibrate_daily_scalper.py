"""
scripts/calibrate_daily_scalper.py
Calibração de Scalper Intradiário de Frequência Moderada (~5 scalps por dia no portfólio).
Testa combinações de sensibilidade de Volume, VWAP, Bollinger Bands e RSI ágil (9 períodos)
com execução MAKER (0.02%) e ordens limite.
"""

import asyncio
import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer

async def run_scalper_calibration():
    loader = DataLoader()
    fe = FeatureEngineer()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    print("==========================================================================================")
    print("⚡ CALIBRAÇÃO DE SCALPER INTRIADIÁRIO (~1.5 a 2 trades/dia por par = ~5 trades/dia total)")
    print("==========================================================================================")
    
    # Testa nas últimas 5.000 velas de 15m (~52 dias de mercado contínuo)
    all_results = []
    
    for sym in symbols:
        df = await loader.get_candles(sym, "15m", limit=5000, exchange="binance")
        if len(df) < 2000:
            df = await loader.get_candles(sym, "15m", limit=5000, exchange="bitget")
            
        if len(df) < 1000:
            continue
            
        df_feat = fe.compute_all_features(df, drop_na=True)
        total_days = len(df_feat) / (24 * 4) # 96 velas de 15m por dia
        
        # Testamos 3 configurações de calibração para encontrar a taxa ideal de ~1.5 trades/dia
        configs = [
            {"name": "Calibracao_A_Sensivel", "vol_z": 0.3, "vwap_dist": 0.4, "rsi_low": 38, "rsi_high": 62, "tp_atr": 0.8, "sl_atr": 0.8},
            {"name": "Calibracao_B_Equilibrada", "vol_z": 0.6, "vwap_dist": 0.6, "rsi_low": 35, "rsi_high": 65, "tp_atr": 1.0, "sl_atr": 0.9},
            {"name": "Calibracao_C_Seletiva", "vol_z": 0.9, "vwap_dist": 0.8, "rsi_low": 32, "rsi_high": 68, "tp_atr": 1.2, "sl_atr": 1.0},
        ]
        
        # Custos: Execução MAKER (0.02% por perna) + slippage leve (0.01% para ordem limite)
        maker_cost_roundtrip = (0.0002 + 0.0001) * 2.0  # ~0.06% total ida e volta
        
        for cfg in configs:
            trades = []
            in_pos = False
            side = None
            entry_p = 0.0
            sl = 0.0
            tp1 = 0.0
            tp1_hit = False
            pos_qty = 1.0
            realized_pnl = 0.0
            
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
                    # Long Scalp: Tendência favorável ou neutra + retração rápida no VWAP
                    if close >= sma_200 * 0.995 and rsi <= cfg["rsi_low"] and vwap_dist <= -cfg["vwap_dist"] and vol_z >= cfg["vol_z"]:
                        in_pos = True
                        side = "LONG"
                        entry_p = next_row["open"]
                        sl = entry_p - cfg["sl_atr"] * atr
                        tp1 = entry_p + cfg["tp_atr"] * atr
                        tp1_hit = False
                        pos_qty = 1.0
                        realized_pnl = - (0.0002 + 0.0001)
                    # Short Scalp: Resistência + esticamento acima do VWAP
                    elif close <= sma_200 * 1.005 and rsi >= cfg["rsi_high"] and vwap_dist >= cfg["vwap_dist"] and vol_z >= cfg["vol_z"]:
                        in_pos = True
                        side = "SHORT"
                        entry_p = next_row["open"]
                        sl = entry_p + cfg["sl_atr"] * atr
                        tp1 = entry_p - cfg["tp_atr"] * atr
                        tp1_hit = False
                        pos_qty = 1.0
                        realized_pnl = - (0.0002 + 0.0001)
                else:
                    if side == "LONG":
                        # Bate TP1: realiza 50% e joga SL no Breakeven
                        if not tp1_hit and high >= tp1:
                            tp1_hit = True
                            q = 0.50
                            pos_qty -= q
                            realized_pnl += q * ((tp1 - entry_p) / entry_p - (0.0002 + 0.0001))
                            sl = entry_p * 1.0004 # Breakeven protegido
                            
                        # Continuação ou saída
                        if tp1_hit and high >= entry_p + (cfg["tp_atr"] * 1.8) * atr:
                            realized_pnl += pos_qty * (((entry_p + cfg["tp_atr"] * 1.8 * atr) - entry_p) / entry_p - (0.0002 + 0.0001))
                            in_pos = False
                            trades.append({"pnl": realized_pnl, "win": realized_pnl > 0})
                            continue
                            
                        if low <= sl:
                            exit_p = sl
                            realized_pnl += pos_qty * ((exit_p - entry_p) / entry_p - (0.0002 + 0.0001))
                            in_pos = False
                            trades.append({"pnl": realized_pnl, "win": realized_pnl > 0})
                            
                    elif side == "SHORT":
                        if not tp1_hit and low <= tp1:
                            tp1_hit = True
                            q = 0.50
                            pos_qty -= q
                            realized_pnl += q * ((entry_p - tp1) / entry_p - (0.0002 + 0.0001))
                            sl = entry_p * 0.9996
                            
                        if tp1_hit and low <= entry_p - (cfg["tp_atr"] * 1.8) * atr:
                            realized_pnl += pos_qty * ((entry_p - (entry_p - cfg["tp_atr"] * 1.8 * atr)) / entry_p - (0.0002 + 0.0001))
                            in_pos = False
                            trades.append({"pnl": realized_pnl, "win": realized_pnl > 0})
                            continue
                            
                        if high >= sl:
                            exit_p = sl
                            realized_pnl += pos_qty * ((entry_p - exit_p) / entry_p - (0.0002 + 0.0001))
                            in_pos = False
                            trades.append({"pnl": realized_pnl, "win": realized_pnl > 0})
                            
            if trades:
                df_t = pd.DataFrame(trades)
                wins = df_t[df_t["win"]]
                losses = df_t[~df_t["win"]]
                n_trades = len(df_t)
                trades_per_day = n_trades / total_days
                win_rate = (len(wins) / n_trades) * 100.0
                gross_gains = wins["pnl"].sum()
                gross_loss = abs(losses["pnl"].sum())
                profit_factor = (gross_gains / gross_loss) if gross_loss > 0 else 999.0
                cum_ret = (1.0 + df_t["pnl"]).prod() - 1.0
                
                all_results.append({
                    "symbol": sym,
                    "config": cfg["name"],
                    "dias_amostra": round(total_days, 1),
                    "total_trades": n_trades,
                    "trades_por_dia": round(trades_per_day, 2),
                    "win_rate_pct": round(win_rate, 1),
                    "profit_factor": round(profit_factor, 2),
                    "retorno_liquido_pct": round(cum_ret * 100.0, 1),
                })
                
    await loader.close()
    
    df_res = pd.DataFrame(all_results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(run_scalper_calibration())
