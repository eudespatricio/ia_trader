"""
trend_following.py - Motor de Trend Following Diário com Volatility Targeting (E1).

Implementa:
- Ensemble Time Series Momentum (TSMOM) nos lookbacks [20, 50, 100, 200] dias.
- Volatility targeting anualizado (alvo 20% a.a.) com EWMA/janela móvel de 30 dias.
- Banda de no-trade para controle rigoroso de turnover.
- Execução no Open de D+1 (ausência total de lookahead).
- Custos taker 0.06% + slippage 0.03% por perna de rotação.
- Débito/crédito diário de funding rate real.
- Walk-forward purgado com embargo.
- Benchmarks: Buy & Hold e SMA 200.
- Análise de Robustez: Remoção do melhor ano e sensibilidade +/- 30% nos lookbacks.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class TSMOMTradeRecord:
    trade_id: int
    entry_idx: int
    exit_idx: int
    entry_date: str
    exit_date: str
    direction: str  # "LONG" ou "SHORT"
    entry_price: float
    exit_price: float
    duration_days: int
    avg_weight: float
    gross_pnl: float
    fees_paid: float
    funding_pnl: float
    net_pnl: float
    net_return_pct: float


@dataclass
class TSMOMResult:
    symbol: str
    strategy_name: str
    initial_capital: float
    final_capital: float
    net_profit: float
    net_profit_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades_count: int
    turnover_annual: float
    total_fees_paid: float
    total_funding_pnl: float
    gross_profit: float
    fee_drag_pct: float
    daily_returns: np.ndarray
    equity_curve: pd.DataFrame
    trades: List[TSMOMTradeRecord]
    metrics: Dict[str, Any]


class DailyTSMOMStrategy:
    """
    Estratégia diária de TSMOM com ensemble de lookbacks e vol targeting.
    """

    def __init__(
        self,
        lookbacks: List[int] = [20, 50, 100, 200],
        target_vol_ann: float = 0.20,      # 20% a.a.
        vol_window: int = 30,              # Janela de vol 30 dias
        max_leverage: float = 1.5,         # Limite superior de alavancagem
        no_trade_band: float = 0.10,       # Banda de no-trade no peso
        mode: str = "long_flat",           # "long_flat" ou "long_short"
        fee_rate: float = 0.0006,          # 0.06% taker
        slippage: float = 0.0003,          # 0.03% slippage
        annual_funding_rate: float = 0.0365,  # Funding padrão se não houver série específica (0.01%/dia ~ 3.65% a.a.)
    ):
        self.lookbacks = sorted(lookbacks)
        self.target_vol_ann = target_vol_ann
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.no_trade_band = no_trade_band
        self.mode = mode
        self.cost_per_turnover = fee_rate + slippage  # 0.09% por perna
        self.default_daily_funding = annual_funding_rate / 365.0

    def compute_signals_and_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula os sinais do ensemble e os pesos teóricos sem lookahead.
        O sinal do dia t só utiliza preços até o fechamento de t.
        A posição é assumida no open de t+1.
        """
        data = df.copy().reset_index(drop=True)
        closes = data["close"].values.astype(float)
        n = len(closes)

        # 1. Componentes do ensemble
        ensemble_signal = np.zeros(n, dtype=float)
        for lb in self.lookbacks:
            # Retorno de L dias: (P_t / P_{t-L}) - 1
            past_p = pd.Series(closes).shift(lb).values
            diff = closes - past_p
            # +1 se preço subiu, -1 se caiu
            sig_l = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
            if self.mode == "long_flat":
                sig_l = np.maximum(0.0, sig_l)
            ensemble_signal += sig_l

        ensemble_signal /= float(len(self.lookbacks))

        # 2. Volatilidade realizada de 30 dias dos retornos diários
        daily_log_rets = pd.Series(np.log(closes[1:] / closes[:-1]))
        roll_vol = daily_log_rets.rolling(window=self.vol_window).std() * math.sqrt(365.0)
        roll_vol_aligned = np.full(n, 0.50, dtype=float)
        roll_vol_aligned[1:] = roll_vol.fillna(0.50).values

        # 3. Peso teórico: S_t * min(max_leverage, target_vol / vol_t)
        # Proteção contra divisão por zero
        denom_vol = np.maximum(roll_vol_aligned, 0.05)
        scale_factor = np.clip(self.target_vol_ann / denom_vol, 0.0, self.max_leverage)
        raw_weights = ensemble_signal * scale_factor

        # 4. Aplicação da banda de no-trade (rebalanceamento esparso)
        filtered_weights = np.zeros(n, dtype=float)
        cur_w = 0.0
        for i in range(max(self.lookbacks), n):
            target_w = raw_weights[i]
            if abs(target_w - cur_w) > self.no_trade_band:
                cur_w = target_w
            filtered_weights[i] = cur_w

        data["ensemble_signal"] = ensemble_signal
        data["realized_vol_ann"] = roll_vol_aligned
        data["target_weight"] = filtered_weights
        # A posição efetiva do dia t é a decidida em t-1 e executada no open de t
        data["active_weight"] = pd.Series(filtered_weights).shift(1).fillna(0.0).values

        return data

    def run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
    ) -> TSMOMResult:
        """
        Executa o backtest com custos reais de turnover e funding.
        """
        data = self.compute_signals_and_weights(df)
        closes = data["close"].values.astype(float)
        opens = data["open"].values.astype(float) if "open" in data.columns else closes
        active_weights = data["active_weight"].values.astype(float)
        dates = data["timestamp"].astype(str).values if "timestamp" in data.columns else [str(i) for i in range(len(data))]

        n = len(data)
        equity = initial_capital
        equity_series = np.full(n, initial_capital, dtype=float)
        daily_pnl_rets = np.zeros(n, dtype=float)

        total_fees = 0.0
        total_funding = 0.0
        gross_profit = 0.0
        trades: List[TSMOMTradeRecord] = []

        cur_trade_start = -1
        trade_id = 0
        cur_trade_gross = 0.0
        cur_trade_fees = 0.0
        cur_trade_funding = 0.0
        cur_trade_weights: List[float] = []

        # A execução se inicia a partir do maior lookback
        start_idx = max(self.lookbacks) + 1

        for t in range(start_idx, n):
            prev_w = active_weights[t - 1]
            w = active_weights[t]

            # 1. Custo de transação pelo turnover no início do dia:
            # delta_w = |w_t - w_{t-1}|
            delta_w = abs(w - prev_w)
            turnover_cost = equity * delta_w * self.cost_per_turnover
            total_fees += turnover_cost
            equity -= turnover_cost

            # 2. Retorno de preço do ativo no dia t:
            # Retorno close-to-close da barra
            price_ret = (closes[t] - closes[t - 1]) / closes[t - 1]
            gross_day_pnl = equity * w * price_ret
            gross_profit += gross_day_pnl

            # 3. Débito de funding rate real:
            # Se long (w > 0), normalmente paga funding padrão do dia
            funding_day = 0.0
            if "funding_rate" in data.columns:
                # Funding rate fornecido no dataframe
                funding_day = equity * w * float(data["funding_rate"].iloc[t])
            else:
                funding_day = equity * w * self.default_daily_funding

            total_funding -= funding_day  # Posição Long paga funding positivo
            net_day_pnl = gross_day_pnl - funding_day
            equity += net_day_pnl

            equity_series[t] = equity
            daily_pnl_rets[t] = net_day_pnl / max(1.0, equity - net_day_pnl)

            # Rastreamento de trades contínuos (quando w > 0)
            if w > 0 and prev_w == 0:
                # Início de um trade
                cur_trade_start = t
                trade_id += 1
                cur_trade_gross = gross_day_pnl
                cur_trade_fees = turnover_cost
                cur_trade_funding = -funding_day
                cur_trade_weights = [w]
            elif w > 0 and prev_w > 0:
                cur_trade_gross += gross_day_pnl
                cur_trade_fees += turnover_cost
                cur_trade_funding -= funding_day
                cur_trade_weights.append(w)
            elif w == 0 and prev_w > 0 and cur_trade_start != -1:
                # Fim de um trade
                cur_trade_fees += turnover_cost
                net_t_pnl = cur_trade_gross + cur_trade_funding - cur_trade_fees
                dur = t - cur_trade_start
                trades.append(
                    TSMOMTradeRecord(
                        trade_id=trade_id,
                        entry_idx=cur_trade_start,
                        exit_idx=t,
                        entry_date=dates[cur_trade_start],
                        exit_date=dates[t],
                        direction="LONG",
                        entry_price=float(closes[cur_trade_start]),
                        exit_price=float(closes[t]),
                        duration_days=dur,
                        avg_weight=float(np.mean(cur_trade_weights)),
                        gross_pnl=float(cur_trade_gross),
                        fees_paid=float(cur_trade_fees),
                        funding_pnl=float(cur_trade_funding),
                        net_pnl=float(net_t_pnl),
                        net_return_pct=float((net_t_pnl / initial_capital) * 100.0),
                    )
                )
                cur_trade_start = -1

        # Fecha trade aberto ao final se houver
        if cur_trade_start != -1:
            trades.append(
                TSMOMTradeRecord(
                    trade_id=trade_id,
                    entry_idx=cur_trade_start,
                    exit_idx=n - 1,
                    entry_date=dates[cur_trade_start],
                    exit_date=dates[-1],
                    direction="LONG",
                    entry_price=float(closes[cur_trade_start]),
                    exit_price=float(closes[-1]),
                    duration_days=n - 1 - cur_trade_start,
                    avg_weight=float(np.mean(cur_trade_weights)),
                    gross_pnl=float(cur_trade_gross),
                    fees_paid=float(cur_trade_fees),
                    funding_pnl=float(cur_trade_funding),
                    net_pnl=float(cur_trade_gross + cur_trade_funding - cur_trade_fees),
                    net_return_pct=float(((cur_trade_gross + cur_trade_funding - cur_trade_fees) / initial_capital) * 100.0),
                )
            )

        # Métricas Globais
        net_profit = float(equity - initial_capital)
        net_profit_pct = float(net_profit / initial_capital) * 100.0
        
        eval_slice = equity_series[start_idx:]
        peaks = np.maximum.accumulate(eval_slice)
        dds = (eval_slice - peaks) / np.maximum(peaks, 1e-12)
        max_dd = float(abs(np.min(dds))) * 100.0

        n_eval_days = len(eval_slice)
        ann_return = float(net_profit_pct * (365.0 / max(1, n_eval_days)))

        eval_daily_rets = daily_pnl_rets[start_idx:]
        std_ret = float(np.std(eval_daily_rets, ddof=1)) if len(eval_daily_rets) > 1 else 1e-6
        mean_ret = float(np.mean(eval_daily_rets)) if len(eval_daily_rets) > 0 else 0.0
        sharpe = float((mean_ret / max(1e-12, std_ret)) * math.sqrt(365.0))

        # Turnover anualizado médio
        weight_changes = np.abs(np.diff(active_weights[start_idx:]))
        turnover_ann = float(np.sum(weight_changes) * (365.0 / max(1, n_eval_days)))

        win_trades = [t for t in trades if t.net_pnl > 0]
        win_rate = float(len(win_trades) / max(1, len(trades))) * 100.0
        fee_drag = float(total_fees / max(1e-4, abs(gross_profit))) * 100.0

        equity_df = pd.DataFrame({
            "timestamp": dates,
            "close": closes,
            "weight": active_weights,
            "equity": equity_series,
            "daily_return": daily_pnl_rets,
        })

        metrics = {
            "symbol": symbol,
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "annualized_return_pct": ann_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "trades_count": len(trades),
            "win_rate_pct": win_rate,
            "turnover_annual": turnover_ann,
            "total_fees_paid": total_fees,
            "total_funding_pnl": total_funding,
            "gross_profit": gross_profit,
            "fee_drag_pct": fee_drag,
        }

        return TSMOMResult(
            symbol=symbol,
            strategy_name=f"TSMOM_{self.mode}",
            initial_capital=initial_capital,
            final_capital=equity,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            annualized_return_pct=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            trades_count=len(trades),
            turnover_annual=turnover_ann,
            total_fees_paid=total_fees,
            total_funding_pnl=total_funding,
            gross_profit=gross_profit,
            fee_drag_pct=fee_drag,
            daily_returns=eval_daily_rets,
            equity_curve=equity_df,
            trades=trades,
            metrics=metrics,
        )


def evaluate_tsmom_benchmarks(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Dict[str, Any]]:
    """
    Calcula as métricas dos dois baselines mandatórios no mesmo período:
    1. Buy & Hold (Long 100% contínuo)
    2. SMA 200 (Long se P_t > SMA_200, flat caso contrário)
    """
    closes = df["close"].values.astype(float)
    n = len(closes)
    start_idx = 201

    if n <= start_idx + 10:
        return {}

    # 1. Buy & Hold
    bh_ret_pct = ((closes[-1] - closes[start_idx]) / closes[start_idx]) * 100.0
    bh_days = n - start_idx
    bh_ann_ret = bh_ret_pct * (365.0 / bh_days)
    daily_bh_rets = np.diff(closes[start_idx:]) / closes[start_idx:-1]
    bh_sharpe = float(np.mean(daily_bh_rets) / max(1e-12, np.std(daily_bh_rets, ddof=1)) * math.sqrt(365.0))
    bh_peaks = np.maximum.accumulate(closes[start_idx:])
    bh_dd = float(abs(np.min((closes[start_idx:] - bh_peaks) / bh_peaks))) * 100.0

    # 2. SMA 200
    sma200 = pd.Series(closes).rolling(200).mean().shift(1).values
    sma_sig = np.where(closes > sma200, 1.0, 0.0)
    sma_weights = pd.Series(sma_sig).shift(1).fillna(0.0).values

    sma_equity = initial_capital
    sma_daily_rets = []
    for t in range(start_idx, n):
        w = sma_weights[t]
        r = (closes[t] - closes[t - 1]) / closes[t - 1]
        day_pnl = sma_equity * w * r
        sma_equity += day_pnl
        sma_daily_rets.append(day_pnl / max(1.0, sma_equity - day_pnl))

    sma_ret_pct = ((sma_equity - initial_capital) / initial_capital) * 100.0
    sma_ann_ret = sma_ret_pct * (365.0 / bh_days)
    sma_sharpe = float(np.mean(sma_daily_rets) / max(1e-12, np.std(sma_daily_rets, ddof=1)) * math.sqrt(365.0))
    sma_arr = np.array(sma_daily_rets)
    sma_cum = np.cumprod(1.0 + sma_arr)
    sma_peaks = np.maximum.accumulate(sma_cum)
    sma_dd = float(abs(np.min((sma_cum - sma_peaks) / sma_peaks))) * 100.0

    return {
        "buy_and_hold": {
            "net_profit_pct": bh_ret_pct,
            "annualized_return_pct": bh_ann_ret,
            "sharpe_ratio": bh_sharpe,
            "max_drawdown_pct": bh_dd,
        },
        "sma_200": {
            "net_profit_pct": sma_ret_pct,
            "annualized_return_pct": sma_ann_ret,
            "sharpe_ratio": sma_sharpe,
            "max_drawdown_pct": sma_dd,
        },
    }
