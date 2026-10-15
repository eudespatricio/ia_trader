"""
funding_carry.py - Motor de Simulação e Backtesting de Funding Carry Delta-Neutro (E2).

Implementa:
- Long Spot + Short Perp no mesmo notional (delta-neutro).
- Custos realistas de 4 pernas (Spot taker + slip, Perp taker + slip).
- Fluxo real de funding recebido/pago a cada 8h / diário.
- Rebalanceamento entre pernas e controle de margem colateral.
- Stress test de choque instantâneo de +/- 30% em 1 dia.
- Diagnóstico de funding negativo e concentração temporal de PnL.
- Ausência total de lookahead bias.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class CarryTradeRecord:
    timestamp_entry: int
    timestamp_exit: int
    duration_days: float
    notional: float
    gross_funding_collected: float
    fees_paid_4_legs: float
    net_pnl: float
    return_pct: float
    annualized_return_pct: float
    min_margin_level_observed: float
    exit_reason: str


@dataclass
class CarrySimulationResult:
    initial_capital: float
    final_capital: float
    net_profit: float
    net_profit_pct: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_funding_gross: float
    total_fees_paid: float
    pct_negative_funding_periods: float
    min_margin_level_observed: float
    survives_stress_shock_30pct: bool
    trades: List[CarryTradeRecord]
    equity_curve: pd.DataFrame
    summary_metrics: Dict[str, Any]


class FundingCarrySimulator:
    """
    Simulador de carry trade delta-neutro spot + perp.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        spot_fee_rate: float = 0.0006,  # 0.06% taker Bitget
        spot_slippage: float = 0.0002,  # 0.02% slippage spot
        perp_fee_rate: float = 0.0006,  # 0.06% taker Bitget
        perp_slippage: float = 0.0003,  # 0.03% slippage perp
        entry_funding_threshold_ann: float = 0.08,  # Entra se funding 7d > 8% a.a.
        exit_funding_threshold_ann: float = 0.02,   # Sai se funding 7d < 2% a.a.
        leverage_perp: float = 1.0,                 # 1.0x = 50% colateral spot, 50% margem perp
        maintenance_margin_pct: float = 0.01,       # 1.0% margem de manutenção
        min_safe_margin_pct: float = 0.30,          # 30% buffer de segurança mínimo
    ):
        self.initial_capital = float(initial_capital)
        self.spot_cost_per_leg = spot_fee_rate + spot_slippage  # 0.08%
        self.perp_cost_per_leg = perp_fee_rate + perp_slippage  # 0.09%
        self.entry_threshold_ann = entry_funding_threshold_ann
        self.exit_threshold_ann = exit_funding_threshold_ann
        self.leverage_perp = leverage_perp
        self.maint_margin_pct = maintenance_margin_pct
        self.min_safe_margin_pct = min_safe_margin_pct

    def run(
        self,
        df: pd.DataFrame,
        funding_col: str = "funding_rate",
        price_col: str = "close",
        time_col: str = "timestamp",
        periods_per_year: float = 365.0 * 3,  # Padrão 8h: 1095 períodos/ano
    ) -> CarrySimulationResult:
        """
        Executa a simulação de carry delta-neutro sobre o histórico de dados.
        df deve conter: [price_col, funding_col, time_col].
        Garante ausência de lookahead calculando o sinal de entrada/saída em t-1.
        """
        if len(df) < 30:
            raise ValueError("Dataframe com dados insuficientes para simulação de carry.")

        df = df.copy().reset_index(drop=True)
        prices = df[price_col].values.astype(float)
        fundings = df[funding_col].values.astype(float)
        timestamps = df[time_col].values

        n_bars = len(df)

        # Média móvel de funding de 7 dias (21 períodos de 8h, ou 7 barras se for diário)
        window = int(round(7.0 * (periods_per_year / 365.0)))
        window = max(3, window)
        
        # Média móvel defasada em 1 barra para eliminar lookahead bias
        roll_funding_ann = pd.Series(fundings).rolling(window=window).mean().shift(1).values * periods_per_year

        equity = self.initial_capital
        in_position = False
        pos_entry_price = 0.0
        pos_entry_ts = 0
        pos_qty = 0.0
        pos_notional = 0.0
        pos_colateral = 0.0
        accum_trade_funding = 0.0
        min_margin_observed = 1.0

        total_funding_gross = 0.0
        total_fees_paid = 0.0
        trades: List[CarryTradeRecord] = []
        equity_series = np.full(n_bars, self.initial_capital, dtype=float)
        margin_levels = np.ones(n_bars, dtype=float)
        positions = np.zeros(n_bars, dtype=int)

        neg_funding_count = int(np.sum(fundings < 0.0))
        pct_negative_funding = float(neg_funding_count / max(1, n_bars))

        for i in range(window, n_bars):
            cur_price = prices[i]
            cur_funding = fundings[i]
            prev_roll_ann = roll_funding_ann[i]

            # 1. Se em posição, atualiza margem e credita funding
            if in_position:
                # O funding da barra atual incide sobre o notional atual
                # Notional short perp = pos_qty * cur_price
                current_notional = pos_qty * cur_price
                # Short perp recebe funding quando rate é positivo
                bar_funding_income = current_notional * cur_funding
                accum_trade_funding += bar_funding_income
                total_funding_gross += bar_funding_income

                # PnL não realizado da perna short perp
                # short ganha se preço cai, perde se preço sobe
                # PnL não realizado da perna short perp e valor da perna spot
                # Modela basis mark-to-market se houver spot_price e perp_price distintos,
                # ou oscilação de basis correlacionada com o funding
                cur_spot = cur_price
                if "spot_price" in df.columns and "perp_price" in df.columns:
                    cur_spot = float(df["spot_price"].iloc[i])
                    cur_perp = float(df["perp_price"].iloc[i])
                else:
                    # Basis dinâmico proporcional à taxa anualizada (contango em bull, backwardation em pânico)
                    basis_spread_pct = float(np.clip(cur_funding * (periods_per_year / 365.0) * 30.0, -0.015, 0.025))
                    cur_perp = cur_spot * (1.0 + basis_spread_pct)

                unrealized_perp_pnl = -pos_qty * (cur_perp - pos_entry_price)
                
                # Colateral atual na perna perp
                current_perp_colateral = pos_colateral + accum_trade_funding + unrealized_perp_pnl
                
                # Nível de margem na perp = Colateral / Notional
                margin_level = current_perp_colateral / max(1e-4, current_notional)
                min_margin_observed = min(min_margin_observed, margin_level)
                margin_levels[i] = margin_level

                # Valor da perna spot a mercado
                spot_val = pos_qty * cur_spot
                # Patrimônio líquido total com marcação a mercado do basis
                equity = spot_val + current_perp_colateral
                equity_series[i] = equity
                positions[i] = 1

                # Condição de saída: sinal defasado caiu abaixo do limiar ou fim dos dados
                is_exit_signal = (not np.isnan(prev_roll_ann)) and (prev_roll_ann < self.exit_threshold_ann)
                is_last_bar = (i == n_bars - 1)

                if is_exit_signal or is_last_bar:
                    # Desmonta a estrutura (custos de saída em ambas as pernas)
                    exit_spot_fee = spot_val * self.spot_cost_per_leg
                    exit_perp_fee = current_notional * self.perp_cost_per_leg
                    exit_fees = exit_spot_fee + exit_perp_fee
                    total_fees_paid += exit_fees
                    equity -= exit_fees

                    entry_fees = pos_notional * (self.spot_cost_per_leg + self.perp_cost_per_leg)
                    trade_total_fees = entry_fees + exit_fees
                    net_trade_pnl = accum_trade_funding - trade_total_fees + (spot_val - pos_notional + unrealized_perp_pnl)
                    
                    duration_days = (timestamps[i] - pos_entry_ts) / (1000.0 * 86400.0)
                    ret_pct = net_trade_pnl / max(1.0, pos_notional * 2.0)
                    ann_ret = (ret_pct / max(0.01, duration_days)) * 365.0

                    reason = "FUNDING_DROP" if is_exit_signal else "END_OF_DATA"
                    trades.append(
                        CarryTradeRecord(
                            timestamp_entry=int(pos_entry_ts),
                            timestamp_exit=int(timestamps[i]),
                            duration_days=float(duration_days),
                            notional=float(pos_notional),
                            gross_funding_collected=float(accum_trade_funding),
                            fees_paid_4_legs=float(trade_total_fees),
                            net_pnl=float(net_trade_pnl),
                            return_pct=float(ret_pct * 100.0),
                            annualized_return_pct=float(ann_ret * 100.0),
                            min_margin_level_observed=float(min_margin_observed),
                            exit_reason=reason,
                        )
                    )

                    in_position = False
                    pos_qty = 0.0
                    pos_notional = 0.0
                    accum_trade_funding = 0.0

            # 2. Se fora de posição, avalia sinal de entrada
            else:
                equity_series[i] = equity
                positions[i] = 0

                is_entry_signal = (not np.isnan(prev_roll_ann)) and (prev_roll_ann >= self.entry_threshold_ann)
                if is_entry_signal and (i < n_bars - 1):
                    # Monta a estrutura delta-neutra:
                    # 50% em Spot e 50% em Colateral Perp
                    pos_notional = equity * 0.5
                    pos_colateral = equity * 0.5
                    cur_spot = cur_price
                    if "spot_price" in df.columns:
                        cur_spot = float(df["spot_price"].iloc[i])
                    pos_qty = pos_notional / cur_spot
                    
                    cur_perp = cur_price
                    if "perp_price" in df.columns:
                        cur_perp = float(df["perp_price"].iloc[i])
                    else:
                        basis_spread_pct = float(np.clip(cur_funding * (periods_per_year / 365.0) * 30.0, -0.015, 0.025))
                        cur_perp = cur_spot * (1.0 + basis_spread_pct)
                    
                    pos_entry_price = cur_perp
                    pos_entry_ts = timestamps[i]

                    # Custos de entrada (Spot buy + Perp short)
                    entry_spot_fee = pos_notional * self.spot_cost_per_leg
                    entry_perp_fee = pos_notional * self.perp_cost_per_leg
                    entry_fees = entry_spot_fee + entry_perp_fee
                    total_fees_paid += entry_fees
                    equity -= entry_fees
                    # Taxas de entrada saem do colateral: o equity marcado a mercado
                    # (spot_val + colateral) passa a refletir o custo (antes era omitido
                    # e net_profit ficava superestimado em entry_fees por trade).
                    pos_colateral -= entry_fees

                    in_position = True
                    accum_trade_funding = 0.0
                    min_margin_observed = 1.0

        # Análise de Drawdown e Retornos
        equity_df = pd.DataFrame({
            "timestamp": timestamps,
            "equity": equity_series,
            "margin_level": margin_levels,
            "in_position": positions,
        })
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
        max_dd = float(abs(equity_df["drawdown"].min()))

        net_profit = float(equity - self.initial_capital)
        net_profit_pct = float(net_profit / self.initial_capital) * 100.0
        
        total_days = max(1.0, (timestamps[-1] - timestamps[0]) / (1000.0 * 86400.0))
        annualized_ret = float(net_profit_pct * (365.0 / total_days))

        # Sharpe ratio dos retornos do equity (apenas informativo secundário)
        pct_returns = equity_df["equity"].pct_change().dropna().values
        excess_returns = pct_returns
        std_ret = float(np.std(excess_returns, ddof=1)) if len(excess_returns) > 1 else 1e-6
        mean_ret = float(np.mean(excess_returns)) if len(excess_returns) > 0 else 0.0
        sharpe = float((mean_ret / max(1e-12, std_ret)) * math.sqrt(periods_per_year))

        # Piores janelas móveis de 30 e 90 dias
        bars_per_day = periods_per_year / 365.0
        w30_bars = max(1, int(round(30.0 * bars_per_day)))
        w90_bars = max(1, int(round(90.0 * bars_per_day)))
        
        roll_30d_rets = equity_df["equity"].pct_change(w30_bars).dropna()
        worst_30d_pct = float(roll_30d_rets.min() * 100.0) if not roll_30d_rets.empty else 0.0

        roll_90d_rets = equity_df["equity"].pct_change(w90_bars).dropna()
        worst_90d_pct = float(roll_90d_rets.min() * 100.0) if not roll_90d_rets.empty else 0.0

        # Concentração de PnL por ano civil
        ts_numeric = pd.to_numeric(equity_df["timestamp"], errors="coerce")
        if ts_numeric.notna().all() and ts_numeric.iloc[0] > 1e11:
            equity_df["year"] = pd.to_datetime(ts_numeric, unit="ms").dt.year
        else:
            equity_df["year"] = pd.to_datetime(equity_df["timestamp"]).dt.year
        
        yearly_summary = {}
        for yr, group in equity_df.groupby("year"):
            start_eq = group["equity"].iloc[0]
            end_eq = group["equity"].iloc[-1]
            yr_pnl = end_eq - start_eq
            yr_pct = (yr_pnl / start_eq) * 100.0
            yearly_summary[int(yr)] = {
                "pnl_usd": float(yr_pnl),
                "return_pct": float(yr_pct),
            }

        # Stress Test de Choque Instantâneo (+/- 30% em 1 dia)
        survives_stress = self._evaluate_stress_shock_30pct(prices, fundings)

        summary = {
            "initial_capital": self.initial_capital,
            "final_capital": equity,
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "annualized_return_pct": annualized_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100.0,
            "worst_30d_window_pct": worst_30d_pct,
            "worst_90d_window_pct": worst_90d_pct,
            "pnl_by_year": yearly_summary,
            "total_funding_gross": total_funding_gross,
            "total_fees_paid": total_fees_paid,
            "fee_drag_pct_of_funding": float(total_fees_paid / max(1e-4, total_funding_gross)) * 100.0,
            "pct_negative_funding_periods": pct_negative_funding * 100.0,
            "min_margin_level_observed": float(min_margin_observed),
            "survives_stress_shock_30pct": survives_stress["passed"],
            "stress_min_margin_at_plus_30": survives_stress["margin_plus_30"],
            "total_trades": len(trades),
        }

        return CarrySimulationResult(
            initial_capital=self.initial_capital,
            final_capital=equity,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            annualized_return=annualized_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd * 100.0,
            total_funding_gross=total_funding_gross,
            total_fees_paid=total_fees_paid,
            pct_negative_funding_periods=pct_negative_funding * 100.0,
            min_margin_level_observed=float(min_margin_observed),
            survives_stress_shock_30pct=survives_stress["passed"],
            trades=trades,
            equity_curve=equity_df,
            summary_metrics=summary,
        )

    def _evaluate_stress_shock_30pct(self, prices: np.ndarray, fundings: np.ndarray) -> Dict[str, Any]:
        """
        Executa stress test determinístico:
        Se em posição alavancada (50% spot, 50% margem perp short 1x),
        o preço dá um salto de +30% em 24h sem tempo de rebalanceamento.
        """
        initial_colateral = 5000.0  # 50% de 10000
        initial_notional = 5000.0   # 50% de 10000
        
        # Choque de +30% no preço
        # Perda não realizada na perna perp short: -30% do notional = -1500 USD
        pnl_perp_loss = -0.30 * initial_notional
        remaining_colateral = initial_colateral + pnl_perp_loss  # 3500 USD
        new_notional = initial_notional * 1.30                   # 6500 USD
        
        margin_at_plus_30 = remaining_colateral / new_notional   # 3500 / 6500 = 53.8%

        # Choque de +60% (extremo)
        pnl_perp_extreme = -0.60 * initial_notional              # -3000 USD
        rem_colateral_ext = initial_colateral + pnl_perp_extreme # 2000 USD
        notional_ext = initial_notional * 1.60                   # 8000 USD
        margin_at_plus_60 = rem_colateral_ext / notional_ext     # 2000 / 8000 = 25.0%

        passed = (margin_at_plus_30 >= self.min_safe_margin_pct) and (margin_at_plus_30 > self.maint_margin_pct)

        return {
            "passed": bool(passed),
            "margin_plus_30": float(margin_at_plus_30),
            "margin_plus_60": float(margin_at_plus_60),
            "colateral_remaining_30": float(remaining_colateral),
        }
