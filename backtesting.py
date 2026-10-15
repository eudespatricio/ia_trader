"""
Módulo de Backtesting Institucional de Alta Precisão para o IA Trader (Fase 3 & Fase 5).
Simula operações em derivativos futuros de criptomoedas com rigor metodológico:
1. Taxas Reais Bitget (Taker ~0.06% = 0.0006 na entrada e na saída).
2. Slippage realista (default 0.02% = 0.0002).
3. Funding Rate contínuo a cada 8h (default 0.01% = 0.0001).
4. Ausência Estrita de Lookahead Bias na Execução:
   - Sinais gerados no fechamento da vela t são executados estritamente na ABERTURA da vela t+1.
5. Tratamento Intra-Barra Conservador (Mandatório):
   - A própria vela de entrada (i+1) é avaliada para SL/TP após a abertura.
   - Em caso de colisão intra-barra (ambos SL e TP tocados na mesma vela), assume-se estritamente que o SL ocorreu primeiro.
6. Gestão de Risco ATR & Trailing Stop:
   - Take Profit (TP) e Stop Loss (SL) calculados dinamicamente via múltiplos de ATR.
   - Suporte a Trailing Stop móvel baseado em múltiplos de ATR.
7. Métricas Completas:
   - Win Rate, Profit Factor, Sharpe Ratio anualizado, Max Drawdown, Lucro Líquido ($ e %) e F1-score.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ia_trader.backtesting")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [backtesting] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TradeRecord:
    """Registro detalhado de uma operação executada no backtest."""
    trade_id: int
    direction: str              # 'LONG' ou 'SHORT'
    entry_idx: int
    entry_timestamp: int
    entry_price: float
    exit_idx: int
    exit_timestamp: int
    exit_price: float
    size: float
    pnl_gross: float
    fee_total: float
    funding_total: float
    pnl_net: float
    return_pct: float
    exit_reason: str            # 'TP', 'SL', 'TRAILING_SL', 'END_OF_DATA', 'SIGNAL_REVERSAL', 'HORIZON_TIMEOUT'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Consolidado de métricas e performance histórica do backtest."""
    initial_capital: float
    final_capital: float
    net_profit: float
    net_profit_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    f1_score: float
    trades: List[TradeRecord]
    equity_curve: List[float]
    gross_profit: float = 0.0
    total_fees: float = 0.0
    exit_reasons: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["trades"] = [t.to_dict() for t in self.trades]
        return res


def _timeframe_hours(tf: str) -> float:
    """Converte '15m', '1h', '4H', '1d' em horas."""
    t = str(tf).strip().lower()
    unit = t[-1]
    mult = {"m": 1 / 60.0, "h": 1.0, "d": 24.0, "w": 168.0}
    if unit not in mult or not t[:-1].isdigit() or int(t[:-1]) <= 0:
        raise ValueError(f"Timeframe inválido: {tf!r}")
    return int(t[:-1]) * mult[unit]


class BacktestEngine:
    """
    Motor de backtest institucional com modelagem estrita de microestrutura.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0006,        # Taker Bitget: 0.06%
        slippage: float = 0.0002,        # Slippage: 0.02%
        funding_rate_8h: float = 0.0001, # Funding Rate a cada 8h: 0.01%
        leverage: float = 1.0,           # Alavancagem (Padrão 1x = Não alavancado)
        tp_atr_multiple: float = 2.0,    # Múltiplo de Take Profit
        sl_atr_multiple: float = 1.5,    # Múltiplo de Stop Loss
        trailing_stop: bool = False,     # Trailing Stop opcional
        trailing_atr_multiple: float = 1.5,
        risk_per_trade: Optional[float] = 0.01,  # Risco fixo institucional (ex: 1% do capital arriscado no SL). Se None, usa 95% do capital
        allow_reversal: bool = False,            # Se False, não fecha por sinal contrário; sai apenas por TP, SL, Trailing ou timeout
        timeframe: str = "1h",                  # Timeframe das barras (funding 8h e anualização do Sharpe)
        max_holding_bars: Optional[int] = None,  # Limite máximo de barras mantidas antes de saída forçada ("HORIZON_TIMEOUT")
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.funding_rate_8h = funding_rate_8h
        self.leverage = leverage
        self.tp_atr_multiple = tp_atr_multiple
        self.sl_atr_multiple = sl_atr_multiple
        self.trailing_stop = trailing_stop
        self.trailing_atr_multiple = trailing_atr_multiple
        self.risk_per_trade = risk_per_trade
        self.allow_reversal = allow_reversal
        self.max_holding_bars = max_holding_bars
        self.timeframe = timeframe
        tf_hours = _timeframe_hours(timeframe)
        self.bars_per_funding = max(1, int(round(8.0 / tf_hours)))
        self.periods_per_year = 8760.0 / tf_hours

    def _update_trailing(self, active_trade: Dict[str, Any], high_p: float, low_p: float, atr: float) -> None:
        """V9: atualiza peak/SL de trailing com dados de uma barra JÁ testada; vale a partir da próxima."""
        if not self.trailing_stop:
            return
        if active_trade["direction"] == "LONG":
            if high_p > active_trade["peak_price"]:
                active_trade["peak_price"] = high_p
                active_trade["sl"] = max(active_trade["sl"], high_p - self.trailing_atr_multiple * atr)
        else:
            if low_p < active_trade["peak_price"]:
                active_trade["peak_price"] = low_p
                active_trade["sl"] = min(active_trade["sl"], low_p + self.trailing_atr_multiple * atr)

    def _check_bar_exit(
        self,
        active_trade: Dict[str, Any],
        open_p: float,
        high_p: float,
        low_p: float,
        next_atr: float,
    ) -> Tuple[bool, float, str]:
        """
        Avalia saída por SL ou TP dentro da barra com tratamento conservador.
        Se ambos os limiares forem tocados na mesma vela (colisão intra-barra),
        assume-se estritamente que o STOP LOSS ocorreu primeiro.
        """
        direction = active_trade["direction"]
        entry_price = active_trade["entry_price"]
        sl = active_trade["sl"]
        tp = active_trade["tp"]

        # 2. Avaliação de Colisão Intra-Barra e Saídas
        if direction == "LONG":
            hit_sl = low_p <= sl
            hit_tp = high_p >= tp
            if hit_sl and hit_tp:
                # Conservador: SL primeiro
                exit_price = min(open_p, sl) * (1.0 - self.slippage)
                reason = "TRAILING_SL" if (self.trailing_stop and sl > entry_price) else "SL"
                return True, exit_price, reason
            elif hit_sl:
                exit_price = min(open_p, sl) * (1.0 - self.slippage)
                reason = "TRAILING_SL" if (self.trailing_stop and sl > entry_price) else "SL"
                return True, exit_price, reason
            elif hit_tp:
                exit_price = tp * (1.0 - self.slippage)
                return True, exit_price, "TP"

        elif direction == "SHORT":
            hit_sl = high_p >= sl
            hit_tp = low_p <= tp
            if hit_sl and hit_tp:
                # Conservador: SL primeiro
                exit_price = max(open_p, sl) * (1.0 + self.slippage)
                reason = "TRAILING_SL" if (self.trailing_stop and sl < entry_price) else "SL"
                return True, exit_price, reason
            elif hit_sl:
                exit_price = max(open_p, sl) * (1.0 + self.slippage)
                reason = "TRAILING_SL" if (self.trailing_stop and sl < entry_price) else "SL"
                return True, exit_price, reason
            elif hit_tp:
                exit_price = tp * (1.0 + self.slippage)
                return True, exit_price, "TP"

        self._update_trailing(active_trade, high_p, low_p, next_atr)
        return False, 0.0, ""

    def run(
        self,
        df: pd.DataFrame,
        signals: Union[pd.Series, np.ndarray, List[int]],
        atr_col: str = "atr_14",
    ) -> BacktestResult:
        """
        Executa simulação sequencial de negociação:
        - `signals`: Série temporal de sinais gerados ao final de cada vela.
          1 = BUY (LONG), 2 (ou -1) = SELL (SHORT), 0 = HOLD.
        - Execução: estritamente na ABERTURA da vela seguinte (i + 1).
        - Rigor conservador: a própria vela de entrada (i + 1) checa SL/TP imediatamente.
        """
        if len(df) == 0:
            raise ValueError("DataFrame de velas fornecido ao backtest está vazio.")

        signals_arr = np.asarray(signals)
        if len(signals_arr) != len(df):
            raise ValueError(f"Tamanho dos sinais ({len(signals_arr)}) difere do DataFrame ({len(df)}).")

        n_candles = len(df)
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values

        if atr_col in df.columns:
            atrs = df[atr_col].values
        else:
            atrs = closes * 0.01

        capital = float(self.initial_capital)
        equity_curve: List[float] = [capital]
        trades: List[TradeRecord] = []

        active_trade: Optional[Dict[str, Any]] = None
        trade_counter = 0

        for i in range(n_candles - 1):
            curr_signal = int(signals_arr[i])
            next_open = opens[i + 1]
            next_high = highs[i + 1]
            next_low = lows[i + 1]
            next_close = closes[i + 1]
            next_ts = timestamps[i + 1]
            next_atr = atrs[i] if atrs[i] > 0 else (next_open * 0.01)

            # -------------------------------------------------------------
            # 1. TRADE EXISTENTE: Verificação na vela i+1
            # -------------------------------------------------------------
            closed_intrabar = False
            if active_trade is not None:
                # Contabiliza funding rate a cada 8h (bars_per_funding barras)
                active_trade["bars_held"] += 1
                if active_trade["bars_held"] % self.bars_per_funding == 0:
                    funding_fee = (active_trade["entry_price"] * active_trade["size"]) * self.funding_rate_8h
                    # funding>0: long paga (custo), short recebe (crédito); inverso se funding<0
                    if active_trade["direction"] == "LONG":
                        active_trade["funding_total"] += funding_fee
                    else:
                        active_trade["funding_total"] -= funding_fee

                # Checa se há reversão de sinal explícita na abertura da vela (somente se allow_reversal=True)
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                # 1.1 Horizonte máximo de barras (timeout)
                if self.max_holding_bars is not None and active_trade["bars_held"] >= self.max_holding_bars:
                    exit_price = next_open * (1.0 - self.slippage if active_trade["direction"] == "LONG" else 1.0 + self.slippage)
                    exit_reason = "HORIZON_TIMEOUT"
                    trade_closed = True

                # 1.2 Reversão de sinal (se permitida)
                if not trade_closed and self.allow_reversal:
                    reversal = False
                    if active_trade["direction"] == "LONG" and curr_signal in [2, -1]:
                        reversal = True
                    elif active_trade["direction"] == "SHORT" and curr_signal == 1:
                        reversal = True

                    if reversal:
                        exit_price = next_open * (1.0 - self.slippage if active_trade["direction"] == "LONG" else 1.0 + self.slippage)
                        exit_reason = "SIGNAL_REVERSAL"
                        trade_closed = True

                # 1.3 Avalia SL e TP intra-barra caso não tenha saído
                if not trade_closed:
                    trade_closed, exit_price, exit_reason = self._check_bar_exit(
                        active_trade, next_open, next_high, next_low, next_atr
                    )

                if trade_closed:
                    size = active_trade["size"]
                    direction = active_trade["direction"]
                    entry_price = active_trade["entry_price"]

                    exit_fee = (exit_price * size) * self.fee_rate
                    total_fees = active_trade["entry_fee"] + exit_fee
                    funding_total = active_trade["funding_total"]

                    pnl_gross = (exit_price - entry_price) * size if direction == "LONG" else (entry_price - exit_price) * size
                    pnl_net = pnl_gross - total_fees - funding_total
                    ret_pct = pnl_net / (active_trade["allocated_margin"] + 1e-8)

                    capital += pnl_net
                    capital = max(0.0, capital)

                    trade_counter += 1
                    trades.append(TradeRecord(
                        trade_id=trade_counter,
                        direction=direction,
                        entry_idx=active_trade["entry_idx"],
                        entry_timestamp=active_trade["entry_ts"],
                        entry_price=entry_price,
                        exit_idx=i + 1,
                        exit_timestamp=next_ts,
                        exit_price=exit_price,
                        size=size,
                        pnl_gross=pnl_gross,
                        fee_total=total_fees,
                        funding_total=funding_total,
                        pnl_net=pnl_net,
                        return_pct=ret_pct,
                        exit_reason=exit_reason,
                    ))
                    # A10a: saída intrabarra (SL/TP/trailing) ocorre APÓS o open; não reentrar no open de i+1
                    closed_intrabar = exit_reason in ("SL", "TP", "TRAILING_SL")
                    active_trade = None

            # -------------------------------------------------------------
            # 2. ENTRADA EM NOVO TRADE (Executada estritamente na vela i+1)
            # -------------------------------------------------------------
            if active_trade is None and capital > 10.0 and not closed_intrabar:
                if curr_signal in [1, 2, -1]:
                    direction = "LONG" if curr_signal == 1 else "SHORT"

                    # Preço de execução na abertura de i+1 com slippage
                    if direction == "LONG":
                        entry_price = next_open * (1.0 + self.slippage)
                        tp = entry_price + (self.tp_atr_multiple * next_atr)
                        sl = entry_price - (self.sl_atr_multiple * next_atr)
                    else:
                        entry_price = next_open * (1.0 - self.slippage)
                        tp = entry_price - (self.tp_atr_multiple * next_atr)
                        sl = entry_price + (self.sl_atr_multiple * next_atr)

                    sl_dist = abs(entry_price - sl)

                    # Dimensionamento de posição: Risco Fixo Institucional ou Margem Fixa
                    if self.risk_per_trade is not None and self.risk_per_trade > 0 and sl_dist > 1e-8:
                        dollar_risk = capital * self.risk_per_trade
                        desired_size = dollar_risk / sl_dist
                        max_notional = capital * self.leverage
                        max_size = max_notional / entry_price
                        size = min(desired_size, max_size)
                        allocated_margin = (size * entry_price) / self.leverage
                    else:
                        allocated_margin = capital * 0.95
                        position_notional = allocated_margin * self.leverage
                        size = position_notional / entry_price

                    entry_fee = (entry_price * size) * self.fee_rate

                    active_trade = {
                        "direction": direction,
                        "entry_idx": i + 1,
                        "entry_ts": next_ts,
                        "entry_price": entry_price,
                        "size": size,
                        "allocated_margin": allocated_margin,
                        "entry_fee": entry_fee,
                        "funding_total": 0.0,
                        "bars_held": 0,
                        "tp": tp,
                        "sl": sl,
                        "peak_price": entry_price,
                    }

                    # =============================================================
                    # REQUISITO MANDATÓRIO: A VELA DE ENTRADA (i+1) CHECA SL/TP!
                    # =============================================================
                    hit_exit, exit_price, exit_reason = self._check_bar_exit(
                        active_trade, next_open, next_high, next_low, next_atr
                    )

                    if hit_exit:
                        exit_fee = (exit_price * size) * self.fee_rate
                        total_fees = entry_fee + exit_fee
                        pnl_gross = (exit_price - entry_price) * size if direction == "LONG" else (entry_price - exit_price) * size
                        pnl_net = pnl_gross - total_fees
                        ret_pct = pnl_net / (allocated_margin + 1e-8)

                        capital += pnl_net
                        capital = max(0.0, capital)

                        trade_counter += 1
                        trades.append(TradeRecord(
                            trade_id=trade_counter,
                            direction=direction,
                            entry_idx=i + 1,
                            entry_timestamp=next_ts,
                            entry_price=entry_price,
                            exit_idx=i + 1,
                            exit_timestamp=next_ts,
                            exit_price=exit_price,
                            size=size,
                            pnl_gross=pnl_gross,
                            fee_total=total_fees,
                            funding_total=0.0,
                            pnl_net=pnl_net,
                            return_pct=ret_pct,
                            exit_reason=exit_reason,
                        ))
                        active_trade = None

            equity_curve.append(capital)

        # Encerramento forçado de trade residual no fim da série
        if active_trade is not None:
            direction = active_trade["direction"]
            entry_price = active_trade["entry_price"]
            size = active_trade["size"]
            last_close = closes[-1]
            exit_price = last_close * (1.0 - self.slippage if direction == "LONG" else 1.0 + self.slippage)
            exit_fee = (exit_price * size) * self.fee_rate
            total_fees = active_trade["entry_fee"] + exit_fee
            pnl_gross = (exit_price - entry_price) * size if direction == "LONG" else (entry_price - exit_price) * size
            pnl_net = pnl_gross - total_fees - active_trade["funding_total"]

            capital += pnl_net
            capital = max(0.0, capital)

            trade_counter += 1
            trades.append(TradeRecord(
                trade_id=trade_counter,
                direction=direction,
                entry_idx=active_trade["entry_idx"],
                entry_timestamp=active_trade["entry_ts"],
                entry_price=entry_price,
                exit_idx=n_candles - 1,
                exit_timestamp=timestamps[-1],
                exit_price=exit_price,
                size=size,
                pnl_gross=pnl_gross,
                fee_total=total_fees,
                funding_total=active_trade["funding_total"],
                pnl_net=pnl_net,
                return_pct=pnl_net / (active_trade["allocated_margin"] + 1e-8),
                exit_reason="END_OF_DATA",
            ))
            equity_curve[-1] = capital

        # -------------------------------------------------------------
        # 3. CONSOLIDAÇÃO DE MÉTRICAS ESTATÍSTICAS
        # -------------------------------------------------------------
        trades_count = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl_net > 0)
        losing_trades = sum(1 for t in trades if t.pnl_net <= 0)
        win_rate = (winning_trades / trades_count) if trades_count > 0 else 0.0

        gross_profit = sum(t.pnl_gross for t in trades)
        total_fees = sum(t.fee_total + t.funding_total for t in trades)
        exit_reasons: Dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        gross_profits = sum(t.pnl_net for t in trades if t.pnl_net > 0)
        gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (99.0 if gross_profits > 0 else 0.0)

        net_profit = capital - self.initial_capital
        net_profit_pct = (net_profit / self.initial_capital) * 100.0

        eq_series = pd.Series(equity_curve)
        pct_returns = eq_series.pct_change().dropna()
        if len(pct_returns) > 1 and pct_returns.std() > 0:
            sharpe_ratio = float(np.sqrt(self.periods_per_year) * (pct_returns.mean() / pct_returns.std()))
        else:
            sharpe_ratio = 0.0

        running_max = eq_series.cummax()
        drawdowns = (eq_series - running_max) / (running_max + 1e-8)
        max_drawdown_pct = float(abs(drawdowns.min())) * 100.0 if not drawdowns.empty else 0.0

        # A10c: a fórmula anterior era degenerada (recall = wins/(2*trades-wins) usava a própria
        # contagem de trades). Sem classes verdadeiras no nível de trade, F1 não é definido aqui;
        # o campo é mantido por compatibilidade e fixado em 0.0 (use model_trainer para F1 do classificador).
        f1 = 0.0

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=capital,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            trades_count=trades_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            f1_score=f1,
            trades=trades,
            equity_curve=equity_curve,
            gross_profit=gross_profit,
            total_fees=total_fees,
            exit_reasons=exit_reasons,
        )

        return result
