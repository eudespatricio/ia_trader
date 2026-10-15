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
7. Modo `pilot_parity=True` (ver `_PilotSimulator`): reproduz o engine do piloto (smart_trade.py é a fonte de
   verdade): escada de saída TP1 40%→BE, TP2 30%→trailing, runner 30% em TP3/trailing; lote=margem;
   alavancagem por leverage_policy; liquidação; maker/taker por saída; margem agregada e teto de posições.
8. Métricas Completas:
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
    # --- modo pilot_parity (defaults mantêm compatibilidade do modo legado) ---
    # exit_reason em pilot_parity = motivo da ÚLTIMA perna (STOP_LOSS, BREAKEVEN_STOP, TRAILING_STOP,
    # TAKE_PROFIT_3, LIQUIDATION, HORIZON_TIMEOUT, END_OF_DATA); exit_price = preço médio das pernas.
    symbol: str = ""
    leverage: float = 1.0
    margin: float = 0.0
    legs: int = 1

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
        max_holding_bars: Optional[int] = None,  # Limite máximo de barras mantidas antes de saída forçada ("HORIZON_TIMEOUT").
                                                 # pilot_parity: None = horizonte do perfil (STRATEGY_PROFILES, 48); 0 = sem timeout
        *,
        pilot_parity: bool = False,              # True = reproduz o engine do piloto (smart_trade.PaperTradingEngine)
        strategy_type: str = "DIRECTIONAL",      # perfil de STRATEGY_PROFILES (DIRECTIONAL | SCALP | GOLDEN_POT)
        entry_mode: Optional[str] = None,        # "signal_close" (piloto: entra em close(t)) | "next_open" (usa open[t+1] bruto)
        lot_margin: float = 100.0,               # lote = MARGEM em USD (smart_trade.PAPER_LOT_USD)
        max_leverage: float = 5.0,               # teto (leverage_policy.MAX_LEVERAGE)
        use_leverage_policy: bool = True,        # com `confidences`, alavancagem = decide_leverage(); senão `leverage`
        maker_fee: float = 0.0002,               # TP1/TP2/TP3 (ordens limite)
        maker_slippage: float = 0.0001,          # saída maker TP3 (smart_trade.MAKER_SLIPPAGE_PCT); `slippage` = mercado
        max_margin_utilization_pct: float = 0.85,
        max_concurrent_positions: Optional[int] = 5,
        daily_quota: Optional[int] = None,       # None = padrão do piloto (DIRECTIONAL sem cota; SCALP 5; GOLDEN_POT 3)
        cooldown_bars: Optional[int] = None,     # None = padrão (DIRECTIONAL 0; SCALP 4; GOLDEN_POT 16 barras)
        trailing_activation: str = "tp2",        # pilot_parity: "tp2" (padrão do piloto) | "tp1" (trailing liga já no TP1; experimento offline)
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

        # ---- modo pilot_parity ----
        self.pilot_parity = bool(pilot_parity)
        if self.pilot_parity:
            from smart_trade import STRATEGY_PROFILES
            if strategy_type not in STRATEGY_PROFILES:
                raise ValueError(f"strategy_type inválido: {strategy_type!r}")
            if entry_mode not in (None, "signal_close", "next_open"):
                raise ValueError(f"entry_mode inválido: {entry_mode!r}")
        if trailing_activation not in ("tp2", "tp1"):
            raise ValueError(f"trailing_activation inválido: {trailing_activation!r}")
        self.trailing_activation = trailing_activation
        self.strategy_type = strategy_type
        self.entry_mode = entry_mode or "signal_close"
        self.lot_margin = lot_margin
        self.max_leverage = max_leverage
        self.use_leverage_policy = use_leverage_policy
        self.maker_fee = maker_fee
        self.maker_slippage = maker_slippage
        self.max_margin_utilization_pct = max_margin_utilization_pct
        self.max_concurrent_positions = max_concurrent_positions
        _defaults_quota = {"DIRECTIONAL": None, "SCALP": 5, "GOLDEN_POT": 3}
        _defaults_cd = {"DIRECTIONAL": 0, "SCALP": 4, "GOLDEN_POT": 16}
        self.daily_quota = daily_quota if daily_quota is not None else _defaults_quota.get(strategy_type)
        self.cooldown_bars = cooldown_bars if cooldown_bars is not None else _defaults_cd.get(strategy_type, 0)

    def run_portfolio(
        self,
        data: Dict[str, pd.DataFrame],
        signals: Dict[str, Union[pd.Series, np.ndarray, List[int]]],
        confidences: Optional[Dict[str, Union[pd.Series, np.ndarray, List[float]]]] = None,
        atr_col: str = "atr_14",
        funding: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> BacktestResult:
        """Multi-ativo com SALDO ÚNICO, margem agregada e teto de posições simultâneas (só `pilot_parity=True`).

        Ordem por timestamp: (1) ticks das posições abertas com a barra FECHADA; (2) sinais -> entradas
        (como o daemon: update_active_positions -> evaluate_*). `confidences[s][i]` = confiança do comitê no
        sinal da barra i (alimenta leverage_policy.decide_leverage); sem ela, usa `leverage` fixa.
        `funding[s]` = (timestamps_ms, taxas por 8h) REAIS; liquidação em 00/08/16h UTC, taxa mais próxima (±10min),
        ausente -> `funding_rate_8h` (fallback, como o piloto). Entrada em close(t)/open(t+1): sinal da última barra é ignorado.
        """
        if not self.pilot_parity:
            raise ValueError("run_portfolio exige pilot_parity=True (o modo legado é single-asset).")
        return _PilotSimulator(self).run(data, {k: np.asarray(v) for k, v in signals.items()}, confidences, atr_col, funding)

    def _pilot_result(self, trades: List[TradeRecord], equity_curve: List[float], capital: float) -> BacktestResult:
        """Consolida métricas do modo pilot_parity (mesma definição do modo legado)."""
        trades_count = len(trades)
        winning = sum(1 for t in trades if t.pnl_net > 0)
        losing = sum(1 for t in trades if t.pnl_net <= 0)
        exit_reasons: Dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        gp = sum(t.pnl_net for t in trades if t.pnl_net > 0)
        gl = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))
        profit_factor = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
        net_profit = capital - self.initial_capital
        eq = pd.Series(equity_curve, dtype=float)
        rets = eq.pct_change().dropna()
        sharpe = float(np.sqrt(self.periods_per_year) * (rets.mean() / rets.std())) if (len(rets) > 1 and rets.std() > 0) else 0.0
        dd = (eq - eq.cummax()) / (eq.cummax() + 1e-8)
        return BacktestResult(
            initial_capital=self.initial_capital, final_capital=capital, net_profit=net_profit,
            net_profit_pct=net_profit / self.initial_capital * 100.0,
            win_rate=(winning / trades_count) if trades_count else 0.0, profit_factor=profit_factor,
            sharpe_ratio=sharpe, max_drawdown_pct=float(abs(dd.min())) * 100.0 if not dd.empty else 0.0,
            trades_count=trades_count, winning_trades=winning, losing_trades=losing, f1_score=0.0,
            trades=trades, equity_curve=equity_curve, gross_profit=sum(t.pnl_gross for t in trades),
            total_fees=sum(t.fee_total + t.funding_total for t in trades), exit_reasons=exit_reasons,
        )

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
        confidences: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        funding: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> BacktestResult:
        """
        Executa simulação sequencial de negociação:
        - `signals`: Série temporal de sinais gerados ao final de cada vela.
          1 = BUY (LONG), 2 (ou -1) = SELL (SHORT), 0 = HOLD.
        - Execução: estritamente na ABERTURA da vela seguinte (i + 1).
        - Rigor conservador: a própria vela de entrada (i + 1) checa SL/TP imediatamente.
        """
        if self.pilot_parity:
            return self.run_portfolio({"X": df}, {"X": signals}, None if confidences is None else {"X": confidences},
                                      atr_col=atr_col, funding=None if funding is None else {"X": funding})
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


# =============================================================================
# MODO `pilot_parity=True`: réplica do engine do piloto (smart_trade.PaperTradingEngine)
# =============================================================================
# Fonte de verdade da semântica = smart_trade.py (importado, NÃO duplicado: perfis, frações da ladder, colchão
# de breakeven, taxas por motivo de saída, slippage de mercado/maker, preço de liquidação, horizonte e intervalo
# de funding vêm de lá). Convenções (iguais ao piloto, ver smart_trade "PARIDADE BACKTEST x PILOTO"):
#  * decisão no CLOSE da vela fechada t; entrada a mercado em close(t) (== open(t+1) em cripto 24/7) + slippage
#    de mercado simétrico; entry_timestamp = ts(t+1); a vela t+1 é a 1ª pós-entrada e já testa stops/TPs;
#  * ordem por barra: funding (UTC 00/08/16h) -> timeout -> liquidação -> ladder (stop vigente ANTES dos TPs);
#  * ladder: TP1 40% (maker, sem slippage) -> SL=BE(+0.04%); TP2 30% (maker) -> ativa trailing; runner 30% sai em
#    TP3 (maker, slippage 0.01%) ou trailing (taker); trailing = mult x ATR de ENTRADA (congelado), alimentado só
#    por barras já fechadas e nunca afrouxa; stop gap-aware no stop pré-existente (min(sl, open) em LONG);
#  * taker (SL/BE/TRAILING/timeout/LIQUIDATION) com slippage de mercado; lote = MARGEM; alavancagem decidida por
#    leverage_policy.decide_leverage (confiança do sinal; máx 5x) e reduzida até SL ficar antes da liquidação;
#  * saldo único, margem agregada <= 85% do saldo, teto de posições simultâneas, cotas/cooldown opcionais.
# NÃO modelado (por design): RollbackGuard/circuit breaker (usa relógio de parede + DB do piloto, não é função
# das velas); `position_size` do SignalGenerator como teto de margem (o lote do piloto o domina: lote=100 <<).
class _PilotPos:
    """Estado de uma posição com ladder (espelha PaperPosition)."""
    __slots__ = (
        "sym", "side", "long", "sgn", "entry_idx", "entry_ts", "entry", "qty0", "qty", "notional", "margin", "lev",
        "sl", "tp1", "tp2", "tp3", "tp1_hit", "tp2_hit", "trail_active", "trail_stop", "trail_mult", "entry_atr",
        "trail_ref", "trail_bar", "liq", "gross", "fees", "funding", "fund_cursor", "exit_qty_px", "n_legs",
        "last_reason",
    )


class _PilotSimulator:
    """Simulador sequencial multi-ativo com saldo único (réplica do piloto). Uso interno de BacktestEngine."""

    def __init__(self, eng: "BacktestEngine"):
        import smart_trade as st
        import leverage_policy as lp
        self.st, self.lp = st, lp
        self.e = eng
        self.prof = st.STRATEGY_PROFILES[eng.strategy_type]
        self.meta = st.PaperTradingEngine._ENTRY_META[eng.strategy_type]
        self.bar_ms = int(round(_timeframe_hours(eng.timeframe) * 3_600_000))
        self.max_hold = self.prof.max_holding_bars if eng.max_holding_bars is None else int(eng.max_holding_bars)
        self.funding: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.balance = float(eng.initial_capital)

    # ---- funding real (UTC 00/08/16h), taxa do cache; fallback = funding_rate_8h ---------------------------
    def _rate_at(self, sym: str, settle_ms: int) -> float:
        series = self.funding.get(sym)
        if series is not None and len(series[0]):
            ts, rt = series
            j = int(np.searchsorted(ts, settle_ms))
            best, bd = None, None
            for c in (j - 1, j):
                if 0 <= c < len(ts):
                    d = abs(int(ts[c]) - settle_ms)
                    if d <= self.st.FUNDING_RATE_MATCH_TOLERANCE_MS and (bd is None or d < bd):
                        best, bd = float(rt[c]), d
            if best is not None:
                return best
        return float(self.e.funding_rate_8h)

    def _apply_funding(self, p: _PilotPos, now_ms: int, mark: float) -> None:
        iv = self.st.FUNDING_INTERVAL_MS
        settle = (int(p.fund_cursor) // iv + 1) * iv
        n = 0
        while settle <= now_ms and n < self.st.FUNDING_MAX_SETTLEMENTS_PER_TICK:
            amount = -p.sgn * self._rate_at(p.sym, settle) * p.qty * mark   # <0 = paga
            self.balance += amount
            p.funding -= amount          # funding_total = custo (positivo = pago), como no modo legado
            p.fund_cursor = float(settle)
            n += 1
            settle += iv
        if settle <= now_ms:
            p.fund_cursor = float(now_ms)

    # ---- pernas de saída ---------------------------------------------------------------------------
    def _leg(self, p: _PilotPos, level: float, qty: float, reason: str, partial: bool) -> float:
        """Realiza `qty` ao nível `level`; devolve o PnL líquido da perna. Parcial TP1/TP2: sem slippage (piloto)."""
        e = self.e
        maker = reason in self.st.MAKER_EXIT_REASONS
        if partial:
            px = level
        else:
            slip = e.maker_slippage if maker else e.slippage
            px = level * (1.0 - slip) if p.long else level * (1.0 + slip)
        fee = qty * px * (e.maker_fee if maker else e.fee_rate)
        gross = p.sgn * (px - p.entry) * qty
        p.gross += gross
        p.fees += fee
        p.exit_qty_px += px * qty
        p.qty -= qty
        p.n_legs += 1
        p.last_reason = reason
        return gross - fee

    def _ratchet(self, p: _PilotPos, bar_high: float, bar_low: float) -> None:
        """Incorpora extremos de uma barra FECHADA ao trailing. Nunca afrouxa."""
        dist = p.trail_mult * p.entry_atr
        if dist <= 0.0:
            return
        if p.long:
            p.trail_ref = max(p.trail_ref, bar_high)
            cand = p.trail_ref - dist
            if cand > p.trail_stop:
                p.trail_stop = cand
            if p.trail_stop > p.sl:
                p.sl = p.trail_stop
        else:
            p.trail_ref = bar_low if p.trail_ref <= 0.0 else min(p.trail_ref, bar_low)
            cand = p.trail_ref + dist
            if p.trail_stop <= 0.0 or cand < p.trail_stop:
                p.trail_stop = cand
            if p.trail_stop > 0.0 and (p.sl <= 0.0 or p.trail_stop < p.sl):
                p.sl = p.trail_stop

    def _stop_reason(self, p: _PilotPos) -> str:
        if not p.tp1_hit:
            return "STOP_LOSS"
        if p.trail_active and p.trail_stop > 0.0 and p.sl == p.trail_stop:
            return "TRAILING_STOP"
        return "BREAKEVEN_STOP"

    # ---- UMA barra FECHADA para uma posição ------------------------------------------------------------
    def process_bar(self, p: _PilotPos, ts: int, o: float, h: float, l: float, c: float
                    ) -> Optional[Tuple[str, float]]:
        """None se segue aberta; senão (motivo, preço de referência do fechamento do restante)."""
        st = self.st
        long, sgn = p.long, p.sgn
        self._apply_funding(p, int(ts), c)
        if self.max_hold > 0 and int(ts) >= int(p.entry_ts) + self.max_hold * self.bar_ms:
            return "HORIZON_TIMEOUT", o
        if p.liq > 0.0 and ((long and l <= p.liq and p.liq >= p.sl) or ((not long) and h >= p.liq and p.liq <= p.sl)):
            return "LIQUIDATION", p.liq
        if p.trail_active and p.trail_bar is not None:   # a barra anterior (fechada) alimenta o trailing
            self._ratchet(p, p.trail_bar[0], p.trail_bar[1])
            p.trail_bar = None

        def stop_hit() -> bool:
            return (l <= p.sl) if long else (h >= p.sl)

        def reached(level: float) -> bool:
            return level > 0.0 and ((h >= level) if long else (l <= level))

        def stop_close(pre_existing: bool) -> Tuple[str, float]:
            # stop não preenche fora do range da barra; gap de abertura só vale p/ o stop que já existia antes da barra
            if long:
                fill = min(p.sl, h)
                if pre_existing and o > 0.0:
                    fill = min(fill, o)
            else:
                fill = max(p.sl, l)
                if pre_existing and o > 0.0:
                    fill = max(fill, o)
            return self._stop_reason(p), fill

        if stop_hit():
            return stop_close(True)
        if not p.tp1_hit and reached(p.tp1):
            p.tp1_hit = True
            be = p.entry * (1.0 + sgn * st.BREAKEVEN_FEE_BUFFER_PCT)
            p.sl = max(p.sl, be) if long else min(p.sl, be)          # nunca afrouxa
            self.balance += self._leg(p, p.tp1, min(p.qty0 * st.EXIT_TP1_FRACTION, p.qty), "TAKE_PROFIT_1", True)
            if self.e.trailing_activation == "tp1" and p.trail_mult > 0.0 and p.entry_atr > 0.0:
                p.trail_active = True       # experimento: trailing liga já no TP1 (mult x ATR de entrada, nunca afrouxa)
                p.trail_ref = p.tp1
                p.trail_bar = None
                self._ratchet(p, p.tp1, p.tp1)
            if stop_hit():
                return stop_close(False)
        if p.tp1_hit and not p.tp2_hit and reached(p.tp2):
            p.tp2_hit = True
            self.balance += self._leg(p, p.tp2, min(p.qty0 * st.EXIT_TP2_FRACTION, p.qty), "TAKE_PROFIT_2", True)
            if p.trail_mult > 0.0 and p.entry_atr > 0.0:
                if p.trail_active:           # já ligado no TP1: só incorpora o extremo TP2 (fechado nesta barra)
                    self._ratchet(p, p.tp2, p.tp2)
                else:
                    p.trail_active = True
                    p.trail_ref = p.tp2
                    p.trail_bar = None
                    self._ratchet(p, p.tp2, p.tp2)
        if p.tp2_hit and reached(p.tp3):
            return "TAKE_PROFIT_3", p.tp3
        if p.trail_active:   # esta barra só vira stop quando a seguinte chegar
            p.trail_bar = (h, l)
        return None

    # ---- abertura -----------------------------------------------------------------------------------------
    def try_open(self, sym: str, idx: int, direction: int, entry_ts: int, sig_close: float, entry_raw: float,
                 atr: float, conf: Optional[float], regime_ok: Optional[bool], positions: Dict[str, _PilotPos],
                 state: Dict[str, Any]) -> Optional[_PilotPos]:
        e, st, lp = self.e, self.st, self.lp
        if sym in positions:
            return None
        if e.max_concurrent_positions and len(positions) >= e.max_concurrent_positions:
            return None
        if e.cooldown_bars and idx - state["last_entry_idx"].get(sym, -10 ** 9) < e.cooldown_bars:
            return None
        day = pd.Timestamp(int(entry_ts), unit="ms", tz="UTC").strftime("%Y-%m-%d")
        if e.daily_quota is not None and state["quota"].get(day, 0) >= e.daily_quota:
            return None
        long = direction == 1
        side = "LONG" if long else "SHORT"
        sgn = 1.0 if long else -1.0
        max_margin = self.balance * e.max_margin_utilization_pct
        margin_used = sum(q.margin for q in positions.values())
        free_margin = max(0.0, max_margin - margin_used)
        if conf is not None and e.use_leverage_policy:      # alavancagem decidida pelo Smart Trader
            atr_pct = atr / sig_close if sig_close > 0 else 0.0
            lev_req = lp.decide_leverage(float(conf), atr_pct=atr_pct, regime_alignment=regime_ok)
        else:
            lev_req = e.leverage
        lev_req = lp.clamp_leverage(lev_req, min(e.max_leverage, lp.MAX_LEVERAGE))
        margin = min(e.lot_margin, free_margin * 0.95, self.balance * self.meta["cap"])
        if margin < 10.0:
            return None
        if self.meta["maker_entry"]:
            slip, fee_rate = st.MAKER_SLIPPAGE_PCT, e.maker_fee
        else:
            slip, fee_rate = e.slippage, e.fee_rate
        exec_px = entry_raw * (1.0 + slip) if long else entry_raw * (1.0 - slip)
        if not (atr > 0 and np.isfinite(atr)):
            atr = entry_raw * self.meta["atr_fallback_pct"]
        sl = exec_px - sgn * self.prof.sl_atr_mult * atr
        if e.strategy_type == "DIRECTIONAL":   # SL do comitê (SignalGenerator): ancorado no close do sinal
            sig_sl = sig_close - sgn * self.prof.sl_atr_mult * atr
            if (long and sig_sl < exec_px) or ((not long) and sig_sl > exec_px):
                sl = sig_sl
        lev = lev_req          # SL sempre antes da liquidação (reduz em passos de 0.5, como o piloto)
        while lev > 1.0:
            liq = st.PaperPosition.calc_liquidation_price(side, exec_px, lev)
            if (sl > liq) if long else (sl < liq):
                break
            lev -= 0.5
        lev = max(1.0, lev)
        notional = margin * lev
        qty = notional / exec_px
        fee = notional * fee_rate
        if margin_used + margin + fee > max_margin:
            return None
        p = _PilotPos()
        p.sym, p.side, p.long, p.sgn = sym, side, long, sgn
        p.entry_idx, p.entry_ts, p.entry = idx, entry_ts, exec_px
        p.qty0 = p.qty = qty
        p.notional, p.margin, p.lev = notional, margin, lev
        p.sl = sl
        p.tp1 = exec_px + sgn * self.prof.tp1_atr_mult * atr
        p.tp2 = exec_px + sgn * self.prof.tp2_atr_mult * atr
        p.tp3 = exec_px + sgn * self.prof.tp3_atr_mult * atr
        p.tp1_hit = p.tp2_hit = p.trail_active = False
        p.trail_stop = 0.0
        p.trail_mult = self.prof.trail_atr_mult
        p.entry_atr = atr          # ATR congelado na entrada (base do trailing)
        p.trail_ref = 0.0
        p.trail_bar = None
        p.liq = st.PaperPosition.calc_liquidation_price(side, exec_px, lev) if lev > 1.0 else 0.0
        p.fees = fee
        p.gross = p.funding = 0.0
        p.fund_cursor = float(entry_ts)
        p.exit_qty_px = 0.0
        p.n_legs = 0
        p.last_reason = ""
        self.balance -= fee
        positions[sym] = p
        state["last_entry_idx"][sym] = idx
        state["quota"][day] = state["quota"].get(day, 0) + 1
        return p

    # ---- loop principal --------------------------------------------------------------------------------------
    def run(self, data: Dict[str, pd.DataFrame], signals: Dict[str, np.ndarray],
            confidences: Optional[Dict[str, np.ndarray]], atr_col: str,
            funding: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> "BacktestResult":
        e = self.e
        self.funding = funding or {}
        self.balance = float(e.initial_capital)
        arr: Dict[str, Dict[str, Any]] = {}
        for s, df in data.items():
            if len(df) == 0:
                raise ValueError("DataFrame de velas fornecido ao backtest está vazio.")
            if len(signals[s]) != len(df):
                raise ValueError(f"Tamanho dos sinais ({len(signals[s])}) difere do DataFrame ({len(df)}).")
            closes = df["close"].to_numpy(float)
            arr[s] = {
                "ts": df["timestamp"].to_numpy(), "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float),
                "l": df["low"].to_numpy(float), "c": closes, "sig": np.asarray(signals[s]).astype(int),
                "atr": df[atr_col].to_numpy(float) if atr_col in df.columns else closes * 0.01,
                "sma": df["sma_200"].to_numpy(float) if "sma_200" in df.columns else None,
                "conf": (np.asarray(confidences[s], dtype=float) if confidences and s in confidences else None),
            }
        index = {s: {int(t): k for k, t in enumerate(a["ts"])} for s, a in arr.items()}
        all_ts = sorted({t for m in index.values() for t in m})
        positions: Dict[str, _PilotPos] = {}
        state: Dict[str, Any] = {"last_entry_idx": {}, "quota": {}}
        trades: List[TradeRecord] = []
        equity_curve: List[float] = [self.balance]

        def finalize(p: _PilotPos, reason: str, ref: float, ts: int, idx: int) -> None:
            if p.qty > 0:
                self.balance += self._leg(p, ref, p.qty, reason, partial=False)
            self.balance = max(0.0, self.balance)
            exit_qty = p.qty0 - p.qty
            pnl_net = p.gross - p.fees - p.funding
            trades.append(TradeRecord(
                trade_id=len(trades) + 1, direction=p.side, entry_idx=p.entry_idx, entry_timestamp=int(p.entry_ts),
                entry_price=p.entry, exit_idx=idx, exit_timestamp=int(ts),
                exit_price=(p.exit_qty_px / exit_qty) if exit_qty > 0 else p.entry, size=p.qty0, pnl_gross=p.gross,
                fee_total=p.fees, funding_total=p.funding, pnl_net=pnl_net,
                return_pct=pnl_net / (p.margin + 1e-8), exit_reason=reason,
                symbol=p.sym, leverage=p.lev, margin=p.margin, legs=p.n_legs,
            ))
            del positions[p.sym]

        last_idx: Dict[str, int] = {}
        for t in all_ts:
            present = [s for s in arr if t in index[s]]
            for s in present:                              # 1) barras FECHADAS -> posições abertas
                k = index[s][t]
                last_idx[s] = k
                a = arr[s]
                p = positions.get(s)
                if p is None or k < p.entry_idx:
                    continue
                out = self.process_bar(p, t, a["o"][k], a["h"][k], a["l"][k], a["c"][k])
                if out is not None:
                    finalize(p, out[0], out[1], t, k)
            for s in present:                              # 2) sinais -> entradas (close(t) == open(t+1))
                k = index[s][t]
                a = arr[s]
                sg = int(a["sig"][k])
                if sg not in (1, 2, -1) or k + 1 >= len(a["ts"]):
                    continue
                sg = 1 if sg == 1 else 2
                conf = None if a["conf"] is None else float(a["conf"][k])
                regime = None
                if a["sma"] is not None:
                    sma, cl = float(a["sma"][k]), float(a["c"][k])
                    if np.isfinite(sma) and np.isfinite(cl) and sma > 0:
                        regime = (cl >= sma) if sg == 1 else (cl <= sma)
                raw = float(a["o"][k + 1]) if e.entry_mode == "next_open" else float(a["c"][k])
                self.try_open(s, k + 1, 1 if sg == 1 else -1, int(a["ts"][k + 1]), float(a["c"][k]), raw,
                              float(a["atr"][k]), conf, regime, positions, state)
            unreal = sum(p.sgn * (arr[s]["c"][last_idx[s]] - p.entry) * p.qty
                         for s, p in positions.items() if s in last_idx)
            equity_curve.append(self.balance + unreal)

        for s in list(positions):                          # fim dos dados: fecha a mercado o que sobrou
            a = arr[s]
            k = len(a["ts"]) - 1
            finalize(positions[s], "END_OF_DATA", float(a["c"][k]), int(a["ts"][k]), k)
        if equity_curve:
            equity_curve[-1] = self.balance
        return e._pilot_result(trades, equity_curve, self.balance)
