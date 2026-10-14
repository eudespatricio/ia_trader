"""
scripts/run_paper_simulation.py - Daemon Autônomo de Simulação em Tempo Real (Paper Trading).
Executa o ciclo contínuo de trading simulado com microestrutura institucional:
1. Carry Arbitrage em STANDBY (conforme diretriz estratégica).
2. Foco Total em:
   - Scalper Cirúrgico Intradiário (15m): saídas parciais (TP1 50% + SL Breakeven + TP2),
     execução Maker (0.02%) e cota diária rigorosa (máximo 5 scalps/dia).
   - Swing Trade de Momentum (1h/4h): rompimentos com surto de volume e tendência macro.
   - Radar Golden Pots integrado para descoberta dinâmica dos melhores pares do mercado.
3. Persistência contínua em SQLite (market_cache.db) refletida no Dashboard Web em tempo real.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from agents.golden_pot_agent import GoldenPotAgent
from signal_generator import TradeSignal
from smart_trade import PaperTradingEngine

logger = logging.getLogger("ia_trader.paper_simulation")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [paper_daemon] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PaperTradingDaemon:
    """
    Daemon de execução contínua em Paper Trading.
    Supervisiona posições abertas, varre o mercado e despacha ordens simuladas.
    """

    def __init__(
        self,
        poll_interval_seconds: int = 15,
        engine: Optional[PaperTradingEngine] = None,
        loader: Optional[DataLoader] = None,
        fe: Optional[FeatureEngineer] = None,
        golden_pot_agent: Optional[GoldenPotAgent] = None,
    ):
        self.poll_interval = poll_interval_seconds
        self.engine = engine or PaperTradingEngine()
        self.loader = loader or DataLoader()
        self.fe = fe or FeatureEngineer()
        self.scanner = golden_pot_agent or GoldenPotAgent(min_volume_usd=20_000_000.0)
        self.is_running = False

        # Pares fixos monitorados continuamente
        self.core_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    async def update_active_positions(self) -> None:
        """Atualiza todas as posições abertas com os preços de mercado mais recentes em paralelo."""
        open_positions = list(self.engine.positions.values())
        if not open_positions:
            return

        symbols_to_update = list({p.symbol for p in open_positions if p.status == "OPEN"})

        async def _update_symbol(sym: str) -> None:
            try:
                df = await self.loader.get_candles(sym, "15m", limit=5, exchange="binance")
                if df.empty:
                    df = await self.loader.get_candles(sym, "15m", limit=5, exchange="bitget")
                if not df.empty:
                    latest = df.iloc[-1]
                    high = float(latest["high"])
                    low = float(latest["low"])
                    close = float(latest["close"])
                    ts = int(latest["timestamp"])

                    triggered = self.engine.update_market_tick(
                        symbol=sym,
                        high=high,
                        low=low,
                        close=close,
                        timestamp=ts,
                    )
                    for t in triggered:
                        logger.info(f"[{sym}] GATILHO SIMULADO: {t.action} @ ${t.price:,.2f} | PnL: ${t.pnl:+.2f}")
            except Exception as e:
                logger.debug(f"Aviso ao atualizar posições para {sym}: {e}")

        await asyncio.gather(*[_update_symbol(s) for s in symbols_to_update], return_exceptions=True)

    async def evaluate_golden_pot_opportunities(self) -> None:
        """
        Avalia o mercado para oportunidades de GOLDEN POT (Varredura de Alta Convicção).
        Possui cota diária separada e independente (máx 3/dia) que NÃO consome a cota de scalps padrões!
        Valida desvios de VWAP, POI (Order Blocks / FVG) e Liquidity Sweeps (captura de liquidez).
        """
        now_ts = int(time.time() * 1000)
        can_exec, reason = self.engine.can_execute_golden_pot(now_ts)
        if not can_exec:
            return  # Teto diário de Golden Pots atingido

        settings = self.engine.get_simulation_settings()
        op_mode = settings.get("operation_mode", "INSTITUTIONAL")
        vwap_pot_thresh = 1.2 if op_mode == "DEMO" else 2.5
        rsi_pot_os = 42.0 if op_mode == "DEMO" else 38.0
        rsi_pot_ob = 58.0 if op_mode == "DEMO" else 62.0

        scan_res = await self.scanner.scan_market(force_refresh=False)
        top_pots = scan_res.get("top_scalps", [])[:4]
        if not top_pots:
            return

        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            return  # Margem livre insuficiente

        for pot in top_pots:
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                break

            can_exec, _ = self.engine.can_execute_golden_pot(now_ts)
            if not can_exec:
                break

            sym = pot["symbol"]
            # Se já há posição aberta neste par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                df = await self.loader.get_candles(sym, "15m", limit=100, exchange="binance")
                if len(df) < 50:
                    df = await self.loader.get_candles(sym, "15m", limit=100, exchange="bitget")
                if len(df) < 50:
                    continue

                df_feat = self.fe.compute_all_features(df, drop_na=True)
                if df_feat.empty:
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                vwap_dist = float(latest.get("dist_vwap_pct", 0.0))
                rsi = float(latest.get("rsi_14", 50.0))
                atr = float(latest.get("atr_14", close * 0.008))
                vol_z = float(latest.get("volume_zscore_20", 0.0))

                # Confluências de POI e Zonas de Liquidez
                sweep_bull = int(latest.get("liquidity_sweep_bull", 0))
                sweep_bear = int(latest.get("liquidity_sweep_bear", 0))
                at_bull_poi = int(latest.get("at_bullish_poi", 0))
                at_bear_poi = int(latest.get("at_bearish_poi", 0))

                # Gatilho de Golden Pot: Desvio expressivo de VWAP OU Liquidity Sweep confirmado em POI
                direction = None
                rationale_extra = []

                if (vwap_dist <= -vwap_pot_thresh and rsi <= rsi_pot_os) or (sweep_bull == 1 and at_bull_poi == 1):
                    direction = "BUY"
                    if sweep_bull == 1: rationale_extra.append("Liquidity Sweep de Fundo (SSL)")
                    if at_bull_poi == 1: rationale_extra.append("Em POI Institucional (OB/FVG)")
                elif (vwap_dist >= vwap_pot_thresh and rsi >= rsi_pot_ob) or (sweep_bear == 1 and at_bear_poi == 1):
                    direction = "SELL"
                    if sweep_bear == 1: rationale_extra.append("Liquidity Sweep de Topo (BSL)")
                    if at_bear_poi == 1: rationale_extra.append("Em POI Institucional (OB/FVG)")

                if direction:
                    extra_str = f" | {', '.join(rationale_extra)}" if rationale_extra else ""
                    sig = TradeSignal(
                        signal_id=f"gp-{sym}-{int(time.time())}",
                        timestamp=int(latest["timestamp"]),
                        symbol=sym,
                        direction=direction,
                        entry_price=close,
                        stop_loss=close - (1.0 * atr) if direction == "BUY" else close + (1.0 * atr),
                        take_profit=close + (2.5 * atr) if direction == "BUY" else close - (2.5 * atr),
                        atr=atr,
                        confidence=0.90,
                        consensus_score=0.85 if direction == "BUY" else -0.85,
                        position_size=min(1200.0, self.engine.available_margin * 0.95, self.engine.balance * 0.20),
                        risk_dollars=50.0,
                        rationale=f"Golden Pot Anomalia: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f}{extra_str}",
                    )
                    pos = self.engine.execute_golden_pot_signal(
                        signal=sig,
                        current_price=close,
                        timestamp=int(latest["timestamp"]),
                        tp1_atr_mult=1.2,
                        tp2_atr_mult=2.5,
                        sl_atr_mult=1.0,
                    )
                    if pos:
                        logger.info(
                            f"[{sym}] 🌟 GOLDEN POT DISPARADO: {direction} @ ${close:,.2f} | "
                            f"Cota Golden Pots: {self.engine.daily_golden_pots_count}/{self.engine.max_daily_golden_pots}"
                        )
            except Exception as e:
                logger.debug(f"Aviso ao avaliar golden pot para {sym}: {e}")

    async def evaluate_scalp_opportunities(self) -> None:
        """
        Avalia o mercado para scalps padrões em pares CORE (ETH, SOL, BTC).
        Consome a cota diária de scalps (mínimo 2, máximo 5/dia).
        """
        now_ts = int(time.time() * 1000)
        open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
        if open_count >= self.engine.max_concurrent_positions:
            return  # Teto de posições simultâneas atingido

        if self.engine.available_margin < 200.0:
            return  # Margem livre insuficiente

        can_exec, reason = self.engine.can_execute_scalp(now_ts)
        if not can_exec:
            return  # Teto diário de 5 scalps atingido

        for sym in ["ETHUSDT", "SOLUSDT", "BTCUSDT"]:
            open_count = len([p for p in self.engine.positions.values() if p.status == "OPEN"])
            if open_count >= self.engine.max_concurrent_positions:
                break

            can_exec, _ = self.engine.can_execute_scalp(now_ts)
            if not can_exec:
                break

            # Se já há scalp aberto no par, pula
            if any(p.symbol == sym and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                df = await self.loader.get_candles(sym, "15m", limit=100, exchange="binance")
                if len(df) < 50:
                    df = await self.loader.get_candles(sym, "15m", limit=100, exchange="bitget")
                if len(df) < 50:
                    continue

                df_feat = self.fe.compute_all_features(df, drop_na=True)
                if df_feat.empty:
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                vwap_dist = float(latest.get("dist_vwap_pct", 0.0))
                rsi = float(latest.get("rsi_14", 50.0))
                atr = float(latest.get("atr_14", close * 0.005))
                vol_z = float(latest.get("volume_zscore_20", 0.0))

                settings = self.engine.get_simulation_settings()
                vwap_dev = float(settings.get("scalp_vwap_dev_pct", 2.0))
                rsi_os = float(settings.get("scalp_rsi_oversold", 35.0))
                rsi_ob = float(settings.get("scalp_rsi_overbought", 65.0))
                vol_z_min = float(settings.get("scalp_volume_z", 0.2))

                direction = None
                if vwap_dist <= -vwap_dev and rsi <= rsi_os and vol_z >= vol_z_min:
                    direction = "BUY"
                elif vwap_dist >= vwap_dev and rsi >= rsi_ob and vol_z >= vol_z_min:
                    direction = "SELL"

                if direction:
                    sig = TradeSignal(
                        signal_id=f"scalp-{sym}-{int(time.time())}",
                        timestamp=int(latest["timestamp"]),
                        symbol=sym,
                        direction=direction,
                        entry_price=close,
                        stop_loss=close - (0.9 * atr) if direction == "BUY" else close + (0.9 * atr),
                        take_profit=close + (1.8 * atr) if direction == "BUY" else close - (1.8 * atr),
                        atr=atr,
                        confidence=0.85,
                        consensus_score=0.80 if direction == "BUY" else -0.80,
                        position_size=min(1500.0, self.engine.available_margin * 0.95, self.engine.balance * 0.20),
                        risk_dollars=50.0,
                        rationale=f"Scalp Padrão Core: VWAP {vwap_dist:+.2f}% | RSI {rsi:.1f} | Vol Z {vol_z:+.1f}",
                    )
                    pos = self.engine.execute_scalp_signal(
                        signal=sig,
                        current_price=close,
                        timestamp=int(latest["timestamp"]),
                        tp1_atr_mult=1.0,
                        tp2_atr_mult=1.8,
                        sl_atr_mult=0.9,
                    )
                    if pos:
                        logger.info(
                            f"[{sym}] ⚡ SCALP PADRÃO DISPARADO: {direction} @ ${close:,.2f} | "
                            f"Cota Scalps: {self.engine.daily_scalps_count}/5"
                        )
            except Exception as e:
                logger.debug(f"Aviso ao avaliar scalp padrão para {sym}: {e}")

    async def evaluate_swing_opportunities(self) -> None:
        """Avalia oportunidades de Swing Trade (1h/4h) com o Comitê de Decisão."""
        for sym in self.core_symbols:
            if any(p.symbol == sym and p.strategy_type == "DIRECTIONAL" and p.status == "OPEN" for p in self.engine.positions.values()):
                continue

            try:
                df = await self.loader.get_candles(sym, "1h", limit=100, exchange="binance")
                if len(df) < 50:
                    df = await self.loader.get_candles(sym, "1h", limit=100, exchange="bitget")
                if len(df) < 50:
                    continue

                df_feat = self.fe.compute_all_features(df, drop_na=True)
                if df_feat.empty:
                    continue

                latest = df_feat.iloc[-1]
                close = float(latest["close"])
                ts = int(latest["timestamp"])

                eval_res = self.engine.core.evaluate_multi_strategy(
                    symbol=sym,
                    current_price=close,
                    context=latest.to_dict(),
                    annualized_funding_rate=0.0, # Carry em standby
                    timestamp=ts,
                )

                dir_sig = eval_res.get("directional_signal", {})
                if dir_sig.get("direction") in ["BUY", "SELL"] and not dir_sig.get("is_vetoed", False):
                    # Exige confiança institucional (>= 0.70)
                    if dir_sig.get("confidence", 0.0) >= 0.70:
                        sig = TradeSignal(**dir_sig)
                        pos = self.engine.execute_directional_signal(sig, current_price=close, timestamp=ts)
                        if pos:
                            logger.info(f"[{sym}] SWING TRADE DISPARADO: {sig.direction} @ ${close:,.2f}")
            except Exception as e:
                logger.debug(f"Aviso ao avaliar swing para {sym}: {e}")

    async def run_loop(self, max_cycles: int = 0) -> None:
        """Loop principal do daemon em tempo real."""
        self.is_running = True
        logger.info("================================================================================")
        logger.info("🚀 MOTOR DE PAPER TRADING INICIADO EM MODO DEMO / SIMULAÇÃO CONTÍNUA")
        logger.info(f"   Saldo Inicial: ${self.engine.balance:,.2f} USD | Modo: 100% Simulação (Zero Risco)")
        logger.info(f"   Estratégias Ativas: Scalper Cirúrgico (15m, Máx 5/dia) & Swing Trade (1h/4h)")
        logger.info("   Arbitragem de Carry: STANDBY")
        logger.info(f"   Dashboard ao vivo em: http://localhost:5000")
        logger.info("================================================================================")

        cycle = 0
        try:
            while self.is_running:
                cycle += 1
                logger.debug(f"--- Ciclo de Simulação #{cycle} ---")
                self.engine.sync_settings_from_db()

                # 1. Atualiza posições abertas e testa gatilhos de saída (SL, TP1, TP2) em paralelo
                await self.update_active_positions()

                # 2. Avalia novas oportunidades de Golden Pots (Cota separada e independente: máx 3/dia)
                await self.evaluate_golden_pot_opportunities()

                # 3. Avalia novas oportunidades de Scalper Cirúrgico em pares Core (Cota padrão: 2 a 5/dia)
                await self.evaluate_scalp_opportunities()

                # 4. Avalia novas oportunidades de Swing Trade
                await self.evaluate_swing_opportunities()

                # 5. Registra métricas e snapshot no banco
                m = self.engine.get_summary_metrics()
                scalp_stat = self.engine.get_daily_scalp_status()
                logger.info(
                    f"[STATUS DEMO] Equity: ${m['current_equity']:,.2f} | "
                    f"PnL: ${m['net_profit']:+,.2f} ({m['return_pct']:+.2f}%) | "
                    f"Scalps Hoje: {scalp_stat['daily_scalps_count']}/{scalp_stat['max_daily_scalps']} ({scalp_stat['quota_status']}) | "
                    f"Golden Pots: {m['daily_golden_pots_count']}/{m['max_daily_golden_pots']} | "
                    f"Posições Abertas: {m['open_positions']} | Trades Totais: {m['total_trades_count']}"
                )

                if max_cycles > 0 and cycle >= max_cycles:
                    logger.info(f"Limite de {max_cycles} ciclos atingido. Encerrando execução.")
                    break

                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Daemon de Paper Trading recebendo sinal de interrupção...")
        finally:
            self.is_running = False
            await self.loader.close()
            await self.scanner.close()
            logger.info("Daemon de Paper Trading encerrado graciosamente.")


async def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    daemon = PaperTradingDaemon(poll_interval_seconds=poll)

    # Tratamento de sinais POSIX para encerramento gracioso
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: setattr(daemon, "is_running", False))
        except NotImplementedError:
            pass

    await daemon.run_loop(max_cycles=cycles)


if __name__ == "__main__":
    asyncio.run(main())
