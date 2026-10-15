"""
Suíte de Testes da Fase 6: Motor de Execução e Simulação em Tempo Real (Paper Trading).
Valida:
1. Trava de Segurança Mandatória: Proibição estrita de Live Trading com fundos reais.
2. Roteamento e Dimensionamento de Ordens Simuladas com Custos Institucionais (Taker 0.06%, Slippage).
3. Estrutura de Carry Delta-Neutro Spot+Perp em Paper Trading.
4. Execução de Gatilhos Intra-Barra (Stop Loss e Take Profit com colisão conservadora).
5. Acúmulo de Funding Rate Real em posições Delta-Neutras.
6. Persistência e Integridade de Tabelas SQLite (paper_positions, paper_trades, paper_portfolio).
"""

import os
import sys
import tempfile
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/eudespatricio/ia_trader")

from smart_trade import PaperTradingEngine, PaperPosition, PaperTrade
from signal_generator import TradeSignal


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def engine(temp_db):
    return PaperTradingEngine(
        initial_capital=10000.0,
        db_path=temp_db,
        taker_fee=0.0006,
        slippage_spot=0.0002,
        slippage_perp=0.0003,
        simulation_mode=True,
    )


class TestPaperTradingSafetyAndInitialization:
    """Valida as diretrizes de governança e inicialização segura."""

    def test_simulation_mode_safety_lock(self, temp_db):
        """Tentativa de instanciar sem modo de simulação deve ser sumariamente bloqueada."""
        with pytest.raises(PermissionError) as exc_info:
            PaperTradingEngine(db_path=temp_db, simulation_mode=False)
        assert "bloqueado pelo protocolo de governança" in str(exc_info.value)

    def test_engine_initial_state_and_tables(self, engine, temp_db):
        assert engine.balance == 10000.0
        assert engine.equity == 10000.0
        assert engine.margin_used == 0.0
        assert len(engine.positions) == 0

        # Verifica criação das tabelas SQLite
        with sqlite3.connect(temp_db) as con:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            assert "paper_positions" in tables
            assert "paper_trades" in tables
            assert "paper_portfolio" in tables


class TestOrderExecutionAndCosts:
    """Valida execução com taxas e slippage institucionais."""

    def test_execute_directional_long(self, engine):
        sig = TradeSignal(
            signal_id="sig-01",
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=58500.0,
            take_profit=63000.0,
            atr=1000.0,
            confidence=0.75,
            consensus_score=0.65,
            position_size=1000.0,
            risk_dollars=100.0,
            rationale="Compra institucional por rompimento",
        )

        pos = engine.execute_directional_signal(sig, current_price=60000.0, timestamp=1700000000000)
        assert pos is not None
        assert pos.side == "LONG"
        # Preço com slippage de 0.02%
        assert pos.entry_price == pytest.approx(60000.0 * 1.0002)
        # Taxa de 0.06% deduzida
        expected_fee = 100.0 * 0.0006  # lote padrão PAPER_LOT_USD=100 (margem) com 1x
        assert engine.total_fees_paid == pytest.approx(expected_fee)
        assert engine.balance == pytest.approx(10000.0 - expected_fee)
        assert len(engine.positions) == 1

    def test_execute_directional_short(self, engine):
        sig = TradeSignal(
            signal_id="sig-02",
            timestamp=1700000000000,
            symbol="ETHUSDT",
            direction="SELL",
            entry_price=3000.0,
            stop_loss=3100.0,
            take_profit=2800.0,
            atr=80.0,
            confidence=0.80,
            consensus_score=-0.70,
            position_size=800.0,
            risk_dollars=80.0,
            rationale="Venda por rejeição em resistência",
        )

        pos = engine.execute_directional_signal(sig, current_price=3000.0, timestamp=1700000000000)
        assert pos is not None
        assert pos.side == "SHORT"
        # Preço de venda com slippage de mercado simétrico (0.02%, igual ao long e ao BacktestEngine)
        assert pos.entry_price == pytest.approx(3000.0 * (1.0 - 0.0002))

    def test_execute_carry_allocation_delta_neutral(self, engine):
        pos = engine.execute_carry_allocation(
            symbol="BTCUSDT",
            current_price=60000.0,
            timestamp=1700000000000,
            allocation_usd=2000.0,
            rationale="Oportunidade Carry Delta-Neutro 14% a.a.",
        )
        assert pos is not None
        assert pos.side == "DELTA_NEUTRAL_CARRY"
        assert pos.notional == 2000.0
        # Taxa das duas pernas (0.06% em cada perna de $1000 = $1.20)
        assert engine.total_fees_paid == pytest.approx(2000.0 * 0.0006)


class TestIntraBarSimulationAndTriggers:
    """Valida gatilhos intra-barra e fluxo de funding."""

    def test_stop_loss_trigger(self, engine):
        sig = TradeSignal(
            signal_id="sig-03",
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
            atr=1000.0,
            confidence=0.70,
            consensus_score=0.60,
            position_size=1000.0,
            risk_dollars=100.0,
            rationale="Compra",
        )
        engine.execute_directional_signal(sig, current_price=60000.0, timestamp=1700000000000)

        # Vela toca o Stop Loss (low = 57500 <= 58000)
        triggered = engine.update_market_tick(
            symbol="BTCUSDT",
            high=60500.0,
            low=57500.0,
            close=57800.0,
            timestamp=1700003600000,
        )
        assert len(triggered) == 1
        assert triggered[0].action == "STOP_LOSS"
        assert len(engine.positions) == 0
        assert engine.total_realized_pnl < 0.0

    def test_take_profit_trigger(self, engine):
        sig = TradeSignal(
            signal_id="sig-04",
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
            atr=1000.0,
            confidence=0.70,
            consensus_score=0.60,
            position_size=1000.0,
            risk_dollars=100.0,
            rationale="Compra",
        )
        engine.execute_directional_signal(sig, current_price=60000.0, timestamp=1700000000000)

        # E7: DIRECTIONAL usa a ladder comum (TP1 1.5xATR, TP2 2.5xATR, TP3 4.0xATR; o take_profit
        # único do sinal não é mais o alvo). Vela forte (low acima do breakeven) percorre TP1->TP2->TP3.
        triggered = engine.update_market_tick(
            symbol="BTCUSDT",
            high=64500.0,
            low=61000.0,
            close=64100.0,
            timestamp=1700003600000,
        )
        assert [t.action for t in triggered] == ["TP1_PARTIAL", "TP2_PARTIAL", "TAKE_PROFIT_3"]
        assert len(engine.positions) == 0
        assert engine.total_realized_pnl > 0.0

    def test_carry_funding_rate_accrual(self, engine):
        engine.execute_carry_allocation(
            symbol="BTCUSDT",
            current_price=60000.0,
            timestamp=1700000000000,
            allocation_usd=2000.0,
        )

        # 3 períodos de funding de 8h (+0.01% por período)
        funding_rate = 0.0001
        for i in range(3):
            engine.update_market_tick(
                symbol="BTCUSDT",
                high=60500.0,
                low=59500.0,
                close=60100.0,
                timestamp=1700000000000 + (i + 1) * 28800000,
                funding_rate_period=funding_rate,
            )

        pos = list(engine.positions.values())[0]
        # Perp short de $1000 notional * 0.0001 * 3 = $0.30
        assert pos.funding_collected == pytest.approx(0.30)
        assert pos.unrealized_pnl == pytest.approx(0.30)

        # Encerra o carry e valida que o funding vira lucro realizado
        trade = engine.close_position(pos.position_id, exit_price=60100.0, timestamp=1700000000000 + 4 * 28800000, reason="CLOSE_CARRY")
        assert trade is not None
        # Lucro líquido = $0.30 de funding - $1.20 de taxa de saída
        assert engine.total_realized_pnl == pytest.approx(0.30 - (2000.0 * 0.0006))


class TestDailyScalperQuotaAndExecution:
    """Valida a cota diária (mínimo 2, máximo 5) e a mecânica cirúrgica de scalp."""

    def test_daily_scalper_strict_quota_limit(self, engine):
        """Valida que o 6º trade no mesmo dia é bloqueado pela trava de cota diária."""
        base_ts = 1700000000000  # Mesmo dia UTC
        
        # Simula 5 scalps executados no mesmo dia
        for i in range(5):
            sig = TradeSignal(
                signal_id=f"scalp-sig-{i}",
                timestamp=base_ts + i * 1000,
                symbol=f"PAIR_{i}USDT",
                direction="BUY",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                atr=2.0,
                confidence=0.8,
                consensus_score=0.7,
                position_size=500.0,
                risk_dollars=50.0,
                rationale="Scalp teste",
            )
            pos = engine.execute_scalp_signal(sig, current_price=100.0, timestamp=base_ts + i * 1000)
            assert pos is not None
            assert pos.strategy_type == "SCALP"

        assert engine.daily_scalps_count == 5
        status = engine.get_daily_scalp_status()
        assert status["quota_status"] == "LIMITE_ATINGIDO"
        assert status["can_trade_more"] is False
        assert status["target_reached"] is True

        # Tentativa do 6º trade no mesmo dia UTC deve ser sumariamente bloqueada
        sig6 = TradeSignal(
            signal_id="scalp-sig-6",
            timestamp=base_ts + 6000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit=62000.0,
            atr=500.0,
            confidence=0.85,
            consensus_score=0.8,
            position_size=500.0,
            risk_dollars=50.0,
            rationale="Scalp 6 excedente",
        )
        pos6 = engine.execute_scalp_signal(sig6, current_price=60000.0, timestamp=base_ts + 6000)
        assert pos6 is None
        assert engine.daily_scalps_count == 5

    def test_daily_quota_rollover_on_new_utc_day(self, engine):
        """Valida que a virada de dia UTC reseta a contagem de scalps."""
        # Dia 1: 2023-11-14 (1700000000000 ms)
        ts_day1 = 1700000000000
        for i in range(5):
            sig = TradeSignal(
                signal_id=f"day1-{i}",
                timestamp=ts_day1 + i * 1000,
                symbol=f"SYM_{i}USDT",
                direction="BUY",
                entry_price=50.0,
                stop_loss=49.0,
                take_profit=52.0,
                atr=1.0,
                confidence=0.8,
                consensus_score=0.7,
                position_size=200.0,
                risk_dollars=20.0,
                rationale="Dia 1",
            )
            engine.execute_scalp_signal(sig, current_price=50.0, timestamp=ts_day1 + i * 1000)

        assert engine.daily_scalps_count == 5

        # Dia 2: 24 horas depois (+86400000 ms)
        ts_day2 = ts_day1 + 86400000
        sig_day2 = TradeSignal(
            signal_id="day2-1",
            timestamp=ts_day2,
            symbol="ETHUSDT",
            direction="BUY",
            entry_price=3000.0,
            stop_loss=2950.0,
            take_profit=3100.0,
            atr=50.0,
            confidence=0.8,
            consensus_score=0.7,
            position_size=300.0,
            risk_dollars=30.0,
            rationale="Dia 2 trade 1",
        )
        pos_day2 = engine.execute_scalp_signal(sig_day2, current_price=3000.0, timestamp=ts_day2)
        assert pos_day2 is not None
        assert engine.daily_scalps_count == 1
        assert engine.current_day_utc != "1970-01-01"

    def test_scalp_staged_tp1_breakeven_and_tp2(self, engine):
        """Ladder E7: TP1 (40%) + SL no Breakeven; TP2 (30%) é parcial e ativa o trailing (runner 30% segue aberto)."""
        ts = 1700000000000
        sig = TradeSignal(
            signal_id="scalp-precision-01",
            timestamp=ts,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=59100.0,
            take_profit=61800.0,
            atr=1000.0,
            confidence=0.85,
            consensus_score=0.75,
            position_size=1000.0,
            risk_dollars=50.0,
            rationale="Setup Scalper Reversão VWAP",
        )

        # Executa scalp com TP1 = 1.0 * ATR (+1000), TP2 = 1.8 * ATR (+1800), SL = 0.9 * ATR (-900)
        pos = engine.execute_scalp_signal(
            sig,
            current_price=60000.0,
            timestamp=ts,
            tp1_atr_mult=1.0,
            tp2_atr_mult=1.8,
            sl_atr_mult=0.9,
        )
        assert pos is not None
        initial_qty = pos.quantity
        assert pos.tp1_price == pytest.approx(pos.entry_price + 1000.0)
        assert pos.tp2_price == pytest.approx(pos.entry_price + 1800.0)
        assert pos.tp1_hit is False

        # 1. Vela atinge TP1: high >= tp1_price
        triggers = engine.update_market_tick(
            symbol="BTCUSDT",
            high=pos.tp1_price + 50.0,
            low=pos.entry_price + 100.0,
            close=pos.tp1_price,
            timestamp=ts + 900000,
        )
        assert len(triggers) == 1

        assert triggers[0].action == "TP1_PARTIAL"
        assert triggers[0].pnl > 0.0

        pos_after_tp1 = engine.positions[pos.position_id]
        assert pos_after_tp1.tp1_hit is True
        assert pos_after_tp1.quantity == pytest.approx(initial_qty * 0.6)  # E7: TP1 realiza 40% (antes 50%)
        # SL agora está em breakeven protegido (>= entry_price)
        assert pos_after_tp1.stop_loss > pos.entry_price

        # 2. Próxima vela atinge TP2: high >= tp2_price
        triggers2 = engine.update_market_tick(
            symbol="BTCUSDT",
            high=pos_after_tp1.tp2_price + 50.0,
            low=pos.entry_price + 500.0,
            close=pos_after_tp1.tp2_price,
            timestamp=ts + 1800000,
        )
        assert len(triggers2) == 1
        assert triggers2[0].action == "TP2_PARTIAL"   # E7: TP2 é parcial (antes encerrava a posição)
        assert pos.position_id in engine.positions     # runner de 30% segue com trailing
        assert engine.positions[pos.position_id].trailing_active is True
        assert engine.positions[pos.position_id].quantity == pytest.approx(initial_qty * 0.3)
        assert engine.total_realized_pnl > 0.0


class TestPaperTradingConcurrencyAndIntegrityStress:
    """
    Testes de Estresse, Concorrência e Integridade Transacional (Auditoria Contínua):
    1. Prevenção estrita de ultrapassar a cota máxima de 5 scalps sob rajada de chamadas simultâneas.
    2. Bloqueio concorrente de posições duplicadas no mesmo par.
    3. Integridade e consistência bidirecional entre memória e tabelas SQLite (paper_positions, paper_trades).
    4. Garantia de rollback em memória quando o SQLite falha (I/O, database locked).
    5. Transição atômica de cota na virada de dia UTC sob concorrência.
    """

    def test_concurrent_scalp_orders_strictly_respect_daily_quota(self, engine, temp_db):
        """
        Dispara 20 threads tentando executar scalps simultaneamente no mesmo dia UTC.
        Garante que NUNCA mais de 5 scalps sejam aceitos e que a cota seja exatamente respeitada.
        """
        num_threads = 20
        timestamp = 1700000000000
        results = [None] * num_threads

        def worker(thread_idx: int):
            sig = TradeSignal(
                signal_id=f"burst-sig-{thread_idx}",
                timestamp=timestamp,
                symbol=f"BURST_{thread_idx}USDT",
                direction="BUY",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                atr=2.0,
                confidence=0.85,
                consensus_score=0.75,
                position_size=300.0,
                risk_dollars=30.0,
                rationale=f"Scalp Rajada Thread {thread_idx}",
            )
            pos = engine.execute_scalp_signal(sig, current_price=100.0, timestamp=timestamp)
            results[thread_idx] = pos

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful = [p for p in results if p is not None]
        assert len(successful) == 5, f"Esperado exatamente 5 ordens executadas, obtido: {len(successful)}"
        assert engine.daily_scalps_count == 5

        # Valida consistência na tabela paper_positions do SQLite
        with sqlite3.connect(temp_db) as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM paper_positions WHERE strategy_type='SCALP'")
            db_positions_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM paper_trades WHERE action LIKE 'OPEN_SCALP%'")
            db_trades_count = cur.fetchone()[0]

        assert db_positions_count == 5, f"Esperado 5 posições no SQLite, obtido: {db_positions_count}"
        assert db_trades_count == 5, f"Esperado 5 trades no SQLite, obtido: {db_trades_count}"

    def test_concurrent_duplicate_symbol_scalp_rejection(self, engine):
        """10 threads tentando abrir scalp no mesmo par (BTCUSDT) concorrentemente: apenas 1 pode passar."""
        num_threads = 10
        timestamp = 1700000000000
        results = [None] * num_threads

        def worker(thread_idx: int):
            sig = TradeSignal(
                signal_id=f"dup-sig-{thread_idx}",
                timestamp=timestamp,
                symbol="BTCUSDT",
                direction="BUY",
                entry_price=60000.0,
                stop_loss=59000.0,
                take_profit=62000.0,
                atr=500.0,
                confidence=0.85,
                consensus_score=0.75,
                position_size=400.0,
                risk_dollars=40.0,
                rationale="Scalp concorrente duplicado",
            )
            pos = engine.execute_scalp_signal(sig, current_price=60000.0, timestamp=timestamp)
            results[thread_idx] = pos

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful = [p for p in results if p is not None]
        assert len(successful) == 1, f"Apenas 1 scalp deveria ser aceito no mesmo par, obtido: {len(successful)}"
        assert engine.daily_scalps_count == 1

    def test_sqlite_tables_referential_and_data_integrity(self, engine, temp_db):
        """Valida que todos os registros em memória refletem fielmente as tabelas SQLite."""
        ts = 1700000000000
        # Abre 1 direcional e 1 scalp
        sig_dir = TradeSignal(
            signal_id="dir-int-01",
            timestamp=ts,
            symbol="ETHUSDT",
            direction="BUY",
            entry_price=3000.0,
            stop_loss=2900.0,
            take_profit=3200.0,
            atr=50.0,
            confidence=0.8,
            consensus_score=0.7,
            position_size=500.0,
            risk_dollars=50.0,
            rationale="Direcional Integridade",
        )
        pos_dir = engine.execute_directional_signal(sig_dir, current_price=3000.0, timestamp=ts)

        sig_scalp = TradeSignal(
            signal_id="scalp-int-01",
            timestamp=ts,
            symbol="SOLUSDT",
            direction="SELL",
            entry_price=150.0,
            stop_loss=153.0,
            take_profit=144.0,
            atr=3.0,
            confidence=0.85,
            consensus_score=-0.75,
            position_size=300.0,
            risk_dollars=30.0,
            rationale="Scalp Integridade",
        )
        pos_scalp = engine.execute_scalp_signal(sig_scalp, current_price=150.0, timestamp=ts)

        # Consulta o banco e valida espelhamento
        with sqlite3.connect(temp_db) as con:
            cur = con.cursor()
            cur.execute("SELECT position_id, symbol, side, status, notional FROM paper_positions ORDER BY rowid")
            rows = cur.fetchall()

        assert len(rows) == 2
        assert rows[0][0] == pos_dir.position_id
        assert rows[0][1] == "ETHUSDT"
        assert rows[0][3] == "OPEN"
        assert rows[1][0] == pos_scalp.position_id
        assert rows[1][1] == "SOLUSDT"
        assert rows[1][3] == "OPEN"

        # Fecha uma posição e valida atualização de status e registro de saída
        trade_exit = engine.close_position(pos_dir.position_id, exit_price=3100.0, timestamp=ts + 3600000, reason="MANUAL_CLOSE")
        assert trade_exit is not None

        with sqlite3.connect(temp_db) as con:
            cur = con.cursor()
            cur.execute("SELECT status FROM paper_positions WHERE position_id = ?", (pos_dir.position_id,))
            status_in_db = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM paper_trades WHERE position_id = ?", (pos_dir.position_id,))
            trades_for_pos = cur.fetchone()[0]

        assert status_in_db == "CLOSED"
        assert trades_for_pos == 2  # 1 de abertura e 1 de fechamento

    def test_database_failure_rollback_integrity(self, engine):
        """
        Garante que, se o SQLite falhar ao persistir, o estado em memória é revertido (Rollback Integrity):
        - A cota diária NÃO é incrementada.
        - As taxas NÃO são deduzidas do saldo.
        - Nenhuma posição fantasma é criada em engine.positions.
        - Nenhum trade fantasma é adicionado a engine.trade_history.
        """
        initial_balance = engine.balance
        initial_quota = engine.daily_scalps_count
        initial_positions_count = len(engine.positions)
        initial_trades_count = len(engine.trade_history)

        sig = TradeSignal(
            signal_id="fail-sig-01",
            timestamp=1700000000000,
            symbol="ADAUSDT",
            direction="BUY",
            entry_price=0.50,
            stop_loss=0.48,
            take_profit=0.55,
            atr=0.01,
            confidence=0.8,
            consensus_score=0.7,
            position_size=200.0,
            risk_dollars=20.0,
            rationale="Simulação de Falha de Banco",
        )

        with patch.object(engine, "_save_trade", side_effect=sqlite3.OperationalError("database is locked")):
            pos = engine.execute_scalp_signal(sig, current_price=0.50, timestamp=1700000000000)

            # Deve retornar None e reverter alterações em memória
            assert pos is None
            assert engine.daily_scalps_count == initial_quota
            assert engine.balance == pytest.approx(initial_balance)
            assert len(engine.positions) == initial_positions_count
            assert len(engine.trade_history) == initial_trades_count

    def test_close_position_database_failure_rollback_integrity(self, engine):
        """Se a persistência falhar ao encerrar posição, a posição deve permanecer ABERTA em memória."""
        sig = TradeSignal(
            signal_id="close-fail-sig",
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=60000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
            atr=1000.0,
            confidence=0.8,
            consensus_score=0.7,
            position_size=500.0,
            risk_dollars=50.0,
            rationale="Posição para teste de falha ao fechar",
        )
        pos = engine.execute_directional_signal(sig, current_price=60000.0, timestamp=1700000000000)
        assert pos is not None
        initial_balance = engine.balance

        with patch.object(engine, "_save_trade", side_effect=sqlite3.OperationalError("disk I/O error")):
            trade = engine.close_position(pos.position_id, exit_price=62000.0, timestamp=1700003600000, reason="TAKE_PROFIT")

            assert trade is None
            # Posição permanece na memória como OPEN
            assert pos.position_id in engine.positions
            assert engine.positions[pos.position_id].status == "OPEN"
            assert engine.balance == pytest.approx(initial_balance)

    def test_midnight_utc_quota_rollover_concurrency(self, engine):
        """
        Testa virada de dia sob rajada concorrente:
        10 threads no segundo final do Dia 1 (23:59:59.999 UTC) -> exatamente 5 executam, 5 bloqueadas.
        10 threads no primeiro milissegundo do Dia 2 (00:00:00.001 UTC) -> contador reseta e exatamente 5 executam.
        """
        # 2023-11-14 23:59:59.999 UTC = 1700006399999
        # (P2: a cota diária é do dia da ENTRADA = ts(decisão)+15m; a decisão ocorre 15m antes do fim do dia 1)
        ts_day1 = 1700006399999 - 900000
        # 2023-11-15 00:00:00.001 UTC = 1700006400001
        ts_day2 = 1700006400001

        day1_results = [None] * 10
        def worker_day1(idx: int):
            sig = TradeSignal(
                signal_id=f"d1-sig-{idx}",
                timestamp=ts_day1,
                symbol=f"D1_PAIR_{idx}USDT",
                direction="BUY",
                entry_price=10.0,
                stop_loss=9.5,
                take_profit=11.0,
                atr=0.5,
                confidence=0.8,
                consensus_score=0.7,
                position_size=100.0,
                risk_dollars=10.0,
                rationale="Fim do Dia 1",
            )
            day1_results[idx] = engine.execute_scalp_signal(sig, current_price=10.0, timestamp=ts_day1)

        t1 = [threading.Thread(target=worker_day1, args=(i,)) for i in range(10)]
        for t in t1: t.start()
        for t in t1: t.join()

        assert len([p for p in day1_results if p is not None]) == 5
        assert engine.daily_scalps_count == 5

        # Agora virada do dia
        day2_results = [None] * 10
        def worker_day2(idx: int):
            sig = TradeSignal(
                signal_id=f"d2-sig-{idx}",
                timestamp=ts_day2,
                symbol=f"D2_PAIR_{idx}USDT",
                direction="BUY",
                entry_price=20.0,
                stop_loss=19.0,
                take_profit=22.0,
                atr=1.0,
                confidence=0.8,
                consensus_score=0.7,
                position_size=100.0,
                risk_dollars=10.0,
                rationale="Início do Dia 2",
            )
            day2_results[idx] = engine.execute_scalp_signal(sig, current_price=20.0, timestamp=ts_day2)

        t2 = [threading.Thread(target=worker_day2, args=(i,)) for i in range(10)]
        for t in t2: t.start()
        for t in t2: t.join()

        assert len([p for p in day2_results if p is not None]) == 5
        assert engine.daily_scalps_count == 5
        assert len(engine.trade_history) == 10

    def test_golden_pot_independent_quota_does_not_impact_standard_scalps(self, engine):
        """
        REGRA MANDATÓRIA: Golden Pots possuem cota própria (máx 3/dia) e NÃO impactam
        a cota diária de scalps padrões (máx 5/dia), e vice-versa.
        """
        ts = 1700000000000

        # 1. Esgota a cota de 5 scalps padrões
        for i in range(5):
            sig = TradeSignal(
                signal_id=f"std-scalp-{i}",
                timestamp=ts,
                symbol=f"STD_PAIR_{i}USDT",
                direction="BUY",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=104.0,
                atr=2.0,
                confidence=0.8,
                consensus_score=0.7,
                position_size=200.0,
                risk_dollars=20.0,
                rationale="Scalp padrão",
            )
            pos = engine.execute_scalp_signal(sig, current_price=100.0, timestamp=ts)
            assert pos is not None

        assert engine.daily_scalps_count == 5
        can_scalp, reason_s = engine.can_execute_scalp(ts)
        assert can_scalp is False
        assert "Limite diário estrito" in reason_s

        # 2. Golden Pot DEVE continuar 100% liberado!
        can_gp, reason_gp = engine.can_execute_golden_pot(ts)
        assert can_gp is True
        assert reason_gp == "OK"
        assert engine.daily_golden_pots_count == 0

        # Executa 3 Golden Pots
        gp_positions = []
        for j in range(3):
            sig_gp = TradeSignal(
                signal_id=f"gp-sig-{j}",
                timestamp=ts,
                symbol=f"GP_PAIR_{j}USDT",
                direction="BUY",
                entry_price=50.0,
                stop_loss=48.0,
                take_profit=54.0,
                atr=1.5,
                confidence=0.9,
                consensus_score=0.85,
                position_size=300.0,
                risk_dollars=30.0,
                rationale="Golden Pot Anomaly",
            )
            pos_gp = engine.execute_golden_pot_signal(sig_gp, current_price=50.0, timestamp=ts)
            assert pos_gp is not None
            assert pos_gp.strategy_type == "GOLDEN_POT"
            gp_positions.append(pos_gp)

        assert engine.daily_golden_pots_count == 3
        # Scalps continuam exatamente em 5 (não foi impactado!)
        assert engine.daily_scalps_count == 5

        # 4º Golden Pot deve ser barrado pelo teto independente
        sig_gp_blocked = TradeSignal(
            signal_id="gp-sig-4",
            timestamp=ts,
            symbol="GP_PAIR_BLOCKEDUSDT",
            direction="BUY",
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=54.0,
            atr=1.5,
            confidence=0.9,
            consensus_score=0.85,
            position_size=300.0,
            risk_dollars=30.0,
            rationale="Golden Pot Excedente",
        )
        pos_gp_blocked = engine.execute_golden_pot_signal(sig_gp_blocked, current_price=50.0, timestamp=ts)
        assert pos_gp_blocked is None
        assert engine.daily_golden_pots_count == 3

        # Métricas de telemetria refletem ambas as cotas
        metrics = engine.get_summary_metrics()
        assert metrics["daily_scalps_count"] == 5
        assert metrics["daily_scalps_limit_reached"] is True
        assert metrics["daily_golden_pots_count"] == 3
        assert metrics["daily_golden_pots_limit_reached"] is True

    def test_golden_pot_staged_exits(self, engine):
        """Valida a ladder E7 (TP1 40% + Breakeven + TP2 30% + trailing no runner) para Golden Pots."""
        ts = 1700000000000
        sig = TradeSignal(
            signal_id="gp-exit-test",
            timestamp=ts,
            symbol="SOLUSDT",
            direction="BUY",
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=105.0,
            atr=2.0,
            confidence=0.9,
            consensus_score=0.85,
            position_size=1000.0,
            risk_dollars=50.0,
            rationale="Golden Pot Staged Exit",
        )
        pos = engine.execute_golden_pot_signal(
            sig, current_price=100.0, timestamp=ts, tp1_atr_mult=1.0, tp2_atr_mult=2.5, sl_atr_mult=1.0
        )
        assert pos is not None
        assert pos.strategy_type == "GOLDEN_POT"
        initial_qty = pos.quantity

        # 1. Preço atinge TP1 (+1.0x ATR = $102.0)
        trades_tp1 = engine.update_market_tick(
            symbol="SOLUSDT", high=102.5, low=100.1, close=102.0, timestamp=ts + 900000 + 60000   # 1ª barra pós-entrada (entry = ts + 15m)
        )
        assert any(t.action == "TP1_PARTIAL" for t in trades_tp1)
        assert pos.tp1_hit is True
        assert pos.quantity == pytest.approx(initial_qty * 0.6)
        # Stop loss deve ter sido elevado para breakeven
        assert pos.stop_loss > pos.entry_price

        # 2. Preço atinge TP2 (+2.5x ATR = $105.0)
        trades_tp2 = engine.update_market_tick(
            symbol="SOLUSDT", high=105.5, low=101.5, close=105.0, timestamp=ts + 900000 + 120000
        )
        assert any(t.action == "TP2_PARTIAL" for t in trades_tp2)   # E7: TP2 parcial; runner segue (TP3 = 6 ATR)
        assert pos.position_id in engine.positions and pos.trailing_active

    def test_portfolio_snapshot_compression(self, engine, temp_db):
        """Valida compressão inteligente de snapshots de portfólio no SQLite."""
        ts = 1700000000000
        # Primeiro snapshot gravado obrigatoriamente
        engine._record_portfolio_snapshot(ts)

        # Chamada imediata sem alteração de saldo/posições (delta t = 10s < 300s)
        engine._record_portfolio_snapshot(ts + 10000)
        engine._record_portfolio_snapshot(ts + 20000)

        with sqlite3.connect(temp_db) as con:
            count = con.execute("SELECT COUNT(*) FROM paper_portfolio").fetchone()[0]
            # Redundâncias ignoradas: apenas 1 snapshot no banco!
            assert count == 1

        # Avança 301 segundos -> agora deve registrar
        engine._record_portfolio_snapshot(ts + 301000)
        with sqlite3.connect(temp_db) as con:
            count = con.execute("SELECT COUNT(*) FROM paper_portfolio").fetchone()[0]
            assert count == 2



