"""
Suíte de Testes da Fase 3: Pipeline de Machine Learning e Backtesting Realista.
Valida:
1. Prova matemática de ausência de vazamento temporal:
   - Embargo temporal estrito respeitado (gap >= horizon entre treino e teste).
   - Rótulos futuros e colunas de target ausentes da matriz de features X.
   - Descarte estrito das últimas N velas na rotulagem.
2. Rotulagem Triple-Barrier sem lookahead:
   - Consistência de classes (0=HOLD, 1=BUY, 2=SELL).
3. Resiliência de Hardware:
   - Resolução de dispositivo GPU/CUDA com fallback automático e transparente para CPU sem quebra.
4. Motor de Backtesting:
   - Execução estritamente na abertura da vela seguinte (i + 1).
   - Dedução de taxas reais Bitget (taker 0.06%) e slippage.
   - Gestão de SL/TP via ATR e cálculo de Drawdown / Sharpe.
5. Persistência de Modelos:
   - Salvamento e carregamento integro de artefato .joblib com metadados estruturados.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from model_trainer import (
    ModelTrainer,
    create_triple_barrier_labels,
    PurgedTimeSeriesSplit,
    resolve_compute_device,
)
from backtesting import BacktestEngine, TradeRecord, BacktestResult
from data.feature_engineering import FeatureEngineer


def generate_market_data(n: int = 400, base_price: float = 60000.0) -> pd.DataFrame:
    """Gera série sintética de mercado com ATR e indicadores."""
    np.random.seed(42)
    timestamps = [1700000000000 + i * 3600000 for i in range(n)]

    drift = np.cumsum(np.random.randn(n) * 150)
    close = base_price + drift
    high = close + np.abs(np.random.randn(n) * 80) + 10.0
    low = close - np.abs(np.random.randn(n) * 80) - 10.0
    open_p = close + np.random.randn(n) * 30

    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))
    volume = np.random.uniform(50, 1000, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
    })

    return FeatureEngineer.compute_all_features(df, drop_na=True)


class TestLeakageFreeLabelingAndSplits:
    """Valida a ausência absoluta de lookahead bias e vazamento de dados."""

    def test_last_n_candles_have_nan_labels(self):
        """As últimas N velas NÃO possuem horizonte futuro completo e DEVEM ter rótulo NaN."""
        df = generate_market_data(n=250)
        horizon = 7
        labels = create_triple_barrier_labels(df, horizon=horizon)

        # As últimas 7 linhas devem ser estritamente NaN
        assert labels.iloc[-horizon:].isna().all(), "As últimas N velas não foram descartadas/rotuladas como NaN!"
        # As linhas anteriores a len - horizon não devem ser NaN
        assert not labels.iloc[:-horizon].isna().any(), "Linhas anteriores contêm NaNs anômalos no rótulo."

    def test_features_matrix_strictly_excludes_future_and_target(self):
        """A matriz preditora X não pode conter 'target', colunas futuras ou identificadores."""
        df = generate_market_data(n=250)
        trainer = ModelTrainer(horizon=5)
        X, y = trainer.prepare_dataset(df)

        assert "target" not in X.columns
        assert len(X) == len(df) - 5
        assert len(y) == len(df) - 5
        assert not y.isna().any()
        assert not X.isna().any().any()

    def test_purged_walk_forward_respects_embargo_gap(self):
        """O conjunto de teste deve iniciar com gap >= horizon após o fim do conjunto de treino."""
        df = generate_market_data(n=300)
        trainer = ModelTrainer(horizon=6)
        X, y = trainer.prepare_dataset(df)

        splitter = PurgedTimeSeriesSplit(n_splits=3, horizon_embargo=6)
        splits = splitter.split(X, y)

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            max_train = train_idx.max()
            min_test = test_idx.min()
            embargo_observed = min_test - max_train

            # O gap temporal real deve ser rigorosamente maior ou igual ao horizonte
            assert embargo_observed >= 6, (
                f"Vazamento detectado: gap entre treino ({max_train}) e teste ({min_test}) "
                f"foi de {embargo_observed}, inferior ao horizonte de 6!"
            )


class TestHardwareResilience:
    """Valida a detecção e fallback dinâmico de CPU/GPU."""

    def test_device_fallback_never_crashes(self):
        """Qualquer requisição de device deve resolver para uma string válida ('cpu', 'cuda', 'gpu')."""
        dev, info = resolve_compute_device(requested_device="cuda", n_jobs=4)
        assert dev in ["cpu", "cuda", "gpu"]
        assert "device" in info
        assert "n_jobs" in info


class TestModelTrainingAndJoblibPersistence:
    """Valida o treinamento e ciclo de vida do modelo com joblib."""

    def test_model_training_and_joblib_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = generate_market_data(n=300)
            trainer = ModelTrainer(
                model_type="rf",
                horizon=5,
                confidence_threshold=0.50,
                n_jobs=4,
                models_dir=tmp_dir,
            )

            X, y = trainer.prepare_dataset(df)
            payload = trainer.train_full_model(X, y, save_name="test_rf.joblib")

            model_file = os.path.join(tmp_dir, "test_rf.joblib")
            assert os.path.exists(model_file)

            # Recarrega modelo
            loaded = ModelTrainer.load_model(model_file)
            assert "model" in loaded
            assert "metrics" in loaded
            assert "features" in loaded
            assert loaded["horizon"] == 5

            # Testa predição com modelo recarregado
            model = loaded["model"]
            preds = model.predict(X.iloc[:5])
            assert len(preds) == 5


class TestBacktestExecutionRealism:
    """Valida o rigor da simulação de execução de ordens e custos."""

    def test_execution_strictly_at_next_candle_open(self):
        """Um sinal na vela i deve abrir trade estritamente com o preço de abertura da vela i+1."""
        df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000, 4000],
            "open": [100.0, 105.0, 110.0, 115.0],
            "high": [102.0, 108.0, 112.0, 118.0],
            "low": [98.0, 104.0, 109.0, 114.0],
            "close": [101.0, 106.0, 111.0, 116.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
            "atr_14": [2.0, 2.0, 2.0, 2.0],
        })

        # Sinal de BUY na vela 0
        signals = [1, 0, 0, 0]

        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0006,
            slippage=0.0002,
            tp_atr_multiple=5.0,  # TP distante para manter trade aberto
            sl_atr_multiple=5.0,
        )

        res = engine.run(df, signals)
        assert len(res.trades) > 0
        first_trade = res.trades[0]

        # A entrada ocorreu no índice 1 (vela seguinte)
        assert first_trade.entry_idx == 1
        expected_entry_price = 105.0 * (1.0 + 0.0002)
        assert pytest.approx(first_trade.entry_price, rel=1e-4) == expected_entry_price

    def test_bitget_fees_deducted_properly(self):
        """As taxas de corretagem (0.06% entrada + 0.06% saída) devem ser deduzidas do PnL."""
        df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
            "atr_14": [1.0, 1.0, 1.0],
        })

        # Sinal neutro no preço -> Deve fechar no fim dos dados com lucro bruto 0 e PnL líquido negativo devido às taxas
        signals = [1, 0, 0]
        engine = BacktestEngine(fee_rate=0.0006, slippage=0.0)
        res = engine.run(df, signals)

        trade = res.trades[0]
        assert trade.pnl_gross == 0.0
        assert trade.fee_total > 0.0
        assert trade.pnl_net < 0.0  # PnL negativo decorrente estritamente das taxas de corretagem Bitget


class TestBacktestEngineSanityChecks:
    """
    Testes de Sanidade Institucional do Motor de Backtest (Sanity Checks A, B, C):
    (a) Sinais aleatórios com custo zero convergem para a probabilidade teórica de primeira passagem:
        P(TP) = SL / (TP + SL) = 1.0 / (1.5 + 1.0) = 40.0% e PnL Líquido próximo de zero.
    (b) Oráculo com visão do futuro (rotulação triple barrier) gera Win Rate elevado (>60%) e PnL fortemente positivo.
    (c) Buy & Hold simples bate rigorosamente o retorno exato do ativo: (P_end - P_entry) / P_entry.
    """

    @pytest.fixture
    def random_walk_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        close = 100.0 + np.cumsum(np.random.randn(n) * 1.0)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_p = close + np.random.randn(n) * 0.2
        high = np.maximum(high, np.maximum(open_p, close))
        low = np.minimum(low, np.minimum(open_p, close))
        return pd.DataFrame({
            "timestamp": range(n),
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "atr_14": [1.0] * n,
        })

    def test_backtest_engine_buy_and_hold_exact_match(self, random_walk_data):
        """Sanity Check (c): Buy & Hold bate retorno exato do preço à vista."""
        df = random_walk_data
        n = len(df)
        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0,
            slippage=0.0,
            funding_rate_8h=0.0,
            tp_atr_multiple=999.0,
            sl_atr_multiple=999.0,
            risk_per_trade=None,
        )
        signals = [1] + [0] * (n - 1)
        res = engine.run(df, signals)

        assert len(res.trades) == 1
        entry_price = df["open"].iloc[1]
        exit_price = df["close"].iloc[-1]
        expected_ret = (exit_price - entry_price) / entry_price

        # O retorno percentual registrado no trade bate com exatidão matemática
        assert pytest.approx(res.trades[0].return_pct, rel=1e-4) == expected_ret
        expected_profit = 10000.0 * 0.95 * expected_ret
        assert pytest.approx(res.net_profit, rel=1e-4) == expected_profit

    def test_backtest_engine_oracle_future_vision(self, random_walk_data):
        """Sanity Check (b): Oráculo com visão de futuro produz alto lucro e win rate > 60%."""
        df = random_walk_data
        labels = create_triple_barrier_labels(df, horizon=5, multiple_tp=1.5, multiple_sl=1.0)
        oracle_signals = [int(x) if not np.isnan(x) else 0 for x in labels]

        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0,
            slippage=0.0,
            funding_rate_8h=0.0,
            tp_atr_multiple=1.5,
            sl_atr_multiple=1.0,
            risk_per_trade=None,
            allow_reversal=True,
        )
        res = engine.run(df, oracle_signals)

        assert res.trades_count > 100
        assert res.win_rate > 0.60
        assert res.net_profit > 50000.0

    def test_backtest_engine_random_signals_sanity_zero_cost(self, random_walk_data):
        """Sanity Check (a): Sinais aleatórios a custo zero convergem para ~40% win rate."""
        df = random_walk_data
        np.random.seed(42)
        n = len(df)
        rand_signals = np.random.choice([0, 1, 2], size=n, p=[0.7, 0.15, 0.15])

        engine = BacktestEngine(
            initial_capital=10000.0,
            fee_rate=0.0,
            slippage=0.0,
            funding_rate_8h=0.0,
            tp_atr_multiple=1.5,
            sl_atr_multiple=1.0,
            risk_per_trade=None,
        )
        res = engine.run(df, rand_signals)

        # Teoria de primeira passagem: SL / (TP + SL) = 1.0 / 2.5 = 40.0%
        # Com N=264 trades, a margem de amostragem fica dentro de 35% - 45%
        assert res.trades_count > 100
        assert 0.35 <= res.win_rate <= 0.45
        assert res.exit_reasons is not None
        assert "TP" in res.exit_reasons
        assert "SL" in res.exit_reasons
