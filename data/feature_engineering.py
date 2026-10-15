"""
Módulo de Engenharia de Features e Indicadores Técnicos Locais (Fase 2).
Calcula indicadores 100% locais e vetorizados utilizando as bibliotecas `ta` e `numpy`
a partir de DataFrames OHLCV normalizados (do DataLoader).

Princípios Fundamentais:
1. Desempenho e Baixa Latência: Execução vetorizada local (< 10ms para 500 velas).
2. Ausência Total de Lookahead Bias: Alinhamento macroeconômico estritamente por release_date (backward).
3. Robustez Numérica: Sanitização contra infs e NaNs residuais, com suporte a drop de janelas de aquecimento.
4. Base Mandatória para SL/TP: Cálculo estrito de ATR (Average True Range) e bandas de volatilidade.
5. Integração com Microestrutura: Spread e Imbalance de livro de ordens (OrderBookDepth).
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

from .data_integrations import OrderBookDepth

logger = logging.getLogger("ia_trader.feature_engineering")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [feature_engineering] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FeatureEngineer:
    """
    Motor Vetorizado de Engenharia de Features e Indicadores Técnicos.
    Gera atributos de tendência, osciladores, volatilidade, microestrutura e macroeconomia.
    """

    FEATURE_COLUMNS = [
        # Tendência
        "sma_20", "sma_50", "sma_200",
        "ema_9", "ema_21",
        "macd_line", "macd_signal", "macd_hist", "macd_diff",
        "dist_sma_20", "dist_sma_50", "dist_sma_200",
        "dist_ema_9", "dist_ema_21",
        # Osciladores
        "rsi_14", "stoch_k", "stoch_d",
        "rsi_overbought", "rsi_oversold",
        # Volatilidade
        "atr_14", "atr_pct", "natr_14",
        "bb_high", "bb_mid", "bb_low", "bb_width", "bb_pband",
        # Volume & Microestrutura
        "obv", "vol_sma_20", "vol_sma_ratio", "volume_zscore_20",
        "dist_vwap_pct", "ob_spread", "ob_imbalance",
        # Retornos e Dinâmica de Preço
        "returns", "log_returns", "volatility_20", "hl_spread",
        # Regime de Financiamento (Funding Carry)
        "funding_rate", "funding_zscore",
        # Fatores Macroeconômicos Anti-Lookahead
        "macro_cpi", "macro_fedfunds", "macro_vix",
        "macro_dxy", "macro_sp500", "macro_regime_risk_off"
    ]

    INSTITUTIONAL_FEATURE_COLUMNS = [
        # 1. Tendência & Regime Estrutural
        "sma_200", "dist_sma_200", "sma_50", "dist_sma_50", "ema_9", "ema_21",
        # 2. Estrutura & Reversão à Média (VWAP e Bandas)
        "dist_vwap_pct", "bb_low", "bb_high", "bb_pband",
        # 3. Momentum Contínuo
        "rsi_14",
        # 4. Volume Institucional & Anomalias
        "volume_zscore_20", "vol_sma_20", "obv", "vol_sma_ratio",
        # 5. Volatilidade & Dimensionamento de Risco (Risk Sizing)
        "natr_14", "atr_14", "bb_width", "volatility_20",
        # 6. Regime de Carry & Alavancagem
        "funding_zscore", "funding_rate",
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Retorna a lista padronizada de nomes de features numéricas para modelos de ML."""
        return list(cls.FEATURE_COLUMNS)

    @classmethod
    def get_institutional_feature_names(cls) -> List[str]:
        """Retorna a lista curada de features institucionais de alto sinal e baixo ruído."""
        return list(cls.INSTITUTIONAL_FEATURE_COLUMNS)

    @classmethod
    def add_trend_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores de tendência:
        - SMA (20, 50, 200)
        - EMA (9, 21)
        - MACD (Linha 12/26, Sinal 9, Histograma)
        - Distância percentual do preço de fechamento para cada média
        """
        close = df["close"]

        # Médias Móveis Simples (SMA)
        df["sma_20"] = ta.trend.SMAIndicator(close=close, window=20, fillna=False).sma_indicator()
        df["sma_50"] = ta.trend.SMAIndicator(close=close, window=50, fillna=False).sma_indicator()
        df["sma_200"] = ta.trend.SMAIndicator(close=close, window=200, fillna=False).sma_indicator()

        # Médias Móveis Exponenciais (EMA)
        df["ema_9"] = ta.trend.EMAIndicator(close=close, window=9, fillna=False).ema_indicator()
        df["ema_21"] = ta.trend.EMAIndicator(close=close, window=21, fillna=False).ema_indicator()

        # MACD (12, 26, 9)
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["macd_diff"] = df["macd_hist"]  # Alias de conveniência

        # Distâncias percentuais normalizadas (Zero-Lookahead, excelente para ML)
        df["dist_sma_20"] = (close - df["sma_20"]) / (df["sma_20"] + 1e-8)
        df["dist_sma_50"] = (close - df["sma_50"]) / (df["sma_50"] + 1e-8)
        df["dist_sma_200"] = (close - df["sma_200"]) / (df["sma_200"] + 1e-8)
        df["dist_ema_9"] = (close - df["ema_9"]) / (df["ema_9"] + 1e-8)
        df["dist_ema_21"] = (close - df["ema_21"]) / (df["ema_21"] + 1e-8)

        return df

    @classmethod
    def add_oscillator_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula osciladores de momento:
        - RSI (14 períodos)
        - Estocástico (%K, %D com janela de suavização 3)
        - Flags binárias de sobrecompra e sobrevenda
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # RSI (14)
        rsi_ind = ta.momentum.RSIIndicator(close=close, window=14, fillna=False)
        df["rsi_14"] = rsi_ind.rsi()

        # Estocástico (14, 3)
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3, fillna=False)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Sinais de Limite
        df["rsi_overbought"] = (df["rsi_14"] >= 70.0).astype(float)
        df["rsi_oversold"] = (df["rsi_14"] <= 30.0).astype(float)

        return df

    @classmethod
    def add_volatility_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas de volatilidade:
        - ATR (14 períodos) - Base mandatória para cálculo dinâmico de SL/TP
        - ATR percentual (relativo ao close)
        - Bandas de Bollinger (superior, média, inferior, largura percentual e posição %B)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # ATR 14 (Base mandatória para SL/TP)
        if len(df) >= 14:
            atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14, fillna=False)
            df["atr_14"] = atr_ind.average_true_range()
        else:
            prev_close = close.shift(1)  # sem bfill: 1a barra usa apenas high-low (anti-lookahead)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr_14"] = tr.expanding(min_periods=1).mean()

        df["atr_pct"] = df["atr_14"] / (close + 1e-8)
        # Normalized ATR % (escala invariante para múltiplos ativos)
        df["natr_14"] = (df["atr_14"] / (close + 1e-8)) * 100.0

        # Bandas de Bollinger (20, 2)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2, fillna=False)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pband"] = bb.bollinger_pband()

        return df

    @classmethod
    def add_volume_and_microstructure(
        cls,
        df: pd.DataFrame,
        order_book: Optional[OrderBookDepth] = None,
    ) -> pd.DataFrame:
        """
        Calcula indicadores de volume e microestrutura:
        - OBV (On-Balance Volume)
        - Volume SMA (20) e Razão de Volume
        - Métricas do Livro de Ordens (Spread e Imbalance)
        """
        close = df["close"]
        volume = df["volume"]

        # OBV
        obv_ind = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume, fillna=False)
        df["obv"] = obv_ind.on_balance_volume()

        # Média Móvel de Volume (20)
        df["vol_sma_20"] = ta.trend.SMAIndicator(close=volume, window=20, fillna=False).sma_indicator()
        df["vol_sma_ratio"] = volume / (df["vol_sma_20"] + 1e-8)

        # Volume Z-score (20 períodos) - Detecção de Anomalias de Volume Institucional
        vol_mean = volume.rolling(window=20, min_periods=2).mean()
        vol_std = volume.rolling(window=20, min_periods=2).std() + 1e-8
        df["volume_zscore_20"] = (volume - vol_mean) / vol_std

        # Distância percentual do Rolling VWAP (24 períodos)
        pv = (close * volume).rolling(window=24, min_periods=2).sum()
        v_sum = volume.rolling(window=24, min_periods=2).sum() + 1e-8
        vwap_24 = pv / v_sum
        df["dist_vwap_pct"] = ((close - vwap_24) / (vwap_24 + 1e-8)) * 100.0

        # Métricas de Microestrutura do Livro de Ofertas
        if order_book is not None:
            # V8: snapshot atual do book só vale para a ÚLTIMA barra (inferência);
            # barras históricas recebem valor neutro para não vazar/difundir o presente.
            df["ob_spread"] = 0.0
            df["ob_imbalance"] = 0.0
            if len(df) > 0:
                df.iloc[-1, df.columns.get_loc("ob_spread")] = float(order_book.spread)
                df.iloc[-1, df.columns.get_loc("ob_imbalance")] = float(order_book.imbalance)
        else:
            if "ob_spread" not in df.columns:
                df["ob_spread"] = 0.0
            if "ob_imbalance" not in df.columns:
                df["ob_imbalance"] = 0.0

        return df

    @classmethod
    def add_returns_and_volatility_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula dinâmicas de retornos e volatilidade de curto prazo:
        - Retorno simples de 1 período
        - Retorno logarítmico
        - Volatilidade histórica móvel (desvio padrão de 20 períodos)
        - High-Low spread da vela (amplitude)
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["returns"] = close.pct_change(fill_method=None)
        df["log_returns"] = np.log(close / close.shift(1).replace(0, np.nan))
        df["volatility_20"] = df["returns"].rolling(window=20, min_periods=2).std()
        df["hl_spread"] = (high - low) / (close + 1e-8)

        return df

    @staticmethod
    def _apply_snapshot_last_row(df: pd.DataFrame, values: Dict[str, float], defaults: Dict[str, float]) -> None:
        """Snapshot pontual (sem timestamp): neutro no histórico, valor real apenas na última barra."""
        for col, dv in defaults.items():
            df[col] = dv
        if len(df) > 0:
            for col, v in values.items():
                df.iloc[-1, df.columns.get_loc(col)] = v

    @classmethod
    def add_macro_features(
        cls,
        df: pd.DataFrame,
        macro_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Codifica fatores macroeconômicos com estrita ausência de lookahead bias.
        Se macro_df for um DataFrame contendo série temporal:
        - Utiliza alinhamento retroativo seguro (merge_asof backward) baseado na release_date.
        - As velas de uma determinada data só recebem dados macro econômicos cuja
          data oficial de divulgação (release_date) seja menor ou igual à data da vela.
        Se macro_df for um dicionário (snapshot pontual do DataLoader):
        - Transmite os valores escalares diretamente de forma consistente.
        Se macro_df for None:
        - Atribui valores neutros padrão preservando a matriz de features estável para o modelo.
        """
        # Valores padrão neutros
        default_cpi = 314.0
        default_fedfunds = 5.33
        default_vix = 18.0
        default_dxy = 103.5
        default_sp500 = 5000.0
        default_risk_off = 0.0

        if macro_df is None:
            df["macro_cpi"] = default_cpi
            df["macro_fedfunds"] = default_fedfunds
            df["macro_vix"] = default_vix
            df["macro_dxy"] = default_dxy
            df["macro_sp500"] = default_sp500
            df["macro_regime_risk_off"] = default_risk_off
            return df

        if isinstance(macro_df, dict):
            # V4: snapshot pontual só se aplica à ÚLTIMA barra (inferência);
            # o histórico recebe valores neutros (não difundir o presente no passado).
            cls._apply_snapshot_last_row(df, {
                "macro_cpi": float(macro_df.get("cpi", default_cpi)),
                "macro_fedfunds": float(macro_df.get("fedfunds", default_fedfunds)),
                "macro_vix": float(macro_df.get("vix", default_vix)),
                "macro_dxy": float(macro_df.get("dxy", default_dxy)),
                "macro_sp500": float(macro_df.get("sp500", default_sp500)),
                "macro_regime_risk_off": 1.0 if macro_df.get("regime") == "risk_off" else 0.0,
            }, {
                "macro_cpi": default_cpi, "macro_fedfunds": default_fedfunds,
                "macro_vix": default_vix, "macro_dxy": default_dxy,
                "macro_sp500": default_sp500, "macro_regime_risk_off": default_risk_off,
            })
            return df

        if isinstance(macro_df, pd.DataFrame):
            if macro_df.empty:
                df["macro_cpi"] = default_cpi
                df["macro_fedfunds"] = default_fedfunds
                df["macro_vix"] = default_vix
                df["macro_dxy"] = default_dxy
                df["macro_sp500"] = default_sp500
                df["macro_regime_risk_off"] = default_risk_off
                return df

            # Prepara a chave temporal para merge anti-lookahead
            m_df = macro_df.copy()

            # Identifica a coluna de data de divulgação no macro_df
            date_col = None
            for candidate in ["release_date", "observation_date", "date", "timestamp"]:
                if candidate in m_df.columns:
                    date_col = candidate
                    break

            if date_col is None:
                # Se não houver coluna temporal no macro_df, assume valores do último registro
                last_row = m_df.iloc[-1]
                cls._apply_snapshot_last_row(df, {
                    "macro_cpi": float(last_row.get("cpi", default_cpi)),
                    "macro_fedfunds": float(last_row.get("fedfunds", default_fedfunds)),
                    "macro_vix": float(last_row.get("vix", default_vix)),
                    "macro_dxy": float(last_row.get("dxy", default_dxy)),
                    "macro_sp500": float(last_row.get("sp500", default_sp500)),
                    "macro_regime_risk_off": float(last_row.get("macro_regime_risk_off", default_risk_off)),
                }, {
                    "macro_cpi": default_cpi, "macro_fedfunds": default_fedfunds,
                    "macro_vix": default_vix, "macro_dxy": default_dxy,
                    "macro_sp500": default_sp500, "macro_regime_risk_off": default_risk_off,
                })
                return df

            # Converte chaves temporais para datetime64[ns]
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                candle_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
            else:
                candle_dt = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")

            if pd.api.types.is_numeric_dtype(m_df[date_col]):
                m_dt = pd.to_datetime(m_df[date_col], unit="ms", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
            else:
                m_dt = pd.to_datetime(m_df[date_col], utc=True).dt.tz_localize(None).astype("datetime64[ns]")

            if date_col != "release_date":
                # V6: sem release_date explícito a data é de OBSERVAÇÃO (ex.: fechamento diário de
                # mercado) -> só fica disponível no dia seguinte: shift conservador de +1 dia.
                logger.warning("macro_df sem 'release_date' (usando '%s'): aplicando shift +1 dia anti-lookahead.", date_col)
                m_dt = m_dt + pd.Timedelta(days=1)
            m_df["_merge_dt"] = m_dt
            m_df = m_df.sort_values("_merge_dt").drop_duplicates(subset=["_merge_dt"])

            # DataFrame temporário para merge_asof backward (Anti-Lookahead Rigoroso)
            temp_left = pd.DataFrame({"_merge_dt": candle_dt, "_orig_idx": df.index}).sort_values("_merge_dt")

            # Mapeamento de colunas macro desejadas
            col_map = {
                "cpi": "macro_cpi",
                "CPIAUCSL": "macro_cpi",
                "fedfunds": "macro_fedfunds",
                "FEDFUNDS": "macro_fedfunds",
                "vix": "macro_vix",
                "VIX": "macro_vix",
                "dxy": "macro_dxy",
                "DXY": "macro_dxy",
                "sp500": "macro_sp500",
                "SP500": "macro_sp500",
            }

            keep_cols = ["_merge_dt"]
            rename_dict = {}
            for col in m_df.columns:
                if col in col_map:
                    keep_cols.append(col)
                    rename_dict[col] = col_map[col]

            m_subset = m_df[keep_cols].rename(columns=rename_dict)

            merged = pd.merge_asof(
                temp_left,
                m_subset,
                on="_merge_dt",
                direction="backward"
            ).sort_values("_orig_idx")

            for target_col, def_val in [
                ("macro_cpi", default_cpi),
                ("macro_fedfunds", default_fedfunds),
                ("macro_vix", default_vix),
                ("macro_dxy", default_dxy),
                ("macro_sp500", default_sp500),
            ]:
                if target_col in merged.columns:
                    df[target_col] = merged[target_col].ffill().fillna(def_val).values
                else:
                    df[target_col] = def_val

            # Regime de risco baseado em VIX e DXY
            df["macro_regime_risk_off"] = (
                (df["macro_vix"] > 25.0) | (df["macro_dxy"] > 105.0)
            ).astype(float)

        return df

    @classmethod
    def add_funding_features(
        cls,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Incorpora histórico de funding rate com alinhamento backward rigoroso (anti-lookahead)
        e calcula o funding_zscore (janela móvel de 90 períodos de 8h / 30 dias).
        """
        if "funding_rate" not in df.columns:
            if funding_df is not None and not funding_df.empty and "timestamp" in funding_df.columns and "funding_rate" in funding_df.columns:
                f_df = funding_df.sort_values("timestamp")[["timestamp", "funding_rate"]].copy()
                f_df["funding_rate"] = pd.to_numeric(f_df["funding_rate"], errors="coerce").fillna(0.0)
                temp = pd.merge_asof(
                    df[["timestamp"]].sort_values("timestamp"),
                    f_df,
                    on="timestamp",
                    direction="backward"
                )
                df["funding_rate"] = temp["funding_rate"].ffill().fillna(0.0).values
            else:
                df["funding_rate"] = 0.0

        f_series = df["funding_rate"].astype(float)
        f_mean = f_series.rolling(window=90, min_periods=2).mean()
        f_std = f_series.rolling(window=90, min_periods=2).std() + 1e-12
        df["funding_zscore"] = (f_series - f_mean) / f_std
        return df

    @classmethod
    def clean_features(cls, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Sanitização e tratamento numérico:
        1. Substituição de infinitos por NaN.
        2. Se drop_na=True e o DataFrame possuir dados suficientes (> 200 velas),
           elimina as primeiras linhas de aquecimento das médias longas (SMA 200).
        3. Para séries mais curtas (<= 200 velas) ou remanescentes, aplica preenchimento
           retroativo (bfill) e prospectivo (ffill) seguro para garantir integridade.
        4. Garante ausência total de NaNs ou infinitos residuais.
        """
        cleaned = df.replace([np.inf, -np.inf], np.nan)

        if drop_na:
            if len(cleaned) > 200:
                # Janela de aquecimento do SMA 200
                cleaned = cleaned.dropna().reset_index(drop=True)
            else:
                # Séries curtas: preserva registros utilizando preenchimento coerente
                cleaned = cleaned.ffill().fillna(0.0).reset_index(drop=True)
        else:
            cleaned = cleaned.fillna(0.0)

        # Garantia final absoluta contra NaNs residuais
        cleaned = cleaned.fillna(0.0)
        return cleaned

    @classmethod
    def compute_all_features(
        cls,
        df_candles: Union[pd.DataFrame, List[Dict[str, Any]]],
        macro_df: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        order_book: Optional[OrderBookDepth] = None,
        funding_df: Optional[pd.DataFrame] = None,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Interface Primária Padronizada do Módulo:
        `FeatureEngineer.compute_all_features(df_candles, macro_df=None, order_book=None, funding_df=None) -> pd.DataFrame`

        Parâmetros:
        - `df_candles`: DataFrame contendo colunas ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        - `macro_df`: Opcional. DataFrame de dados macroeconômicos com release_date ou dicionário de contexto.
        - `order_book`: Opcional. Contrato OrderBookDepth com spread e imbalance do livro de ofertas.
        - `funding_df`: Opcional. DataFrame com histórico de taxas de financiamento (funding rate).
        - `drop_na`: Se True, remove as linhas iniciais de warmup (SMA 200) se len > 200, ou faz bfill seguro.

        Retorna:
        - pd.DataFrame enriquecido com todos os indicadores técnicos locais e features sem lookahead bias.
        """
        if not HAS_TA:
            raise ImportError("A biblioteca 'ta' é mandatória para o cálculo de indicadores técnicos locais.")

        if isinstance(df_candles, list):
            df = pd.DataFrame(df_candles)
        else:
            df = df_candles.copy()

        if df.empty:
            logger.warning("DataFrame de velas vazio fornecido a compute_all_features.")
            return pd.DataFrame()

        # Validação de integridade das colunas mandatórias
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame de velas não contém as colunas mandatórias: {missing_cols}")

        # Garante ordenação cronológica estrita ascendente
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Garante coerência geométrica dos dados numéricos
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 1. Indicadores de Tendência
        df = cls.add_trend_indicators(df)

        # 2. Osciladores de Momento
        df = cls.add_oscillator_indicators(df)

        # 3. Métricas de Volatilidade (ATR mandatório para SL/TP e Bandas de Bollinger)
        df = cls.add_volatility_indicators(df)

        # 4. Volume e Microestrutura (OBV, Volume SMA ratio, Z-score, VWAP e OrderBookDepth)
        df = cls.add_volume_and_microstructure(df, order_book=order_book)

        # 5. Retornos e Variações de Preço
        df = cls.add_returns_and_volatility_features(df)

        # 6. Fatores Macroeconômicos com Defasagem (Anti-Lookahead)
        df = cls.add_macro_features(df, macro_df=macro_df)

        # 7. Regime de Financiamento / Funding Carry
        df = cls.add_funding_features(df, funding_df=funding_df)

        # 8. Estrutura de Mercado, Suporte/Resistência, Zonas de Liquidez & POI (Smart Money Concepts)
        try:
            from data.market_structure import MarketStructureDetector
            ms_detector = MarketStructureDetector()
            df = ms_detector.compute_all_structure_features(df)
        except Exception as e:
            logger.debug(f"Aviso ao calcular features de estrutura de mercado: {e}")

        # 9. Limpeza e Tratamento Numérico
        df = cls.clean_features(df, drop_na=drop_na)

        return df
