"""Testes das correções V1-V8 (lookahead)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_integrations import (  # noqa: E402
    CandleData, FredConnector, drop_open_candles, timeframe_to_ms,
)
from data.feature_engineering import FeatureEngineer  # noqa: E402
from data.data_integrations import OrderBookDepth  # noqa: E402
from tests.test_lookahead_perturbation_audit import make_candles, feats  # noqa: E402


def _rd(sid, d):
    return FredConnector(api_key="x")._calculate_effective_release_date(sid, d)


def test_v1_release_dates():
    assert _rd("FEDFUNDS", "2024-01-01") == "2024-02-01"   # média mensal: após fim do mês
    assert _rd("FEDFUNDS", "2024-12-01") == "2025-01-01"
    assert _rd("CPIAUCSL", "2024-01-01") == "2024-02-20"
    assert _rd("DGS10", "2024-03-05") == "2024-03-06"
    assert _rd("T10Y2Y", "2024-03-05") == "2024-03-06"


def test_v2_v3_no_backfill_short_series():
    df = make_candles(60)
    out = FeatureEngineer.clean_features(
        pd.DataFrame({"a": [None, None, 5.0, 6.0]}), drop_na=True)
    assert list(out["a"]) == [0.0, 0.0, 5.0, 6.0]


def test_v4_dict_snapshot_last_row_only():
    df = make_candles(50)
    a = feats(df, macro_df={"cpi": 400.0, "vix": 50.0})
    assert a["macro_vix"].iloc[-1] == 50.0 and (a["macro_vix"].iloc[:-1] != 50.0).all()


def test_v6_missing_release_date_shifts_one_day():
    df = make_candles(100)
    ts = pd.to_datetime(df["timestamp"].iloc[10], unit="ms", utc=True)
    macro = pd.DataFrame({"date": [ts], "vix": [77.0]})
    a = feats(df, macro_df=macro)
    # candle idx 10 (mesmo instante da observação) ainda não vê o valor; 24h depois vê
    assert a["macro_vix"].iloc[10] != 77.0
    assert a["macro_vix"].iloc[10 + 24] == 77.0


def test_v7_drop_open_candles():
    H = 3_600_000
    cs = [CandleData(timestamp=i * H, open=1, high=1, low=1, close=1, volume=1, quote_volume=1,
                     symbol="X", timeframe="1h") for i in range(3)]
    out = drop_open_candles(cs, "1h", now_ms=2 * H + 10)  # candle 2 em formação
    assert [c.timestamp for c in out] == [0, H]
    assert timeframe_to_ms("4h") == 4 * H
    assert drop_open_candles(cs, "weird", now_ms=0) == cs


def test_v8_order_book_last_row_only():
    df = make_candles(50)
    ob = OrderBookDepth(**{k: v for k, v in _ob_kwargs().items()})
    out = FeatureEngineer.compute_all_features(df, order_book=ob, drop_na=False)
    assert (out["ob_imbalance"].iloc[:-1] == 0.0).all()
    assert out["ob_imbalance"].iloc[-1] == float(ob.imbalance)


def _ob_kwargs():
    import dataclasses
    from data.data_integrations import OrderBookDepth as O
    kw = {}
    for f in dataclasses.fields(O):
        if f.name in ("bids", "asks"):
            kw[f.name] = [(100.0, 5.0)] if f.name == "bids" else [(100.5, 1.0)]
        elif f.type in ("str", str):
            kw[f.name] = "X"
        elif f.default is not dataclasses.MISSING:
            continue
        else:
            kw[f.name] = 1.0
    return kw
