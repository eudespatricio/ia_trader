import numpy as np
import statistical_rigor as sr


def test_pbo_keeps_recent_bars_and_no_global_seed():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, (103, 4))
    st = np.random.get_state()[1].copy()
    sr.probability_of_backtest_overfitting(m, n_splits=10)
    sr.bootstrap_performance_metrics(m[:, 0], n_iterations=50)
    assert (np.random.get_state()[1] == st).all()


def test_block_bootstrap_ci():
    r = np.random.default_rng(1).normal(0.001, 0.01, 200)
    out = sr.bootstrap_performance_metrics(r, n_iterations=100, block_size=10)
    assert out["sharpe_ci"][0] <= out["sharpe_ci"][1]
    assert out == sr.bootstrap_performance_metrics(r, n_iterations=100, block_size=10)


def test_log_trial_unique_ids_threaded(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(sr, "TRIAL_LOG_PATH", str(tmp_path / "t.csv"))
    ids = []
    def w():
        ids.append(sr.log_trial("p", "e", "s", "btc", "1h", {}, "x", True, 1, 0, 0, 0, 0, 0, "L"))
    ts = [threading.Thread(target=w) for _ in range(12)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(set(ids)) == 12
