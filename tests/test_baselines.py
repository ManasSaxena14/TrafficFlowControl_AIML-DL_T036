"""Phase 3 checks: the feature table is aligned and leak-free, and the simple baselines behave."""
import numpy as np
import pytest

from src.data import preprocess as pp
from src.data.load import load_speed
from src.eval.evaluate import evaluate_rows
from src.features import build as fb
from src.models import simple

SUBSET = 3000


@pytest.fixture(scope="module")
def cfg():
    return pp.load_config()


@pytest.fixture(scope="module")
def speed():
    return load_speed().iloc[:SUBSET]


@pytest.fixture(scope="module")
def built(cfg, speed):
    ds = pp.build_dataset(speed, cfg, "strict")
    adj = np.eye(207, dtype=np.float32)
    static = fb.static_features(speed, adj, ds["train_end"])
    s = ds["splits"]["val"]
    F = fb.build_features(s["X"], ds["scaler"], static)
    Y, M = fb.flatten_targets(s["Y"], s["M_y"])
    return ds, s, static, F, Y, M


def col(name):
    return fb.FEATURE_NAMES.index(name)


def test_feature_names_unique_and_counted():
    assert len(set(fb.FEATURE_NAMES)) == len(fb.FEATURE_NAMES) == 46
    assert fb.FEATURE_NAMES[:12] == fb.LAGS and fb.DYNAMIC == fb.FEATURE_NAMES[:41]


def test_lags_match_raw_speed_at_the_right_time(built, speed):
    _, s, _, F, _, _ = built
    v = speed.values
    for w in (0, 100, 250):
        t = int(s["origin_idx"][w])
        for k in (0, 1, 5, 11):
            real = v[t - k] > 0
            got = F[w * 207:(w + 1) * 207, col(f"speed_t-{k}" if k else "speed_t")]
            assert np.allclose(got[real], v[t - k][real], atol=1e-3)           # lag k = speed k steps BEFORE the origin
            assert np.array_equal(F[w * 207:(w + 1) * 207, col(f"real_t-{k}" if k else "real_t")], real.astype(np.float32))


def test_target_rows_follow_feature_rows(built):
    _, s, _, _, Y, M = built
    for w, n, h in [(0, 0, 0), (3, 17, 5), (40, 206, 11)]:
        a, b = Y[w * 207 + n, h], s["Y"][w, h, n]
        assert (np.isnan(a) and np.isnan(b)) or a == b
        assert M[w * 207 + n, h] == s["M_y"][w, h, n]


def test_static_features_use_train_only(speed, cfg):
    adj = np.eye(207, dtype=np.float32)
    end = pp.split_bounds(len(speed), cfg["split"])["train"][1]
    a = fb.static_features(speed, adj, end)
    changed = speed.copy()
    changed.iloc[end:] = 5.0
    assert np.array_equal(a, fb.static_features(changed, adj, end))


def test_persistence_repeats_last_value(built):
    _, _, _, F, _, _ = built
    P = simple.persistence(F)
    assert P.shape == (len(F), 12) and (P == F[:, [0]]).all()


def test_historical_average_is_train_only_and_shaped(speed, cfg, built):
    ds, s, *_ = built
    end = ds["train_end"]
    m = simple.HistoricalAverage().fit(speed, end)
    changed = speed.copy()
    changed.iloc[end:] = 1.0
    assert np.array_equal(m.table, simple.HistoricalAverage().fit(changed, end).table)
    P = m.predict(s["target_time"])
    assert P.shape == (len(s["origin_idx"]) * 207, 12) and np.isfinite(P).all()


def test_evaluate_rows_ignores_masked_and_scores_perfect_as_zero():
    rng = np.random.default_rng(1)
    Y = rng.uniform(10, 70, (500, 12)).astype(np.float32)
    M = rng.random(Y.shape) > 0.3
    Yn = np.where(M, Y, np.nan)
    df = evaluate_rows(Yn, np.where(M, Y, 999.0), M)
    assert list(df.horizon) == [str(h) for h in range(1, 13)] + ["avg"]
    assert df[["mae", "rmse", "mape"]].abs().max().max() == 0


def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    N, S = 3, 400
    F = rng.normal(size=(S * N, len(fb.FEATURE_NAMES))).astype(np.float32)
    Y = np.repeat((2.0 * F[:, [0]] + 5.0), 12, axis=1)
    M = np.ones_like(Y, dtype=bool)
    m = simple.RidgePerSensor(n_sensors=N, alphas=(0.1,)).fit(F, Y, M, F, Y, M)
    assert np.abs(m.predict(F) - Y).mean() < 0.05
