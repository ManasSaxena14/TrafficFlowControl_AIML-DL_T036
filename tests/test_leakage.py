"""Phase 2 gate: nothing the model is scored on may influence anything it is trained on."""
import numpy as np
import pytest

from src.data import preprocess as pp
from src.data.load import load_all, load_distances
from src.eval.metrics import masked_mae, masked_mape, masked_rmse

SUBSET = 3000   # ~10 days: enough for all three blocks, small enough to build in seconds


@pytest.fixture(scope="module")
def cfg():
    return pp.load_config()


@pytest.fixture(scope="module")
def data():
    return load_all()


@pytest.fixture(scope="module")
def full_plan(cfg, data):
    speed, _, _ = data
    return pp.plan_windows(len(speed), cfg, "strict")[0]


@pytest.fixture(scope="module")
def small(cfg, data):
    speed, _, _ = data
    sub = speed.iloc[:SUBSET]
    return sub, pp.build_dataset(sub, cfg, "strict")


def targets(origins, hor=12):
    return origins[:, None] + np.arange(1, hor + 1)


def test_target_timestamps_strictly_ordered(full_plan):
    t = {k: targets(v) for k, v in full_plan.items()}
    assert t["train"].max() < t["val"].min() <= t["val"].max() < t["test"].min()


def test_no_target_timestamp_in_two_splits(full_plan):
    sets = {k: set(targets(v).ravel().tolist()) for k, v in full_plan.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"]) and not (sets["train"] & sets["test"])


def test_purge_removes_exactly_the_boundary_windows(cfg, data):
    """DCRNN-style window splitting leaks; strict mode does not (this is why the purge exists)."""
    speed, _, _ = data
    dc = pp.plan_windows(len(speed), cfg, "dcrnn")[0]
    assert set(targets(dc["train"]).ravel().tolist()) & set(targets(dc["val"]).ravel().tolist())
    b = pp.split_bounds(len(speed), cfg["split"])
    strict = pp.plan_windows(len(speed), cfg, "strict")[0]
    for k, (a, z) in b.items():
        tt = targets(strict[k])
        assert tt.min() >= a and tt.max() <= z - 1


def test_scaler_is_train_slice_nanmean(small, cfg):
    sub, ds = small
    v = sub.values.astype("float32")
    raw = np.where(v > 0, v, np.nan)
    tr = raw[:ds["train_end"]]
    assert ds["scaler"]["mu"] == pytest.approx(float(np.nanmean(tr)))
    assert ds["scaler"]["sd"] == pytest.approx(float(np.nanstd(tr)))
    assert ds["train_end"] == pp.split_bounds(len(sub), cfg["split"])["train"][1]


def test_scaler_ignores_val_and_test(small, cfg):
    sub, ds = small
    changed = sub.copy()
    changed.iloc[ds["train_end"]:] = 99.0           # wreck everything after the train block
    ds2 = pp.build_dataset(changed, cfg, "strict")
    assert ds2["scaler"] == ds["scaler"]


def test_inputs_never_see_the_future(small, cfg):
    """Altering readings after t0 must not change the inputs of any window that ends at or before t0."""
    sub, ds = small
    t0 = 2600
    changed = sub.copy()
    changed.iloc[t0 + 1:] = 12.0
    ds2 = pp.build_dataset(changed, cfg, "strict")
    for k in ("val", "test"):
        o = ds["splits"][k]["origin_idx"]
        assert np.array_equal(o, ds2["splits"][k]["origin_idx"])
        keep = o <= t0
        assert np.array_equal(ds["splits"][k]["X"][keep], ds2["splits"][k]["X"][keep])
    assert (ds["splits"]["val"]["origin_idx"] <= t0).any()   # the check is not vacuous


def test_masked_positions_contribute_zero_to_metrics():
    rng = np.random.default_rng(0)
    y = rng.uniform(5, 70, (4, 12, 20))
    p = y + rng.normal(0, 3, y.shape)
    m = rng.random(y.shape) > 0.2
    base = (masked_mae(y, p, m), masked_rmse(y, p, m), masked_mape(y, p, m))
    y2, p2 = y.copy(), p.copy()
    y2[~m], p2[~m] = np.nan, 1e6              # perturb masked points wildly (NaN targets, absurd forecasts)
    assert (masked_mae(y2, p2, m), masked_rmse(y2, p2, m), masked_mape(y2, p2, m)) == base


def test_outages_are_nan_targets_and_flagged_inputs(small):
    _, ds = small
    s = ds["splits"]["train"]
    assert not (s["Y"][s["M_y"]] <= 0).any()          # no real target is 0 mph
    assert np.isnan(s["Y"][~s["M_y"]]).all()          # masked targets stay NaN
    assert np.isfinite(s["X"]).all()                   # inputs are imputed, never NaN
    assert set(np.unique(s["X"][..., 1])) <= {0.0, 1.0}


def test_sensor_order_matches_adjacency(data):
    speed, graph, _ = data
    assert list(speed.columns) == graph.sensor_ids
    arrays = pp.adjacency_arrays(graph.adj, graph.sensor_ids)
    assert list(arrays["sensor_ids"]) == list(speed.columns)
    assert arrays["adj"].shape[0] == speed.shape[1]


def test_rebuilt_adjacency_equals_pickle(cfg, data):
    _, graph, _ = data
    rebuilt = pp.rebuild_adjacency(load_distances(), graph.sensor_ids, cfg["adjacency"]["kernel_threshold"])
    assert np.allclose(rebuilt, graph.adj, atol=1e-5)
    assert not np.allclose(rebuilt, rebuilt.T)         # still directed


def test_shapes_and_cyclic_features(small, cfg):
    _, ds = small
    for s in ds["splits"].values():
        S = len(s["origin_idx"])
        assert s["X"].shape == (S, 12, 207, pp.N_FEATURES)
        assert s["Y"].shape == (S, 12, 207) and s["M_y"].shape == s["Y"].shape
        assert s["target_time"].shape == (S, 12)
    import pandas as pd
    f = pp.time_features(pd.DatetimeIndex(["2012-03-01 23:55", "2012-03-02 00:05"]))
    assert abs(f[0, 0] - f[1, 0]) < 0.05 and abs(f[0, 1] - f[1, 1]) < 1e-3   # 23:55 ~ 00:05


def test_full_split_sizes(full_plan):
    """Purge costs 11 windows per boundary: block of n steps -> n - 11 windows (train also loses its first 11 to history)."""
    n = {k: len(v) for k, v in full_plan.items()}
    assert n == {"train": 23990 - 11 - 12, "val": 3428 - 11, "test": 6854 - 11}
