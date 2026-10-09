"""Phase 1 data checks. These guard the assumptions every later phase relies on."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import eda
from src.data.load import load_all, load_distances


@pytest.fixture(scope="module")
def data():
    return load_all()


def test_shapes(data):
    speed, graph, loc = data
    assert speed.shape == (34272, 207)
    assert graph.adj.shape == (207, 207)
    assert loc.shape == (207, 2)


def test_sensor_order_matches_graph(data):
    speed, graph, loc = data
    assert list(speed.columns) == graph.sensor_ids
    assert list(loc.index) == graph.sensor_ids
    assert all(graph.id_to_idx[s] == i for i, s in enumerate(graph.sensor_ids))


def test_regular_five_minute_clock(data):
    speed, _, _ = data
    assert speed.index.is_monotonic_increasing
    assert (speed.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()


def test_values_are_speeds(data):
    speed, _, _ = data
    v = speed.values
    assert not np.isnan(v).any()
    assert v.min() >= 0 and v.max() <= 100
    assert 0.07 < (v == 0).mean() < 0.09  # ~8% outages, encoded as 0


def test_adjacency_is_directed_kernel(data):
    _, graph, _ = data
    A = graph.adj
    assert np.allclose(np.diag(A), 1.0)
    assert not np.allclose(A, A.T)               # direction matters: keep it
    nz = A[A > 0]
    assert nz.min() >= 0.1 and nz.max() <= 1.0  # Gaussian kernel, thresholded at 0.1


def test_adjacency_matches_distances(data):
    """Every non-zero off-diagonal entry must correspond to a road distance from i to j."""
    _, graph, _ = data
    d = load_distances()
    pairs = set(zip(d["from"], d["to"]))
    ids = graph.sensor_ids
    for i, j in np.argwhere(graph.adj > 0)[:300]:
        if i != j:
            assert (ids[i], ids[j]) in pairs


def test_masked_treats_zero_as_missing(data):
    speed, _, _ = data
    m = eda.masked(speed.iloc[:500])
    assert m.isna().sum().sum() == (speed.iloc[:500] == 0).sum().sum()


def test_train_slice_is_chronological_prefix(data):
    speed, _, _ = data
    tr = eda.train_slice(speed)
    assert len(tr) == round(0.7 * len(speed))
    assert tr.index[-1] < speed.index[len(tr)]


def test_lagged_corr_sign_convention():
    """corr(x[t-k], y[t]) peaks at k = +3 when y is x delayed by 3 steps."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = np.roll(x, 3)
    best = max(range(-6, 7), key=lambda k: eda.lagged_corr(x, y, k))
    assert best == 3
