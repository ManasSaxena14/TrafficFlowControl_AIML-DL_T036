"""Phase 4 checks: neighbour direction, neighbour features, onset labelling and residual correlation."""
import numpy as np

from src.analysis import gap

# toy road: 0 -> 1 -> 2 (adj[i, j] > 0 means j is DOWNSTREAM of i); sensor 3 is isolated
ADJ = np.eye(4, dtype=np.float32)
ADJ[0, 1] = ADJ[1, 2] = 0.9


def test_direction_convention():
    down, up = gap.neighbour_weights(ADJ, "down"), gap.neighbour_weights(ADJ, "up")
    assert list(np.flatnonzero(down[0])) == [1] and list(np.flatnonzero(up[1])) == [0]   # 1 is downstream of 0; 0 upstream of 1
    assert down[3].sum() == 0 and up[3].sum() == 0                                         # isolated sensor, and self excluded


def test_neighbour_features_values_and_nan_for_isolated():
    lags = np.zeros((1, 4, 12), np.float32)
    lags[0, :, 0] = [60, 20, 55, 50]          # speed now
    lags[0, :, 3] = [60, 60, 55, 50]          # speed 15 min ago: sensor 1 just dropped by 40
    w = gap.neighbour_weights(ADJ, "down")
    f = gap.neighbour_features(lags, w, gap.neighbour_index(w))
    assert f[0, 0, 0] == 20 and f[0, 0, 4] == 20 and f[0, 0, 5] == -40      # sensor 0 sees its downstream neighbour crash
    assert f[0, 1, 0] == 55                                                  # sensor 1's downstream neighbour is sensor 2
    assert np.isnan(f[0, 2]).all() and np.isnan(f[0, 3]).all()               # no downstream link / isolated


def test_add_neighbour_columns_row_layout():
    rng = np.random.default_rng(0)
    adj = np.eye(gap.N_SENSORS, dtype=np.float32)
    adj[0, 1] = 1.0
    F = rng.uniform(10, 70, (5 * gap.N_SENSORS, 20)).astype(np.float32)
    out = gap.add_neighbour_columns(F, adj, ["down"], chunk=2)
    assert out.shape == (5 * gap.N_SENSORS, 6)
    for w in range(5):
        assert out[w * gap.N_SENSORS, 0] == F[w * gap.N_SENSORS + 1, 0]       # sensor 0's downstream mean = sensor 1's speed


def test_onset_labels_cover_target_not_origin():
    grid = np.zeros((100, 2), bool)
    grid[50:57, 0] = True
    lab = gap.onset_labels(np.array([40, 49]), grid, horizon=12)         # (2 windows * 2 sensors, 12)
    assert lab.shape == (4, 12)
    assert lab[0, 9:].all() and not lab[0, :9].any()                      # origin 40: targets 50, 51, 52 are flagged (h = 10..12)
    assert lab[2, :7].all() and not lab[2, 7:].any()                      # origin 49: targets 50..56 flagged (h = 1..7)
    assert not lab[1].any() and not lab[3].any()                          # sensor 1 never has an onset


def test_masked_corr_matches_pairwise_reference():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(500, 4))
    E[:, 1] += E[:, 0]                                                    # sensors 0 and 1 share errors
    M = rng.random(E.shape) > 0.2
    C = gap.masked_corr_matrix(E, M)
    both = M[:, 0] & M[:, 1]
    assert np.isclose(C[0, 1], np.corrcoef(E[both, 0], E[both, 1])[0, 1])
    assert C[0, 1] > 0.5 > abs(C[0, 2])


def test_residual_correlation_decays_with_hops():
    rng = np.random.default_rng(2)
    n, T = gap.N_SENSORS, 800
    adj = np.eye(n, dtype=np.float32)
    for i in range(0, 10):
        adj[i, i + 1] = 1                                                 # chain 0-1-...-10, rest isolated
    E = rng.normal(size=(T, n))
    E[:, :11] += 2 * rng.normal(size=(T, 1))                              # strong common error component on the chain only
    tab = gap.residual_correlation_by_hop(E, np.ones_like(E, bool), adj).set_index("group")
    assert tab.loc["1 link", "mean_corr"] > 0.5 > abs(tab.loc["not connected", "mean_corr"])
