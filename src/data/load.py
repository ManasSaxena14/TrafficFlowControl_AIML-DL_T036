"""Loading METR-LA speed data, the sensor graph and sensor locations.

Every later phase imports from here, so the column order of the speed matrix is
guaranteed to match the row/column order of the adjacency matrix.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _decode(arr: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_speed(path: str | Path = RAW_DIR / "metr-la.h5") -> pd.DataFrame:
    """Return the speed matrix: rows = 5-min timestamps, columns = sensor IDs (str), values = mph.

    The file was written by pandas 0.15 under Python 2 and modern pandas cannot parse
    its byte-string metadata, so we read the HDF5 "fixed" layout directly with h5py.
    A value of 0 means the sensor was down; it is left as 0 here and masked in Phase 2.
    """
    try:
        df = pd.read_hdf(path)
        df.columns = df.columns.astype(str)
        return df
    except (TypeError, ValueError):
        pass
    with h5py.File(path, "r") as f:
        g = f["df"]
        values = g["block0_values"][:]
        items = _decode(g["block0_items"][:])
        columns = _decode(g["axis0"][:])
        index = pd.to_datetime(g["axis1"][:], unit="ns")
    df = pd.DataFrame(values, index=index, columns=items)[columns]
    df.index.name = "timestamp"
    df.columns.name = "sensor_id"
    return df


@dataclass
class SensorGraph:
    sensor_ids: list[str]
    id_to_idx: dict[str, int]
    adj: np.ndarray  # (N, N); adj[i, j] > 0 means a short road path from sensor i to sensor j


def load_graph(path: str | Path = RAW_DIR / "adj_mx.pkl") -> SensorGraph:
    """Load the DCRNN adjacency: Gaussian kernel of road-network distance, directed, self-loops = 1."""
    with open(path, "rb") as f:
        sensor_ids, id_to_idx, adj = pickle.load(f, encoding="latin1")
    sensor_ids = [str(s) for s in sensor_ids]
    return SensorGraph(sensor_ids, {str(k): int(v) for k, v in id_to_idx.items()}, adj.astype(np.float32))


def load_locations(path: str | Path = RAW_DIR / "graph_sensor_locations.csv",
                   sensor_ids: list[str] | None = None) -> pd.DataFrame:
    """Sensor latitude/longitude, indexed by sensor ID and (optionally) reordered to graph order."""
    loc = pd.read_csv(path)
    loc["sensor_id"] = loc["sensor_id"].astype(str)
    loc = loc.set_index("sensor_id")[["latitude", "longitude"]]
    return loc.loc[sensor_ids] if sensor_ids is not None else loc


def load_distances(path: str | Path = RAW_DIR / "distances_la_2012.csv") -> pd.DataFrame:
    """Pairwise road-network distances (from, to, cost in metres). Used to rebuild the adjacency in Phase 2."""
    d = pd.read_csv(path)
    d["from"] = d["from"].astype(str)
    d["to"] = d["to"].astype(str)
    return d


def load_all() -> tuple[pd.DataFrame, SensorGraph, pd.DataFrame]:
    """Load speed, graph and locations, and fail loudly if their sensor orders disagree."""
    speed = load_speed()
    graph = load_graph()
    if list(speed.columns) != graph.sensor_ids:
        raise ValueError("Speed columns are not in adjacency order; a GNN would train on a scrambled graph.")
    loc = load_locations(sensor_ids=graph.sensor_ids)
    return speed, graph, loc
