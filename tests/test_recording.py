import numpy as np
import pytest

from wbc.recording import (
    HDF5Backend,
    MemoryBackend,
    NullRecorder,
    Recorder,
    RunReader,
    run_metadata,
)


def test_ragged_rates_keep_their_own_time_base(tmp_path):
    path = tmp_path / "log.h5"
    rec = Recorder(HDF5Backend(path), chunk=16)
    rec.register("fast", x=3)
    rec.register("slow", y=())

    for i in range(100):
        rec.record("fast", i * 0.002, x=np.full(3, float(i)))
        if i % 5 == 0:
            rec.record("slow", i * 0.002, y=float(i))
    rec.close({"note": "ragged"})

    with RunReader(path) as run:
        assert set(run.channels()) == {"fast", "slow"}
        assert len(run["fast"]) == 100
        assert len(run["slow"]) == 20
        assert run["fast"].x[7, 1] == 7.0
        assert run.meta["note"] == "ragged"


def test_chunk_boundary_is_seamless(tmp_path):
    path = tmp_path / "log.h5"
    rec = Recorder(HDF5Backend(path), chunk=8)
    rec.register("c", v=2)
    for i in range(37):
        rec.record("c", float(i), v=np.array([i, -i], dtype=float))
    rec.close()

    with RunReader(path) as run:
        view = run["c"]
        assert len(view) == 37
        assert np.array_equal(view.t, np.arange(37, dtype=float))
        assert np.array_equal(view.v[:, 0], np.arange(37, dtype=float))


def test_zero_order_hold_lookup(tmp_path):
    path = tmp_path / "log.h5"
    rec = Recorder(HDF5Backend(path))
    rec.register("c", v=())
    for i in range(10):
        rec.record("c", float(i), v=float(i))
    rec.close()

    with RunReader(path) as run:
        assert run.at("c", 3.9, field="v") == 3.0
        assert run.at("c", -1.0, field="v") == 0.0
        assert run.at("c", 99.0, field="v") == 9.0
        assert np.array_equal(run.at("c", np.array([0.5, 7.2]), field="v"), [0.0, 7.0])


def test_schema_violations_raise_immediately():
    rec = Recorder(MemoryBackend())
    rec.register("c", v=3)

    with pytest.raises(ValueError):
        rec.record("c", 0.0, v=np.zeros(4))
    with pytest.raises(ValueError):
        rec.record("c", 0.0, w=np.zeros(3))
    with pytest.raises(KeyError):
        rec.record("missing", 0.0, v=np.zeros(3))
    with pytest.raises(ValueError):
        rec.register("c", v=3)


def test_null_recorder_matches_the_interface():
    rec = NullRecorder()
    rec.register("anything", v=3)
    rec.record("anything", 0.0, v=np.zeros(3))
    rec.flush()
    rec.close({"meta": 1})


def test_metadata_captures_versions_and_hashes(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("hello")
    meta = run_metadata(config={"a": 1}, files=[target], repo_root=tmp_path)
    assert meta["config"]["a"] == 1
    assert meta["files"]["f.txt"] is not None
    assert meta["versions"]["numpy"] is not None
    assert "created" in meta
