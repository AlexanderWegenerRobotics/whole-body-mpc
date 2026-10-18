from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

ShapeSpec = int | tuple[int, ...] | None


def _shape(spec: ShapeSpec) -> tuple[int, ...]:
    if spec is None or spec == ():
        return ()
    if isinstance(spec, (int, np.integer)):
        return (int(spec),)
    return tuple(int(s) for s in spec)


def _plain(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


class Backend(Protocol):
    def create(self, channel: str, fields: dict[str, tuple[int, ...]]) -> None: ...
    def extend(self, channel: str, t: np.ndarray, block: dict[str, np.ndarray]) -> None: ...
    def finish(self, meta: dict | None) -> None: ...


class MemoryBackend:
    def __init__(self) -> None:
        self.channels: dict[str, dict[str, list[np.ndarray]]] = {}
        self.times: dict[str, list[np.ndarray]] = {}
        self.meta: dict | None = None

    def create(self, channel: str, fields: dict[str, tuple[int, ...]]) -> None:
        self.channels[channel] = {name: [] for name in fields}
        self.times[channel] = []

    def extend(self, channel: str, t: np.ndarray, block: dict[str, np.ndarray]) -> None:
        self.times[channel].append(t.copy())
        for name, arr in block.items():
            self.channels[channel][name].append(arr.copy())

    def finish(self, meta: dict | None) -> None:
        self.meta = meta

    def arrays(self, channel: str) -> dict[str, np.ndarray]:
        out = {"t": np.concatenate(self.times[channel])}
        for name, blocks in self.channels[channel].items():
            out[name] = np.concatenate(blocks)
        return out


class HDF5Backend:
    def __init__(self, path: Path, compression: str | None = "gzip") -> None:
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        self._compression = compression

    def create(self, channel: str, fields: dict[str, tuple[int, ...]]) -> None:
        group = self._h5.create_group(channel)
        group.create_dataset(
            "t", shape=(0,), maxshape=(None,), dtype="f8",
            chunks=(1024,), compression=self._compression,
        )
        for name, shape in fields.items():
            group.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape, dtype="f8",
                chunks=(min(1024, 2 ** 20 // max(1, int(np.prod(shape, dtype=int)))) or 1,) + shape,
                compression=self._compression,
            )

    def extend(self, channel: str, t: np.ndarray, block: dict[str, np.ndarray]) -> None:
        group = self._h5[channel]
        n = len(t)
        dset = group["t"]
        start = dset.shape[0]
        dset.resize(start + n, axis=0)
        dset[start:] = t
        for name, arr in block.items():
            dset = group[name]
            dset.resize(start + n, axis=0)
            dset[start:] = arr

    def finish(self, meta: dict | None) -> None:
        if meta is not None:
            self._h5.attrs["meta"] = json.dumps(_plain(meta), indent=2)
        self._h5.close()


class _Channel:
    def __init__(self, name: str, fields: dict[str, tuple[int, ...]], chunk: int) -> None:
        self.name = name
        self.fields = fields
        self.chunk = chunk
        self.n = 0
        self.t = np.empty(chunk)
        self.buf = {k: np.empty((chunk,) + s) for k, s in fields.items()}


class Recorder:
    def __init__(self, backend: Backend, chunk: int = 2048) -> None:
        self.backend = backend
        self.chunk = chunk
        self._channels: dict[str, _Channel] = {}
        self._closed = False

    def register(self, channel: str, **fields: ShapeSpec) -> None:
        if channel in self._channels:
            raise ValueError(f"channel {channel!r} already registered")
        if not fields:
            raise ValueError(f"channel {channel!r} has no fields")
        shapes = {name: _shape(spec) for name, spec in fields.items()}
        self._channels[channel] = _Channel(channel, shapes, self.chunk)
        self.backend.create(channel, shapes)

    def record(self, channel: str, t: float, **values: Any) -> None:
        ch = self._channels.get(channel)
        if ch is None:
            raise KeyError(
                f"channel {channel!r} was never registered "
                f"(have: {sorted(self._channels)})"
            )
        if values.keys() != ch.fields.keys():
            missing = sorted(ch.fields.keys() - values.keys())
            extra = sorted(values.keys() - ch.fields.keys())
            raise ValueError(f"{channel}: missing={missing} unexpected={extra}")

        i = ch.n
        ch.t[i] = t
        for name, value in values.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != ch.fields[name]:
                raise ValueError(
                    f"{channel}.{name}: expected {ch.fields[name]}, got {arr.shape}"
                )
            ch.buf[name][i] = arr

        ch.n += 1
        if ch.n == ch.chunk:
            self._flush_channel(ch)

    def _flush_channel(self, ch: _Channel) -> None:
        if ch.n == 0:
            return
        block = {name: buf[: ch.n] for name, buf in ch.buf.items()}
        self.backend.extend(ch.name, ch.t[: ch.n], block)
        ch.n = 0

    def flush(self) -> None:
        for ch in self._channels.values():
            self._flush_channel(ch)

    def close(self, meta: dict | None = None) -> None:
        if self._closed:
            return
        self.flush()
        self.backend.finish(meta)
        self._closed = True

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class NullRecorder:
    def register(self, channel: str, **fields: ShapeSpec) -> None:
        return None

    def record(self, channel: str, t: float, **values: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self, meta: dict | None = None) -> None:
        return None

    def __enter__(self) -> "NullRecorder":
        return self

    def __exit__(self, *exc) -> None:
        return None


class ChannelView:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    @property
    def t(self) -> np.ndarray:
        return self._arrays["t"]

    @property
    def fields(self) -> list[str]:
        return [k for k in self._arrays if k != "t"]

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __len__(self) -> int:
        return len(self._arrays["t"])

    def __repr__(self) -> str:
        return f"<ChannelView n={len(self)} fields={self.fields}>"


class RunReader:
    def __init__(self, path: Path) -> None:
        import h5py

        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        raw = self._h5.attrs.get("meta")
        self.meta: dict = json.loads(raw) if raw else {}
        self._cache: dict[str, ChannelView] = {}

    def channels(self) -> list[str]:
        return sorted(self._h5.keys())

    def __getitem__(self, channel: str) -> ChannelView:
        view = self._cache.get(channel)
        if view is None:
            if channel not in self._h5:
                raise KeyError(f"no channel {channel!r} (have: {self.channels()})")
            group = self._h5[channel]
            view = ChannelView({k: group[k][...] for k in group.keys()})
            self._cache[channel] = view
        return view

    def at(self, channel: str, t: np.ndarray | float, field: str | None = None):
        view = self[channel]
        idx = np.clip(np.searchsorted(view.t, t, side="right") - 1, 0, len(view) - 1)
        if field is not None:
            return view[field][idx]
        return {name: view[name][idx] for name in view.fields}

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "RunReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def run_metadata(
    config: dict | None = None,
    files: Iterable[Path] = (),
    repo_root: Path | None = None,
) -> dict:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "sha": _git(["rev-parse", "HEAD"], root),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
            "dirty": bool(_git(["status", "--porcelain"], root)),
        },
        "versions": {
            name: _version(name)
            for name in ("numpy", "mujoco", "pin", "pinocchio", "crocoddyl", "h5py")
        },
        "files": {str(Path(f).name): _sha256(f) for f in files},
        "config": _plain(config or {}),
    }


def write_meta(path: Path, meta: dict) -> None:
    import yaml

    Path(path).write_text(yaml.safe_dump(_plain(meta), sort_keys=False))


def make_run_dir(root: Path, name: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = Path(root) / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path
