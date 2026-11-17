"""Batch-one CPU decoder backed by the SHRF Rust cdylib."""
import ctypes
import math
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch


class _Config(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "d", "trees", "depth", "r", "q", "m", "beam", "lag", "shrinkage", "context_lags")] + [
        (name, ctypes.c_double) for name in (
            "rho", "sketch_rho", "tau", "kappa_min", "kappa_max", "gate_power", "decay")]


_PTR = ctypes.POINTER(ctypes.c_double)
_NATIVE = {"win32": "shrf_core.dll", "darwin": "libshrf_core.dylib"}


class RustSHRF:
    def __init__(self, layer, library_path=None):
        root = Path(__file__).resolve().parent.parent
        if library_path is None:
            library_path = (root / "rust" / "shrf-core" / "target" / "release" /
                            _NATIVE.get(sys.platform, "libshrf_core.so"))
        if not Path(library_path).is_file():
            raise FileNotFoundError("Build the decoder with cargo build --release --manifest-path rust/shrf-core/Cargo.toml")
        lib = self.lib = ctypes.CDLL(str(Path(library_path).resolve()))
        if not hasattr(lib, "shrf_abi_version") or lib.shrf_abi_version() != 5:
            raise RuntimeError("Rebuild the native library for the current SHRF ABI")
        lib.shrf_new.argtypes = [_Config]
        lib.shrf_new.restype = ctypes.c_void_p
        lib.shrf_set_weights.argtypes = [ctypes.c_void_p, _PTR, ctypes.c_size_t]
        lib.shrf_set_weights.restype = ctypes.c_bool
        lib.shrf_step.argtypes = [ctypes.c_void_p, _PTR, _PTR, ctypes.c_double]
        lib.shrf_step.restype = ctypes.c_bool
        lib.shrf_decode.argtypes = [ctypes.c_void_p, _PTR, _PTR, ctypes.c_size_t, _PTR]
        lib.shrf_decode.restype = ctypes.c_size_t
        for fn in (lib.shrf_reset, lib.shrf_free):
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = None
        c = layer.cfg
        self._lock = Lock()
        self.d = c.d_model
        self._handle = lib.shrf_new(_Config(c.d_model, c.n_trees, c.depth, c.r_key, c.d_value,
                                          c.sketch_width, c.beam, c.write_lag, int(c.use_shrinkage),
                                          c.context_lags, c.ridge, c.sketch_ridge, c.tau,
                                          c.kappa_min, c.kappa_max, c.gate_power, c.decay))
        if not self._handle:
            raise ValueError("Invalid native decoder configuration")
        pieces = [layer.W_k, layer.W_v.weight, layer.W_o.weight, layer.write_gate.weight,
                  layer.write_gate.bias, layer.routing.dirs, layer.routing.theta]
        if layer.sketch is not None:
            pieces += [layer.sketch.omega, layer.sketch.phase]
        if layer.context_mix is not None:
            pieces.append(layer.context_mix)
        weight = np.concatenate([p.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
                                 for p in pieces])
        if not self.lib.shrf_set_weights(self._handle, self._ptr(weight), weight.size):
            self.close()
            raise ValueError("Native model weights are invalid or use an incompatible layout")

    @staticmethod
    def _ptr(array):
        return array.ctypes.data_as(_PTR)

    def step(self, x, *, write_weight=1.0):
        if x.shape != (self.d,) or x.device.type != "cpu" or not x.is_floating_point():
            raise ValueError("Native decoder expects a floating-point CPU tensor of shape [d_model]")
        if not math.isfinite(write_weight) or not 0 <= write_weight <= 1:
            raise ValueError("write_weight must be finite and in [0, 1]")
        src = x.detach().to(torch.float64).contiguous().numpy()
        out = np.empty(self.d, dtype=np.float64)
        with self._lock:
            if not self.lib.shrf_step(self._live_handle(), self._ptr(src), self._ptr(out), write_weight):
                raise ValueError("Native input or projection must be finite and within the accumulation range")
        return torch.from_numpy(out).to(x.dtype)

    def _live_handle(self):
        if self._handle is None:
            raise RuntimeError("Native decoder is closed")
        return self._handle

    def prefill(self, x, *, write_weight=None):
        """Stream [time, width] through the same state in one FFI call."""
        if x.ndim != 2 or x.shape[1] != self.d or x.device.type != "cpu" or not x.is_floating_point():
            raise ValueError("Native prefill expects a floating-point CPU tensor of shape [time, d_model]")
        weight = None
        if write_weight is not None:
            if write_weight.shape != (x.shape[0],) or write_weight.device.type != "cpu":
                raise ValueError("Native prefill weights must be a CPU tensor of shape [time]")
            weight = write_weight.detach().to(torch.float64).contiguous().numpy()
            if not (np.isfinite(weight) & (weight >= 0) & (weight <= 1)).all():
                raise ValueError("write_weight must be finite and in [0, 1]")
        src = x.detach().to(torch.float64).contiguous().numpy()
        out = np.empty_like(src)
        with self._lock:
            accepted = self.lib.shrf_decode(self._live_handle(), self._ptr(src), self._ptr(out), len(src),
                                            None if weight is None else self._ptr(weight))
            if accepted != len(src):
                raise ValueError(f"Native prefill rejected input or projection after {accepted} accepted tokens")
        return torch.from_numpy(out).to(x.dtype)

    def reset(self):
        with self._lock:
            self.lib.shrf_reset(self._live_handle())

    def close(self):
        with self._lock:
            if self._handle is not None:
                self.lib.shrf_free(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
