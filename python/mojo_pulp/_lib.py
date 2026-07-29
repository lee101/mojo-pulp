"""ctypes bridge to the Mojo simplex kernel."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
from numbers import Integral

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "solver.mojo")
LIB = os.environ.get("MOJO_PULP_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pulp.so"
)

I = ctypes.c_int64
F = ctypes.c_double


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PULP_LIB"):
        if os.path.isfile(LIB):
            return LIB
        raise BuildError(f"MOJO_PULP_LIB does not name a file: {LIB}")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run `pixi run build` first")
    script = os.path.join(ROOT, "build", "build.sh")
    try:
        proc = subprocess.run(
            ["bash", script], capture_output=True, text=True, timeout=1800
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuildError(f"failed to run Mojo build: {error}") from error
    if proc.returncode != 0 or not os.path.exists(LIB):
        output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        raise BuildError(output[:4000] or f"Mojo build exited with status {proc.returncode}")
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        fn = _library.mp_solve_lp
        fn.argtypes = (
            [ctypes.c_void_p] * 4
            + [I] * 2
            + [ctypes.c_void_p] * 4
            + [I, F]
        )
        fn.restype = I
    return _library


def solve_dense(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    senses: np.ndarray,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    senses = np.asarray(senses)
    if a.ndim != 2:
        raise ValueError("a must be a two-dimensional matrix")
    m, n = a.shape
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},)")
    if c.shape != (n,):
        raise ValueError(f"c must have shape ({n},)")
    if senses.shape != (m,):
        raise ValueError(f"senses must have shape ({m},)")

    def safely_representable_as_float64(array: np.ndarray) -> bool:
        if np.issubdtype(array.dtype, np.integer):
            return bool(np.all(array >= -(2**53)) and np.all(array <= 2**53))
        return bool(
            np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize <= 8
        )

    if not all(safely_representable_as_float64(array) for array in (a, b, c)):
        raise TypeError("a, b, and c must be safely convertible to float64")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(
        np.isfinite(c)
    ):
        raise ValueError("a, b, and c must contain only finite values")
    if not np.issubdtype(senses.dtype, np.integer):
        raise TypeError("senses must have an integer dtype")
    if not np.all(np.isin(senses, (-1, 0, 1))):
        raise ValueError("senses entries must be -1, 0, or 1")
    if not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer")
    max_iter = int(max_iter)
    if max_iter < 0 or max_iter > np.iinfo(np.int64).max:
        raise ValueError("max_iter must be between 0 and INT64_MAX")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive number")

    # These local owners stay alive until the synchronous ctypes call returns.
    # At least one element is allocated for inputs that the kernel does not
    # dereference when m or n is zero, keeping every C ABI pointer non-null.
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    senses = np.ascontiguousarray(senses, dtype=np.int64)
    a_ffi = a if a.size else np.empty(1, dtype=np.float64)
    b_ffi = b if b.size else np.empty(1, dtype=np.float64)
    c_ffi = c if c.size else np.empty(1, dtype=np.float64)
    senses_ffi = senses if senses.size else np.empty(1, dtype=np.int64)
    cols = n + 2 * m + 1
    tableau = np.empty((m + 1, cols), dtype=np.float64)
    basis = np.empty(m, dtype=np.int64)
    solution = np.empty(n, dtype=np.float64)
    stats = np.zeros(4, dtype=np.float64)

    def ptr(array: np.ndarray) -> ctypes.c_void_p:
        return ctypes.c_void_p(array.ctypes.data)

    status = int(
        lib().mp_solve_lp(
            ptr(a_ffi),
            ptr(b_ffi),
            ptr(c_ffi),
            ptr(senses_ffi),
            m,
            n,
            ptr(tableau),
            ptr(basis),
            ptr(solution),
            ptr(stats),
            max_iter,
            tolerance,
        )
    )
    return status, solution, stats
