"""Honest end-to-end benchmarks against PuLP 2.8 with system CBC."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_pulp as mp  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def dense_model(module, rows=60, cols=100):
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, size=(rows, cols))
    b = rng.uniform(15.0, 30.0, size=rows)
    c = rng.uniform(0.0, 1.0, size=cols)
    x = [module.LpVariable(f"x_{j}", lowBound=0, upBound=2) for j in range(cols)]
    problem = module.LpProblem("dense", module.LpMaximize)
    problem += module.lpDot(c, x)
    for i in range(rows):
        problem += module.lpDot(a[i], x) <= b[i]
    return problem


def transport_model(module, size=12):
    rng = np.random.default_rng(7)
    supply = rng.integers(20, 50, size=size)
    demand = supply.copy()
    costs = rng.integers(1, 30, size=(size, size))
    x = module.LpVariable.dicts(
        "ship", (range(size), range(size)), lowBound=0
    )
    problem = module.LpProblem("transport", module.LpMinimize)
    problem += module.lpSum(
        costs[i, j] * x[i][j] for i in range(size) for j in range(size)
    )
    for i in range(size):
        problem += module.lpSum(x[i][j] for j in range(size)) == supply[i]
    for j in range(size):
        problem += module.lpSum(x[i][j] for i in range(size)) == demand[j]
    return problem


def assignment_model(module, size=8):
    rng = np.random.default_rng(11)
    costs = rng.integers(1, 100, size=(size, size))
    x = module.LpVariable.dicts(
        "assign", (range(size), range(size)), cat=module.LpBinary
    )
    problem = module.LpProblem("assignment", module.LpMinimize)
    problem += module.lpSum(
        costs[i, j] * x[i][j] for i in range(size) for j in range(size)
    )
    for i in range(size):
        problem += module.lpSum(x[i][j] for j in range(size)) == 1
    for j in range(size):
        problem += module.lpSum(x[i][j] for i in range(size)) == 1
    return problem


def tiny_model(module):
    p = module.LpProblem("tiny", module.LpMaximize)
    x = module.LpVariable("x", lowBound=0)
    y = module.LpVariable("y", lowBound=0)
    p += 3 * x + 2 * y
    p += x + y <= 4
    p += x <= 2
    p += y <= 3
    return p


def benchmark():
    mojo_solver = mp.MOJO_CMD(msg=False)
    pulp_solver = pulp.COIN_CMD(msg=False)
    mp_problem = tiny_model(mp)
    mp_problem.solve(mojo_solver)

    cases = []

    ours = dense_model(mp)
    theirs = dense_model(pulp)
    cases.append(
        (
            "Dense LP solve (60 rows, 100 vars)",
            lambda problem=ours: problem.solve(mojo_solver),
            lambda problem=theirs: problem.solve(pulp_solver),
        )
    )

    ours = transport_model(mp)
    theirs = transport_model(pulp)
    cases.append(
        (
            "Transportation solve (12 x 12)",
            lambda problem=ours: problem.solve(mojo_solver),
            lambda problem=theirs: problem.solve(pulp_solver),
        )
    )

    ours = assignment_model(mp)
    theirs = assignment_model(pulp)
    cases.append(
        (
            "Binary assignment solve (8 x 8)",
            lambda problem=ours: problem.solve(mojo_solver),
            lambda problem=theirs: problem.solve(pulp_solver),
        )
    )

    ours = tiny_model(mp)
    theirs = tiny_model(pulp)
    cases.append(
        (
            "Tiny LP solve",
            lambda problem=ours: problem.solve(mojo_solver),
            lambda problem=theirs: problem.solve(pulp_solver),
        )
    )

    cases.append(
        (
            "Dense model construction",
            lambda: dense_model(mp),
            lambda: dense_model(pulp),
        )
    )

    cpu = platform.processor()
    if not cpu or cpu == platform.machine():
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = platform.machine()
    print(f"Machine: {cpu} ({platform.platform()})")
    print(f"Reference: PuLP {pulp.__version__} with COIN-OR CBC")
    print()
    print("| benchmark | mojo-pulp | PuLP/CBC | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_fn, pulp_fn in cases:
        mojo_fn()
        pulp_fn()
        mojo_time = timeit(mojo_fn)
        pulp_time = timeit(pulp_fn)
        ratio = pulp_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{pulp_time * 1e3:.3f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    benchmark()
