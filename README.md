# mojo-pulp

`mojo-pulp` is a standalone implementation of linear and mixed-integer model
building with a dense simplex solver written in Mojo. Its Python layer mirrors
the commonly used PuLP names and signatures, so covered models normally need
only:

```python
import mojo_pulp as pulp
```

The project is useful for small and medium dense problems, repeated solves, and
applications where starting an external solver process costs more than the
optimization itself. It is not a wrapper around PuLP or CBC: the relaxation
solver is compiled Mojo code and the MILP search is part of this package.

## Coverage

The covered modeling API includes:

- `LpProblem`, `LpVariable`, `LpAffineExpression`, and `LpConstraint`
- natural linear algebra and comparisons such as `3*x + y <= 10`
- `lpSum`, `lpDot`, `value`, `valueOrDefault`, and variable factories
  `dict`, `dicts`, and `matrix`
- continuous, integer, and binary variables
- finite, one-sided, fixed, and free variable bounds
- minimization and maximization with `<=`, `==`, and `>=` constraints
- `MOJO_CMD`, the compatibility spelling `PULP_CBC_CMD`, status dictionaries,
  solver lookup, re-solve, rounding, validity, and infeasibility helpers
- dense two-phase primal simplex and deterministic branch-and-bound

The test suite compares numerical results and behavior with the pinned PuLP
2.8.0 and COIN-OR CBC. Each item above is exercised by the suite, including
factory and helper behavior, all bound transformations and row senses, LP and
MILP examples, re-solve and rounding, status/error paths, and malformed FFI
inputs. Numerical parity cases include randomized LPs, blending,
transportation, assignment, knapsack, production, negative right-hand sides,
free variables, infeasible models, and unbounded models.

Not covered are LP/MPS import and export, SOS constraints, elastic constraints,
column generation, callbacks, warm starts, CBC's cuts and presolve controls,
dual prices and reduced costs, or PuLP's other external solver integrations.
`constraint.pi` and `variable.dj` are therefore `None`. The simplex tableau is
dense, so this is not a replacement for a sparse industrial solver on very
large models. Branch-and-bound is deliberately compact and does not implement
modern cutting planes or heuristics.

## Install

The pinned Mojo nightly and all Python/reference dependencies are managed by
Pixi:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-pulp.so`. Set `MOJO_PULP_LIB` to load a
prebuilt shared library from another location.

## Usage

```python
import mojo_pulp as pulp

model = pulp.LpProblem("product_mix", pulp.LpMaximize)
tables = pulp.LpVariable("tables", lowBound=0, cat=pulp.LpInteger)
chairs = pulp.LpVariable("chairs", lowBound=0, cat=pulp.LpInteger)

model += 45 * tables + 30 * chairs
model += 4 * tables + 3 * chairs <= 120, "wood"
model += 2 * tables + chairs <= 50, "labor"

status = model.solve(pulp.MOJO_CMD(msg=False))
print(pulp.LpStatus[status])
print(tables.value(), chairs.value(), pulp.value(model.objective))
```

This prints:

```text
Optimal
15.0 20.0 1275.0
```

## How it works

The Python layer stores sparse affine expressions in insertion-ordered maps.
At solve time it transforms every original variable into one or two
non-negative variables: lower bounds become shifts, upper-only variables are
reflected, and free variables become a positive part minus a negative part.
Finite ranges become additional rows.

The resulting matrix, right-hand side, objective, and row senses are contiguous
NumPy arrays in row-major `float64` layout. `ctypes` passes pointers to their
buffers across a C ABI into one Mojo compilation unit. NumPy owns every
buffer, including the tableau and basis, so the FFI does not allocate or retain
Python memory.

Mojo builds a phase-I tableau with slack, surplus, and artificial columns,
removes zero-valued artificial basic variables, and runs phase II using Bland's
rule to prevent cycling. Integer models repeatedly solve these relaxations in a
depth-first branch-and-bound search. Solver status codes intentionally match
PuLP's values.

## Benchmarks

Measured on 2026-08-26 on an Intel Xeon E5-2697 v4 host running
Linux 6.8.0, using the pinned Mojo `1.1.0.dev2026081105`, PuLP 2.8.0, and
COIN-OR CBC 2.10.13. Times are the best of three runs under the machine-wide
lock from `pixi run bench`.
Solve rows include model-to-matrix conversion for mojo-pulp and PuLP's normal
external CBC invocation.

| benchmark | mojo-pulp | PuLP/CBC | upstream / Mojo |
|---|---:|---:|---:|
| Dense LP solve (60 rows, 100 vars) | 12.944 ms | 61.551 ms | 4.76x |
| Transportation solve (12 x 12) | 0.922 ms | 41.109 ms | 44.60x |
| Binary assignment solve (8 x 8) | 0.916 ms | 48.541 ms | 53.00x |
| Tiny LP solve | 0.070 ms | 36.714 ms | 527.08x |
| Dense model construction | 9.789 ms | 56.658 ms | 5.79x |

The large advantage on tiny solves is mostly avoided model serialization and
process startup, not a claim that this compact simplex outperforms CBC's
algorithms. Dense expression construction directly accumulates dot-product
coefficients instead of allocating and copying a one-term expression for every
product.

No GPU path is included. The dominant pivot update performs about two floating-
point operations while moving roughly 24 bytes per tableau element, or about
0.08 FLOP/byte. That is far below the 2 FLOP/byte threshold where transferring
and launching this kernel on a GPU could be justified, so the solver remains
CPU-only.

Run the benchmark again on the current machine with:

```bash
pixi run bench
```

## License

MIT
