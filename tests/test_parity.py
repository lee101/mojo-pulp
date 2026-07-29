import inspect

import numpy as np
import pulp
import pytest

import mojo_pulp as mp
from mojo_pulp._lib import lib, solve_dense


UPSTREAM_SOLVER = pulp.COIN_CMD(msg=False)
MOJO_SOLVER = mp.MOJO_CMD(msg=False)


def solved_objective(module, problem, solver):
    status = problem.solve(solver)
    return status, module.value(problem.objective)


def test_core_constructor_signatures_cover_upstream_prefixes():
    assert list(inspect.signature(mp.LpVariable).parameters) == list(
        inspect.signature(pulp.LpVariable).parameters
    )
    assert list(inspect.signature(mp.LpProblem).parameters) == list(
        inspect.signature(pulp.LpProblem).parameters
    )
    assert list(inspect.signature(mp.LpConstraint).parameters) == list(
        inspect.signature(pulp.LpConstraint).parameters
    )


def test_affine_expression_arithmetic_matches_upstream():
    mx, my = mp.LpVariable("x"), mp.LpVariable("y")
    px, py = pulp.LpVariable("x"), pulp.LpVariable("y")
    ours = 2 * mx - 3 * my + 7 + mx
    theirs = 2 * px - 3 * py + 7 + px
    assert ours.constant == theirs.constant
    assert {v.name: c for v, c in ours.items()} == {
        v.name: c for v, c in theirs.items()
    }
    assert (ours / 2).constant == pytest.approx(theirs.constant / 2)


def test_lpdot_direct_accumulation_handles_numpy_scalars_and_cancellation():
    x, y = mp.LpVariable("x"), mp.LpVariable("y")
    terms = [x + 2, y - 3, x]
    coefficients = np.array([2, -4, -2], dtype=np.float64)
    result = mp.lpDot(coefficients, terms)
    assert result.constant == pytest.approx(16)
    assert list(result.items()) == [(y, -4)]


def test_lpsum_accumulates_without_aliasing_inputs():
    x, y = mp.LpVariable("x"), mp.LpVariable("y")
    source = 2 * x + 1
    result = mp.lpSum([source, y, 3])
    source.addterm(x, 5)
    assert result.constant == pytest.approx(4)
    assert list(result.items()) == [(x, 2), (y, 1)]


def test_add_in_place_preserves_self_aliasing_behavior():
    x = mp.LpVariable("x")
    expression = 2 * x + 1
    expression.addInPlace(expression, -1)
    assert expression.constant == 0
    assert list(expression.items()) == []


def build_basic(module):
    problem = module.LpProblem("basic", module.LpMaximize)
    x = module.LpVariable("x", lowBound=0)
    y = module.LpVariable("y", lowBound=0)
    problem += 3 * x + 2 * y
    problem += x + y <= 4, "capacity"
    problem += x <= 2
    problem += y <= 3
    return problem, (x, y)


def test_basic_max_lp_matches_pulp():
    ours, ours_vars = build_basic(mp)
    theirs, theirs_vars = build_basic(pulp)
    ours_status, ours_obj = solved_objective(mp, ours, MOJO_SOLVER)
    their_status, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_status == their_status == mp.LpStatusOptimal
    assert ours_obj == pytest.approx(their_obj, abs=1e-9)
    assert [v.value() for v in ours_vars] == pytest.approx(
        [v.value() for v in theirs_vars], abs=1e-9
    )
    assert ours.constraints["capacity"].slack == pytest.approx(
        theirs.constraints["capacity"].slack, abs=1e-9
    )


def build_free_equality(module):
    problem = module.LpProblem("free", module.LpMinimize)
    x = module.LpVariable("x")
    y = module.LpVariable("y", lowBound=0, upBound=5)
    problem += x + 2 * y
    problem += x + y >= 3
    problem += x - y == 1
    return problem, (x, y)


def test_minimization_equality_and_free_variable_match_pulp():
    ours, ours_vars = build_free_equality(mp)
    theirs, theirs_vars = build_free_equality(pulp)
    _, ours_obj = solved_objective(mp, ours, MOJO_SOLVER)
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj, abs=1e-8)
    assert [v.value() for v in ours_vars] == pytest.approx(
        [v.value() for v in theirs_vars], abs=1e-8
    )
    assert ours.infeasibilityGap() <= 1e-9


@pytest.mark.parametrize(
    "low,high,coefficient,expected",
    [
        (-4, -1, 1, -1),
        (None, 3, 1, 3),
        (-2, None, -1, 2),
        (2, 2, 10, 20),
    ],
)
def test_bound_transformations(low, high, coefficient, expected):
    problem = mp.LpProblem("bounds", mp.LpMaximize)
    x = mp.LpVariable("x", lowBound=low, upBound=high)
    problem += coefficient * x
    assert problem.solve(mp.MOJO_CMD(msg=False)) == mp.LpStatusOptimal
    assert mp.value(problem.objective) == pytest.approx(expected)


def test_infeasible_status_matches_pulp():
    def build(module):
        p = module.LpProblem("infeasible")
        x = module.LpVariable("x", lowBound=0)
        p += x
        p += x <= 1
        p += x >= 2
        return p

    ours, theirs = build(mp), build(pulp)
    assert ours.solve(MOJO_SOLVER) == theirs.solve(UPSTREAM_SOLVER)
    assert ours.status == mp.LpStatusInfeasible
    assert mp.LpStatus[ours.status] == pulp.LpStatus[theirs.status]


def test_unbounded_status_matches_pulp():
    def build(module):
        p = module.LpProblem("unbounded", module.LpMaximize)
        x = module.LpVariable("x", lowBound=0)
        p += x
        return p

    ours, theirs = build(mp), build(pulp)
    assert ours.solve(MOJO_SOLVER) == theirs.solve(UPSTREAM_SOLVER)
    assert ours.status == mp.LpStatusUnbounded


def build_blending(module):
    foods = ["oats", "soy", "fish", "oil"]
    cost = dict(zip(foods, [0.50, 0.72, 1.45, 1.10]))
    protein = dict(zip(foods, [13, 44, 62, 0]))
    fat = dict(zip(foods, [7, 1, 12, 100]))
    fiber = dict(zip(foods, [10, 7, 0, 0]))
    x = module.LpVariable.dicts("food", foods, lowBound=0)
    p = module.LpProblem("blend", module.LpMinimize)
    p += module.lpSum(cost[i] * x[i] for i in foods)
    p += module.lpSum(protein[i] * x[i] for i in foods) >= 30
    p += module.lpSum(fat[i] * x[i] for i in foods) >= 8
    p += module.lpSum(fiber[i] * x[i] for i in foods) >= 4
    p += module.lpSum(x.values()) == 1
    return p, x


def test_blending_model_matches_pulp():
    ours, ours_x = build_blending(mp)
    theirs, theirs_x = build_blending(pulp)
    _, ours_obj = solved_objective(mp, ours, MOJO_SOLVER)
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj, rel=1e-7, abs=1e-8)
    assert [ours_x[i].value() for i in ours_x] == pytest.approx(
        [theirs_x[i].value() for i in theirs_x], abs=2e-7
    )


def build_transport(module, integral=False):
    supply = [20, 30, 25]
    demand = [10, 15, 20, 30]
    costs = np.array([[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]])
    category = module.LpInteger if integral else module.LpContinuous
    x = module.LpVariable.dicts(
        "ship", (range(3), range(4)), lowBound=0, cat=category
    )
    p = module.LpProblem("transport", module.LpMinimize)
    p += module.lpSum(costs[i, j] * x[i][j] for i in range(3) for j in range(4))
    for i in range(3):
        p += module.lpSum(x[i][j] for j in range(4)) <= supply[i]
    for j in range(4):
        p += module.lpSum(x[i][j] for i in range(3)) >= demand[j]
    return p


def test_transportation_lp_matches_pulp():
    ours, theirs = build_transport(mp), build_transport(pulp)
    _, ours_obj = solved_objective(mp, ours, MOJO_SOLVER)
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj, abs=1e-7)
    assert ours.valid(1e-7)


def build_knapsack(module):
    values = [8, 11, 6, 4, 14, 7]
    weights = [5, 7, 4, 3, 9, 5]
    x = [
        module.LpVariable(f"take_{i}", cat=module.LpBinary)
        for i in range(len(values))
    ]
    p = module.LpProblem("knapsack", module.LpMaximize)
    p += module.lpDot(values, x)
    p += module.lpDot(weights, x) <= 17
    return p, x


def test_binary_knapsack_matches_pulp():
    ours, ours_x = build_knapsack(mp)
    theirs, theirs_x = build_knapsack(pulp)
    _, ours_obj = solved_objective(mp, ours, mp.MOJO_CMD(msg=False))
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj)
    assert all(x.value() in (0, 1) for x in ours_x)
    assert ours.valid(1e-8)


def test_integer_production_matches_pulp():
    def build(module):
        p = module.LpProblem("production", module.LpMaximize)
        x = module.LpVariable("tables", lowBound=0, cat=module.LpInteger)
        y = module.LpVariable("chairs", lowBound=0, cat=module.LpInteger)
        p += 45 * x + 30 * y
        p += 4 * x + 3 * y <= 120
        p += 2 * x + y <= 50
        return p, (x, y)

    ours, ours_vars = build(mp)
    theirs, theirs_vars = build(pulp)
    _, ours_obj = solved_objective(mp, ours, mp.MOJO_CMD(msg=False))
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj)
    assert [v.value() for v in ours_vars] == pytest.approx(
        [v.value() for v in theirs_vars]
    )


def test_binary_assignment_matches_pulp():
    costs = np.array([[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]])

    def build(module):
        x = module.LpVariable.dicts(
            "assign", (range(4), range(4)), cat=module.LpBinary
        )
        p = module.LpProblem("assignment", module.LpMinimize)
        p += module.lpSum(costs[i, j] * x[i][j] for i in range(4) for j in range(4))
        for i in range(4):
            p += module.lpSum(x[i][j] for j in range(4)) == 1
        for j in range(4):
            p += module.lpSum(x[i][j] for i in range(4)) == 1
        return p

    ours, theirs = build(mp), build(pulp)
    _, ours_obj = solved_objective(mp, ours, mp.MOJO_CMD(msg=False))
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj)
    assert ours.valid(1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_random_bounded_lp_objective_parity(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 2, size=(12, 18))
    feasible = rng.uniform(0, 3, size=18)
    b = a @ feasible + rng.uniform(0.1, 4, size=12)
    c = rng.normal(size=18)

    def build(module):
        x = [
            module.LpVariable(f"x_{i}", lowBound=0, upBound=6)
            for i in range(18)
        ]
        p = module.LpProblem(f"random_{seed}", module.LpMaximize)
        p += module.lpDot(c, x)
        for row, bound in zip(a, b):
            p += module.lpDot(row, x) <= bound
        return p

    ours, theirs = build(mp), build(pulp)
    _, ours_obj = solved_objective(mp, ours, MOJO_SOLVER)
    _, their_obj = solved_objective(pulp, theirs, UPSTREAM_SOLVER)
    assert ours_obj == pytest.approx(their_obj, rel=2e-7, abs=2e-7)
    assert ours.valid(2e-7)


def test_negative_rhs_and_mixed_senses():
    p = mp.LpProblem("negative_rhs", mp.LpMinimize)
    x = mp.LpVariable("x", lowBound=0)
    y = mp.LpVariable("y", lowBound=0)
    p += x + y
    p += -x - y <= -3
    p += -x + y >= -2
    assert p.solve(MOJO_SOLVER) == mp.LpStatusOptimal
    assert mp.value(p.objective) == pytest.approx(3)
    assert p.valid(1e-8)


def test_change_rhs_and_resolve():
    p = mp.LpProblem("resolve", mp.LpMaximize)
    x = mp.LpVariable("x", lowBound=0)
    p += x
    p += x <= 2, "limit"
    p.solve(MOJO_SOLVER)
    assert x.value() == pytest.approx(2)
    p.constraints["limit"].changeRHS(5)
    p.resolve(MOJO_SOLVER)
    assert x.value() == pytest.approx(5)


def test_variable_factories_and_helpers():
    nested = mp.LpVariable.dicts("x", (["a", "b"], range(3)), cat=mp.LpBinary)
    flat = mp.LpVariable.dict("y", (["a", "b"], range(2)), lowBound=-1)
    matrix = mp.LpVariable.matrix("z", (range(2), range(2)))
    assert nested["a"][2].name == "x_a_2"
    assert flat["b", 1].lowBound == -1
    assert matrix[1][0].name == "z_1_0"
    assert mp.listSolvers(onlyAvailable=True) == ["MOJO_CMD", "PULP_CBC_CMD"]
    assert isinstance(mp.getSolver("MOJO_CMD", msg=False), mp.MojoSolver)


def test_iteration_limit_is_reported_not_solved():
    p, _ = build_basic(mp)
    status = p.solve(mp.MOJO_CMD(msg=False, maxIterations=0))
    assert status == mp.LpStatusNotSolved
    assert all(variable.value() is None for variable in p.variables())


@pytest.mark.parametrize(
    "a,b,c,senses,error",
    [
        (np.zeros(2), np.zeros(1), np.zeros(2), np.zeros(1, dtype=int), ValueError),
        (np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.zeros(1, dtype=int), ValueError),
        (np.zeros((1, 2)), np.zeros(1), np.zeros(1), np.zeros(1, dtype=int), ValueError),
        (np.zeros((1, 2)), np.zeros(1), np.zeros(2), np.zeros(2, dtype=int), ValueError),
        (np.array([[np.nan]]), np.zeros(1), np.zeros(1), np.zeros(1, dtype=int), ValueError),
        (
            np.array([[2**53 + 1]], dtype=np.int64),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1, dtype=int),
            TypeError,
        ),
        (
            np.array([[1 + 2j]]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1, dtype=int),
            TypeError,
        ),
        (np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.array([0.0]), TypeError),
        (np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.array([2]), ValueError),
    ],
)
def test_dense_ffi_rejects_invalid_shapes_values_and_senses(a, b, c, senses, error):
    with pytest.raises(error):
        solve_dense(a, b, c, senses, max_iter=10, tolerance=1e-9)


@pytest.mark.parametrize(
    "max_iter,tolerance,error",
    [
        (-1, 1e-9, ValueError),
        (1.5, 1e-9, TypeError),
        (10, 0.0, ValueError),
        (10, np.inf, ValueError),
    ],
)
def test_dense_ffi_rejects_invalid_solver_limits(max_iter, tolerance, error):
    with pytest.raises(error):
        solve_dense(
            np.zeros((0, 0)),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            max_iter=max_iter,
            tolerance=tolerance,
        )


def test_empty_problem_uses_non_null_ffi_buffers():
    problem = mp.LpProblem("empty")
    assert problem.solve(mp.MOJO_CMD(msg=False)) == mp.LpStatusOptimal


def test_documented_helpers_solver_alias_and_rounding():
    x = mp.LpVariable("x", lowBound=2)
    assert mp.value(x) is None
    assert mp.valueOrDefault(x) == 2
    x.varValue = 2.00000001
    x.cat = mp.LpInteger
    problem = mp.LpProblem("helpers")
    problem += x
    problem.roundSolution()
    assert x.value() == 2
    assert problem.valid()
    assert problem.infeasibilityGap() == 0
    assert isinstance(
        mp.getSolverFromDict({"solver": "MOJO_CMD", "msg": False}),
        mp.MojoSolver,
    )
    assert isinstance(mp.PULP_CBC_CMD(msg=False), mp.MojoSolver)


def test_unsupported_or_invalid_solver_options_are_not_silently_ignored():
    with pytest.raises(TypeError, match="unsupported solver option"):
        mp.MOJO_CMD(threads=8)
    with pytest.raises(ValueError, match="maxNodes"):
        mp.MOJO_CMD(maxNodes=-1)
    with pytest.raises(ValueError, match="tolerance"):
        mp.MOJO_CMD(tolerance=np.nan)


def test_invalid_model_enums_are_rejected_before_compilation():
    with pytest.raises(ValueError, match="category"):
        mp.LpVariable("x", cat="SemiContinuous")
    with pytest.raises(ValueError, match="constraint sense"):
        mp.LpConstraint(sense=7)
    with pytest.raises(ValueError, match="objective sense"):
        mp.LpProblem(sense=7)


def test_mojo_abi_rejects_null_pointers_without_dereferencing_them():
    status = lib().mp_solve_lp(
        None, None, None, None, 0, 0, None, None, None, None, 10, 1e-9
    )
    assert status == mp.LpStatusUndefined
