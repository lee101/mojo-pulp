"""PuLP-style algebraic modeling objects."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from . import constants
from ._lib import solve_dense


_NAME_TRANSLATION = str.maketrans(" -+[]", "_____")


def _number(value: Any) -> float:
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("NaN and infinity are not valid model coefficients")
        return result
    raise TypeError(f"expected a number, got {type(value).__name__}")


class LpElement:
    def __init__(self, name: str):
        self.name = str(name).translate(_NAME_TRANSLATION)

    def getName(self) -> str:
        return self.name

    def setName(self, name: str) -> None:
        self.name = str(name)


class LpVariable(LpElement):
    __hash__ = object.__hash__

    def __init__(
        self,
        name: str,
        lowBound: float | None = None,
        upBound: float | None = None,
        cat: str = constants.LpContinuous,
        e: Any = None,
    ):
        super().__init__(name)
        if cat not in {
            constants.LpContinuous,
            constants.LpInteger,
            constants.LpBinary,
        }:
            raise ValueError(f"unsupported variable category {cat!r}")
        if cat == constants.LpBinary:
            lowBound, upBound, cat = 0.0, 1.0, constants.LpInteger
            self._was_binary = True
        else:
            self._was_binary = False
        self.lowBound = None if lowBound is None else _number(lowBound)
        self.upBound = None if upBound is None else _number(upBound)
        if (
            self.lowBound is not None
            and self.upBound is not None
            and self.lowBound > self.upBound
        ):
            raise ValueError(f"lower bound exceeds upper bound for {self.name}")
        self.cat = cat
        self.varValue: float | None = None
        self.dj: float | None = None
        self._lowbound_original = self.lowBound
        self._upbound_original = self.upBound
        if e is not None and hasattr(e, "items"):
            self.addVariableToConstraints(e)

    @classmethod
    def dicts(
        cls,
        name: str,
        indices: Iterable | tuple[Iterable, ...] | None = None,
        lowBound: float | None = None,
        upBound: float | None = None,
        cat: str = constants.LpContinuous,
        indexStart: list | None = None,
    ):
        if indices is None:
            return {}
        dimensions = indices if isinstance(indices, tuple) else (indices,)
        dimensions = tuple(list(dimension) for dimension in dimensions)

        def create(prefix: tuple, depth: int):
            result = {}
            for index in dimensions[depth]:
                key = prefix + (index,)
                if depth + 1 == len(dimensions):
                    suffix = "_".join(str(part) for part in key)
                    result[index] = cls(
                        f"{name}_{suffix}", lowBound, upBound, cat
                    )
                else:
                    result[index] = create(key, depth + 1)
            return result

        return create(tuple(indexStart or ()), 0)

    @classmethod
    def dict(cls, name: str, indices, *args, **kwargs):
        dimensions = indices if isinstance(indices, tuple) else (indices,)
        keys = [()]
        for dimension in dimensions:
            keys = [prefix + (value,) for prefix in keys for value in dimension]
        return {
            key if len(key) > 1 else key[0]: cls(
                f"{name}_{'_'.join(map(str, key))}", *args, **kwargs
            )
            for key in keys
        }

    @classmethod
    def matrix(cls, name: str, indices=None, *args, **kwargs):
        dimensions = indices if isinstance(indices, tuple) else (indices,)

        def create(prefix, depth):
            return [
                cls(f"{name}_{'_'.join(map(str, prefix + (i,)))}", *args, **kwargs)
                if depth + 1 == len(dimensions)
                else create(prefix + (i,), depth + 1)
                for i in dimensions[depth]
            ]

        return create((), 0)

    def addVariableToConstraints(self, e) -> None:
        for constraint, coefficient in e.items():
            constraint.addVariable(self, coefficient)

    def value(self) -> float | None:
        return self.varValue

    def valueOrDefault(self) -> float:
        if self.varValue is not None:
            return self.varValue
        if self.lowBound is not None and self.lowBound > 0:
            return self.lowBound
        if self.upBound is not None and self.upBound < 0:
            return self.upBound
        return 0.0

    def roundedValue(self, eps: float = 1e-5) -> float | None:
        if self.varValue is None:
            return None
        if self.cat == constants.LpInteger and abs(round(self.varValue) - self.varValue) <= eps:
            return float(round(self.varValue))
        return self.varValue

    def setInitialValue(self, val: float, check: bool = True) -> bool:
        val = _number(val)
        valid = (
            (self.lowBound is None or val >= self.lowBound)
            and (self.upBound is None or val <= self.upBound)
        )
        if check and not valid:
            raise ValueError(f"initial value for {self.name} violates its bounds")
        if valid:
            self.varValue = val
        return valid

    def fixValue(self) -> None:
        if self.varValue is not None:
            self.lowBound = self.upBound = self.varValue

    def unfixValue(self) -> None:
        self.lowBound = self._lowbound_original
        self.upBound = self._upbound_original

    def isBinary(self) -> bool:
        return self.cat == constants.LpInteger and self.lowBound == 0 and self.upBound == 1

    def isInteger(self) -> bool:
        return self.cat == constants.LpInteger

    def isFree(self) -> bool:
        return self.lowBound is None and self.upBound is None

    def isConstant(self) -> bool:
        return self.lowBound is not None and self.lowBound == self.upBound

    def valid(self, eps: float) -> bool:
        if self.varValue is None:
            return False
        return (
            (self.lowBound is None or self.varValue >= self.lowBound - eps)
            and (self.upBound is None or self.varValue <= self.upBound + eps)
            and (
                self.cat != constants.LpInteger
                or abs(self.varValue - round(self.varValue)) <= eps
            )
        )

    def infeasibilityGap(self, mip: int = 1) -> float:
        if self.varValue is None:
            raise ValueError("variable has no value")
        gap = 0.0
        if self.lowBound is not None and self.varValue < self.lowBound:
            gap = self.varValue - self.lowBound
        if self.upBound is not None and self.varValue > self.upBound:
            gap = self.varValue - self.upBound
        if mip and self.cat == constants.LpInteger:
            gap = max(gap, abs(self.varValue - round(self.varValue)))
        return gap

    def round(self, epsInt: float = 1e-5, eps: float = 1e-7) -> None:
        if self.varValue is None:
            return
        if self.upBound is not None and self.varValue > self.upBound:
            if self.varValue <= self.upBound + eps:
                self.varValue = self.upBound
        if self.lowBound is not None and self.varValue < self.lowBound:
            if self.varValue >= self.lowBound - eps:
                self.varValue = self.lowBound
        if self.cat == constants.LpInteger and abs(round(self.varValue) - self.varValue) <= epsInt:
            self.varValue = float(round(self.varValue))

    def __repr__(self) -> str:
        return self.name

    def _expression(self):
        return LpAffineExpression(self)

    def __add__(self, other):
        return self._expression() + other

    def __radd__(self, other):
        return self._expression() + other

    def __sub__(self, other):
        return self._expression() - other

    def __rsub__(self, other):
        return other - self._expression()

    def __mul__(self, other):
        return self._expression() * other

    def __rmul__(self, other):
        return self._expression() * other

    def __truediv__(self, other):
        return self._expression() / other

    def __neg__(self):
        return -self._expression()

    def __le__(self, other):
        return self._expression() <= other

    def __ge__(self, other):
        return self._expression() >= other

    def __eq__(self, other):
        return self._expression() == other


class LpAffineExpression:
    def __init__(self, e=None, constant: float = 0.0, name: str | None = None):
        self.name = name
        self.constant = _number(constant)
        self._coefficients: OrderedDict[LpVariable, float] = OrderedDict()
        if e is None:
            return
        if isinstance(e, LpVariable):
            self._coefficients[e] = 1.0
        elif isinstance(e, LpAffineExpression):
            self.constant += e.constant
            self._coefficients.update(e._coefficients)
        elif isinstance(e, Mapping) or hasattr(e, "items"):
            for variable, coefficient in e.items():
                self.addterm(variable, coefficient)
        elif isinstance(e, Real):
            self.constant += _number(e)
        else:
            for variable, coefficient in e:
                self.addterm(variable, coefficient)

    def copy(self):
        return LpAffineExpression(self)

    def addterm(self, key: LpVariable, value: float) -> None:
        self._add_coefficient(key, _number(value))

    def _add_coefficient(self, key: LpVariable, value: float) -> None:
        coefficient = self._coefficients.get(key, 0.0) + value
        if coefficient == 0.0:
            self._coefficients.pop(key, None)
        else:
            self._coefficients[key] = coefficient

    def items(self):
        return self._coefficients.items()

    def keys(self):
        return self._coefficients.keys()

    def values(self):
        return self._coefficients.values()

    def get(self, key, default=None):
        return self._coefficients.get(key, default)

    def __len__(self):
        return len(self._coefficients)

    def __iter__(self):
        return iter(self._coefficients)

    def sorted_keys(self):
        return sorted(self._coefficients, key=lambda variable: variable.name)

    def value(self) -> float | None:
        result = self.constant
        for variable, coefficient in self.items():
            if variable.varValue is None:
                return None
            result += coefficient * variable.varValue
        return result

    def valueOrDefault(self) -> float:
        return self.constant + sum(
            coefficient * variable.valueOrDefault()
            for variable, coefficient in self.items()
        )

    def addInPlace(self, other, sign: int = 1):
        if isinstance(other, LpAffineExpression):
            terms = tuple(other.items()) if other is self else other.items()
            self.constant += sign * other.constant
            for variable, coefficient in terms:
                self._add_coefficient(variable, sign * coefficient)
        elif isinstance(other, LpVariable):
            self._add_coefficient(other, float(sign))
        elif isinstance(other, Real):
            self.constant += sign * _number(other)
        else:
            raise TypeError(
                f"cannot use {type(other).__name__} in a linear expression"
            )
        return self

    def subInPlace(self, other):
        return self.addInPlace(other, -1)

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.copy().addInPlace(other, -1)

    def __rsub__(self, other):
        return _as_expression(other).addInPlace(self, -1)

    def __neg__(self):
        result = LpAffineExpression()
        result.constant = -self.constant
        result._coefficients = OrderedDict(
            (variable, -coefficient) for variable, coefficient in self.items()
        )
        return result

    def __mul__(self, other):
        scalar = _number(other)
        result = LpAffineExpression()
        result.constant = scalar * self.constant
        result._coefficients = OrderedDict(
            (variable, scalar * coefficient)
            for variable, coefficient in self.items()
            if scalar * coefficient != 0.0
        )
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        scalar = _number(other)
        if scalar == 0.0:
            raise ZeroDivisionError
        return self * (1.0 / scalar)

    def __le__(self, other):
        return LpConstraint(self - other, constants.LpConstraintLE)

    def __ge__(self, other):
        return LpConstraint(self - other, constants.LpConstraintGE)

    def __eq__(self, other):
        return LpConstraint(self - other, constants.LpConstraintEQ)

    def __bool__(self):
        return bool(self._coefficients) or self.constant != 0.0

    def __repr__(self) -> str:
        terms = [f"{coefficient:g}*{variable.name}" for variable, coefficient in self.items()]
        if self.constant or not terms:
            terms.append(f"{self.constant:g}")
        return " + ".join(terms).replace("+ -", "- ")


def _as_expression(value) -> LpAffineExpression:
    if isinstance(value, LpAffineExpression):
        return value.copy()
    if isinstance(value, LpVariable):
        return LpAffineExpression(value)
    if isinstance(value, Real):
        return LpAffineExpression(constant=value)
    raise TypeError(f"cannot use {type(value).__name__} in a linear expression")


class LpConstraint:
    def __init__(
        self,
        e=None,
        sense: int = constants.LpConstraintEQ,
        name: str | None = None,
        rhs: float | None = None,
    ):
        self.expr = _as_expression(0 if e is None else e)
        if rhs is not None:
            self.expr.constant -= _number(rhs)
        if sense not in {
            constants.LpConstraintLE,
            constants.LpConstraintEQ,
            constants.LpConstraintGE,
        }:
            raise ValueError(f"unsupported constraint sense {sense!r}")
        self.sense = sense
        self.name = name
        self.pi: float | None = None
        self.slack: float | None = None
        self.modified = True

    @property
    def constant(self):
        return self.expr.constant

    def items(self):
        return self.expr.items()

    def keys(self):
        return self.expr.keys()

    def values(self):
        return self.expr.values()

    def get(self, key, default=None):
        return self.expr.get(key, default)

    def addVariable(self, variable: LpVariable, coefficient: float) -> None:
        self.expr.addterm(variable, coefficient)

    def changeRHS(self, rhs: float) -> None:
        self.expr.constant = -_number(rhs)
        self.modified = True

    def value(self) -> float | None:
        return self.expr.value()

    def valueOrDefault(self) -> float:
        return self.expr.valueOrDefault()

    def valid(self, eps: float = 0.0) -> bool:
        value = self.value()
        if value is None:
            return False
        if self.sense == constants.LpConstraintEQ:
            return abs(value) <= eps
        if self.sense == constants.LpConstraintLE:
            return value <= eps
        return value >= -eps

    def __repr__(self):
        symbol = {
            constants.LpConstraintLE: "<=",
            constants.LpConstraintEQ: "=",
            constants.LpConstraintGE: ">=",
        }[self.sense]
        return f"{self.expr} {symbol} 0"


def lpSum(vector) -> LpAffineExpression:
    result = LpAffineExpression()
    if isinstance(vector, (LpVariable, LpAffineExpression, Real)):
        return result.addInPlace(vector)
    for value in vector:
        result.addInPlace(value)
    return result


def lpDot(v1, v2):
    if isinstance(v1, Iterable) and not isinstance(v1, (str, bytes)):
        result = LpAffineExpression()
        for left, right in zip(v1, v2):
            if isinstance(left, Real):
                scalar = _number(left)
                value = right
            elif isinstance(right, Real):
                scalar = _number(right)
                value = left
            else:
                result.addInPlace(left * right)
                continue

            if isinstance(value, LpVariable):
                result._add_coefficient(value, scalar)
            elif isinstance(value, LpAffineExpression):
                result.constant += scalar * value.constant
                for variable, coefficient in value.items():
                    result._add_coefficient(variable, scalar * coefficient)
            elif isinstance(value, Real):
                result.constant += scalar * _number(value)
            else:
                result.addInPlace(scalar * value)
        return result
    return v1 * v2


def value(x):
    return x.value() if hasattr(x, "value") else x


def valueOrDefault(x):
    return x.valueOrDefault() if hasattr(x, "valueOrDefault") else x


class LpProblem:
    def __init__(self, name: str = "NoName", sense: int = constants.LpMinimize):
        if sense not in {constants.LpMinimize, constants.LpMaximize}:
            raise ValueError(f"unsupported objective sense {sense!r}")
        self.name = str(name)
        self.sense = sense
        self.objective: LpAffineExpression | None = None
        self.constraints: OrderedDict[str, LpConstraint] = OrderedDict()
        self.status = constants.LpStatusNotSolved
        self.sol_status = constants.LpSolutionNoSolutionFound
        self.solutionTime = 0.0
        self.solutionCpuTime = 0.0
        self.numVariables = 0
        self.numConstraints = 0
        self._last_solver = None

    def __iadd__(self, other):
        name = None
        if isinstance(other, tuple):
            other, name = other
        if isinstance(other, LpConstraint):
            self.addConstraint(other, name)
        else:
            self.setObjective(other)
            if name is not None:
                self.objective.name = name
        return self

    def setObjective(self, obj) -> None:
        self.objective = _as_expression(obj)

    def addConstraint(self, constraint: LpConstraint, name: str | None = None) -> None:
        if not isinstance(constraint, LpConstraint):
            raise TypeError("only LpConstraint objects can be added as constraints")
        key = name or constraint.name or f"_C{len(self.constraints) + 1}"
        if key in self.constraints:
            raise ValueError(f"overlapping constraint names: {key}")
        constraint.name = key
        self.constraints[key] = constraint
        self.numConstraints = len(self.constraints)

    def variables(self) -> list[LpVariable]:
        seen: dict[int, LpVariable] = {}
        expressions = [self.objective] if self.objective is not None else []
        expressions += [constraint.expr for constraint in self.constraints.values()]
        for expression in expressions:
            for variable in expression.keys():
                seen[id(variable)] = variable
        result = sorted(seen.values(), key=lambda variable: variable.name)
        self.numVariables = len(result)
        return result

    def variablesDict(self) -> dict[str, LpVariable]:
        return {variable.name: variable for variable in self.variables()}

    def solve(self, solver=None, **kwargs) -> int:
        if solver is None:
            solver = MojoSolver(**kwargs)
        self._last_solver = solver
        return solver.actualSolve(self)

    def resolve(self, solver=None, **kwargs) -> int:
        return self.solve(solver or self._last_solver, **kwargs)

    def roundSolution(self, epsInt: float = 1e-5, eps: float = 1e-7) -> None:
        for variable in self.variables():
            variable.round(epsInt, eps)

    def isMIP(self) -> int:
        return int(any(variable.cat == constants.LpInteger for variable in self.variables()))

    def valid(self, eps: float = 0.0) -> bool:
        return all(variable.valid(eps) for variable in self.variables()) and all(
            constraint.valid(eps) for constraint in self.constraints.values()
        )

    def infeasibilityGap(self, mip: int = 1) -> float:
        gaps = [abs(variable.infeasibilityGap(mip)) for variable in self.variables()]
        for constraint in self.constraints.values():
            val = constraint.value()
            if val is None:
                continue
            if constraint.sense == constants.LpConstraintEQ:
                gaps.append(abs(val))
            elif constraint.sense == constants.LpConstraintLE:
                gaps.append(max(0.0, val))
            else:
                gaps.append(max(0.0, -val))
        return max(gaps, default=0.0)

    def __repr__(self):
        direction = "MINIMIZE" if self.sense == constants.LpMinimize else "MAXIMIZE"
        constraints = "\n".join(
            f"{name}: {constraint}" for name, constraint in self.constraints.items()
        )
        return f"{self.name}:\n{direction}\n{self.objective}\nSUBJECT TO\n{constraints}"


class _CompiledLP:
    def __init__(self, problem: LpProblem, branches):
        self.variables = problem.variables()
        self.transforms: dict[LpVariable, tuple[float, list[tuple[int, float]]]] = {}
        upper_rows: list[tuple[int, float]] = []
        column_count = 0
        for variable in self.variables:
            low, high = variable.lowBound, variable.upBound
            if low is not None and high is not None and low == high:
                self.transforms[variable] = (low, [])
            elif low is not None:
                self.transforms[variable] = (low, [(column_count, 1.0)])
                if high is not None:
                    upper_rows.append((column_count, high - low))
                column_count += 1
            elif high is not None:
                self.transforms[variable] = (high, [(column_count, -1.0)])
                column_count += 1
            else:
                self.transforms[variable] = (
                    0.0,
                    [(column_count, 1.0), (column_count + 1, -1.0)],
                )
                column_count += 2

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        senses: list[int] = []

        def add_expression(expression, sense):
            row = np.zeros(column_count, dtype=np.float64)
            constant = expression.constant
            for variable, coefficient in expression.items():
                shift, terms = self.transforms[variable]
                constant += coefficient * shift
                for index, scale in terms:
                    row[index] += coefficient * scale
            rows.append(row)
            rhs.append(-constant)
            senses.append(sense)

        for constraint in problem.constraints.values():
            add_expression(constraint.expr, constraint.sense)
        for column, bound in upper_rows:
            row = np.zeros(column_count, dtype=np.float64)
            row[column] = 1.0
            rows.append(row)
            rhs.append(bound)
            senses.append(constants.LpConstraintLE)
        for variable, sense, bound in branches:
            expression = LpAffineExpression(variable)
            expression.constant -= bound
            add_expression(expression, sense)

        objective = problem.objective or LpAffineExpression()
        c = np.zeros(column_count, dtype=np.float64)
        self.objective_constant = objective.constant
        for variable, coefficient in objective.items():
            shift, terms = self.transforms[variable]
            self.objective_constant += coefficient * shift
            for index, scale in terms:
                c[index] += coefficient * scale
        if problem.sense == constants.LpMinimize:
            c *= -1.0

        self.a = (
            np.vstack(rows)
            if rows
            else np.empty((0, column_count), dtype=np.float64)
        )
        self.b = np.asarray(rhs, dtype=np.float64)
        self.senses = np.asarray(senses, dtype=np.int64)
        self.c = c

    def values(self, transformed: np.ndarray) -> dict[LpVariable, float]:
        result = {}
        for variable, (shift, terms) in self.transforms.items():
            result[variable] = shift + sum(
                scale * transformed[index] for index, scale in terms
            )
        return result


class MojoSolver:
    name = "MOJO"

    def __init__(
        self,
        mip: bool = True,
        msg: bool = True,
        timeLimit: float | None = None,
        gapRel: float | None = None,
        gapAbs: float | None = None,
        maxNodes: int | None = None,
        maxIterations: int = 100_000,
        tolerance: float = 1e-9,
        **kwargs,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported solver option(s): {names}")
        if timeLimit is not None and (
            not isinstance(timeLimit, Real)
            or not math.isfinite(float(timeLimit))
            or timeLimit < 0
        ):
            raise ValueError("timeLimit must be a finite non-negative number")
        for name, gap in (("gapRel", gapRel), ("gapAbs", gapAbs)):
            if gap is not None and (
                not isinstance(gap, Real)
                or not math.isfinite(float(gap))
                or gap < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if not isinstance(maxIterations, int) or maxIterations < 0:
            raise ValueError("maxIterations must be a non-negative integer")
        if maxNodes is not None and (
            not isinstance(maxNodes, int) or maxNodes < 0
        ):
            raise ValueError("maxNodes must be a non-negative integer")
        if (
            not isinstance(tolerance, Real)
            or not math.isfinite(float(tolerance))
            or tolerance <= 0
        ):
            raise ValueError("tolerance must be a finite positive number")
        self.mip = mip
        self.msg = msg
        self.timeLimit = timeLimit
        self.gapRel = gapRel
        self.gapAbs = gapAbs
        self.maxNodes = 10_000 if maxNodes is None else maxNodes
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.options = {}
        self.node_count = 0
        self.iteration_count = 0

    def available(self) -> bool:
        return True

    def _relaxation(self, problem, branches):
        compiled = _CompiledLP(problem, branches)
        status, transformed, stats = solve_dense(
            compiled.a,
            compiled.b,
            compiled.c,
            compiled.senses,
            max_iter=self.maxIterations,
            tolerance=self.tolerance,
        )
        self.iteration_count += int(stats[1])
        if status != constants.LpStatusOptimal:
            return status, None, None
        values = compiled.values(transformed)
        objective = (problem.objective or LpAffineExpression()).constant
        objective += sum(
            coefficient * values[variable]
            for variable, coefficient in (problem.objective or LpAffineExpression()).items()
        )
        return status, values, objective

    def actualSolve(self, problem: LpProblem, **kwargs) -> int:
        start = time.perf_counter()
        cpu_start = time.process_time()
        self.node_count = 0
        self.iteration_count = 0
        variables = problem.variables()
        integers = [
            variable
            for variable in variables
            if self.mip and variable.cat == constants.LpInteger
        ]
        stack = [()]
        incumbent_values = None
        incumbent_objective = None
        terminal_status = constants.LpStatusInfeasible
        interrupted = False

        while stack:
            if self.node_count >= self.maxNodes:
                interrupted = True
                break
            if self.timeLimit is not None and time.perf_counter() - start >= self.timeLimit:
                interrupted = True
                break
            branches = stack.pop()
            self.node_count += 1
            status, values, objective = self._relaxation(problem, branches)
            if status == constants.LpStatusUnbounded and self.node_count == 1:
                terminal_status = status
                break
            if status != constants.LpStatusOptimal:
                if status == constants.LpStatusNotSolved:
                    interrupted = True
                continue

            terminal_status = constants.LpStatusOptimal
            if incumbent_objective is not None:
                improvement = (
                    objective - incumbent_objective
                    if problem.sense == constants.LpMaximize
                    else incumbent_objective - objective
                )
                absolute_gap = self.gapAbs or 0.0
                relative_gap = (self.gapRel or 0.0) * max(1.0, abs(incumbent_objective))
                if improvement <= max(self.tolerance, absolute_gap, relative_gap):
                    continue

            fractional = [
                (abs(values[variable] - round(values[variable])), variable)
                for variable in integers
                if abs(values[variable] - round(values[variable])) > 1e-7
            ]
            if not fractional:
                incumbent_values = values
                incumbent_objective = objective
                continue

            _, variable = max(fractional, key=lambda item: item[0])
            current = values[variable]
            lower_branch = branches + (
                (variable, constants.LpConstraintLE, math.floor(current)),
            )
            upper_branch = branches + (
                (variable, constants.LpConstraintGE, math.ceil(current)),
            )
            stack.append(upper_branch)
            stack.append(lower_branch)

        if incumbent_values is not None:
            for variable in variables:
                variable.varValue = incumbent_values[variable]
            problem.roundSolution()
            for constraint in problem.constraints.values():
                residual = constraint.value()
                constraint.slack = (
                    -residual
                    if constraint.sense != constants.LpConstraintGE
                    else residual
                )
                constraint.pi = None
            problem.status = (
                constants.LpStatusNotSolved if interrupted else constants.LpStatusOptimal
            )
            problem.sol_status = (
                constants.LpSolutionIntegerFeasible
                if interrupted
                else constants.LpSolutionOptimal
            )
        else:
            for variable in variables:
                variable.varValue = None
            for constraint in problem.constraints.values():
                constraint.slack = None
                constraint.pi = None
            problem.status = (
                constants.LpStatusNotSolved if interrupted else terminal_status
            )
            problem.sol_status = {
                constants.LpStatusInfeasible: constants.LpSolutionInfeasible,
                constants.LpStatusUnbounded: constants.LpSolutionUnbounded,
            }.get(problem.status, constants.LpSolutionNoSolutionFound)

        problem.solutionTime = time.perf_counter() - start
        problem.solutionCpuTime = time.process_time() - cpu_start
        if self.msg:
            label = constants.LpStatus[problem.status]
            print(
                f"Mojo simplex: {label}; {self.node_count} nodes; "
                f"{self.iteration_count} pivots; {problem.solutionTime:.4f}s"
            )
        return problem.status


class MOJO_CMD(MojoSolver):
    pass


class PULP_CBC_CMD(MojoSolver):
    """Compatibility name; this package still runs the Mojo solver."""


def getSolver(solver: str, **kwargs):
    normalized = solver.upper().replace("_CMD", "")
    if normalized in {"MOJO", "PULP_CBC", "CBC"}:
        return MojoSolver(**kwargs)
    raise ValueError(f"unsupported solver {solver!r}")


def getSolverFromDict(data: dict):
    options = dict(data)
    name = options.pop("solver")
    return getSolver(name, **options)


def listSolvers(onlyAvailable: bool = False):
    return ["MOJO_CMD", "PULP_CBC_CMD"]
