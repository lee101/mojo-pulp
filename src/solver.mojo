"""Dense two-phase simplex for the Python modeling layer.

The caller owns the tableau and all result buffers. The kernel uses Bland's
entering rule, which trades a little speed for deterministic anti-cycling.
"""

from std.math import abs

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = 4


def fp(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def pivot(tableau: Ptr, rows: Int, cols: Int, pivot_row: Int, pivot_col: Int):
    var prow = tableau + pivot_row * cols
    var divisor = prow[pivot_col]
    var denom = SIMD[DType.float64, W](divisor)
    var j = 0
    while j + W <= cols:
        prow.store(j, prow.load[width=W](j) / denom)
        j += W
    while j < cols:
        prow[j] /= divisor
        j += 1
    prow[pivot_col] = 1.0

    for r in range(rows):
        if r != pivot_row:
            var row = tableau + r * cols
            var factor = row[pivot_col]
            if factor != 0.0:
                var vf = SIMD[DType.float64, W](factor)
                j = 0
                while j + W <= cols:
                    row.store(
                        j,
                        row.load[width=W](j) - vf * prow.load[width=W](j),
                    )
                    j += W
                while j < cols:
                    row[j] -= factor * prow[j]
                    j += 1
                row[pivot_col] = 0.0


def simplex(
    tableau: Ptr,
    basis: IPtr,
    rows: Int,
    cols: Int,
    entering_cols: Int,
    max_iter: Int,
    tolerance: Float64,
) -> Int:
    """Return pivots, -(pivots+1) for unbounded, max_iter+1 on limit."""
    var iterations = 0
    while iterations < max_iter:
        var entering = -1
        for j in range(entering_cols):
            if tableau[j] < -tolerance:
                entering = j
                break
        if entering < 0:
            return iterations

        var leaving = -1
        var best_ratio = 0.0
        var best_basis = 0
        for r in range(1, rows):
            var coefficient = tableau[r * cols + entering]
            if coefficient > tolerance:
                var ratio = tableau[r * cols + cols - 1] / coefficient
                if (
                    leaving < 0
                    or ratio < best_ratio - tolerance
                    or (
                        abs(ratio - best_ratio) <= tolerance
                        and Int(basis[r - 1]) < best_basis
                    )
                ):
                    leaving = r
                    best_ratio = ratio
                    best_basis = Int(basis[r - 1])
        if leaving < 0:
            return -(iterations + 1)

        pivot(tableau, rows, cols, leaving, entering)
        basis[leaving - 1] = Int64(entering)
        iterations += 1
    return max_iter + 1


@export("mp_solve_lp")
def mp_solve_lp(
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    senses_addr: Int,
    m: Int,
    n: Int,
    tableau_addr: Int,
    basis_addr: Int,
    x_addr: Int,
    stats_addr: Int,
    max_iter: Int,
    tolerance: Float64,
) abi("C") -> Int:
    """Maximize c*x subject to mixed row senses and x >= 0.

    senses are -1 for <=, 0 for equality, and 1 for >=. Status values match
    PuLP: 1 optimal, 0 iteration limit, -1 infeasible, -2 unbounded.
    """
    # Never construct Mojo's non-null pointer type from an invalid C address.
    # Array extents remain the caller's responsibility, as with any C ABI.
    if (
        a_addr == 0
        or b_addr == 0
        or c_addr == 0
        or senses_addr == 0
        or tableau_addr == 0
        or basis_addr == 0
        or x_addr == 0
        or stats_addr == 0
        or m < 0
        or n < 0
        or max_iter < 0
        or tolerance <= 0.0
        or tolerance != tolerance
    ):
        return -3

    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    var senses = ip(senses_addr)
    var tableau = fp(tableau_addr)
    var basis = ip(basis_addr)
    var solution = fp(x_addr)
    var stats = fp(stats_addr)

    var rows = m + 1
    var variable_cols = n + 2 * m
    var cols = variable_cols + 1
    for i in range(rows * cols):
        tableau[i] = 0.0

    var artificial_count = 0
    for i in range(m):
        var sign = 1.0
        var row_sense = senses[i]
        var rhs = b[i]
        if rhs < 0.0:
            sign = -1.0
            rhs = -rhs
            row_sense = -row_sense
        var row = tableau + (i + 1) * cols
        for j in range(n):
            row[j] = sign * a[i * n + j]
        row[cols - 1] = rhs

        var slack_col = n + i
        if row_sense < 0:
            row[slack_col] = 1.0
            basis[i] = Int64(slack_col)
        else:
            if row_sense > 0:
                row[slack_col] = -1.0
            var artificial_col = n + m + i
            row[artificial_col] = 1.0
            basis[i] = Int64(artificial_col)
            artificial_count += 1

    var phase_one_iterations = 0
    if artificial_count > 0:
        for i in range(m):
            var artificial_col = n + m + i
            tableau[artificial_col] = 1.0
            if Int(basis[i]) == artificial_col:
                var row = tableau + (i + 1) * cols
                for j in range(cols):
                    tableau[j] -= row[j]

        var phase_one = simplex(
            tableau, basis, rows, cols, variable_cols, max_iter, tolerance
        )
        if phase_one < 0:
            stats[0] = tableau[cols - 1]
            stats[1] = Float64(-phase_one - 1)
            return -1
        if phase_one > max_iter:
            stats[0] = tableau[cols - 1]
            stats[1] = Float64(max_iter)
            return 0
        phase_one_iterations = phase_one
        if tableau[cols - 1] < -tolerance:
            stats[0] = tableau[cols - 1]
            stats[1] = Float64(phase_one_iterations)
            return -1
        for i in range(m):
            if Int(basis[i]) >= n + m:
                var row_index = i + 1
                var replacement = -1
                for j in range(n + m):
                    if abs(tableau[row_index * cols + j]) > tolerance:
                        replacement = j
                        break
                if replacement >= 0:
                    tableau[row_index * cols + cols - 1] = 0.0
                    pivot(
                        tableau,
                        rows,
                        cols,
                        row_index,
                        replacement,
                    )
                    basis[i] = Int64(replacement)

    for j in range(cols):
        tableau[j] = 0.0
    for j in range(n):
        tableau[j] = -c[j]
    for i in range(m):
        var basic = Int(basis[i])
        if basic < n:
            var coefficient = c[basic]
            if coefficient != 0.0:
                var row = tableau + (i + 1) * cols
                for j in range(cols):
                    tableau[j] += coefficient * row[j]

    var remaining = max_iter - phase_one_iterations
    if remaining < 0:
        remaining = 0
    var phase_two = simplex(
        tableau, basis, rows, cols, n + m, remaining, tolerance
    )
    if phase_two < 0:
        stats[0] = tableau[cols - 1]
        stats[1] = Float64(phase_one_iterations - phase_two - 1)
        return -2
    if phase_two > remaining:
        stats[0] = tableau[cols - 1]
        stats[1] = Float64(max_iter)
        return 0

    for j in range(n):
        solution[j] = 0.0
    for i in range(m):
        var basic = Int(basis[i])
        if basic < n:
            var value = tableau[(i + 1) * cols + cols - 1]
            solution[basic] = value if abs(value) > tolerance else 0.0
    stats[0] = tableau[cols - 1]
    stats[1] = Float64(phase_one_iterations + phase_two)
    stats[2] = Float64(artificial_count)
    return 1
