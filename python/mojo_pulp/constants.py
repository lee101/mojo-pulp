"""PuLP-compatible constants for the covered API."""

VERSION = "0.1.0"

LpMaximize = -1
LpMinimize = 1

LpConstraintLE = -1
LpConstraintEQ = 0
LpConstraintGE = 1

LpContinuous = "Continuous"
LpInteger = "Integer"
LpBinary = "Binary"

LpStatusNotSolved = 0
LpStatusOptimal = 1
LpStatusInfeasible = -1
LpStatusUnbounded = -2
LpStatusUndefined = -3

LpStatus = {
    LpStatusNotSolved: "Not Solved",
    LpStatusOptimal: "Optimal",
    LpStatusInfeasible: "Infeasible",
    LpStatusUnbounded: "Unbounded",
    LpStatusUndefined: "Undefined",
}

LpSolutionNoSolutionFound = 0
LpSolutionOptimal = 1
LpSolutionIntegerFeasible = 2
LpSolutionInfeasible = -1
LpSolutionUnbounded = -2

LpSolution = {
    LpSolutionNoSolutionFound: "No Solution Found",
    LpSolutionOptimal: "Optimal Solution Found",
    LpSolutionIntegerFeasible: "Solution Found",
    LpSolutionInfeasible: "No Solution Exists",
    LpSolutionUnbounded: "Solution is Unbounded",
}
