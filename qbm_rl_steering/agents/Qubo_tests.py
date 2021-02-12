from neal import SimulatedAnnealingSampler
import numpy as np


# Play a bit with the neal SimulatedAnnealingSampler QUBO concept to
# understand it better.

def solve_problem(Q, num_reads=5000):
    """ Plug in Q dictionary with linear and quadratic coefficients. Linear
    coefficients are self-coupling, i.e key looks like this
    (i, i): coupling_value, while quadratic coefficients are given by the
    couplings between nodes, i.e. (i, j) = coupling_value, where i != j. See
    examples below for better understanding. """
    sampler = SimulatedAnnealingSampler()
    samples = list(sampler.sample_qubo(Q, num_reads=num_reads).samples())
    samples_np = np.array([list(s.values()) for s in samples])
    return samples_np


def print_occurences(samples_np):
    """ Count and print how often each solution (here row in the np array)
    appears during the sampling. 'sol' contains the unique rows, i.e our
    solutions that minimize the posed problem, and 'cnt' the corresponding
    count for each solution. """
    sol, cnt = np.unique(samples_np, axis=0, return_counts=True)
    total_count = np.sum(cnt)
    for s, c in zip(sol, cnt):
        print(f'{s}: {100*c/total_count:.1f} % of occurences')


# Case 2, just 2 qubits.
# Try to solve -3x1 - 2x1x2. We expect solution x1 = 1, x2 = 1
# Solution table:
# x1   x2   f(x1, x2)
# 1     0     -3
# 0     1      0
# 1     1     -5
# 0     0      0

# QUBO matrix
# -3 in front of x1 is a linear coefficient, i.e. an on-diagonal term at
# index (0, 0). -2 in front of x1*x2 is a quadratic coefficient and needs to
# be specified as off-diagonal term (but only once I think!) at index (0,
# 1). Could we also specify at index (1, 0)? (lower triang. matrix). Or could
# we even share the coefficient between (0, 1) and (1, 0), but to 50% each?
# What if we don't share the weights on the off-diagonals, but give full
# weight to both entries? Let's try.
print('================')
print('Case 1: 2 qubits')
print('Minimizing f(x1, x2) = -3x1 - 2 x1x2')

# Case 1a: upper triangular matrix
Q = {(0, 0): -3, (0, 1): -2}
samples_np = solve_problem(Q)
# print('Case 1a, samples_np', samples_np)
print('\nCase 1a')
print_occurences(samples_np)

# Case 1b: lower triangular matrix
Q = {(0, 0): -3, (1, 0): -2}
samples_np = solve_problem(Q)
# print('\nCase 1b, samples_np', samples_np)
print('\nCase 1b')
print_occurences(samples_np)

# Case 1c: symmetric matrix, shared weights
Q = {(0, 0): -3, (1, 0): -2/2., (0, 1): -2/2.}
samples_np = solve_problem(Q)
# print('\nCase 1c, samples_np', samples_np)
print('\nCase 1c')
print_occurences(samples_np)

# Case 1d: symmetric matrix, but full weights either side
# I would expect this is solving a different problem, namely
# f(x1, x2) = -3 * x1 - 4 * x1 * x2
Q = {(0, 0): -3, (1, 0): -2, (0, 1): -2}
samples_np = solve_problem(Q)
# print('\nCase 1d, samples_np', samples_np)
print('\nCase 1d')
print_occurences(samples_np)

# Of course they all give the same solution, because even Case 4 is still the
# minimum for 1, 1. But the energy would be different in my opinion.


# Case 2: 3 qubits.
# Think of a problem where this is not the case.
# e.g. -3 x1 + x1 x2 - x2 x3
# and then we split the last weight in front of x2 x3.

# Solution table:
# 0  0  0    0   0
# 1  0  0   -3  -3   # Equally good solution
# 0  1  0    0   0
# 0  0  1    0   0
# 1  1  0   -2  -2
# 0  1  1   -1  -2
# 1  0  1   -3  -3   # Equally good solution
# 1  1  1   -3  -4   # Will be the only solution when using symmetric matrix.

print('\n\n================')
print('Case 2: 3 qubits')
print('Minimizing f(x1, x2, x3) = -3x1 + x1 x2 - x2 x3')
# Case 2a: upper triangular matrix
Q = {(0, 0): -3, (0, 1): 1, (1, 2): -1}
samples_np = solve_problem(Q)
# print('Case 2a, samples_np', samples_np)
print('\nCase 2a')
print_occurences(samples_np)

# Case 2b: lower triangular matrix
Q = {(0, 0): -3, (0, 1): 1, (2, 1): -1}
samples_np = solve_problem(Q)
# print('Case 2b, samples_np', samples_np)
print('\nCase 2b')
print_occurences(samples_np)

# Case 2c: symmetric matrix, shared weights
Q = {(0, 0): -3, (0, 1): 1, (1, 2): -1/2., (2, 1): -1/2.}
samples_np = solve_problem(Q)
# print('Case 2c, samples_np', samples_np)
print('\nCase 2c')
print_occurences(samples_np)

# Case 2d: symmetric matrix, but full weights either side
# I would expect this is solving a different problem, namely
# f(x1, x2, x3) = -3 * x1 + x1 * x2 - 2 x2 * x3
# and it will have a different solution (see table above)
Q = {(0, 0): -3, (0, 1): 1, (1, 2): -1., (2, 1): -1.}
samples_np = solve_problem(Q)
# print('Case 2d, samples_np', samples_np)
print('\nCase 2d')
print_occurences(samples_np)


# CONFIRMED!
# TODO: this means that I think there is a bug in the create_general_Q_from
#  method.
