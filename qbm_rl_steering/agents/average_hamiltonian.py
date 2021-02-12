import numpy as np
import math
import random
from neal import SimulatedAnnealingSampler
from typing import Union


# Copied from Mircea
def initialise_weights():

    # ==============================
    # Couplings between hidden nodes
    w_hh = dict()

    # This first nested loop initializes weights to fully connect the nodes in
    # the two unit cells of the Chimera graph (see Fig. 2 in the paper). The
    # indexing of the nodes is starting at the top left (node 0) and goes
    # down vertically (blue nodes), and then to the right (first red node is
    # index 4). These are 32 couplings = 2 * 4**2
    for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
        for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            w_hh[(i, j)] = 2 * random.random() - 1
            w_hh[(ii, jj)] = 2 * random.random() - 1

    # This is the for loop that connects the 4 red nodes form the first unit
    # cell of the Chimera graph on the left (Fig. 2) to the blue nodes of the
    # second unit on the right, i.e. unit 4 to unit 12; unit 5 to unit 13, etc.
    # These are 4 additional couplings
    for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
        w_hh[(i, j)] = 2 * random.random() - 1

    # We get a total of 32 + 4 = 36 hidden couplings defined by weights w_hh.

    # ==========================================
    # Couplings between visible (that is the 'input' (=state layer) and the
    # 'output' (=action layer) and hidden nodes
    w_vh = dict()

    n_bits_observation_space = 8
    n_bits_action_space = 2

    # Dense connection between the state nodes (visible, input) and the BLUE
    # hidden nodes (all 8 of them) of the Chimera graph. Blue nodes have
    # indices [0, 1, 2, 3, 12, 13, 14, 15]. We hence have connections between
    # the state nodes [0, 1, ..., n_bits_observation_space] to all of the blue
    # nodes.
    # This is n_bits_observation_space * 8 = 64 couplings (here)
    for j in (tuple(range(4)) + tuple(range(12, 16))):
        for i in range(n_bits_observation_space):
            w_vh[(i, j,)] = 2 * random.random() - 1

    # Dense connection between the action nodes (visible, output) and the RED
    # hidden nodes (all 8 of them) of the Chimera graph. Red nodes have indices
    # [4, 5, 6, 7, 8, 9, 10, 11]. We hence have connections between the
    # action nodes [n_bits_observation_space, ..,
    # n_bits_observation_space + n_bits_action_space] (here: [8, 9]) to all of
    # the red nodes.
    # This is n_bits_action_space * 8 = 16 couplings (here)
    for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
        for i in range(
                n_bits_observation_space,
                n_bits_observation_space + n_bits_action_space):
            w_vh[(i, j,)] = 2 * random.random() - 1

    # We get a total of 64 + 16 = 80 couplings (here) defined by weights w_vh.
    return w_hh, w_vh


w_hh, w_vh = initialise_weights()
print('w_hh.keys()', w_hh.keys())
print('w_vh.keys()', w_vh.keys())
print('#couplings hh', len(w_hh.keys()))
print('#couplings vh', len(w_vh.keys()))


def convert_bits(vec: Union[tuple, np.ndarray], mapping: dict = {0: -1}) \
        -> tuple:
    """ Remap values in vec (bits) according to the mapping dict. Default
    mapping means that 0s are transformed into -1s.
    :param vec input np.ndarray or tuple that needs to be converted.
    :param mapping dictionary that contains the remapping of the values
    :return remapped vector as tuple. """
    vec = np.array(vec)
    for inp, out in mapping.items():
        vec[vec == inp] = out
    return tuple(vec)


def make_action_binary(action_index: int, n_bits_action_space: int = 2) -> \
        tuple:
    """ Similar to make_state_discrete_binary. Convert action_index to a
    binary vector using 0s and 1s. Conversion of 0s to -1s will be done in a
    separate function.
    :param action_index: index of action (integer). See which index
    corresponds to which action in env.action_map.
    :param n_bits_action_space: number of bits required to describe action
    space in binary.
    :return binary vector that encodes the action_index """
    binary_fmt = f'0{n_bits_action_space}b'
    action_binary = tuple([int(i) for i in format(action_index, binary_fmt)])
    return action_binary


def create_visible_iterable(state: np.ndarray, action: int,
                            n_bits_action_space: int = 2) -> tuple:
    """ Take state (e.g. directly from environment), and action_index (
    following env.action_map), and concatenate them to the visible_iterable
    tuple required for the create_general_Q_from(..) function. This also
    converts all 0s appearing in state and action_index (once made binary) to
    -1s.
    :param state: state in binary-encoded vector, as obtained directly from
    the environment (either through .reset(), or .step())
    :param action: index of action as used in environment
    :return tuple, concatenation of state and action, following the
    convention of Mircea's code on QBM. """
    s = convert_bits(state)
    a = convert_bits(make_action_binary(action, n_bits_action_space))
    return s + a


# Copied from Mircea
def create_general_Q_from(Q_hh, Q_vh, visible_iterable):
    """ Creates a weight dict that can be used with the DWAVE API. As the
    visible units are clamped, they are incorporated into biases.
    :param Q_hh: Contains key pairs (i,j) where i < j for hidden-hidden weights.
    :param Q_vh: Contains key pairs (visible, hidden) for visible-hidden
    weights.
    :param visible_iterable: Contains -1/1 values. """
    # TODO: in DWAVE documentation it says that QUBO requires upper-diagonal
    #  matrix Q (NxN upper triangular matrix) of weights (as dictionary)
    #  Biases (aka linear coefficients) are self-to-self, i.e. {(1, 1): w} is a
    #  self coupling that means a bias of w. They correspond to the diagonal
    #  entries of the Q matrix / dict.

    Q = dict()
    for k_pair, w in Q_hh.items():
        # Add flipped couplings (e.g. (8, 12) with weight w, add also
        # (12, 8) with weight w).
        # TODO: Do we really have to specify the connections in 'both
        #  directions'? These are the couplings between the hidden nodes
        #  specified in the previous for-loop. They are quadratic (rather
        #  than bias) couplings since they are not self-couplings. Still not
        #  sure why we have to create both Q[(i, j)] and Q[(j, i)] with same
        #  weight? Somewhere in the documentation it said that if you prefer
        #  a symmetric matrix (rather than an upper triangular one), you can
        #  do that by splitting the weight half between the upper and lower
        #  triangular entries (that means we would have to set w/2 in the
        #  line below). I think this is confirmed now - check the
        #  Qubo_tests.py file to see that. Just thinking now that maybe
        #  because we anyway are tweaking / updating the weights, this is not
        #  relevant. But we can maybe save computational time by specifying
        #  fewer couplings?
        # Q[k_pair] = Q[(k_pair[1], k_pair[0],)] = w
        Q[k_pair] = w

    # What we do here is add the couplings of visible nodes (both actions and
    # states) as a bias to the nodes of the Chimera graph. The reason why its
    # a bias is because they are added as self-couplings, through the k_pair[
    # 1]. The index 1 specifies the index of the node on the Chimera graph.
    # The index 0 is the visible state / action node. Everything else that
    # doesn't use self-coupling is treated as quadratic terms of the
    # minimzation problem (e.g. couplings between hidden nodes defined above)
    # TODO: This one here looks correct. I am still not sure, however,
    #  what this sentence in the docstring above means: "As the visible units
    #  are clamped, they are incorporated into biases." I see that this is
    #  what's done here; we take the visible nodes, i.e. states and actions,
    #  and add them as biases to the nodes of the Chimera graph that they are
    #  connected to.
    for k_pair, w in Q_vh.items():
        if (k_pair[1], k_pair[1],) not in Q:
            Q[(k_pair[1], k_pair[1],)] = w * visible_iterable[k_pair[0]]
        else:
            Q[(k_pair[1], k_pair[1],)] += w * visible_iterable[k_pair[0]]
    return Q


"""
Trying to implement the effective Hamiltonian calculation according to eq. (9)
in the paper: https://arxiv.org/pdf/1706.00074.pdf
:param r: number of replicas (number of 'copies' of the 2D Ising model into 
the third dimension)
:param w_hh: weights between hidden nodes
:param w_vh: weights between visible and hidden nodes
"""

# The samples contain the measured spin states of all the HIDDEN nodes.
# sample_count is given by the number of times we want to run for a good
# estimate of the average effective Hamiltonian multiplied by the number of
# replicas of the Ising model we are using.
r = 10  # 10 replicas
n_measurements_for_average = 5  # number of measurements for calc. of
# average
sample_count = r * n_measurements_for_average

# To run the sampler, we need to set a certain state and action, convert the
# action index into a binary vector, concatenate state and action binary
# vector and replace all 0s by -1s. Looks correct.
# TODO: Why do we have to make them -1, 1 ?
action = 2
state = np.array([0, 1, 1, 0, 0, 1, 1, 0])
vis_iterable = create_visible_iterable(state, action)
print('vis_iterable', vis_iterable)

# Next we need to bring the weights dictionary into a form that can be
# understood by the DWAVE API. For this we use the create_general_Q_from(..)
# TODO: what is num_sweeps?
general_Q = create_general_Q_from(w_hh, w_vh, vis_iterable)
print('general_Q', general_Q)
samples = list(SimulatedAnnealingSampler().sample_qubo(
    general_Q, num_reads=sample_count).samples())

# Samples are provided in a list of dictionaries. Each dictionary corresponds
# to 1 sample. The dictionary keys are the indices of the hidden nodes [0, 1,
# ..., 15], and the values are the corresponding spins [0, 1].
# print('len(samples)', len(samples))
# print('samples', samples)

# Now we have everything to calculate the effective Hamiltonian
# We start with only 1 measurement , i.e. no statistics for calc. the average
# We already use r = 10 replicas.

# k: summation index over replicas, runs from 1 to r
# h, h': summation indices over the nodes, resp. spin configurations
# w_plus: w_plus parameter explained in the paper. Log must be log_10,
# since in the paper otherwise they use ln for natural log.
# capital_gamma: parameter as in paper
# beta: parameter as in paper (some kind of temperature)

# Hyperbolic cotangent: coth(x) = cosh (x) / sinh(x)
big_gamma = 0.5
beta = 2.0
n_replicas = 10

# Calculate w_plus
x = big_gamma * beta / n_replicas
coth_term = math.cosh(x) / math.sinh(x)
w_plus = math.log10(coth_term) / (2. * beta)

# We will work with {-1, 1} rather than {0, 1}, so we remap all the sample
# spin configurations
# In view of the sum that we need to make, it's probably better to turn the
# dictionary of samples into a numpy array, where the axis 0 is the index
# avg., axis 1 is index k (replica), and axis 2 is index h
samples_np = np.array([list(s.values()) for s in samples])
samples_np[samples_np == 0] = -1  # swap 0s for -1s

# TODO: why are almost all the samples exactly identical => actually this
#  means that there is exactly one solution that minimizes the problem. Of
#  course we have to sample enough times to be sure of that. This will also
#  impact the average of the effective Hamiltonian.
# TODO: Is it OK to treat replicas in the same way as the 'independent'
#  measurements (for avg.) that we do?
for i in range(sample_count-1):
    if not np.array_equal(samples_np[i, :], samples_np[i+1, :]):
        print('Different samples found')
        print(samples_np[i, :])
        print(samples_np[i+1, :])

# Reshape
samples_np = samples_np.reshape(n_measurements_for_average, n_replicas, -1)

# Will loop over the i_avg: index of the measurement instance for averaging
# at the end.
# TODO: No need to loop, can probably also use np function mean with correct
#  axis => Yes works. See below. For better readability, could still have the
#  for-loop over i_avg, but will be slower.
# i_avg = 0

# Sum over h (hidden nodes) and k (replicas) of hk * h(k+1). The result has
# dimensions (n_measurements_for_average,)
# TODO: I think there is a typo indeed in eq. 9. The summation in w+(.. + ..)
#  of the first term should only go from k=1 to r-1
sum_hk_hkplus1 = np.sum(np.sum(
    samples_np[:, :-1, :] * samples_np[:, 1:, :], axis=1), axis=-1)

# Sum over h of h1 * hr
# The result has dimensions (n_measurements_for_average,)
sum_h1_hr = np.sum(samples_np[:, -1, :] * samples_np[:, 0, :], axis=-1)

# This should be the term of eq. (9) on the second line
# w+ * (sum h sum k .. + sum h .. )
# The result has dimensions (n_measurements_for_average,)
sum_wplus = -w_plus * (sum_hk_hkplus1 + sum_h1_hr)

# TODO: do summations also for line 1 in eq. (9)
# Is the first term sum over h, h' only a sum over the keys with different
# indices, while the second term with sum over v, h is summing up samples
# where indices of keys are identical?! Looks a bit like it
# I will use the original coupling weights w_hh, w_vh, rather than the
# general_Q dictionary. The idea is to be closer to the notation in the paper
# and to avoid using the weighted sums of the clamped visible nodes directly.
# This will also avoid having to check if a pair has already been added,
# since in the original w_hh each coupling is only defined once.

# - Remember: in samples_np, the last axis (axis=2) is h, i.e. the node
# - Here we assume that all replicas have the same weights (and I think that's
#   true)
# - In the end we want something that has shape (n_measurements_for_average, )
# h_sum_1 is the first double sum that appears in eq. (9)
h_sum_1 = 0.
for h, h_prime in w_hh.keys():
    weight = w_hh[(h, h_prime)]
    h_val = samples_np[:, :, h]
    h_prime_val = samples_np[:, :, h_prime]
    h_sum_1 += weight * h_val * h_prime_val

# Sum over replicas (sum over k), divide by n_replicas (= r in eq. (9)),
# and negate.
h_sum_1 = -np.sum(h_sum_1, axis=-1) / n_replicas
# print(h_sum_1.shape)  # Result has shape (n_measurements_for_average,)

# h_sum_2 is the second double sum that appears in eq. (9)
h_sum_2 = 0.
for v, h in w_vh.keys():
    weight = w_vh[(v, h)]
    v_val = vis_iterable[v]
    h_val = samples_np[:, :, h]
    h_sum_2 += weight * v_val * h_val

# Sum over replicas (sum over k), divide by n_replicas (= r) and negate.
h_sum_2 = -np.sum(h_sum_2, axis=-1) / n_replicas
# print(h_sum_2.shape)  # Result has shape (n_measurements_for_average,)

# Sum up all the terms, and then perform mean over n_measurements_for_average
effective_hamiltonian = h_sum_1 + h_sum_2 + sum_wplus
average_effective_hamiltonian = np.mean(effective_hamiltonian)
print('average_effective_hamiltonian', average_effective_hamiltonian)


import qbm_rl_steering.agents.qbmq_utils as utl
mircea_hamil = utl.get_3d_hamiltonian_average_value(
    samples, general_Q, n_replicas, n_measurements_for_average, big_gamma, beta)
print(mircea_hamil)


def get_average_effective_hamiltonian(
        samples: np.ndarray, w_hh: dict, w_vh: dict, visible_nodes: tuple,
        big_gamma: float, beta: float) -> float:
    """ This method calculates the average effective Hamiltonian as given in
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf , using samples
    obtained from DWAVE QUBO sample method.
    :param samples: samples returned by the DWAVE sample() method, but
    converted to numpy array and reshaped to
    (n_measurements_for_average, n_replicas, n_nodes).
    :param w_hh: coupling dictionary (quadratic coefficients) between the
    hidden nodes. The key is (h, h') where h and h' are in range [0, ..,
    n_nodes] and h != h' (no self-coupling here).
    :param w_vh: couplings between the visible nodes (states and actions) and
    the corresponding hidden nodes. The key is (v, h), where v is the index
    of the visible node in range [0, .., n_bits_observation_space, .. ,
    n_bits_observation_space + n_bits_action_space]. h is in range [0, ..,
    n_nodes].
    :param visible_nodes: tuple of inputs, i.e. visible nodes, given by
    binary vectors (with -1 and +1) of the states and action vectors
    concatenated (length: n_bits_observation_space + n_bits_action_space).
    :param big_gamma: hyperparameter; strength of the transverse field (
    virtual, average
    value), see paper for more details.
    :param beta: hyperparameter; inverse temperature used for simulated
    annealing
    :return: single float; effective Hamiltonian averaged over the individual
    measurements.
    """
    _, n_replicas, _ = samples.shape

    # FIRST TERM, sum over h, h' in Eq. (9)
    h_sum_1 = 0.
    for (h, h_prime), w in w_hh.items():
        h_sum_1 += w * samples[:, :, h] * samples[:, :, h_prime]

    # SECOND term, sum over v, h in Eq. (9)
    h_sum_2 = 0.
    for (v, h), w in w_vh.items():
        h_sum_2 += w * visible_nodes[v] * samples_np[:, :, h]

    # Sum over replicas (sum over k), divide by n_replicas, and negate
    # This corresponds to the first line in Eq. (9)
    # h_sum_12 has shape (n_measurements_for_average,)
    h_sum_12 = -np.sum(h_sum_1 + h_sum_2, axis=-1) / n_replicas

    # THIRD term, [-w_plus * (sum_hk_hkplus1 + sum_h1_hr)], in Eq. (9)
    # h_sum_3 has shape (n_measurements_for_average,)
    x = big_gamma * beta / n_replicas
    coth_term = math.cosh(x) / math.sinh(x)
    w_plus = math.log10(coth_term) / (2. * beta)

    hk_hkplus1_sum = np.sum(np.sum(
        samples[:, :-1, :] * samples[:, 1:, :], axis=1), axis=-1)
    h1_hr_sum = np.sum(samples[:, -1, :] * samples[:, 0, :], axis=-1)
    h_sum_3 = -w_plus * (hk_hkplus1_sum + h1_hr_sum)

    # Take average over n_measurements_for_average
    return float(np.mean(h_sum_12 + h_sum_3))


print(get_average_effective_hamiltonian(
    samples_np, w_hh, w_vh, vis_iterable, big_gamma, beta))
