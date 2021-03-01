from typing import Tuple, Dict
import numpy as np

# The DWAVE-neal is NOT a quantum annealing simulator (SQA, transverse field
# decay), but it does simulated annealing (SA, temperature decay)
from neal import SimulatedAnnealingSampler


class SA:
    def __init__(self, beta: Tuple[float, float], n_replicas: int,
                 n_nodes: int = 16, beta_schedule: str = 'linear') -> None:
        """
        Initialize a simulated annealer (SA) using the dwave neal library.
        :param beta: inverse temperature. First and last entry of tuple
        correspond to the initial and final inv. temperature of the annealing
        process.
        :param n_replicas: number of replications of the graph in the extended
        dimension (number of Trotter slices). Believe that n_replicas = 1 in
        classical annealing.
        :param n_nodes: the number of qubits of the chip / problem to be
        simulated. Default is set to 16 (which corresponds to the 2 unit
        cells of the DWAVE-2000). This number is also the size of the square
        QUBO matrix that defines the couplings between the qubits / nodes.
        :param beta_schedule: defines how beta increases (i.e. temperature
        decays) throughout the course of the annealing.
        """
        self.n_nodes = n_nodes

        if n_replicas > 1:
            print("! Using classical annealing: setting n_replicas = 1.")
        self.n_replicas = 1

        # D-WAVE SA
        self.annealer = SimulatedAnnealingSampler()

        # Inverse temperature tuple
        if type(beta) is not tuple:
            raise ValueError("When using SA, specify beta as a tuple of two "
                             "floats to implement the temperature decay "
                             "required for annealing.")

        if beta[0] > beta[1]:
            print("! Your initial beta is larger than the final one, i.e. the "
                  "initial temperature is smaller than the final one. This is "
                  "contrary to the typical annealing process.")

        self.beta_schedule = beta_schedule
        self.beta_init = beta[0]
        self.beta_final = beta[1]

        # For classical annealing we should use big_gamma == 0 throughout.
        # This will not go into the annealer, but is important to be set for
        # the calculation of the effective Hamiltonian, for example.
        print(" ! Warning: using classical annealer: setting big_gamma = 0.")
        self.big_gamma_init = 0.
        self.big_gamma_final = 0.

    def anneal(self, qubo_dict: Dict, n_meas_for_average: int,
               *args, **kwargs) -> np.ndarray:
        """
        Run the CLASSICAL AnnealingSampler with the DWAVE QUBO method and
        generate all the samples (= spin configurations at hidden nodes of
        Chimera graph, with values {-1, +1}). The DWAVE sample() method
        provides samples in a list of dictionaries and the values are the
        corresponding spins {0, 1}. We will work with {-1, 1} rather than
        {0, 1}, so we remap all the sampled spin configurations and turn
        the list of dictionaries into a 3D numpy array.
        :param qubo_dict: Dictionary of the coupling weights of the graph. It
        corresponds to an upper triangular matrix, where the self-coupling
        weights (linear coefficients) are on the diagonal, i.e. (i, i) keys,
        and the quadratic coefficients are on the off-diagonal, i.e. (i, j)
        keys with i < j. As the visible nodes are clamped, they are
        incorporated into biases, i.e. self-coupling weights of the hidden
        nodes they are connected to.
        :param n_meas_for_average: number of times we run an independent
        annealing process from start to end.
        :param: *args, **kwargs: accept these to match interface among
        annealers.
        :return Spin configurations, i.e. {-1, 1} of all the nodes and
        n_replicas (# Trotter slices) for all the n_meas_for_average runs we
        do. np array with dimensions
        (n_meas_for_average, n_replicas, n_hidden_nodes).
        """
        # TODO: we run all the n_meas_for_average at once. I hope that's OK.
        num_reads = n_meas_for_average * self.n_replicas
        spin_configurations = list(self.annealer.sample_qubo(
            Q=qubo_dict, num_reads=num_reads,
            beta_schedule_type=self.beta_schedule,
            beta_range=(self.beta_init, self.beta_final)).samples())

        # Convert to np array and flip all the 0s to -1s
        spin_configurations = np.array([
            list(s.values()) for s in spin_configurations])
        spin_configurations[spin_configurations == 0] = -1
        spin_configurations = spin_configurations.reshape(
            (n_meas_for_average, self.n_replicas, self.n_nodes))

        return spin_configurations
