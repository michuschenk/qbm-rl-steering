from typing import Tuple, Dict
import numpy as np

# The DWAVE-neal is NOT a quantum annealing simulator (SQA, transverse field
# decay) in my understanding, but it does simulated annealing (SA, temperature
# decay)
from neal import SimulatedAnnealingSampler

# Found this library that does SQA (is CUDA enabled in principle)
import sqaod as sq

# TODO: implement D-WAVE QPU annealer for test on the hardware.


class SA:
    def __init__(self, beta: Tuple[float, float], n_replicas: int,
                 n_nodes: int = 16) -> None:
        """
        Initialize a simulated annealer (SA) using the dwave neal library.
        :param beta: inverse temperature. First and last entry of tuple
        correspond to the initial and final inv. temperature of the annealing
        process.
        :param n_replicas: number of replications of the graph in the extended
        dimension (number of Trotter slices)
        :param n_nodes: the number of qubits of the chip / problem to be
        simulated. Default is set to 16 (which corresponds to the 2 unit
        cells of the DWAVE-2000). This number is also the size of the square
        QUBO matrix that defines the couplings between the qubits / nodes.
        """
        # TODO: does it make sense to have Trotter slices != 1 when using
        #  classical annealing?
        self.n_nodes = n_nodes
        self.n_replicas = n_replicas

        # D-WAVE SA
        self.annealer = SimulatedAnnealingSampler()

        # Inverse temperature tuple
        if type(beta) is not tuple:
            raise ValueError("When using SA, specify beta as a tuple of two "
                             "floats to implement the temperature decay "
                             "required for annealing.")

        self.beta_schedule = 'linear'
        self.beta_init = beta[0]
        self.beta_final = beta[1]

        # For classical annealing we should use big_gamma == 0 throughout
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


class SQA:
    def __init__(self, big_gamma: Tuple[float, float], beta: float,
                 n_replicas: int, n_nodes: int = 16):
        """
        Initialize a simulated quantum annealer (SQA).
        :param big_gamma: Transverse field; first entry is initial gamma and
        second entry is final gamma, following a linear schedule for now as
        explained in the paper.
        :param beta: inverse temperature (note that this parameter is kept
        constant in SQA other than in SA).
        :param n_replicas: number of replications of the graph in the extended
        dimension (number of Trotter slices)
        :param n_nodes: the number of qubits of the chip / problem to be
        simulated. Default is set to 16 (which corresponds to the 2 unit
        cells of the DWAVE-2000). This number is also the size of the square
        QUBO matrix that defines the couplings between the qubits / nodes.
        """
        self.n_nodes = n_nodes
        self.n_replicas = n_replicas

        # Initialize the SQ annealer object
        architecture = self._set_architecture()
        self.annealer = architecture.dense_graph_annealer()

        # Transverse field decay during annealing
        if type(big_gamma) is not tuple:
            raise ValueError("When using SQA, specify big_gamma as a tuple to "
                             "implement transverse field decay required for "
                             "annealing.")
        self.big_gamma_schedule = 'linear'
        self.big_gamma_init = big_gamma[0]
        self.big_gamma_final = big_gamma[1]

        # Inverse temperature (assumed constant)
        if type(beta) is tuple:
            raise ValueError("When using SQA, beta must be single float "
                             "(constant inverse temperature).")
        self.beta_init = beta
        self.beta_final = beta

    @staticmethod
    def _set_architecture():
        """
        Define where to run the annealing process (CPU or NVidia GPU, if
        available).
        :return the corresponding architecture object defined in sqaod (either
        cpu or gpu)
        """
        arch = sq.cpu
        if sq.is_cuda_available():
            import sqaod.cuda
            arch = sqaod.cuda
        return arch

    def set_seed(self, val: int):
        """
        Set random seed of the annealer (this is optional)
        :param val: the seed value (an integer)
        """
        self.annealer.seed(val)

    def _set_qubo_matrix(self, qubo_dict: Dict):
        """
        Update the QUBO matrix (the couplings of the graph)
        :param qubo_dict: QUBO dictionary defining the couplings. We will
        turn that into an upper triangular NxN matrix, where N is the number
        of nodes on the graph.
        """
        qubo_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i, j in qubo_dict.keys():
            qubo_matrix[i, j] = qubo_dict[(i, j)]

        # We fix that we always want to minimize
        # TODO: Is it even relevant for us whether we do minimize or maximize?
        self.annealer.set_qubo(qubo_matrix, sq.minimize)

    def anneal(self, qubo_dict: Dict, n_meas_for_average: int, n_steps: int) \
            -> np.ndarray:
        """
        Run the actual QUANTUM annealing process with decaying transverse field
        big_gamma and constant inverse temperature beta.
        :param qubo_dict: Dictionary of the coupling weights of the graph. It
        corresponds to an upper triangular matrix, where the self-coupling
        weights (linear coefficients) are on the diagonal, i.e. (i, i) keys,
        and the quadratic coefficients are on the off-diagonal, i.e. (i, j)
        keys with i < j. As the visible nodes are clamped, they are
        incorporated into biases, i.e. self-coupling weights of the hidden
        nodes they are connected to.
        :param n_meas_for_average: number of times we run an independent
        annealing process from start to end.
        :param n_steps: number of steps that one annealing process should take
        (~annealing time). This adapts the big_gamma decay schedule accordingly.
        :return Spin configurations, i.e. {-1, 1} of all the nodes and
        n_replicas (# Trotter slices) for all the n_meas_for_average runs we
        do. np array with dimensions
        (n_meas_for_average, n_replicas, n_hidden_nodes).
        """
        # Transverse field strength schedule
        if self.big_gamma_schedule == 'linear':
            big_gamma_step = (
                    (self.big_gamma_final - self.big_gamma_init) / n_steps)
        else:
            raise NotImplementedError(
                "big_gamma_schedule other than linear not implemented.")

        # Set the QUBO matrix
        self._set_qubo_matrix(qubo_dict)

        # Set number of Trotter slices (called replicas in the paper)
        # This has to be done after setting the QUBO matrix, otherwise sqaod
        # automatically sets n_trotters to n_nodes / 4 for some reason (see
        # sqaod doc.)
        self.annealer.set_preferences(n_trotters=self.n_replicas)

        # Annealing
        spin_configurations = np.empty(
            (n_meas_for_average, self.n_replicas, self.n_nodes))

        for i in range(n_meas_for_average):
            # Run one full annealing process
            self.annealer.prepare()
            self.annealer.randomize_spin()

            big_gamma = self.big_gamma_init
            for j in range(n_steps):
                self.annealer.anneal_one_step(big_gamma, self.beta_final)
                big_gamma += big_gamma_step

            # Get the spin configurations at the end of this annealing run
            # .get_q() returns the spin configurations as np matrix with shape
            # (n_replicas, n_hidden_nodes).
            # Note that .get_x() would return the bits (i.e. -1 spins are 0s)
            spin_configurations[i, :, :] = self.annealer.get_q()

        return spin_configurations
