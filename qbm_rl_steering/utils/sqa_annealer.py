from typing import Tuple, Dict
import numpy as np

try:
    # Found this library that does SQA (is CUDA enabled in principle)
    import sqaod as sq
except ImportError:
    pass


class SQA:
    def __init__(self, big_gamma: Tuple[float, float], beta: float,
                 n_replicas: int, n_nodes: int = 16,
                 big_gamma_schedule: str = 'linear') -> None:
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
        :param big_gamma_schedule: defines how big_gamma decays throughout
        the course of the annealing.
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
        self.big_gamma_schedule = big_gamma_schedule
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
            big_gamma_vs_t = np.linspace(
                self.big_gamma_init, self.big_gamma_final, n_steps)
        elif self.big_gamma_schedule == 'logarithmic':
            # gamma(t) = c / log(t + 2) + d
            const_c = ((self.big_gamma_final - self.big_gamma_init) /
                       (1./np.log(2+n_steps) - 1./np.log(2)))
            const_d = self.big_gamma_init - const_c / np.log(2)
            t = np.arange(n_steps)
            big_gamma_vs_t = const_d + const_c / (np.log(2 + t))
        elif self.big_gamma_schedule == 'sqrt':
            # gamma(t) = c / sqrt(t+1) + d
            const_c = ((self.big_gamma_final - self.big_gamma_init) /
                       (1./np.sqrt(1. + n_steps) - 1.))
            const_d = self.big_gamma_init - const_c
            t = np.arange(n_steps)
            big_gamma_vs_t = const_d + const_c / np.sqrt(t + 1.)
        else:
            raise NotImplementedError(
                "big_gamma_schedule other than linear, logarithmic, or sqrt "
                "not implemented.")

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

            for j in range(n_steps):
                self.annealer.anneal_one_step(
                    big_gamma_vs_t[j], self.beta_final)

            # Get the spin configurations at the end of this annealing run
            # .get_q() returns the spin configurations as np matrix with shape
            # (n_replicas, n_hidden_nodes).
            # Note that .get_x() would return the bits (i.e. -1 spins are 0s)
            spin_configurations[i, :, :] = self.annealer.get_q()

        return spin_configurations
