from typing import Dict
import numpy as np

import qiskit as qsk
import qiskit.utils as utl
import qiskit.algorithms as alg
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class QAOA:
    def __init__(self, n_nodes: int = 16, solver: str = 'QAOA',
                 simulator: str = 'statevector', n_shots: int = 10,
                 beta_final: float = 2.) -> None:
        """
        Initialize a QAOA problem or NumPy Eigensolver using Qiskit library.
        :param n_nodes: number of qubits. This number is also
        the size of the square QUBO matrix that defines the couplings between
        the qubits / nodes.
        :param solver: either 'QAOA' or the classical 'NumPyEigensolver'
        :param simulator: 'qasm' or 'statevector'. The latter is quite slow.
        :param n_shots: number of samples per solver iteration, comparable to
        annealing_steps?
        :param beta_final: inverse temperature. To be understood as a
        hyperparameter here as it does not go into the calculation. But in
        the free energy calc. we have the entropy term which is to be divided
        by beta_final. Maybe we have to drop that additional term for QAOA?
        """
        self.n_nodes = n_nodes
        self.n_replicas = 1

        # Define the solver
        if solver == 'QAOA':
            quantum_instance = utl.QuantumInstance(backend=qsk.Aer.get_backend(simulator + '_simulator'))
            qaoa_problem = alg.QAOA(quantum_instance=quantum_instance)
        elif solver == 'NumPyEigensolver':
            qaoa_problem = alg.NumPyMinimumEigensolver()
        else:
            raise NotImplementedError("Requested solver is not implemented. "
                                      "Use either QAOA_qasm_simulator or "
                                      "NumPyEigensolver")
        self.solver = MinimumEigenOptimizer(qaoa_problem)

        self.beta_final = beta_final
        self.big_gamma_final = 0.

    def _reformulate_qubo(self, qubo_dict) -> QuadraticProgram:
        """ Translate qubo_dict into a Qiskit compatible format, i.e. a
        QuadraticProgram.
        :param qubo_dict: Dictionary of the coupling weights of the graph. It
        corresponds to an upper triangular matrix, where the self-coupling
        weights (linear coefficients) are on the diagonal, i.e. (i, i) keys,
        and the quadratic coefficients are on the off-diagonal, i.e. (i, j)
        keys with i < j. As the visible nodes are clamped, they are
        incorporated into biases, i.e. self-coupling weights of the hidden
        nodes they are connected to.
        :return QuadraticProgram (the Qiskit QUBO formulation)
        """
        qubo_problem = QuadraticProgram()

        # Define all the "qubits" / binary variables
        for i in range(self.n_nodes):
            qubo_problem.binary_var('x' + str(i))

        # Find all the weights in the qubo dict where keys are identical.
        # These are the biases => sort them in the same manner as we created
        # the binary variables above.
        # All the other key pairs are the coupling terms, they go in the
        # quadratic_terms dict.
        linear_terms = np.zeros(self.n_nodes)
        quadratic_terms = {}
        for (j, k), w in qubo_dict.items():
            if j == k:
                linear_terms[j] = w
            else:
                quadratic_terms[('x' + str(j), 'x' + str(k))] = w

        qubo_problem.minimize(linear=linear_terms, quadratic=quadratic_terms)

        return qubo_problem

    def sample(self, qubo_dict: Dict, n_meas_for_average: int,
               *args, **kwargs) -> np.ndarray:
        """
        Run the QAOA QUBO method and generate all the samples (= spin
        configurations at hidden nodes of Chimera graph, with values {-1,
        +1}). Remap all the sampled spin configurations and turn the list
        into a 3D numpy array.
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
        samplers.
        :return Spin configurations, i.e. {-1, 1} of all the nodes and
        n_replicas (# Trotter slices) for all the n_meas_for_average runs we
        do. np array with dimensions
        (n_meas_for_average, n_replicas, n_hidden_nodes).
        """
        qubo_problem = self._reformulate_qubo(qubo_dict)
        num_reads = n_meas_for_average * self.n_replicas

        spin_configurations = []
        for i in range(num_reads):
            spin_configurations.append(
                list(self.solver.solve(qubo_problem).x))

        # Convert to np array and flip all the 0s to -1s
        spin_configurations = np.array(spin_configurations)
        spin_configurations[spin_configurations == 0] = -1

        spin_configurations = spin_configurations.reshape(
            (n_meas_for_average, self.n_replicas, self.n_nodes))

        return spin_configurations
