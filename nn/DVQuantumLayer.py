import pennylane as qml
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import unitary_group
from scipy.linalg import polar
from qiskit_ibm_runtime import QiskitRuntimeService

class DVQuantumLayer(nn.Module):
    def __init__(self, args, diff_method="parameter-shift"):
        super().__init__()
        self.num_qubits = args["num_qubits"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.shots = args.get("shots", 1024) # Hardware requires shots
        self.q_ansatz = args["q_ansatz"]
        self.problem = args["problem"]
        self.encoding = args.get("encoding", "angle")
        
        # IBM Quantum Specific Arguments
        self.use_ibm_hardware = args.get("use_ibm_hardware", False)
        self.ibm_token = args.get("ibm_token", None)
        self.ibm_backend = args.get("ibm_backend", "ibmq_qasm_simulator")
        self.ibm_instance = args.get("ibm_instance", None)

        if self.q_ansatz == "layered":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 4,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "alternate":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (self.num_qubits * 4) - 4,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "cascade":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 3,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "farhi":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (2 * self.num_qubits - 2),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "sim_circ_15":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 2,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "cross_mesh":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        else:
            self.params = None

        if not hasattr(self, "params") or self.params is None:
            raise ValueError(
                "Parameters are not initialized. Check the q_ansatz value."
            )
        self._initialize_weights()

        if self.num_qubits >= 4:
            seed = args.get("seed", None)
            self.haar_seed1 = seed
            self.haar_seed2 = seed + 1 if seed is not None else None
        else:
            self.haar_seed1 = None
            self.haar_seed2 = None

        # DEVICE INITIALIZATION
        if self.use_ibm_hardware:
            print(f"Initializing IBM Quantum Backend: {self.ibm_backend}")
            is_local_simulator = False
            backend = None
            try:
                if self.ibm_instance:
                    service = QiskitRuntimeService(instance=self.ibm_instance, token=self.ibm_token)
                else:
                    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.ibm_token)
                try:
                    backend = service.backend(self.ibm_backend)
                    print(f"Using IBM Quantum backend: {backend.name}")
                except Exception as backend_error:
                    print(f"Warning: Backend '{self.ibm_backend}' not found: {backend_error}")
                    available_backends = [b.name for b in service.backends()]
                    print(f"Available backends: {available_backends[:10]}")
                    if available_backends:
                        print(f"Attempting to use first available backend: {available_backends[0]}")
                        try:
                            backend = service.backend(available_backends[0])
                            print(f"Using IBM Quantum backend: {backend.name}")
                        except Exception:
                            print("Falling back to local default.qubit simulator")
                            is_local_simulator = True
                    else:
                        print("Falling back to local default.qubit simulator")
                        is_local_simulator = True
            except Exception as e:
                print(f"Warning: Could not connect to IBM Quantum service: {e}")
                print("Falling back to local default.qubit simulator")
                is_local_simulator = True
            
            if is_local_simulator:
                self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=self.shots)
                local_diff_method = "finite-diff"
                self.use_batch_processing = False
            else:
                self.dev = qml.device(
                    "qiskit.remote", 
                    wires=self.num_qubits, 
                    shots=self.shots,
                    backend=backend
                )
                local_diff_method = "parameter-shift"
                self.use_batch_processing = False
        else:
            self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)
            local_diff_method = "backprop"
            self.use_batch_processing = True

        self.circuit = qml.QNode(
            self._quantum_circuit, self.dev, interface="torch", diff_method=local_diff_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_batch_processing:
            result = self.circuit(x)
            return torch.stack(result) if isinstance(result, list) else result
        else:
            batch_size = x.shape[0]
            outputs = []
            for idx, sample in enumerate(x):
                if idx % max(1, batch_size // 4) == 0 or idx == batch_size - 1:
                    print(f"Processing sample {idx+1}/{batch_size}...", end='\r', flush=True)
                result = self.circuit(sample)
                if isinstance(result, list):
                    if all(isinstance(r, torch.Tensor) for r in result):
                        outputs.append(torch.stack(result))
                    else:
                        outputs.append(torch.stack([torch.tensor(float(r), dtype=torch.float32) for r in result]))
                else:
                    if isinstance(result, torch.Tensor):
                        outputs.append(result)
                    else:
                        outputs.append(torch.tensor(float(result), dtype=torch.float32))
            if batch_size > 10:
                print(f"Processed {batch_size}/{batch_size} samples.        ")
            return torch.stack(outputs)

    def _quantum_circuit(self, x):
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(
                x, wires=range(self.num_qubits), normalize=True, pad_with=0.0
            )
        else:
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits), rotation="X")

        if self.q_ansatz == "layered":
            for layer in range(self.num_quantum_layers):
                self.layered(self.params[layer])
        elif self.q_ansatz == "alternate":
            for layer in range(self.num_quantum_layers):
                self.alternate(self.params[layer])
        elif self.q_ansatz == "cascade":
            for layer in range(self.num_quantum_layers):
                self.cascade(self.params[layer])
        elif self.q_ansatz == "farhi":
            for layer in range(self.num_quantum_layers):
                self.farhi_ansatz(self.params[layer])
        elif self.q_ansatz == "sim_circ_15":
            for layer in range(self.num_quantum_layers):
                self.create_sim_circuit_15(self.params[layer])
        elif self.q_ansatz == "cross_mesh":
            for layer in range(self.num_quantum_layers):
                self.create_cross_mesh(self.params[layer])

        if self.haar_seed1 is not None and self.haar_seed2 is not None:
            random_state1 = np.random.RandomState(self.haar_seed1)
            random_state2 = np.random.RandomState(self.haar_seed2)
            u1_unitary = unitary_group.rvs(4, random_state=random_state1)
            u2_unitary = unitary_group.rvs(4, random_state=random_state2)
            qml.QubitUnitary(u1_unitary, wires=[0, 1])
            qml.QubitUnitary(u2_unitary, wires=[2, 3])

        if self.num_qubits > 0:
            qml.Hadamard(wires=self.num_qubits - 1)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _initialize_weights(self):
        if self.q_ansatz == "farhi":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (2 * self.num_qubits - 2))
            )
        elif self.q_ansatz in ["sim_circ_15"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 2)
            )
        elif self.q_ansatz in [ "alternate"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4) - 4)
            )
        elif self.q_ansatz in ["layered"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4))
            )
        elif self.q_ansatz == "cascade":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 3)
            )
        elif self.q_ansatz == "cross_mesh":
            torch.nn.init.xavier_normal_(
                self.params.view(
                    self.num_quantum_layers, (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)
                )
            )
        else:
            raise ValueError("Invalid q_ansatz value.", self.q_ansatz)

    def layered(self, params):
        assert params is not None and len(params) == self.num_qubits * 4, (
            "The number of parameters must be equal to 4* num_qubits."
        )
        param_idx = 0
        for qubit_id in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1
        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])
        for qubit_id in range(self.num_qubits):
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1

    def alternate(self, params):
        assert params is not None and len(params) == (self.num_qubits * 4) - 4, (
            "The number of parameters must be equal to  num_qubits * 4."
        )
        param_idx = 0  
        def build_tdcnot(ctrl, tgt):
            nonlocal param_idx  
            qml.RY(params[param_idx], wires=ctrl)
            param_idx += 1
            qml.RY(params[param_idx], wires=tgt)
            param_idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[param_idx], wires=ctrl)
            param_idx += 1
            qml.RZ(params[param_idx], wires=tgt)
            param_idx += 1
        for i in range(self.num_qubits - 1)[::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)
        for i in range(self.num_qubits)[1::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)

    def cascade(self, params):
        param_counter = 0
        def add_rotations():
            nonlocal param_counter
            for i in range(0, self.num_qubits):
                qml.RX(params[param_counter], wires=i)
                param_counter += 1
            for i in range(0, self.num_qubits):
                qml.RZ(params[param_counter], wires=i)
                param_counter += 1
        def add_entangling_gates():
            nonlocal param_counter
            qml.CRX(params[param_counter], wires=[self.num_qubits - 1, 0])
            param_counter += 1
            for i in reversed(range(1, self.num_qubits)):
                qml.CRX(params[param_counter], wires=[i - 1, i])
                param_counter += 1
        add_rotations()
        add_entangling_gates()

    def farhi_ansatz(self, params):
        param_counter = 0
        if len(params) != (2 * self.num_qubits - 2):
            raise ValueError("Insufficient parameters for RXX and RZX gates")
        def RXX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=wires)
        def RZX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires)
        for i in range(self.num_qubits - 1):
            RXX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1
        for i in range(self.num_qubits - 1):
            RZX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1

    def create_sim_circuit_15(self, params):
        if params is None or len(params) != 2 * self.num_qubits:
            raise ValueError("Insufficient parameters for RXX and RZX gates")
        param_index = 0
        def apply_rotations():
            nonlocal param_index
            for i in range(self.num_qubits):
                qml.RY(params[param_index], wires=i)
                param_index += 1
        def apply_entangling_block1():
            for i in reversed(range(self.num_qubits)):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
        def apply_entangling_block2():
            for i in range(self.num_qubits):
                control_qubit = (i + self.num_qubits - 1) % self.num_qubits
                target_qubit = (control_qubit + 3) % self.num_qubits
                qml.CNOT(wires=[control_qubit, target_qubit])
        apply_rotations()
        apply_entangling_block1()
        apply_rotations()
        apply_entangling_block2()

    def create_cross_mesh(self, params):
        param_idx = 0
        expected_params = (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)
        if params is None or len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters but got {params.shape}"
            )
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        for i in range(self.num_qubits - 1, -1, -1):
            for j in range(self.num_qubits - 1, -1, -1):
                if j != i:
                    qml.CRZ(params[param_idx], wires=[i, j])
                    param_idx += 1
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

    def quantum_tanh_n_qubits(self, params, scale=1.0):
        if self.num_qubits is None:
            raise ValueError("Wires cannot be None.")
        if params is None:
            n_params = self.num_qubits * (self.num_qubits - 1) // 2
            params = [scale * np.pi / 2.0 * index for index in range(n_params)]
        for index in range(self.num_qubits):
            qml.PhaseShift(np.sin(params[index]) * np.pi, wires=index)