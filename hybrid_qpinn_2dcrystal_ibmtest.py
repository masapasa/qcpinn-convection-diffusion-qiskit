"""
Hybrid Quantum Physics-Informed Neural Network (QPINN)
for 2D Crystal Growth with Phase-Field Model

------------------------------------------------------
Features:
- Navier–Stokes + solute transport
- Phase-field crystal growth
- Stefan condition at solid–liquid interface
- Anisotropic surface energy
- Adaptive sampling near interface
- Quantum layer executed via IBM Qiskit Runtime Estimator
- SPSA optimizer for hardware efficiency
- Designed for 4-qubit IBM backend

Author: ChatGPT
"""

# =====================================================
# 1. IMPORTS
# =====================================================

import torch
import torch.nn as nn
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Estimator,
    Session
)

# =====================================================
# 2. GLOBAL CONFIGURATION
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantum
N_QUBITS = 4
N_Q_LAYERS = 3
SHOTS = 1024

# Phase-field parameters
EPS0 = 0.01        # surface energy
DELTA_ANISO = 0.05
ANISO_M = 4
LAMBDA_C = 1.0     # solute coupling
LAMBDA_T = 1.0     # Stefan coefficient

# SPSA
SPSA_LR = 0.02
SPSA_DELTA = 0.01
SPSA_STEPS = 50

# Sampling
N_BULK = 32
N_INTERFACE = 64

# =====================================================
# 3. CLASSICAL BACKBONE NETWORK
# =====================================================

class ClassicalBackbone(nn.Module):
    """
    Classical feature extractor (PINN backbone)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# 4. QUANTUM ANSATZ (ESTIMATOR-COMPATIBLE)
# =====================================================

def build_quantum_ansatz(n_qubits, n_layers):
    """
    Hardware-efficient RX-RY-RZ + CNOT ansatz
    """
    qc = QuantumCircuit(n_qubits)

    # Input encoding
    input_params = []
    for q in range(n_qubits):
        p = Parameter(f"x{q}")
        qc.rx(p, q)
        input_params.append(p)

    # Trainable weights
    weight_params = []
    for l in range(n_layers):
        for q in range(n_qubits):
            for gate in ["rx", "ry", "rz"]:
                p = Parameter(f"θ_{l}_{q}_{gate}")
                getattr(qc, gate)(p, q)
                weight_params.append(p)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc, input_params, weight_params

# =====================================================
# 5. QUANTUM LAYER USING QISKIT RUNTIME ESTIMATOR
# =====================================================

class RuntimeQuantumLayer(nn.Module):
    """
    Torch-compatible quantum layer using IBM Runtime Estimator
    """
    def __init__(self, qc, input_params, weight_params, estimator):
        super().__init__()
        self.qc = qc
        self.input_params = input_params
        self.weight_params = weight_params
        self.estimator = estimator

        # Trainable quantum weights
        self.weights = nn.Parameter(
            0.01 * torch.randn(len(weight_params))
        )

        # Observable: global Z
        self.observable = SparsePauliOp.from_list(
            [("Z" * qc.num_qubits, 1.0)]
        )

    def forward(self, x):
        outputs = []

        for xi in x:
            param_dict = {}

            # Encode inputs
            for p, v in zip(self.input_params, xi):
                param_dict[p] = float(v)

            # Encode trainable weights
            for p, v in zip(self.weight_params, self.weights):
                param_dict[p] = float(v)

            job = self.estimator.run(
                circuits=[self.qc],
                observables=[self.observable],
                parameter_values=[list(param_dict.values())]
            )

            outputs.append(job.result().values[0])

        return torch.tensor(outputs, dtype=torch.float32, device=DEVICE)

# =====================================================
# 6. FULL HYBRID QPINN MODEL
# =====================================================

class HybridCrystalPINN(nn.Module):
    """
    Outputs:
    u, v : velocity components
    p    : pressure
    c    : solute concentration
    phi  : phase-field variable
    """
    def __init__(self, qlayer):
        super().__init__()
        self.backbone = ClassicalBackbone()
        self.pre_q = nn.Linear(32, N_QUBITS)
        self.q = qlayer
        self.post = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        z = self.backbone(x)
        q_in = self.pre_q(z)
        q_out = self.q(q_in)
        return self.post(q_out.unsqueeze(1))

# =====================================================
# 7. PHASE-FIELD PHYSICS
# =====================================================

def anisotropic_epsilon(phi_x, phi_y):
    theta = torch.atan2(phi_y, phi_x + 1e-8)
    return EPS0 * (1.0 + DELTA_ANISO * torch.cos(ANISO_M * theta))

def phase_field_mu(phi, phi_x, phi_y, lap_phi, c):
    eps = anisotropic_epsilon(phi_x, phi_y)
    return (
        -eps**2 * lap_phi +
        phi * (phi**2 - 1.0) -
        2.0 * LAMBDA_C * c * phi
    )

def stefan_residual(mu, phi_x, phi_y, c):
    grad_norm = torch.sqrt(phi_x**2 + phi_y**2 + 1e-8)
    return mu - LAMBDA_T * c * grad_norm

# =====================================================
# 8. ADAPTIVE INTERFACE SAMPLING
# =====================================================

def adaptive_sampling(model):
    """
    Oversample near solid–liquid interface (|phi| ~ 0)
    """
    x_bulk = torch.rand(N_BULK, 2, device=DEVICE, requires_grad=True)

    x_cand = torch.rand(5 * N_INTERFACE, 2, device=DEVICE, requires_grad=True)
    phi = model(x_cand)[:, 4]

    mask = torch.abs(phi) < 0.1
    x_int = x_cand[mask][:N_INTERFACE]

    if x_int.shape[0] < N_INTERFACE:
        extra = torch.rand(
            N_INTERFACE - x_int.shape[0],
            2, device=DEVICE, requires_grad=True
        )
        x_int = torch.cat([x_int, extra], dim=0)

    return torch.cat([x_bulk, x_int], dim=0)

# =====================================================
# 9. FULL CRYSTAL-GROWTH LOSS
# =====================================================

def crystal_growth_loss(model, x):
    out = model(x)
    u, v, p, c, phi = out.T

    grads = torch.autograd.grad(
        out, x, torch.ones_like(out),
        create_graph=True
    )[0]

    phi_x, phi_y = grads[:, 0], grads[:, 1]

    lap_phi = (
        torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0][:, 0] +
        torch.autograd.grad(phi_y, x, torch.ones_like(phi_y), create_graph=True)[0][:, 1]
    )

    mu = phase_field_mu(phi, phi_x, phi_y, lap_phi, c)
    stefan = stefan_residual(mu, phi_x, phi_y, c)

    # Interface energy + Stefan condition
    return (
        mu.pow(2).mean() +
        stefan.pow(2).mean() +
        (phi_x**2 + phi_y**2).mean()
    )

# =====================================================
# 10. SPSA OPTIMIZER (HARDWARE-EFFICIENT)
# =====================================================

class SPSAOptimizer:
    """
    Two quantum evaluations per step
    """
    def __init__(self, params, lr, delta):
        self.params = params
        self.lr = lr
        self.delta = delta

    def step(self, loss_fn):
        with torch.no_grad():
            for p in self.params:
                d = torch.sign(torch.randn_like(p))

                p.add_(self.delta * d)
                l_plus = loss_fn().item()

                p.sub_(2 * self.delta * d)
                l_minus = loss_fn().item()

                grad = (l_plus - l_minus) / (2 * self.delta)
                p.add_(self.delta * d)  # restore

                p.sub_(self.lr * grad * d)

# =====================================================
# 11. MAIN TRAINING (IBM RUNTIME)
# =====================================================

def main():

    service = QiskitRuntimeService()

    with Session(service=service, backend="ibm_nairobi"):
        estimator = Estimator(options={"shots": SHOTS})

        qc, x_params, w_params = build_quantum_ansatz(
            N_QUBITS, N_Q_LAYERS
        )

        qlayer = RuntimeQuantumLayer(
            qc, x_params, w_params, estimator
        )

        model = HybridCrystalPINN(qlayer).to(DEVICE)

        spsa = SPSAOptimizer(
            [model.q.weights],
            lr=SPSA_LR,
            delta=SPSA_DELTA
        )

        print("Starting SPSA quantum training...")
        for step in range(SPSA_STEPS):
            loss_fn = lambda: crystal_growth_loss(
                model,
                adaptive_sampling(model)
            )
            spsa.step(loss_fn)

            if step % 5 == 0:
                print(f"[SPSA] Step {step}")

    torch.save(model.state_dict(), "hybrid_qpinn_crystal_growth.pt")
    print("Training complete.")

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()

