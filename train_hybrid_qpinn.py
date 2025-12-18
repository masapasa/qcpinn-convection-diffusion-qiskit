#!/usr/bin/env python3
"""
Hybrid Quantum Physics-Informed Neural Network (QPINN) Trainer

A single-file implementation for training hybrid quantum-classical neural networks
to solve 2D Partial Differential Equations (PDEs).

Supports:
- Local GPU (CUDA) for fast simulation
- IBM Quantum hardware (4+ qubits) for real quantum execution

PDE: 2D Convection-Diffusion equation
     ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)

Usage:
    # GPU Simulator (fast)
    python train_hybrid_qpinn.py --device cuda --epochs 5000

    # IBM Quantum Hardware
    python train_hybrid_qpinn.py --use-ibm --ibm-token YOUR_TOKEN --ibm-backend ibm_torino --epochs 100

Author: Generated with Claude Code
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pennylane as qml
from scipy.stats import unitary_group

# Optional: IBM Quantum support
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("Warning: qiskit-ibm-runtime not installed. IBM hardware support disabled.")


# =============================================================================
# 1. CONFIGURATION (ARGPARSE)
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum PINN Trainer for 2D PDEs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Device settings
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device (auto-detects CUDA if available)")

    # IBM Quantum settings
    parser.add_argument("--use-ibm", action="store_true",
                        help="Use IBM Quantum hardware instead of simulator")
    parser.add_argument("--ibm-token", type=str, default=None,
                        help="IBM Quantum API token")
    parser.add_argument("--ibm-backend", type=str, default="ibm_torino",
                        help="IBM Quantum backend name")
    parser.add_argument("--ibm-instance", type=str, default=None,
                        help="IBM Quantum instance (optional)")

    # Quantum circuit settings
    parser.add_argument("--num-qubits", type=int, default=4,
                        help="Number of qubits in quantum circuit")
    parser.add_argument("--ansatz", type=str, default="cascade",
                        choices=["cascade", "layered", "alternate", "farhi",
                                 "sim_circ_15", "cross_mesh"],
                        help="Quantum circuit ansatz type")
    parser.add_argument("--encoding", type=str, default="angle",
                        choices=["angle", "amplitude"],
                        help="Input encoding method")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Measurement shots for hardware execution")

    # Training settings
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Network architecture
    parser.add_argument("--hidden-dim", type=int, default=50,
                        help="Hidden dimension of classical layers")

    # Output settings
    parser.add_argument("--print-every", type=int, default=100,
                        help="Print loss every N epochs")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Base output directory")

    # PDE parameters
    parser.add_argument("--diffusion-coef", type=float, default=0.01,
                        help="Diffusion coefficient D")

    return parser.parse_args()


# =============================================================================
# 2. ANALYTICAL SOLUTION & DATA SAMPLER
# =============================================================================

def analytical_solution(t, x, y, D=0.01):
    """
    Analytical solution for 2D convection-diffusion equation.
    u(t, x, y) = sin(πx) * sin(πy) * exp(-2π²Dt)

    Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
    """
    pi = np.pi
    return np.sin(pi * x) * np.sin(pi * y) * np.exp(-2 * pi**2 * D * t)


def analytical_solution_torch(X, D=0.01):
    """Torch version of analytical solution."""
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    pi = torch.pi
    return torch.sin(pi * x) * torch.sin(pi * y) * torch.exp(-2 * pi**2 * D * t)


class DataSampler:
    """Sampler for generating training data from coordinate domains."""

    def __init__(self, coords, func, device="cpu", D=0.01):
        """
        Args:
            coords: Tensor of shape (2, dim) with [min_coords, max_coords]
            func: Function to compute target values
            device: Torch device
            D: Diffusion coefficient
        """
        self.coords = coords
        self.func = func
        self.device = device
        self.D = D
        self.dim = coords.shape[1]

    def sample(self, N):
        """Sample N random points from the domain."""
        rand_vals = torch.rand(N, self.dim, device=self.device)
        X = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * rand_vals
        y = self.func(X, self.D) if self.D is not None else self.func(X)
        return X, y


def create_samplers(device, D=0.01):
    """Create data samplers for initial, boundary, and domain conditions."""

    # Initial condition: t=0, x∈[0,1], y∈[0,1]
    ics_coords = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                               dtype=torch.float32, device=device)
    ics_sampler = DataSampler(ics_coords, analytical_solution_torch, device, D)

    # Boundary conditions (u=0 at all boundaries)
    # x=0 boundary
    bc1_coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                               dtype=torch.float32, device=device)
    # x=1 boundary
    bc2_coords = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                               dtype=torch.float32, device=device)
    # y=0 boundary
    bc3_coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                               dtype=torch.float32, device=device)
    # y=1 boundary
    bc4_coords = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                               dtype=torch.float32, device=device)

    def zero_bc(X, D=None):
        return torch.zeros((X.shape[0], 1), device=X.device)

    bc_samplers = [
        DataSampler(bc1_coords, zero_bc, device, None),
        DataSampler(bc2_coords, zero_bc, device, None),
        DataSampler(bc3_coords, zero_bc, device, None),
        DataSampler(bc4_coords, zero_bc, device, None),
    ]

    # Domain (residual) sampler: t∈[0,1], x∈[0,1], y∈[0,1]
    dom_coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                               dtype=torch.float32, device=device)

    def zero_residual(X, D=None):
        return torch.zeros((X.shape[0], 1), device=X.device)

    res_sampler = DataSampler(dom_coords, zero_residual, device, None)

    return ics_sampler, bc_samplers, res_sampler, dom_coords


# =============================================================================
# 3. QUANTUM ANSATZES
# =============================================================================

class QuantumAnsatzes:
    """Collection of parameterized quantum circuit ansatzes."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def get_num_params(self, ansatz_type):
        """Return number of parameters for given ansatz type."""
        n = self.num_qubits
        params_map = {
            "cascade": 3 * n,
            "layered": 4 * n,
            "alternate": max(0, 4 * n - 4),
            "farhi": max(0, 2 * n - 2),
            "sim_circ_15": 2 * n,
            "cross_mesh": 4 * n + n * (n - 1),
        }
        return params_map.get(ansatz_type, 3 * n)

    def cascade(self, params):
        """Cascade ansatz with ring CRX connectivity. Parameters: 3n"""
        n = self.num_qubits
        idx = 0

        # RX layer
        for i in range(n):
            qml.RX(params[idx], wires=i)
            idx += 1

        # RZ layer
        for i in range(n):
            qml.RZ(params[idx], wires=i)
            idx += 1

        # CRX entangling layer (ring topology)
        qml.CRX(params[idx], wires=[n - 1, 0])
        idx += 1
        for i in reversed(range(1, n)):
            qml.CRX(params[idx], wires=[i - 1, i])
            idx += 1

    def layered(self, params):
        """Layered ansatz with RZ-RX rotations and CNOT ring. Parameters: 4n"""
        n = self.num_qubits
        idx = 0

        # Layer 1: RZ-RX
        for i in range(n):
            qml.RZ(params[idx], wires=i)
            idx += 1
            qml.RX(params[idx], wires=i)
            idx += 1

        # CNOT ring
        for i in range(n):
            qml.CNOT(wires=[i, (i + 1) % n])

        # Layer 2: RX-RZ
        for i in range(n):
            qml.RX(params[idx], wires=i)
            idx += 1
            qml.RZ(params[idx], wires=i)
            idx += 1

    def alternate(self, params):
        """Alternating TDCNOT blocks. Parameters: 4n-4"""
        n = self.num_qubits
        idx = 0

        def build_tdcnot(ctrl, tgt):
            nonlocal idx
            qml.RY(params[idx], wires=ctrl)
            idx += 1
            qml.RY(params[idx], wires=tgt)
            idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[idx], wires=ctrl)
            idx += 1
            qml.RZ(params[idx], wires=tgt)
            idx += 1

        # Even pairs: (0,1), (2,3), ...
        for i in range(0, n - 1, 2):
            build_tdcnot(i, i + 1)

        # Odd pairs: (1,2), (3,4), ...
        for i in range(1, n - 1, 2):
            build_tdcnot(i, i + 1)

    def farhi(self, params):
        """Farhi-inspired ansatz with RXX and RZX gates. Parameters: 2n-2"""
        n = self.num_qubits
        idx = 0

        def RXX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        def RZX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        # RXX gates (hub-and-spoke from last qubit)
        for i in range(n - 1):
            RXX(params[idx], wires=[n - 1, i])
            idx += 1

        # RZX gates
        for i in range(n - 1):
            RZX(params[idx], wires=[n - 1, i])
            idx += 1

    def sim_circ_15(self, params):
        """Hardware-efficient circuit 15. Parameters: 2n"""
        n = self.num_qubits
        idx = 0

        # First RY layer
        for i in range(n):
            qml.RY(params[idx], wires=i)
            idx += 1

        # Ring CNOT (reverse)
        for i in reversed(range(n)):
            qml.CNOT(wires=[i, (i + 1) % n])

        # Second RY layer
        for i in range(n):
            qml.RY(params[idx], wires=i)
            idx += 1

        # Cross-layer CNOT
        for i in range(n):
            ctrl = (i + n - 1) % n
            tgt = (ctrl + 3) % n
            qml.CNOT(wires=[ctrl, tgt])

    def cross_mesh(self, params):
        """Full all-to-all CRZ connectivity. Parameters: 4n + n(n-1)"""
        n = self.num_qubits
        idx = 0

        # RX layer
        for i in range(n):
            qml.RX(params[idx], wires=i)
            idx += 1

        # RZ layer
        for i in range(n):
            qml.RZ(params[idx], wires=i)
            idx += 1

        # All-to-all CRZ
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if j != i:
                    qml.CRZ(params[idx], wires=[i, j])
                    idx += 1

        # Final RX layer
        for i in range(n):
            qml.RX(params[idx], wires=i)
            idx += 1

        # Final RZ layer
        for i in range(n):
            qml.RZ(params[idx], wires=i)
            idx += 1

    def apply(self, ansatz_type, params):
        """Apply the specified ansatz."""
        ansatz_map = {
            "cascade": self.cascade,
            "layered": self.layered,
            "alternate": self.alternate,
            "farhi": self.farhi,
            "sim_circ_15": self.sim_circ_15,
            "cross_mesh": self.cross_mesh,
        }
        ansatz_fn = ansatz_map.get(ansatz_type, self.cascade)
        ansatz_fn(params)


# =============================================================================
# 4. QUANTUM LAYER
# =============================================================================

class QuantumLayer(nn.Module):
    """Hybrid quantum-classical layer using PennyLane."""

    def __init__(self, num_qubits, ansatz_type="cascade", encoding="angle",
                 use_ibm=False, ibm_token=None, ibm_backend=None,
                 ibm_instance=None, shots=1024, seed=42):
        super().__init__()

        self.num_qubits = num_qubits
        self.ansatz_type = ansatz_type
        self.encoding = encoding
        self.shots = shots
        self.use_ibm = use_ibm
        self.use_batch_processing = True

        # Initialize ansatzes helper
        self.ansatzes = QuantumAnsatzes(num_qubits)
        num_params = self.ansatzes.get_num_params(ansatz_type)

        # Trainable quantum parameters
        self.params = nn.Parameter(torch.randn(num_params) * 0.1)

        # Haar unitary seeds for 4+ qubits
        self.haar_seed1 = seed if num_qubits >= 4 else None
        self.haar_seed2 = seed + 1 if num_qubits >= 4 else None

        # Initialize quantum device
        self._init_device(use_ibm, ibm_token, ibm_backend, ibm_instance, shots)

        # Create QNode
        self.circuit = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface="torch",
            diff_method=self.diff_method
        )

    def _init_device(self, use_ibm, ibm_token, ibm_backend, ibm_instance, shots):
        """Initialize PennyLane quantum device."""

        if use_ibm and IBM_AVAILABLE:
            print(f"Initializing IBM Quantum Backend: {ibm_backend}")
            try:
                # Connect to IBM Quantum
                if ibm_instance:
                    service = QiskitRuntimeService(instance=ibm_instance, token=ibm_token)
                else:
                    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)

                backend = service.backend(ibm_backend)
                print(f"Connected to IBM Quantum backend: {backend.name}")

                self.dev = qml.device(
                    "qiskit.remote",
                    wires=self.num_qubits,
                    shots=shots,
                    backend=backend
                )
                self.diff_method = "parameter-shift"
                self.use_batch_processing = False

            except Exception as e:
                print(f"Warning: IBM connection failed: {e}")
                print("Falling back to local simulator")
                self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=shots)
                self.diff_method = "finite-diff"
                self.use_batch_processing = False
        else:
            # Local simulator (fast, supports backprop)
            self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)
            self.diff_method = "backprop"
            self.use_batch_processing = True

    def _quantum_circuit(self, x):
        """Define the quantum circuit."""

        # Input encoding
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(
                x, wires=range(self.num_qubits), normalize=True, pad_with=0.0
            )
        else:  # angle encoding
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits), rotation="X")

        # Apply ansatz
        self.ansatzes.apply(self.ansatz_type, self.params)

        # Apply Haar unitaries for 4+ qubits (adds expressivity)
        if self.haar_seed1 is not None and self.haar_seed2 is not None:
            rs1 = np.random.RandomState(self.haar_seed1)
            rs2 = np.random.RandomState(self.haar_seed2)
            u1 = unitary_group.rvs(4, random_state=rs1)
            u2 = unitary_group.rvs(4, random_state=rs2)
            qml.QubitUnitary(u1, wires=[0, 1])
            qml.QubitUnitary(u2, wires=[2, 3])

        # Final Hadamard on last qubit
        if self.num_qubits > 0:
            qml.Hadamard(wires=self.num_qubits - 1)

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x):
        """Forward pass through quantum layer."""

        if self.use_batch_processing:
            # Simulator: process entire batch
            result = self.circuit(x)
            if isinstance(result, list):
                return torch.stack(result).T
            return result
        else:
            # Hardware: sequential processing
            batch_size = x.shape[0]
            outputs = []

            for idx, sample in enumerate(x):
                if idx % max(1, batch_size // 4) == 0:
                    print(f"\rProcessing sample {idx+1}/{batch_size}...", end="", flush=True)

                result = self.circuit(sample)

                if isinstance(result, list):
                    if all(isinstance(r, torch.Tensor) for r in result):
                        outputs.append(torch.stack(result))
                    else:
                        outputs.append(torch.tensor([float(r) for r in result],
                                                    dtype=torch.float32))
                else:
                    outputs.append(result if isinstance(result, torch.Tensor)
                                   else torch.tensor(float(result), dtype=torch.float32))

            if batch_size > 10:
                print(f"\rProcessed {batch_size}/{batch_size} samples.          ")

            return torch.stack(outputs)


# =============================================================================
# 5. HYBRID QPINN MODEL
# =============================================================================

class HybridQPINN(nn.Module):
    """Hybrid Quantum Physics-Informed Neural Network."""

    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args
        self.num_qubits = args.num_qubits
        self.hidden_dim = args.hidden_dim

        # Classical preprocessor: (3) -> (hidden) -> (num_qubits)
        self.preprocessor = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_qubits),
        ).to(device)

        # Quantum layer
        self.quantum_layer = QuantumLayer(
            num_qubits=args.num_qubits,
            ansatz_type=args.ansatz,
            encoding=args.encoding,
            use_ibm=args.use_ibm,
            ibm_token=args.ibm_token,
            ibm_backend=args.ibm_backend,
            ibm_instance=args.ibm_instance,
            shots=args.shots,
            seed=args.seed
        )

        # Classical postprocessor: (num_qubits) -> (hidden) -> (1)
        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        ).to(device)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=500
        )

        # Loss history
        self.loss_history = []

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the hybrid network."""
        # Preprocessing
        preprocessed = self.preprocessor(x)

        # Quantum processing
        quantum_out = self.quantum_layer(preprocessed)
        quantum_out = quantum_out.to(dtype=torch.float32, device=self.device)

        # Handle shape variations from quantum layer
        if quantum_out.shape[0] == self.num_qubits and quantum_out.dim() == 2:
            quantum_out = quantum_out.T
        quantum_out = quantum_out.view(-1, self.num_qubits)

        # Postprocessing
        output = self.postprocessor(quantum_out)
        return output


# =============================================================================
# 6. PDE OPERATOR
# =============================================================================

def diffusion_operator(model, t, x, y, D=0.01):
    """
    Compute PDE residual using automatic differentiation.

    PDE: ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)
    Residual: r = ∂u/∂t - D(∂²u/∂x² + ∂²u/∂y²)
    """
    # Ensure gradients are tracked
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    # Forward pass
    X = torch.cat([t, x, y], dim=1)
    u = model(X)

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True, retain_graph=True)[0]

    # PDE residual: ∂u/∂t - D(∂²u/∂x² + ∂²u/∂y²) = 0
    residual = u_t - D * (u_xx + u_yy)

    return u, residual


# =============================================================================
# 7. TRAINING FUNCTION
# =============================================================================

def train(model, args, ics_sampler, bc_samplers, res_sampler, output_dir):
    """Training loop with physics-informed loss."""

    print("\n" + "="*60)
    print("TRAINING HYBRID QUANTUM PINN")
    print("="*60)

    D = args.diffusion_coef
    batch_size = args.batch_size
    device = model.device

    training_start = time.time()
    epoch_times = []

    for epoch in range(args.epochs + 1):
        epoch_start = time.time()

        model.optimizer.zero_grad()

        # Sample training data
        X_ics, u_ics = ics_sampler.sample(batch_size // 3)
        X_res, _ = res_sampler.sample(batch_size)

        # Sample from all boundary conditions
        X_bc_list, u_bc_list = [], []
        for bc_sampler in bc_samplers:
            X_bc, u_bc = bc_sampler.sample(batch_size // 12)
            X_bc_list.append(X_bc)
            u_bc_list.append(u_bc)
        X_bc = torch.cat(X_bc_list, dim=0)
        u_bc = torch.cat(u_bc_list, dim=0)

        # Enable gradients for PDE residual computation
        X_ics.requires_grad_(True)
        X_bc.requires_grad_(True)
        X_res.requires_grad_(True)

        # Forward pass for initial and boundary conditions
        u_ics_pred = model(X_ics)
        u_bc_pred = model(X_bc)

        # PDE residual
        t_r, x_r, y_r = X_res[:, 0:1], X_res[:, 1:2], X_res[:, 2:3]
        _, residual = diffusion_operator(model, t_r, x_r, y_r, D)

        # Compute losses
        loss_ics = model.loss_fn(u_ics_pred, u_ics)
        loss_bc = model.loss_fn(u_bc_pred, u_bc)
        loss_res = model.loss_fn(residual, torch.zeros_like(residual))

        # Weighted composite loss
        loss = 2.0 * loss_res + 4.0 * loss_bc + 2.0 * loss_ics

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimization step
        model.optimizer.step()
        model.scheduler.step(loss)

        # Track loss
        model.loss_history.append(loss.item())

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Logging
        if epoch % args.print_every == 0 or epoch == 0:
            total_time = time.time() - training_start
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta = avg_epoch_time * (args.epochs - epoch)
            lr = model.optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:5d}/{args.epochs} [{100*epoch/args.epochs:5.1f}%] | "
                  f"Loss: {loss.item():.2e} | Res: {loss_res.item():.2e} | "
                  f"BC: {loss_bc.item():.2e} | IC: {loss_ics.item():.2e} | "
                  f"LR: {lr:.2e} | Time: {total_time:.1f}s | ETA: {eta:.1f}s")

            # Save checkpoint
            if epoch > 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "loss": loss.item(),
                    "loss_history": model.loss_history,
                }, os.path.join(output_dir, "checkpoint.pth"))

    print(f"\nTraining completed in {time.time() - training_start:.1f}s")

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    return model


# =============================================================================
# 8. EVALUATION & VISUALIZATION
# =============================================================================

def evaluate(model, args, dom_coords, output_dir):
    """Evaluate model and generate visualizations."""

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    device = model.device
    D = args.diffusion_coef

    # Create evaluation grid
    num_points = 20
    t_vals = torch.linspace(0, 1, num_points, device=device)
    x_vals = torch.linspace(0, 1, num_points, device=device)
    y_vals = torch.linspace(0, 1, num_points, device=device)

    # Evaluate at t=0.5
    t_eval = 0.5

    # Create meshgrid for x, y at fixed t
    X_mesh, Y_mesh = torch.meshgrid(x_vals, y_vals, indexing='ij')
    X_flat = X_mesh.flatten()
    Y_flat = Y_mesh.flatten()
    T_flat = torch.full_like(X_flat, t_eval)

    X_eval = torch.stack([T_flat, X_flat, Y_flat], dim=1).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_eval)

    # Analytical solution
    u_analytical = analytical_solution_torch(X_eval, D)

    # Move to CPU for numpy operations
    u_pred_np = u_pred.cpu().numpy().reshape(num_points, num_points)
    u_analytical_np = u_analytical.cpu().numpy().reshape(num_points, num_points)
    X_np = X_mesh.cpu().numpy()
    Y_np = Y_mesh.cpu().numpy()

    # Compute L2 error
    error = np.linalg.norm(u_analytical_np - u_pred_np) / np.linalg.norm(u_analytical_np)
    print(f"Relative L2 Error at t={t_eval}: {error*100:.4f}%")

    # Plot loss history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Plot error vs analytical
    plt.subplot(1, 2, 2)
    error_map = np.abs(u_analytical_np - u_pred_np)
    plt.imshow(error_map, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
    plt.colorbar(label='|Error|')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Absolute Error at t={t_eval}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "training_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Contour plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Predicted
    c1 = axes[0].contourf(X_np, Y_np, u_pred_np, levels=50, cmap='viridis')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Predicted u(t={t_eval}, x, y)")

    # Analytical
    c2 = axes[1].contourf(X_np, Y_np, u_analytical_np, levels=50, cmap='viridis')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"Analytical u(t={t_eval}, x, y)")

    # Error
    c3 = axes[2].contourf(X_np, Y_np, error_map, levels=50, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Absolute Error (L2: {error*100:.2f}%)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contour_plots.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "contour_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")

    return error


# =============================================================================
# 9. MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""

    # Parse arguments
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "="*60)
    print("HYBRID QUANTUM PINN TRAINER")
    print("="*60)
    print(f"Device: {device}")
    print(f"Use IBM Quantum: {args.use_ibm}")
    if args.use_ibm:
        print(f"IBM Backend: {args.ibm_backend}")
    print(f"Qubits: {args.num_qubits}")
    print(f"Ansatz: {args.ansatz}")
    print(f"Encoding: {args.encoding}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Diffusion coefficient: {args.diffusion_coef}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save configuration
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for key, value in vars(args).items():
            if key == "ibm_token" and value:
                f.write(f"{key}: ****\n")
            else:
                f.write(f"{key}: {value}\n")

    # Create data samplers
    ics_sampler, bc_samplers, res_sampler, dom_coords = create_samplers(
        device, D=args.diffusion_coef
    )

    # Create model
    model = HybridQPINN(args, device).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = model.quantum_layer.params.numel()
    print(f"Total parameters: {total_params}")
    print(f"Quantum parameters: {quantum_params}")

    # Train
    model = train(model, args, ics_sampler, bc_samplers, res_sampler, output_dir)

    # Evaluate
    error = evaluate(model, args, dom_coords, output_dir)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final L2 Error: {error*100:.4f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
