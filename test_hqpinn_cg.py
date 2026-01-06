"""
Hybrid Quantum PINN for 2D Crystal Growth
----------------------------------------

Features:
- Classical PINN pretraining (local)
- Quantum-only fine tuning on IBM hardware
- Adaptive shot scheduling
- Layer-wise quantum early stopping
- Trains last quantum layers first
- Navier–Stokes + particle transport residual structure

Author: ChatGPT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from typing import List

# ============================================================
# CONFIGURATION
# ============================================================

N_QUBITS = 4
N_Q_LAYERS = 3

LOCAL_PRETRAIN_EPOCHS = 300
HARDWARE_EPOCHS_PER_LAYER = 40

INITIAL_SHOTS = 512
MAX_SHOTS = 4096
SHOT_INCREASE_FACTOR = 2

NOISE_EVAL_REPEATS = 5
NOISE_THRESHOLD_FACTOR = 2.0
EARLY_STOP_PATIENCE = 3

IBM_BACKEND = "ibmq_jakarta"   # change if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DEVICE FACTORY
# ============================================================

def make_device(backend="local", shots=None):
    if backend == "local":
        return qml.device("default.qubit", wires=N_QUBITS)
    elif backend == "ibm":
        return qml.device(
            "qiskit.ibmq",
            wires=N_QUBITS,
            backend=IBM_BACKEND,
            shots=shots,
            ibmq_api=True
        )
    else:
        raise ValueError("Unknown backend")

# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def make_qnode(dev):
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for i in range(N_QUBITS):
            qml.RX(inputs[i], wires=i)

        for l in range(N_Q_LAYERS):
            for q in range(N_QUBITS):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            for q in range(N_QUBITS - 1):
                qml.CNOT(wires=[q, q+1])

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return circuit

def make_quantum_layer(dev):
    qnode = make_qnode(dev)
    return qml.qnn.TorchLayer(
        qnode,
        weight_shapes={"weights": (N_Q_LAYERS, N_QUBITS, 3)}
    )

# ============================================================
# HYBRID MODEL
# ============================================================

class HybridCrystalPINN(nn.Module):
    def __init__(self, qlayer):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh()
        )

        self.pre_q = nn.Linear(16, N_QUBITS)
        self.q_layer = qlayer

        # Outputs: u,v (velocity), p (pressure), c (concentration)
        self.post_q = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        z = self.encoder(x)
        q_in = self.pre_q(z)
        q_out = torch.stack([self.q_layer(q_in[i]) for i in range(x.shape[0])])
        return self.post_q(q_out)

# ============================================================
# PDE RESIDUALS (Navier–Stokes + Transport)
# ============================================================

def pde_residual(model, x, nu=0.01, D=0.01):
    """
    Returns combined residual norm:
    - Incompressible Navier–Stokes
    - Particle transport equation
    """

    out = model(x)
    u, v, p, c = out[:,0], out[:,1], out[:,2], out[:,3]

    grads = torch.autograd.grad(
        out, x, torch.ones_like(out), create_graph=True
    )[0]

    u_x, u_y = grads[:,0], grads[:,1]
    v_x, v_y = grads[:,0], grads[:,1]
    c_x, c_y = grads[:,0], grads[:,1]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0]
    u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1]

    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0]
    v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1]

    c_xx = torch.autograd.grad(c_x, x, torch.ones_like(c_x), create_graph=True)[0][:,0]
    c_yy = torch.autograd.grad(c_y, x, torch.ones_like(c_y), create_graph=True)[0][:,1]

    # Navier–Stokes residuals
    res_u = u * u_x + v * u_y + p - nu * (u_xx + u_yy)
    res_v = u * v_x + v * v_y + p - nu * (v_xx + v_yy)
    div_free = u_x + v_y

    # Transport equation
    res_c = u * c_x + v * c_y - D * (c_xx + c_yy)

    return (
        res_u.pow(2).mean() +
        res_v.pow(2).mean() +
        div_free.pow(2).mean() +
        res_c.pow(2).mean()
    )

# ============================================================
# SAMPLING
# ============================================================

def sample_domain(n):
    x = torch.rand(n, 2, device=DEVICE, requires_grad=True)
    return x

# ============================================================
# STAGE 1: CLASSICAL ONLY
# ============================================================

def train_classical_only(model):
    for p in model.q_layer.parameters():
        p.requires_grad = False

    opt = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.pre_q.parameters()) +
        list(model.post_q.parameters()),
        lr=1e-3
    )

    for ep in range(LOCAL_PRETRAIN_EPOCHS):
        x = sample_domain(64)
        opt.zero_grad()
        loss = pde_residual(model, x)
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(f"[Stage 1] Epoch {ep} Loss {loss.item():.3e}")

# ============================================================
# SHOT NOISE ESTIMATION
# ============================================================

def estimate_loss_noise(model, x):
    losses = []
    with torch.no_grad():
        for _ in range(NOISE_EVAL_REPEATS):
            losses.append(pde_residual(model, x).item())
    return torch.tensor(losses).std().item()

# ============================================================
# STAGE 2: QUANTUM LAYER-WISE TRAINING
# ============================================================

def train_quantum_layerwise(model):

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.pre_q.parameters():
        p.requires_grad = False
    for p in model.post_q.parameters():
        p.requires_grad = False

    current_shots = INITIAL_SHOTS

    for layer_idx in reversed(range(N_Q_LAYERS)):
        print(f"\n=== Training quantum layer {layer_idx} ===")

        patience = 0
        best_loss = float("inf")

        while True:
            dev = make_device("ibm", shots=current_shots)
            qlayer = make_quantum_layer(dev)
            qlayer.load_state_dict(model.q_layer.state_dict())
            model.q_layer = qlayer.to(DEVICE)

            opt = optim.Adam(model.q_layer.parameters(), lr=1e-3)

            for ep in range(HARDWARE_EPOCHS_PER_LAYER):
                x = sample_domain(16)
                opt.zero_grad()
                loss = pde_residual(model, x)
                loss.backward()

                # Mask gradients except selected layer
                with torch.no_grad():
                    for name, p in model.q_layer.named_parameters():
                        if name == "weights":
                            p.grad[:layer_idx] = 0
                            p.grad[layer_idx+1:] = 0

                opt.step()

                noise = estimate_loss_noise(model, x)

                print(
                    f"[Layer {layer_idx}] "
                    f"epoch {ep} "
                    f"loss={loss.item():.3e} "
                    f"noise≈{noise:.2e} "
                    f"shots={current_shots}"
                )

                if best_loss - loss.item() > NOISE_THRESHOLD_FACTOR * noise:
                    best_loss = loss.item()
                    patience = 0
                else:
                    patience += 1

                if patience >= EARLY_STOP_PATIENCE:
                    break

            if current_shots < MAX_SHOTS:
                current_shots *= SHOT_INCREASE_FACTOR
                print(f"↑ Increasing shots to {current_shots}")
            else:
                print(f"✓ Layer {layer_idx} converged")
                break

# ============================================================
# MAIN
# ============================================================

def main():
    dev_local = make_device("local")
    qlayer = make_quantum_layer(dev_local)
    model = HybridCrystalPINN(qlayer).to(DEVICE)

    print("=== Stage 1: Classical Pretraining ===")
    train_classical_only(model)

    torch.save(model.state_dict(), "stage1_classical.pt")

    print("\n=== Stage 2: Quantum Layer-wise Training ===")
    train_quantum_layerwise(model)

    torch.save(model.state_dict(), "final_hybrid_crystal_growth.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()

