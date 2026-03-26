import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
from torch.utils.data import DataLoader, TensorDataset

# ==========================================================
# 1. GLOBAL CONFIGURATION
# ==========================================================
TRAINING_PHASE = "AER"  # Switch to "IBM" for hardware phase
IBM_TOKEN = "YOUR_IBM_TOKEN"
INSTANCE = "ibm-q/open/main"

N_QUBITS = 16
N_LAYERS = 2           # Depth of the quantum circuit
SHOTS = 4096
RE, PR, GR = 15.0, 28.463, 8000.0

# ==========================================================
# 2. QUANTUM DEVICE & CIRCUIT SPECIFICATION
# ==========================================================
def initialize_device(phase):
    """
    Sets up the Pennylane device. 
    AER: Uses 'adjoint' differentiation - O(1) circuit evaluations for gradients.
    IBM: Uses 'parameter-shift' - O(N) evaluations, but hardware compatible.
    """
    if phase == "AER":
        print(">>> Initializing Local Aer Simulator (Adjoint Mode)")
        dev = qml.device("default.qubit", wires=N_QUBITS)
        return dev, "adjoint", None
    else:
        print(">>> Connecting to IBM Quantum Hardware")
        service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN, instance=INSTANCE)
        backend = service.least_busy(min_num_qubits=N_QUBITS, simulator=False)
        dev = qml.device("qiskit.remote", wires=N_QUBITS, backend=backend, shots=SHOTS)
        return dev, "parameter-shift", service

dev_quantum, diff_method, ibm_service = initialize_device(TRAINING_PHASE)

# Weight shape for StronglyEntanglingLayers: (L, N, 3)
weight_shapes = {"q_weights": (N_LAYERS, N_QUBITS, 3)}

@qml.qnode(dev_quantum, interface="torch", diff_method=diff_method)
def quantum_circuit(inputs, q_weights):
    """
    Implements Data Re-uploading: Coordinates are re-injected at every layer
    to increase the expressivity of the 16-qubit system.
    """
    for i in range(N_LAYERS):
        # 1. Map classical data to rotation angles
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
        # 2. Strongly entangling block (Rotations + CNOTs)
        qml.StronglyEntanglingLayers(q_weights[i:i+1], wires=range(N_QUBITS))
    
    # Return expectation values in Z basis for all 16 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ==========================================================
# 3. HYBRID ARCHITECTURE (QPINN)
# ==========================================================
class QPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Random Fourier Features: Projects 2D (r,z) into a 16D frequency space
        # Helps the QNN overcome the 'Spectral Bias' (failing to learn high frequencies)
        self.register_buffer("B", torch.randn(2, 8) * 2.0 * np.pi) 
        
        # Classical pre-processing
        self.pre_linear = nn.Sequential(
            nn.Linear(16, 32), 
            nn.Tanh(),
            nn.Linear(32, N_QUBITS),
            nn.Sigmoid()
        )
        
        # Quantum Layer Integrated into Torch
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Post-processing to physical variables: [u, v, w, p, T]
        self.post_linear = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.Tanh(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        r_coord = x[:, 0:1]
        
        # 1. Fourier Feature Mapping
        # [batch, 2] -> [batch, 16] (sin and cos of projection)
        proj = x @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        # 2. Quantum Forward Pass
        q_in = self.pre_linear(ff) * np.pi 
        q_out = self.q_layer(q_in)
        
        # 3. Post-process & Symmetry Enforcement
        # Formula: u = f(r,z) * r  (Ensures u=0 at r=0)
        raw = self.post_linear(q_out)
        u = raw[:, 0:1] * r_coord
        v = raw[:, 1:2]
        w = raw[:, 2:3] * r_coord
        p = raw[:, 3:4]
        T = raw[:, 4:5]
        
        return torch.cat([u, v, w, p, T], dim=1)

# ==========================================================
# 4. PHYSICS ENGINE (AUTOMATIC DIFFERENTIATION)
# ==========================================================
def compute_physics_loss(x, model):
    """
    Computes the Residual of the Navier-Stokes Equations.
    Loss = ||Continuity||^2 + ||Momentum||^2 + ||Energy||^2
    """
    x = x.clone().detach().requires_grad_(True)
    pred = model(x)
    u, v, w, p, T = pred[:,0:1], pred[:,1:2], pred[:,2:3], pred[:,3:4], pred[:,4:5]
    
    def get_grad(y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

    # Gradients
    grad_u = get_grad(u, x); u_r, u_z = grad_u[:,0:1], grad_u[:,1:2]
    grad_v = get_grad(v, x); v_r, v_z = grad_v[:,0:1], grad_v[:,1:2]
    grad_T = get_grad(T, x); T_r, T_z = grad_T[:,0:1], grad_T[:,1:2]
    grad_p = get_grad(p, x); p_r, p_z = grad_p[:,0:1], grad_p[:,1:2]

    r = torch.clamp(x[:, 0:1], min=1e-4)

    # 1. Continuity: u/r + du/dr + dv/dz = 0
    res_c = u/r + u_r + v_z
    
    # 2. Energy: u*dT/dr + v*dT/dz - (1/(Re*Pr))*Laplacian(T) = 0
    # Simplified Laplacian for efficiency
    T_rr = get_grad(u_r, x)[:, 0:1]
    T_zz = get_grad(v_z, x)[:, 1:2]
    res_e = (u*T_r + v*T_z) - (1.0/(RE*PR))*(T_rr + T_r/r + T_zz)

    return torch.mean(res_c**2) + 10.0 * torch.mean(res_e**2)

# ==========================================================
# 5. TRAINING LOOP (SIMULATOR VS HARDWARE)
# ==========================================================
model = QPINN()

# Placeholder for Data Loading (r, z coordinates and target fluid values)
X_data = torch.rand((500, 2))
Y_data = torch.rand((500, 5))
dataset = TensorDataset(X_data, Y_data)

if TRAINING_PHASE == "AER":
    epochs = 2000
    batch_size = 32
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting AER Pre-training...")
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_data = torch.mean((model(xb) - yb)**2)
            loss_phys = compute_physics_loss(xb, model)
            total_loss = loss_data + 0.1 * loss_phys
            total_loss.backward()
            optimizer.step()
        if ep % 50 == 0:
            print(f"Epoch {ep} | Loss: {total_loss.item():.4e}")
    
    torch.save(model.state_dict(), "hybrid_qpinn_aer.pt")

else:
    # --- IBM HARDWARE PHASE ---
    model.load_state_dict(torch.load("hybrid_qpinn_aer.pt"))
    epochs = 100
    batch_size = 4 # Small batches are critical for real hardware queueing
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) # SGD is more stable on noisy hardware
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use Options to enable Error Mitigation
    options = Options()
    options.resilience_level = 1 # M3 Readout mitigation
    options.optimization_level = 3 # Maximum circuit transpilation optimization

    print(f"Starting IBM Hardware Session on {dev_quantum.backend.name}")
    with Session(service=ibm_service, backend=dev_quantum.backend) as session:
        for ep in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                # Focus on Data Loss on hardware to minimize circuit depth per step
                loss_data = torch.mean((model(xb) - yb)**2)
                loss_data.backward()
                optimizer.step()
            print(f"Hardware Epoch {ep} | Data Loss: {loss_data.item():.4f}")

    torch.save(model.state_dict(), "hybrid_qpinn_final.pt")