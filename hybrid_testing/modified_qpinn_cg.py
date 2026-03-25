import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

# ----------------------------------------------------------
# Device & Quantum Setup
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaling to 16 qubits
n_qubits = 16  
n_layers = 2  # Reduced layers slightly for 16 qubits to mitigate Barren Plateaus
# Use 'lightning.qubit' for high-performance simulation if available, else 'default.qubit'
try:
    dev_quantum = qml.device("lightning.qubit", wires=n_qubits)
except:
    dev_quantum = qml.device("default.qubit", wires=n_qubits)

# ----------------------------------------------------------
# Physical constants (Dimensionless)
# ----------------------------------------------------------
Re = 15.0     
Pr = 28.463   
Gr = 8000.0    

# ----------------------------------------------------------
# Data Loading & Scaling 
# ----------------------------------------------------------
# [Assuming cz_melt_raw.txt exists]
try:
    data = pd.read_csv("cz_melt_raw.txt", comment="%", sep='\s+', header=None)
    data = data.iloc[:, [0, 1, 4, 5, 6, 7, 8]]
    data.columns = ["r","z","u","v","w","p","T"]

    L_inf = data["r"].max()
    v_inf = max(data["u"].abs().max(), data["v"].abs().max(), data["w"].abs().max())
    P_inf = data["p"].abs().max()
    T_min, T_max = data["T"].min(), data["T"].max()

    data["r"] /= L_inf
    data["z"] /= L_inf
    data["u"] /= v_inf
    data["v"] /= v_inf
    data["w"] /= v_inf
    data["p"] /= P_inf
    data["T"] = (data["T"] - T_min) / (T_max - T_min)

    X_train = torch.tensor(data[["r","z"]].values, dtype=torch.float32).to(device)
    Y_train = torch.tensor(data[["u","v","w","p","T"]].values, dtype=torch.float32).to(device)
except FileNotFoundError:
    print("Warning: Data file not found. Using dummy data for structural test.")
    X_train = torch.rand(1000, 2).to(device)
    Y_train = torch.rand(1000, 5).to(device)

# ----------------------------------------------------------
# Variational Quantum Circuit (VQC)
# ----------------------------------------------------------
@qml.qnode(dev_quantum, interface="torch", diff_method="adjoint") # Adjoint is faster for large qubit counts
def quantum_circuit(inputs, weights):
    # Angle Embedding maps 16 features to 16 qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # Strongly Entangling Layers for high expressivity
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Return expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ----------------------------------------------------------
# Hybrid QPINN Architecture (16 Qubits)
# ----------------------------------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=32, scale=10):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Hybrid_QPINN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        
        # 1. Classical Feature Extraction
        self.fourier = FourierFeatures(in_dim=2, mapping_size=32, scale=5)
        # Expanded encoder to map 64 fourier features to 16 qubits
        self.enc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_qubits)
        )
        
        # 2. Quantum Layer (16 Qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # 3. Post-Quantum & SI-Gating
        # Measurements from 16 qubits provide a 16-dimensional feature vector
        self.post_dense = nn.Linear(n_qubits, 64)
        self.gate_m = nn.Linear(64, 64)
        self.gate_n = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        r_coord = x[:, 0:1]
        
        # Mapping to Quantum Space
        x_f = self.fourier(x)
        # Ensuring inputs are in [0, pi] to avoid periodic aliasing in AngleEmbedding
        q_in = torch.sigmoid(self.enc(x_f)) * np.pi 
        
        # Quantum forward pass
        q_out = self.q_layer(q_in)
        
        # SI-Gating Logic
        h = self.act(self.post_dense(q_out))
        m = torch.sigmoid(self.gate_m(h)) # Using sigmoid for gating stability
        n = torch.sigmoid(self.gate_n(h))
        h = h * m + (1 - h) * n
        
        out = self.output_layer(h)
        
        # Axisymmetric Hard Constraints (Ensures zero velocity/flux at r=0)
        vr = out[:, 0:1] * r_coord
        vz = out[:, 1:2]
        vtheta = out[:, 2:3] * r_coord
        p = out[:, 3:4]
        T = out[:, 4:5]
        
        return torch.cat([vr, vz, vtheta, p, T], dim=1)

model = Hybrid_QPINN().to(device)

# ----------------------------------------------------------
# Physics & Loss Logic
# ----------------------------------------------------------
class CoupledAdaptiveWeighting(nn.Module):
    def __init__(self, target_ratio=100.0): # Slightly lower ratio for larger circuits
        super().__init__()
        self.log_eps_data = nn.Parameter(torch.zeros(1))
        self.ratio_multiplier = np.sqrt(target_ratio)
        
    def forward(self, l_data, l_phys):
        eps_data = torch.exp(self.log_eps_data)
        eps_phys = eps_data * self.ratio_multiplier
        loss = (0.5 / (eps_data**2)) * l_data + \
               (0.5 / (eps_phys**2)) * l_phys + \
               torch.log(eps_data * eps_phys)
        return loss

loss_balancer = CoupledAdaptiveWeighting().to(device)

def gradients(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def physics_loss(x):
    # Ensure graph is tracked
    x = x.clone().detach().requires_grad_(True)
    pred = model(x)
    vr, vz, vtheta, P, T = pred[:,0:1], pred[:,1:2], pred[:,2:3], pred[:,3:4], pred[:,4:5]
    
    r_denom = torch.clamp(x[:, 0:1], min=1e-4)

    # Gradients
    g_vr = gradients(vr, x); vr_r, vr_z = g_vr[:,0:1], g_vr[:,1:2]
    g_vz = gradients(vz, x); vz_r, vz_z = g_vz[:,0:1], g_vz[:,1:2]
    g_vt = gradients(vtheta, x); vt_r, vt_z = g_vt[:,0:1], g_vt[:,1:2]
    g_P = gradients(P, x); P_r, P_z = g_P[:,0:1], g_P[:,1:2]
    g_T = gradients(T, x); T_r, T_z = g_T[:,0:1], g_T[:,1:2]

    # Second Gradients
    vr_rr = gradients(vr_r, x)[:,0:1]; vr_zz = gradients(vr_z, x)[:,1:2]
    vz_rr = gradients(vz_r, x)[:,0:1]; vz_zz = gradients(vz_z, x)[:,1:2]
    vt_rr = gradients(vt_r, x)[:,0:1]; vt_zz = gradients(vt_z, x)[:,1:2]
    T_rr = gradients(T_r, x)[:,0:1]; T_zz = gradients(T_z, x)[:,1:2]

    # Navier-Stokes + Energy
    continuity = vr_r + vr/r_denom + vz_z
    mom_r = (vr*vr_r + vz*vr_z - (vtheta**2)/r_denom + P_r - (1/Re)*(vr_rr + vr_r/r_denom - vr/(r_denom**2) + vr_zz))
    swirl = (vr*vt_r + vz*vt_z + (vr*vtheta)/r_denom - (1/Re)*(vt_rr + vt_r/r_denom - vtheta/(r_denom**2) + vt_zz))
    mom_z = (vr*vz_r + vz*vz_z + P_z - (1/Re)*(vz_rr + vz_r/r_denom + vz_zz) - (Gr/(Re**2))*T)
    energy = (vr*T_r + vz*T_z - (1/(Pr*Re))*(T_rr + T_r/r_denom + T_zz))

    return (continuity.pow(2).mean() + mom_r.pow(2).mean() + 
            mom_z.pow(2).mean() + swirl.pow(2).mean() + 10.0 * energy.pow(2).mean())

# ----------------------------------------------------------
# Training Loop
# ----------------------------------------------------------
optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 8e-4}, # Slightly lower LR for 16-qubit stability
    {'params': loss_balancer.parameters(), 'lr': 1e-4}
])

# Batch size adjusted for 16 qubits. 
# If simulation is too slow, reduce to 128.
batch_size = 256 

print(f"Starting Hybrid QPINN Training with {n_qubits} Qubits...")

for epoch in range(5001):
    # Stochastic sampling
    idx = torch.randperm(X_train.shape[0])[:batch_size]
    xb, yb = X_train[idx], Y_train[idx]
    
    optimizer.zero_grad()
    
    # Forward & Data Loss
    pred_b = model(xb)
    l_d = torch.mean((pred_b - yb)**2)
    
    # Physics Loss
    l_p = physics_loss(xb)
    
    # Adaptive Balanced Loss
    total_loss = loss_balancer(l_d, l_p)
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:04d} | Loss: {total_loss.item():.4e} | Data: {l_d.item():.4e} | Phys: {l_p.item():.4e}")

# Save the 16-qubit model
torch.save(model.state_dict(), "hybrid_qpinn_16q.pt")