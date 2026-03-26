import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from qiskit_ibm_runtime import QiskitRuntimeService

# ----------------------------------------------------------
# 1. Configuration & Mode Switch
# ----------------------------------------------------------
# Set to 'AER' for first 2000 epochs, then 'IBM' for last 100
TRAINING_PHASE = "AER"  # Options: "AER", "IBM"
IBM_TOKEN = "YOUR_IBM_TOKEN_HERE"

n_qubits = 16
n_layers = 2
shots = 4096

def get_quantum_device(phase):
    if phase == "AER":
        print("Initializing Aer Simulator...")
        return qml.device("qiskit.aer", wires=n_qubits, shots=shots), "parameter-shift"
    else:
        print("Connecting to Real IBM Hardware...")
        service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
        backend = service.least_busy(min_num_qubits=n_qubits, simulator=False)
        print(f"Connected to: {backend.name}")
        return qml.device("qiskit.remote", wires=n_qubits, backend=backend, shots=shots), "parameter-shift"

dev_quantum, diff_method = get_quantum_device(TRAINING_PHASE)

# ----------------------------------------------------------
# 2. Physical Constants
# ----------------------------------------------------------
Re, Pr, Gr = 15.0, 28.463, 8000.0

# ----------------------------------------------------------
# 3. Data Loading (Czochralski Melt Flow)
# ----------------------------------------------------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(data[["r","z"]].values, dtype=torch.float32).to(device)
Y_train = torch.tensor(data[["u","v","w","p","T"]].values, dtype=torch.float32).to(device)

# ----------------------------------------------------------
# 4. Hybrid Architecture
# ----------------------------------------------------------
@qml.qnode(dev_quantum, interface="torch", diff_method=diff_method)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, mapping_size=32, scale=5):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, mapping_size) * scale)
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Hybrid_QPINN(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.fourier = FourierFeatures(mapping_size=32)
        self.enc = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, n_qubits))
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.post_dense = nn.Linear(n_qubits, 64)
        self.gate_m = nn.Linear(64, 64)
        self.gate_n = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        r_coord = x[:, 0:1]
        q_in = torch.sigmoid(self.enc(self.fourier(x))) * np.pi 
        q_out = self.q_layer(q_in)
        
        h = self.act(self.post_dense(q_out))
        m, n = torch.sigmoid(self.gate_m(h)), torch.sigmoid(self.gate_n(h))
        h = h * m + (1 - h) * n
        
        out = self.output_layer(h)
        # Axisymmetric Hard Constraints
        return torch.cat([out[:,0:1]*r_coord, out[:,1:2], out[:,2:3]*r_coord, out[:,3:4], out[:,4:5]], dim=1)

model = Hybrid_QPINN().to(device)

# ----------------------------------------------------------
# 5. Physics Engine
# ----------------------------------------------------------
def gradients(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def physics_loss(x, model_obj):
    x = x.clone().detach().requires_grad_(True)
    pred = model_obj(x)
    vr, vz, vtheta, P, T = pred[:,0:1], pred[:,1:2], pred[:,2:3], pred[:,3:4], pred[:,4:5]
    r_denom = torch.clamp(x[:, 0:1], min=1e-4)

    g_vr = gradients(vr, x); vr_r, vr_z = g_vr[:,0:1], g_vr[:,1:2]
    g_vz = gradients(vz, x); vz_r, vz_z = g_vz[:,0:1], g_vz[:,1:2]
    g_vt = gradients(vtheta, x); vt_r, vt_z = g_vt[:,0:1], g_vt[:,1:2]
    g_P = gradients(P, x); P_r, P_z = g_P[:,0:1], g_P[:,1:2]
    g_T = gradients(T, x); T_r, T_z = g_T[:,0:1], g_T[:,1:2]

    vr_rr = gradients(vr_r, x)[:,0:1]; vr_zz = gradients(vr_z, x)[:,1:2]
    vz_rr = gradients(vz_r, x)[:,0:1]; vz_zz = gradients(vz_z, x)[:,1:2]
    vt_rr = gradients(vt_r, x)[:,0:1]; vt_zz = gradients(vt_z, x)[:,1:2]
    T_rr = gradients(T_r, x)[:,0:1]; T_zz = gradients(T_z, x)[:,1:2]

    continuity = vr_r + vr/r_denom + vz_z
    mom_r = (vr*vr_r + vz*vr_z - (vtheta**2)/r_denom + P_r - (1/Re)*(vr_rr + vr_r/r_denom - vr/(r_denom**2) + vr_zz))
    swirl = (vr*vt_r + vz*vt_z + (vr*vtheta)/r_denom - (1/Re)*(vt_rr + vt_r/r_denom - vtheta/(r_denom**2) + vt_zz))
    mom_z = (vr*vz_r + vz*vz_z + P_z - (1/Re)*(vz_rr + vz_r/r_denom + vz_zz) - (Gr/(Re**2))*T)
    energy = (vr*T_r + vz*T_z - (1/(Pr*Re))*(T_rr + T_r/r_denom + T_zz))

    return (continuity.pow(2).mean() + mom_r.pow(2).mean() + 
            mom_z.pow(2).mean() + swirl.pow(2).mean() + 10.0 * energy.pow(2).mean())

# ----------------------------------------------------------
# 6. Hybrid Training Execution
# ----------------------------------------------------------
if TRAINING_PHASE == "AER":
    epochs = 2000
    batch_size = 128 # Simulator can handle decent batches
    lr = 1e-3
elif TRAINING_PHASE == "IBM":
    # LOAD PRETRAINED AER WEIGHTS
    model.load_state_dict(torch.load("hybrid_qpinn_aer.pt"))
    epochs = 100
    batch_size = 4 # DRAMATICALLY reduced for real hardware feasibility
    lr = 1e-4      # Lower learning rate for fine-tuning

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"Starting Phase: {TRAINING_PHASE}")
for epoch in range(epochs):
    idx = torch.randperm(X_train.shape[0])[:batch_size]
    xb, yb = X_train[idx], Y_train[idx]
    
    optimizer.zero_grad()
    
    # Data Loss
    l_d = torch.mean((model(xb) - yb)**2)
    
    # Physics Loss (Note: Very slow on IBM hardware)
    l_p = physics_loss(xb, model)
    
    total_loss = l_d + 0.1 * l_p # Simplified weighting for stability
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss.item():.4e} | Data: {l_d.item():.4e}")

# Save results
model_name = "hybrid_qpinn_aer.pt" if TRAINING_PHASE == "AER" else "hybrid_qpinn_final.pt"
torch.save(model.state_dict(), model_name)