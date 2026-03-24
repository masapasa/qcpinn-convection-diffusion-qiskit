import os
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Local Simulation
from qiskit.primitives import StatevectorEstimator as LocalEstimator

# IBM Hardware
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as IBMEstimator

# =========================================================
# 1. Configuration & Physics Constants
# =========================================================

@dataclass
class Config:
    seed: int = 42
    n_qubits: int = 6
    n_q_layers: int = 2
    hidden_dim: int = 64
    
    # Physics (Silicon Melt)
    nu: float = 1.0e-6
    rho: float = 2330.0
    
    # Training
    pretrain_epochs: int = 1000  # Local (Fast)
    finetune_epochs: int = 50    # IBM Hardware (Careful usage)
    lr: float = 3e-4
    
    # IBM Specific
    ibm_token: str = "YOUR_IBM_TOKEN"
    ibm_backend: str = "ibm_brisbane" # Use a 127-qubit Heron/Eagle system
    checkpoint_file: str = "local_pretrain_model.pt"

# =========================================================
# 2. The Model Architecture
# =========================================================

def create_qnn(cfg: Config, estimator_instance):
    """
    Creates a QNN that can be attached to either Aer or IBM Hardware
    """
    qc = QuantumCircuit(cfg.n_qubits)
    inputs = ParameterVector("x", 4)
    weights = ParameterVector("w", cfg.n_qubits * cfg.n_q_layers * 2)

    # Encoding
    for i in range(cfg.n_qubits):
        qc.ry(inputs[i % 4], i)

    # Ansatz (Hardware Efficient)
    w_idx = 0
    for _ in range(cfg.n_q_layers):
        for i in range(cfg.n_qubits):
            qc.rx(weights[w_idx], i); qc.rz(weights[w_idx+1], i)
            w_idx += 2
        for i in range(cfg.n_qubits - 1):
            qc.cz(i, i + 1)

    obs = [SparsePauliOp.from_list([("I"*i + "Z" + "I"*(cfg.n_qubits-i-1), 1.0)]) 
           for i in range(cfg.n_qubits)]

    return EstimatorQNN(
        circuit=qc, observables=obs, input_params=inputs, 
        weight_params=weights, estimator=estimator_instance, input_gradients=True
    )

class QPINN(nn.Module):
    def __init__(self, cfg: Config, qnn_connector: TorchConnector):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh()
        )
        self.latent_to_q = nn.Linear(cfg.hidden_dim, 4)
        self.q_layer = qnn_connector
        self.head = nn.Linear(cfg.hidden_dim + cfg.n_qubits, 5) # [u, v, w, p, T]

    def forward(self, x):
        h = self.trunk(x)
        q_in = torch.pi * torch.tanh(self.latent_to_q(h))
        q_out = self.q_layer(q_in)
        return self.head(torch.cat([h, q_out], dim=-1))

# =========================================================
# 3. Phase 1: Local Pre-training (Aer)
# =========================================================

def run_local_pretraining(cfg: Config, x_data, y_data):
    print("--- STARTING LOCAL PRE-TRAINING (AER SIMULATOR) ---")
    
    aer_estimator = LocalEstimator()
    qnn = create_qnn(cfg, aer_estimator)
    model = QPINN(cfg, TorchConnector(qnn))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    for epoch in range(cfg.pretrain_epochs):
        optimizer.zero_grad()
        
        # Data Loss
        pred = model(x_data)
        loss_data = torch.mean((pred - y_data)**2)
        
        # Physics Loss (Simplified placeholder - use previous full PDE logic)
        loss_physics = torch.tensor(0.0) 
        
        total_loss = loss_data + loss_physics
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.6e}")

    # Save weights for IBM phase
    torch.save(model.state_dict(), cfg.checkpoint_file)
    print(f"Pre-training complete. Weights saved to {cfg.checkpoint_file}")
    return model

# =========================================================
# 4. Phase 2: IBM Fine-tuning (Real Hardware)
# =========================================================

def run_ibm_finetuning(cfg: Config, x_data, y_data):
    print(f"--- STARTING IBM FINE-TUNING ({cfg.ibm_backend}) ---")
    
    service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
    backend = service.backend(cfg.ibm_backend)

    # The Session context keeps your 150-minute window active and avoids requeuing
    with Session(service=service, backend=backend) as session:
        ibm_estimator = IBMEstimator(mode=session)
        ibm_estimator.options.default_shots = 1024
        ibm_estimator.options.optimization_level = 3
        ibm_estimator.options.resilience_level = 1
        
        # Reconstruct model with the IBM Estimator
        qnn_ibm = create_qnn(cfg, ibm_estimator)
        model = QPINN(cfg, TorchConnector(qnn_ibm))
        
        # Load the local weights
        model.load_state_dict(torch.load(cfg.checkpoint_file))
        print("Loaded local weights successfully.")

        # Fine-tune with a smaller learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.1)
        
        for epoch in range(cfg.finetune_epochs):
            optimizer.zero_grad()
            # On hardware, we use a smaller subset of data per epoch to save time
            idx = torch.randperm(x_data.size(0))[:32] 
            
            pred = model(x_data[idx])
            loss = torch.mean((pred - y_data[idx])**2)
            
            loss.backward()
            optimizer.step()
            print(f"IBM Epoch {epoch} | Loss: {loss.item():.6e}")
            
    print("Fine-tuning finished. Saving final model.")
    torch.save(model.state_dict(), "final_quantum_hardware_model.pt")

# =========================================================
# 5. Execution Logic
# =========================================================

if __name__ == "__main__":
    cfg = Config()
    
    # Generate dummy data for demonstration
    # x_train = torch.rand((100, 2))
    # y_train = torch.rand((100, 5))
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cz_melt_raw.txt")
    raw = np.loadtxt(data_path, comments="%")
    x_train = torch.tensor(raw[:, 0:2], dtype=torch.float32)
    y_train = torch.tensor(raw[:, 4:9], dtype=torch.float32)

    # STEP 1: Run locally (Cost: $0, Time: Minutes)
    run_local_pretraining(cfg, x_train, y_train)

    # STEP 2: Run on IBM (Cost: Uses your 150 min, Time: Depends on queue)
    # run_ibm_finetuning(cfg, x_train, y_train) # Uncomment when ready