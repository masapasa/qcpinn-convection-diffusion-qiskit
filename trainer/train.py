import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

# ==========================================
# 1. CONFIGURATION & SEEDING
# ==========================================

class Config:
    # Problem Parameters
    D = 0.01          # Diffusion coefficient
    VX = 1.0          # Velocity x
    VY = 1.0          # Velocity y
    
    # Model Architecture
    N_QUBITS = 4
    N_LAYERS = 2
    CLASSICAL_HIDDEN = 50
    
    # Training
    EPOCHS = 20    # Adjust based on convergence needs
    BATCH_SIZE = 64
    LR = 0.005
    SEED = 42
    
    # Quantum Device
    # Options: "default.qubit" (Local fast), "ibm_brisbane", "ibm_kyoto", etc.
    # Set BACKEND to a specific IBM quantum system name to use hardware.
    # If BACKEND is set to "default.qubit", it runs locally.
    BACKEND = "default.qubit" 
    
    # IBM Quantum Credentials
    # Replace with your actual API Token
    IBM_TOKEN = "rTMZyypbjUUZ9Q_jjydgC-HAkVb2OJd42YAmIRtrDAvl"
    IBM_INSTANCE = None # e.g., "ibm-q/open/main"

    SHOTS = None      # Set to integer (e.g., 1024) for hardware/sampling simulation

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ==========================================
# 2. PHYSICS DEFINITION (EXACT SOLUTION & PDE)
# ==========================================

def exact_u(t, x, y):
    """
    Exact solution for validation: A Gaussian pulse decaying over time.
    u(t,x,y) = exp(-100((x-0.5)^2 + (y-0.5)^2)) * exp(-t)
    """
    return torch.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2)) * torch.exp(-t)

def get_pde_residual(model, t, x, y):
    """
    Computes the residual of the Convection-Diffusion Equation:
    Residual = u_t + v_x*u_x + v_y*u_y - D*(u_xx + u_yy)
    """
    # Enable gradient tracking for inputs
    t.requires_grad_(True)
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Forward pass
    inputs = torch.cat([t, x, y], dim=1)
    u = model(inputs)
    
    # First derivatives
    grads = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    u_y = grads[:, 2:3]
    
    # Second derivatives (w.r.t x and y)
    grad_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grad_x
    
    grad_y = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_yy = grad_y
    
    # PDE Residual
    f = u_t + Config.VX * u_x + Config.VY * u_y - Config.D * (u_xx + u_yy)
    
    return f, u

# ==========================================
# 3. DATA SAMPLER
# ==========================================

class PDESampler:
    def __init__(self, device):
        self.device = device
        
    def sample_domain(self, n):
        """Random points inside the domain: t \\in [0,1], x,y \\in [0,1]"""
        t = torch.rand(n, 1, device=self.device)
        x = torch.rand(n, 1, device=self.device)
        y = torch.rand(n, 1, device=self.device)
        return t, x, y

    def sample_initial(self, n):
        """Points at t=0"""
        t = torch.zeros(n, 1, device=self.device)
        x = torch.rand(n, 1, device=self.device)
        y = torch.rand(n, 1, device=self.device)
        # Exact value at t=0
        u_exact = exact_u(t, x, y)
        return t, x, y, u_exact

    def sample_boundary(self, n):
        """Points at spatial boundaries (Dirichlet)"""
        t = torch.rand(n, 1, device=self.device)
        
        # Randomly select which boundary (left, right, top, bottom)
        side = torch.randint(0, 4, (n, 1), device=self.device)
        
        x = torch.rand(n, 1, device=self.device)
        y = torch.rand(n, 1, device=self.device)
        
        # Enforce boundary coordinates
        x = torch.where(side == 0, torch.zeros_like(x), x) # Left
        x = torch.where(side == 1, torch.ones_like(x), x)  # Right
        y = torch.where(side == 2, torch.zeros_like(y), y) # Bottom
        y = torch.where(side == 3, torch.ones_like(y), y)  # Top
        
        # Exact value at boundaries
        u_exact = exact_u(t, x, y)
        return t, x, y, u_exact

# ==========================================
# 4. HYBRID NEURAL NETWORK MODEL
# ==========================================

class HybridPINN(nn.Module):
    def __init__(self, device_atom):
        super().__init__()
        self.n_qubits = Config.N_QUBITS
        self.n_layers = Config.N_LAYERS
        
        # 1. Classical Pre-processing (Encoder)
        # Inputs: (t, x, y) -> 3 features
        self.encoder = nn.Sequential(
            nn.Linear(3, Config.CLASSICAL_HIDDEN),
            nn.Tanh(),
            nn.Linear(Config.CLASSICAL_HIDDEN, self.n_qubits),
            nn.Tanh() # Scale to [-1, 1] for rotation gates
        )
        
        # 2. Quantum Layer
        self.q_layer = self._build_quantum_layer()
        
        # 3. Classical Post-processing (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_qubits, Config.CLASSICAL_HIDDEN),
            nn.Tanh(),
            nn.Linear(Config.CLASSICAL_HIDDEN, 1) # Output u
        )

    def _build_quantum_layer(self):
        """Builds the PennyLane TorchLayer"""
        if Config.BACKEND == "default.qubit":
            dev = qml.device(Config.BACKEND, wires=self.n_qubits, shots=Config.SHOTS)
            diff_method = "backprop"
        else:
            print(f"--> Initializing IBM Quantum Backend: {Config.BACKEND}")
            # Connect to IBM Quantum
            try:
                if Config.IBM_INSTANCE:
                    service = QiskitRuntimeService(instance=Config.IBM_INSTANCE, token=Config.IBM_TOKEN)
                else:
                    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=Config.IBM_TOKEN)
                
                backend = service.backend(Config.BACKEND)
                print(f"--> Using IBM Quantum backend: {backend.name}")
                
                # Enforce shots for hardware if not set
                shots = Config.SHOTS if Config.SHOTS is not None else 1024
                
                dev = qml.device(
                    "qiskit.remote", 
                    wires=self.n_qubits, 
                    shots=shots,
                    backend=backend
                )
                diff_method = "parameter-shift"
                
            except Exception as e:
                print(f"--> Error connecting to IBM Quantum: {e}")
                print("--> Falling back to 'default.qubit' local simulator.")
                dev = qml.device("default.qubit", wires=self.n_qubits, shots=Config.SHOTS)
                diff_method = "backprop"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            # Encoding: Angle Embedding (inputs are scaled by Tanh already)
            # Use pi scaling to cover full rotation
            for i in range(self.n_qubits):
                qml.RX(inputs[:, i] * np.pi, wires=i)
            
            # Variational Layers (Strongly Entangling inspired)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                    
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Weight shape: (Layers, Qubits, 3 parameters per Rot gate)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x shape: [Batch, 3]
        
        # Classical Encode
        features = self.encoder(x) # -> [Batch, N_Qubits]
        
        # Quantum Process
        # TorchLayer handles batch dimension automatically if configured correctly
        q_out = self.q_layer(features) # -> [Batch, N_Qubits]
        
        # Classical Decode
        u = self.decoder(q_out) # -> [Batch, 1]
        return u

# ==========================================
# 5. TRAINING ENGINE
# ==========================================

def train_model():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using Device: {device}")
    
    set_seed(Config.SEED)
    
    # Initialize Model & Sampler
    model = HybridPINN(device).to(device)
    sampler = PDESampler(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=200)
    
    loss_history = []
    
    print(f"--> Starting Training for {Config.EPOCHS} epochs...")
    print(f"--> Quantum Architecture: {Config.N_QUBITS} Qubits, {Config.N_LAYERS} Layers")
    
    start_time = time.time()
    
    for epoch in range(Config.EPOCHS + 1):
        optimizer.zero_grad()
        
        # 1. Residual Loss (Interior)
        t_res, x_res, y_res = sampler.sample_domain(Config.BATCH_SIZE)
        f_pred, _ = get_pde_residual(model, t_res, x_res, y_res)
        loss_pde = torch.mean(f_pred ** 2)
        
        # 2. Initial Condition Loss (t=0)
        t_ic, x_ic, y_ic, u_exact_ic = sampler.sample_initial(Config.BATCH_SIZE // 2)
        u_pred_ic = model(torch.cat([t_ic, x_ic, y_ic], dim=1))
        loss_ic = torch.mean((u_pred_ic - u_exact_ic) ** 2)
        
        # 3. Boundary Condition Loss (Spatial boundaries)
        t_bc, x_bc, y_bc, u_exact_bc = sampler.sample_boundary(Config.BATCH_SIZE // 2)
        u_pred_bc = model(torch.cat([t_bc, x_bc, y_bc], dim=1))
        loss_bc = torch.mean((u_pred_bc - u_exact_bc) ** 2)
        
        # Total Loss (Weighted)
        loss = loss_pde + 5.0 * loss_ic + 5.0 * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch: {epoch} | Loss: {loss.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | BC: {loss_bc.item():.4e} | "
                  f"Time: {elapsed:.1f}s")
            
    print("--> Training Complete.")
    return model, loss_history

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================

def evaluate(model):
    device = next(model.parameters()).device
    model.eval()
    
    # Create a grid for t=0.5
    t_fixed = 0.5
    n_points = 50
    x = torch.linspace(0, 1, n_points, device=device)
    y = torch.linspace(0, 1, n_points, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    t_tensor = torch.full_like(grid_x, t_fixed)
    
    # Prepare input
    inputs = torch.stack([t_tensor.flatten(), grid_x.flatten(), grid_y.flatten()], dim=1)
    
    with torch.no_grad():
        u_pred = model(inputs).reshape(n_points, n_points).cpu().numpy()
        u_true = exact_u(t_tensor, grid_x, grid_y).reshape(n_points, n_points).cpu().numpy()
        
    abs_error = np.abs(u_true - u_pred)
    mse = np.mean(abs_error**2)
    print(f"Validation MSE at t={t_fixed}: {mse:.4e}")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].contourf(grid_x.cpu(), grid_y.cpu(), u_true, levels=50, cmap='viridis')
    axes[0].set_title(f"Exact u(t={t_fixed})")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(grid_x.cpu(), grid_y.cpu(), u_pred, levels=50, cmap='viridis')
    axes[1].set_title(f"Hybrid PINN u(t={t_fixed})")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(grid_x.cpu(), grid_y.cpu(), abs_error, levels=50, cmap='inferno')
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("hybrid_pinn_result.png")
    print("--> Result plot saved to 'hybrid_pinn_result.png'")
    plt.show()

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Ensure plots work in non-GUI environments
    if os.environ.get('DISPLAY','') == '':
        print('No display found. Using non-interactive Agg backend')
        plt.switch_backend('Agg')

    # Train
    model, history = train_model()
    
    # Save Model
    torch.save(model.state_dict(), "hybrid_pinn_diffusion.pth")
    print("--> Model saved to 'hybrid_pinn_diffusion.pth'")
    
    # Evaluate
    evaluate(model)