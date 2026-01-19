# CLAUDE.md

## Project Overview

QCPINN (Quantum Classical Physics-Informed Neural Networks) - A hybrid quantum-classical framework for solving partial differential equations using quantum computing with Qiskit and IBM Quantum hardware.

## Quick Commands

```bash
# Run main training
python trainer/diffusion_hybrid_trainer.py
l
# Quick test (1 minute)
python trainer/diffusion_hybrid_trainer-simple-1min.py

# Simple training variant
python trainer/diffusion_hybrid_trainer-simple.py
```

## Architecture

```
Input (t, x, y) → Classical Preprocessor → Quantum Layer → Classical Postprocessor → Output u(t,x,y)
                    (Dense + Tanh)       (Variational      (Dense + Tanh)
                                         Quantum Circuit)
```

## Key Files

| File | Purpose |
|------|---------|
| `nn/DVPDESolver.py` | Main discrete-variable solver combining pre/post-processors with quantum layer |
| `nn/DVQuantumLayer.py` | Core quantum circuit implementation with 6 ansatzes |
| `nn/CVPDESolver.py` | Continuous-variable (photonic) quantum solver |
| `nn/ClassicalSolver.py` | Classical baseline using Hopfield networks |
| `nn/pde.py` | PDE operator definitions (Convection-Diffusion, Navier-Stokes, etc.) |
| `nn/hopfield_layer.py` | Modern Hopfield network layer |
| `trainer/diffusion_train.py` | Core training loop with physics-informed loss |
| `trainer/diffusion_hybrid_trainer.py` | Main entry point with full configuration |
| `data/diffusion_dataset.py` | Analytical solutions and sampling utilities |
| `utils/logger.py` | Experiment logging with timestamped output directories |
| `utils/ContourPlotter.py` | 2D/3D visualization of predictions vs analytical solutions |

## Solver Types

- **DV (Discrete Variable)** - Qubit-based quantum circuits, compatible with IBM Quantum hardware
- **CV (Continuous Variable)** - Photonic quantum circuits, simulators only
- **Classical** - Hopfield network baseline for comparison

## Quantum Ansatzes

| Ansatz | Parameters | Description |
|--------|------------|-------------|
| `cascade` | 3n | Ring connectivity with CRX entanglement |
| `layered` | 4n | RZ-RX rotations with CNOT ring |
| `alternate` | 4n-4 | Alternating TDCNOT blocks |
| `farhi` | 2n-2 | RXX and RZX gates |
| `sim_circ_15` | 2n | Hardware-efficient design |
| `cross_mesh` | 4n + n(n-1) | All-to-all CRZ connectivity |

## Supported PDEs

- **Convection-Diffusion**: `∂u/∂t + v·∇u = D∇²u`
- **Navier-Stokes 2D**: Incompressible fluid flow
- **Klein-Gordon**: Nonlinear wave equation
- **Wave Equation**: `∂²u/∂t² = c²∇²u`
- **Helmholtz**: `∇²u + λu = 0`

## Key Patterns

- **Physics-informed loss**: `L = λ_r·L_residual + λ_bc·L_boundary + λ_ic·L_initial`
- **Automatic differentiation** for computing PDE residuals via PyTorch autograd
- **Parameter-shift rule** for gradients on real quantum hardware
- **Gradient clipping** to prevent training instability
- **ReduceLROnPlateau** scheduler for adaptive learning rate

## Configuration

All parameters controlled via `args` dict in trainer files:

```python
args = {
    "batch_size": 64,
    "epochs": 20000,
    "lr": 0.005,
    "num_qubits": 4,
    "num_quantum_layers": 1,
    "q_ansatz": "cascade",
    "solver": "DV",
    "encoding": "angle",  # or "amplitude"
    "use_ibm_hardware": False,
    "ibm_backend": "ibm_torino",
    "shots": 1024,
}
```

## Dependencies

- PyTorch (neural network framework)
- PennyLane (quantum ML framework)
- Qiskit + qiskit-ibm-runtime (IBM Quantum integration)
- pennylane-qiskit (bridge between PennyLane and Qiskit)
- NumPy, SciPy, Matplotlib

## Output Files

Training produces in `checkpoints/` directory:
- `model.pth` - Saved model weights
- `loss_history.pdf` - Training loss curve
- `circuit.pdf` - Quantum circuit visualization
- Contour plots comparing predictions vs analytical solutions

## Hardware vs Simulator

| Mode | Gradient Method | Performance |
|------|-----------------|-------------|
| Simulator | Backpropagation | Fast |
| Hardware | Parameter-shift rule | Slow, accurate |

For hardware runs, reduce `epochs` significantly (e.g., 100 instead of 20000).
