# QCPINN: Quantum Physics-Informed Neural Networks for Convection-Diffusion

Hybrid quantum physics-informed neural networks for solving partial differential equations (PDEs) with support for IBM Quantum hardware via Qiskit Runtime.

## Overview

This project implements hybrid quantum-classical neural networks that combine:
- **Physics-Informed Neural Networks (PINNs)**: Enforcing PDE constraints directly in the loss function
- **Quantum Computing Layers**: Leveraging quantum circuits for enhanced expressivity
- **IBM Quantum Integration**: Running on real quantum hardware via Qiskit Runtime

## Architecture

```
Input (t, x, y) → Classical Preprocessor → Quantum Layer → Classical Postprocessor → Output u(t, x, y)
                      (Dense + Tanh)      (Variational    (Dense + Tanh)
                                          Quantum Circuit)
```

### Supported Quantum Backends

| Solver | Description | Hardware |
|--------|-------------|----------|
| **DV** | Discrete Variable (Qubit-based) | IBM Quantum, Simulators |
| **CV** | Continuous Variable (Photonic) | Simulators only |
| **Classical** | Hopfield-based classical baseline | CPU/GPU |

### Quantum Circuit Ansatzes

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

## Installation

```bash
pip install pennylane-qiskit
pip install torch numpy matplotlib scipy
pip install qiskit-ibm-runtime
```

## Quick Start

### Simulation Mode

```python
from nn.DVPDESolver import DVPDESolver
from utils.logger import Logging
import trainer.diffusion_train as diffusion_train

args = {
    "batch_size": 64,
    "epochs": 5000,
    "lr": 0.005,
    "input_dim": 3,
    "output_dim": 1,
    "num_qubits": 4,
    "hidden_dim": 50,
    "num_quantum_layers": 1,
    "classic_network": [3, 50, 1],
    "q_ansatz": "cascade",
    "solver": "DV",
    "use_ibm_hardware": False,
}

logger = Logging("./checkpoints/diffusion")
model = DVPDESolver(args, logger, device="cpu")
diffusion_train.train(model)
```

### IBM Quantum Hardware

```python
args = {
    # ... same as above ...
    "use_ibm_hardware": True,
    "ibm_token": "YOUR_IBM_QUANTUM_TOKEN",
    "ibm_backend": "ibm_torino",
    "shots": 1024,
    "epochs": 100,  # Reduce for hardware
}

model = DVPDESolver(args, logger, device="cpu")
diffusion_train.train(model)
```

## Project Structure

```
qcpinn-convection-diffusion-qiskit/
├── nn/
│   ├── DVPDESolver.py       # Discrete Variable quantum solver
│   ├── DVQuantumLayer.py    # Qubit-based quantum circuit layer
│   ├── CVPDESolver.py       # Continuous Variable quantum solver
│   ├── CVNeuralNetwork1.py  # Photonic quantum circuits
│   ├── ClassicalSolver.py   # Classical baseline with Hopfield
│   ├── pde.py               # PDE operator definitions
│   └── hopfield_layer.py    # Hopfield network layer
├── trainer/
│   ├── diffusion_train.py           # Training loop
│   └── diffusion_hybrid_trainer.py  # Main training script
├── data/
│   └── diffusion_dataset.py  # Analytical solutions & samplers
├── utils/
│   ├── logger.py             # Logging utilities
│   └── ContourPlotter.py     # Visualization
└── checkpoints/              # Saved models & outputs
```

## Training

Run the main trainer:

```bash
python trainer/diffusion_hybrid_trainer.py
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 64 |
| `epochs` | Number of training epochs | 20000 |
| `lr` | Learning rate | 0.005 |
| `num_qubits` | Number of qubits | 4 |
| `num_quantum_layers` | Quantum circuit depth | 1 |
| `q_ansatz` | Circuit topology | "cascade" |
| `shots` | Measurement shots (hardware) | 1024 |
| `encoding` | Input encoding ("angle"/"amplitude") | "angle" |

## Physics-Informed Loss

The training minimizes a composite loss:

```
L = λ_r · L_residual + λ_bc · L_boundary + λ_ic · L_initial
```

Where:
- `L_residual`: PDE residual at collocation points
- `L_boundary`: Boundary condition enforcement
- `L_initial`: Initial condition enforcement

## Results

After training, the model outputs:
- `model.pth`: Saved model weights
- `loss_history.pdf`: Training loss curve
- `circuit.pdf`: Quantum circuit visualization
- Contour plots comparing predictions vs analytical solutions

## Quantum Gradient Computation

| Mode | Differentiation Method | Performance |
|------|----------------------|-------------|
| Simulator | Backpropagation | Fast |
| Hardware | Parameter-shift rule | Slow, accurate |
| Fallback | Finite difference | Moderate |

## Hardware Considerations

- **Simulators**: Full batch processing, backpropagation gradients
- **Real Hardware**: Sequential sample processing, parameter-shift gradients
- **Recommended**: Start with simulators, reduce epochs for hardware runs

## References

1. [Quantum Physics-Informed Neural Networks for CFD](https://arxiv.org/abs/2304.11247)
2. [Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
3. [PennyLane Documentation](https://pennylane.ai/)
4. [Qiskit Runtime](https://quantum.ibm.com/)

## License

MIT License

