## Quantum-Classical Hybrid PINN for Convection-Diffusion PDEs

We have implemented a hybrid quantum physics-informed neural network (PI-NN) framework for solving convection-diffusion partial differential equations (PDEs), with applications to computational fluid dynamics (CFD) in crystal growth simulations.

### Project Overview
- **Core Technology**: Hybrid quantum-classical neural networks combining classical feedforward layers with discrete-variable (DV) quantum circuits
- **Application Domain**: 2D/3D convection-diffusion PDE solving for crystal growth optimization
- **Framework**: PyTorch + PennyLane/Qiskit integration with physics-informed training
- **Key Components**:
  - Classical preprocessor layers (dense networks with Tanh activation)
  - Quantum layers with configurable ansätze (layered, alternate, cascade topologies)
  - Classical postprocessor layers
  - Physics-informed loss functions enforcing PDE constraints

### Technical Architecture
- **Neural Network Structure**: Classical → Quantum → Classical hybrid architecture
- **Quantum Implementation**: DV quantum circuits with rotation gates (Rx, Ry, Rz), entangling gates (CNOT, CZ), and configurable topologies
- **Training Approach**: Physics-informed learning with PDE residual loss, boundary/initial condition loss, and optional data-driven loss
- **Hardware Support**: Integrated IBM Quantum hardware execution via Qiskit Runtime Service

### Why This Network Might Not Be Optimal for IBM Hardware

1. **High Circuit Depth**: Current implementations use multiple quantum layers with complex entanglement patterns, leading to deep circuits that suffer from NISQ device noise and decoherence
2. **Shot-Based Execution**: Training requires many circuit evaluations with finite shot noise, making real-time gradient computation challenging on current IBM hardware
3. **Parameter-Shift Rule Overhead**: Quantum gradient computation doubles circuit evaluations per parameter, exponentially increasing execution time on hardware
4. **Limited Qubit Count**: Complex PDE domains require many qubits, but current IBM devices have limited qubit counts (20-127 qubits) with connectivity constraints
5. **Training Latency**: Iterative training loops with frequent hardware calls create significant communication overhead

### Best Options for Running on IBM Hardware

1. **Circuit Optimization**:
   - Use cascade topology (ring connectivity) for better hardware efficiency
   - Reduce circuit depth through ansatz optimization
   - Implement variational quantum eigensolver (VQE) inspired architectures

2. **Hybrid Execution Strategy**:
   - Train quantum components on simulators first
   - Use hardware only for inference/validation phases
   - Implement quantum state caching to reduce redundant evaluations

3. **Hardware-Specific Adaptations**:
   - Map to specific IBM backend topologies (e.g., heavy-hex, linear connectivity)
   - Use error mitigation techniques (readout error correction, dynamical decoupling)
   - Leverage pulse-level control for optimized gate implementations

### Do We Have to Focus on the Quantum Part?

**No, the quantum component is not mandatory for this application.** The classical components alone provide significant value:

- Classical dense networks can achieve high accuracy for many PDE problems
- Quantum layers add expressivity for complex nonlinear PDEs but introduce scalability challenges
- The physics-informed training framework works effectively with purely classical architectures
- Current results show classical baselines perform competitively with hybrid approaches for many CFD problems

### Implementation Status
- **Classical Networks**: Multiple variants (CVNeuralNetwork1/2/3) with 2-3 dense layers
- **Quantum Integration**: DVQuantumLayer with IBM hardware support and fallback simulation
- **Training Infrastructure**: Complete physics-informed training loops with loss monitoring
- **Validation**: Checkpointed models with loss history tracking and circuit visualization

### Recommendations for IBM Collaboration
1. **Prioritize Classical Validation**: Establish strong classical baselines before quantum integration
2. **Focus on Hardware-Compatible Circuits**: Develop shallow, noise-resilient quantum circuits optimized for IBM topologies
3. **Explore Hybrid Training Strategies**: Train quantum components offline, deploy inference on hardware
4. **Investigate QPU Acceleration**: Evaluate potential benefits of quantum advantage for specific PDE classes