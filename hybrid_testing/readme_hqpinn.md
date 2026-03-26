# Hybrid-QPINN: Quantum-Enhanced Physics-Informed Neural Networks for Silicon Crystal Growth (Czochralski Method) Simulations 

This repository contains a state-of-the-art **Hybrid Quantum-Physics Informed Neural Network (QPINN)** designed to simulate the fluid dynamics and heat transfer within a **Czochralski (Cz) Crystal Growth Melt**. 

The model leverages **16 qubits** via PennyLane and IBM Qiskit Runtime to solve the Navier-Stokes and Energy equations in a cylindrical coordinate system, utilizing a unique **Data Re-uploading** architecture and **Fourier Feature Mapping**.

---

## 1. Scientific Context: The Czochralski Melt
The Czochralski process is the primary method for growing high-quality single-crystal silicon. The physics involve a rotating crucible and a rotating crystal, creating a complex interaction of:
- **Forced Convection** (due to rotation)
- **Natural Convection** (due to temperature gradients)
- **Centrifugal/Coriolis Forces** (due to the cylindrical reference frame)

## 2. Mathematical Formulation

### A. Governing Equations (Cylindrical Axisymmetry)
The model solves the incompressible, steady-state Navier-Stokes equations in 2D axisymmetry $(r, z)$, where $u$ is radial velocity, $v$ is axial velocity, $w$ is swirl (azimuthal) velocity, $p$ is pressure, and $T$ is temperature.

1. **Continuity Equation:**
   $$\frac{u}{r} + \frac{\partial u}{\partial r} + \frac{\partial v}{\partial z} = 0$$

2. **Momentum Equations ($r, z, \theta$):**
   $$u\frac{\partial u}{\partial r} + v\frac{\partial u}{\partial z} - \frac{w^2}{r} = -\frac{\partial p}{\partial r} + \frac{1}{Re} \left( \frac{\partial^2 u}{\partial r^2} + \frac{1}{r}\frac{\partial u}{\partial r} - \frac{u}{r^2} + \frac{\partial^2 u}{\partial z^2} \right)$$
   $$u\frac{\partial v}{\partial r} + v\frac{\partial v}{\partial z} = -\frac{\partial p}{\partial z} + \frac{1}{Re} \left( \frac{\partial^2 v}{\partial r^2} + \frac{1}{r}\frac{\partial v}{\partial r} + \frac{\partial^2 v}{\partial z^2} \right) + \frac{Gr}{Re^2}T$$
   $$u\frac{\partial w}{\partial r} + v\frac{\partial w}{\partial z} + \frac{uw}{r} = \frac{1}{Re} \left( \frac{\partial^2 w}{\partial r^2} + \frac{1}{r}\frac{\partial w}{\partial r} - \frac{w}{r^2} + \frac{\partial^2 w}{\partial z^2} \right)$$

3. **Energy Equation:**
   $$u\frac{\partial T}{\partial r} + v\frac{\partial T}{\partial z} = \frac{1}{Re \cdot Pr} \left( \frac{\partial^2 T}{\partial r^2} + \frac{1}{r}\frac{\partial T}{\partial r} + \frac{\partial^2 T}{\partial z^2} \right)$$

### B. Quantum Feature Mapping
To overcome the "Spectral Bias" of standard neural networks (the tendency to fail at learning high-frequency details), we use **Random Fourier Features (RFF)** before the quantum layer:
$$\gamma(\mathbf{x}) = [\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})]^T$$
Where $\mathbf{B}$ is a Gaussian projection matrix. This 16-dimensional vector is then mapped to the rotation angles of the 16 qubits.

---

## 3. Hybrid Architecture

The network consists of three distinct stages:
1. **Classical Pre-Processor:** Maps the 2D coordinates $(r, z)$ into a 16-dimensional feature vector using Fourier Mapping and a Tanh-activated MLP.
2. **Quantum Processor (Variational Circuit):** 
   - **Data Re-uploading:** The 16-dimensional features are re-injected into every layer of the circuit to maximize expressivity.
   - **Entanglement:** A `StronglyEntanglingLayers` template creates complex correlations between all 16 qubits using CNOT gates and Euler rotations.
3. **Classical Post-Processor:** Translates the 16 expectation values $\langle Z_i \rangle$ into the 5 physical output variables.

### Hard Physical Constraints
To ensure the solution is physically valid at the axis of symmetry ($r=0$), we enforce:
- $u(0, z) = 0$
- $w(0, z) = 0$
This is done by multiplying the raw network outputs for $u$ and $w$ by the input coordinate $r$.

---

## 4. Training Strategy: The Two-Phase Approach

Quantum hardware is currently limited by noise and queue times. This project uses a sophisticated two-step pipeline:

### Phase 1: High-Speed Simulation (AER)
- **Epochs:** 2000
- **Device:** `default.qubit` (optimized simulator)
- **Differentiation:** **Adjoint Method**. This allows for extremely fast gradient computation by avoiding the $2N$ circuit evaluations required by the parameter-shift rule.
- **Goal:** Learn the underlying physics and reach a stable local minimum.

### Phase 2: Hardware Fine-Tuning (IBM)
- **Epochs:** 100
- **Device:** Real IBM Quantum Hardware (e.g., Eagle or Heron processors).
- **Differentiation:** **Parameter-Shift Rule**.
- **Runtime Optimization:**
    - **Qiskit Runtime Sessions:** Keeps the job "hot" on the processor to avoid re-queueing between batches.
    - **Resilience Level 1:** Enables M3 Readout Error Mitigation to correct for hardware noise.
    - **Optimization Level 3:** High-level circuit transpilation to reduce SWAP gate overhead on the 16-qubit topology.

---

## 5. Requirements
- Python 3.9+
- [PennyLane](https://pennylane.ai/)
- [PennyLane-Qiskit](https://github.com/PennyLaneAI/pennylane-qiskit)
- [Qiskit-IBM-Runtime](https://github.com/Qiskit/qiskit-ibm-runtime)
- PyTorch
- NumPy, Pandas

---

## 6. Usage
1. **Initial Setup:** Install dependencies and add your `IBM_TOKEN` to the configuration section.
2. **Simulate:** Set `TRAINING_PHASE = "AER"` and run to generate `hybrid_qpinn_aer.pt`.
3. **Deploy:** Set `TRAINING_PHASE = "IBM"`. The script will load the pre-trained weights and open a Qiskit Runtime Session for hardware fine-tuning.

---

## 7. Citation/Reference
This implementation follows the principles of **Physics-Informed Neural Networks (PINNs)** introduced by Raissi et al. (2019) and extends them to the quantum domain using the **Data Re-uploading** method proposed by Pérez-Salinas et al. (2020).
