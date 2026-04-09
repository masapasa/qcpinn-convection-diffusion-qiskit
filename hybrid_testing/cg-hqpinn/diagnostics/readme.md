Architecture (126,021 params): 
     
- Input (r,z) → Fourier features (48-dim) → Coord MLP (128-wide, 2 ResBlocks) → splits into quantum path (16 qubits) and classical skip (64-dim) → concatenated (82-dim) → Post MLP → 5 outputs (ur, uz, u_θ, p, T)
    
Key observations from the weights:

- Classical layers (coord_proj, res blocks, post): Well-behaved Gaussian distributions centered at 0 with σ ≈ 0.18–0.35. No dead neurons or exploding weights — healthy training.                        
- Quantum circuit params (q_layer.layer): Mean ≈ 3.3 rad, σ ≈ 1.85. The values span 0–6.3 (roughly 0 to 2π), which is expected — the Rot gate parameters have wrapped around during training. Both layers
show diverse parameter values across all 16 qubits, meaning the circuit isn't stuck in a trivial state.                                                                                                  
- Quantum LayerNorm: Scale (γ) varies from 0.3 to 1.2 across qubits — the network learned to weight some qubits more than others. Qubits 13 and 12 are downweighted (γ ≈ 0.3–0.5), while qubits 3, 9, and
10 are amplified. The biases show a clear pattern of positive/negative shifts per qubit.                                                                                                                 
- Fourier frequencies (B matrix): The frozen random frequencies span ±15, giving the network multi-scale spatial resolution for both r and z coordinates.                                                
- Output layer: uz has the strongest weights (largest magnitude), which makes sense — axial velocity is the dominant flow component in Czochralski melt. ur and u_θ are softer (they're multiplied by r  
in the hard constraint). Temperature (T) has a notable positive bias (0.22).                                                                                                                             
- Weight magnitude summary: The Fourier B matrix and quantum circuit params are the largest-magnitude tensors. Everything else is well-scaled with max values under 2. 