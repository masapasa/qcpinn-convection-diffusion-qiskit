"""Run pre-training diagnostic plots without connecting to IBM hardware."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import pennylane as qml

from CG_HQPINN_IBMtest_16q_effective import (
    load_cz_data, load_checkpoint, make_quantum_layer,
    Hybrid16QPINN, choose_calibration_subset, plot_pretrain_diagnostics,
)

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--load", required=True)
p.add_argument("--out", default="diagnostics")
p.add_argument("--n-qubits", type=int, default=16)
p.add_argument("--n-layers", type=int, default=2)
p.add_argument("--ibm-calib-size", type=int, default=8)
args = p.parse_args()

device = torch.device("cpu")

# Use default.qubit — no IBM connection needed
qdev = qml.device("default.qubit", wires=args.n_qubits)
q_layer = make_quantum_layer(qdev, args.n_qubits, args.n_layers, diff_method="best")
model = Hybrid16QPINN(q_layer=q_layer, n_qubits=args.n_qubits).to(device)

artifacts = load_cz_data(args.data, device)
ckpt_stats = load_checkpoint(args.load, model)
if ckpt_stats:
    artifacts.stats = ckpt_stats

x_calib, y_calib = choose_calibration_subset(artifacts.x, artifacts.y, args.ibm_calib_size)
plot_pretrain_diagnostics(artifacts, model, device, args.out, x_calib, y_calib)
