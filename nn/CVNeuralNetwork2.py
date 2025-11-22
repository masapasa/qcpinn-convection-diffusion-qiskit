import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
class CVNeuralNetwork2(nn.Module):
    def __init__(
        self,
        num_qumodes: int,
        num_layers: int,
        device: str = "cpu",
        cutoff_dim: int = 2,
    ):
        super().__init__()
        self.num_qumodes = num_qumodes
        self.num_layers = num_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        active_sd = 0.1
        passive_sd = 2 * np.pi
        self.num_interfermoter_params = int(
            self.num_qumodes * (self.num_qumodes - 1)
        ) + max(1, self.num_qumodes - 1)
        self.theta_1 = nn.Parameter(
            torch.randn(num_layers, self.num_interfermoter_params, device=self.device)
            * passive_sd,
            requires_grad=True,
        )
        self.theta_2 = nn.Parameter(
            torch.randn(num_layers, self.num_interfermoter_params, device=self.device)
            * passive_sd,
            requires_grad=True,
        )
        self.squeezing_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )
        self.squeezing_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd,
            requires_grad=True,
        )
        self.displacement_r = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )
        self.displacement_phi = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * passive_sd,
            requires_grad=True,
        )
        self.kerr_params = nn.Parameter(
            torch.randn(num_layers, num_qumodes, device=self.device) * active_sd,
            requires_grad=True,
        )
        self.dev = qml.device(
            "strawberryfields.fock", wires=num_qumodes, cutoff_dim=cutoff_dim
        )
        self.circuit = qml.QNode(self._quantum_circuit, self.dev, interface="torch")
        self.activation = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.circuit(sample) for sample in x])
    def _quantum_circuit(self, inputs):
        for i, input_val in enumerate(inputs):
            qml.Displacement(input_val, 0.0, wires=i)
        for layer_idx in range(self.num_layers):
            self.qnn_layer(layer_idx)
        return [
            qml.expval(qml.QuadOperator(wires=wire, phi=0.0))
            for wire in range(self.num_qumodes)
        ]
    def qnn_layer(self, layer_idx):
        self.interferometer(self.theta_1[layer_idx])
        for wire in range(self.num_qumodes):
            qml.Squeezing(
                self.squeezing_r[layer_idx, wire],
                self.squeezing_phi[layer_idx, wire],
                wires=wire,
            )
        self.interferometer(self.theta_2[layer_idx])
        for wire in range(self.num_qumodes):
            qml.Displacement(
                self.displacement_r[layer_idx, wire],
                self.displacement_phi[layer_idx, wire],
                wires=wire,
            )
            qml.Kerr(self.kerr_params[layer_idx, wire], wires=wire)
    def interferometer(self, params):
        qumode_list = list(range(self.num_qumodes))
        theta = params[: self.num_qumodes * (self.num_qumodes - 1) // 2]
        phi = params[
            (self.num_qumodes * (self.num_qumodes - 1) // 2) : (
                self.num_qumodes * (self.num_qumodes - 1)
            )
        ]
        rphi = params[-self.num_qumodes + 1 :]
        if self.num_qumodes == 1:
            qml.Rotation(rphi[0], wires=0)
            return
        n = 0
        for l in range(self.num_qumodes):
            for k, (q1, q2) in enumerate(zip(qumode_list[:-1], qumode_list[1:])):
                if (l + k) % 2 != 1:
                    qml.Beamsplitter(theta[n], phi[n], wires=[q1, q2])
                    n += 1
        for i in range(max(1, self.num_qumodes - 1)):
            qml.Rotation(rphi[i], qumode_list[i])