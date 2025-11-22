import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from utils.logger import Logging
from nn.DVQuantumLayer import DVQuantumLayer
class DVPDESolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=None):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.num_qubits = self.args["num_qubits"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.encoding = self.args.get("encoding", "angle")
        self.draw_quantum_circuit_flag = True
        self.classic_network = self.args["classic_network"]  
        self.total_training_time = 0
        self.total_memory_peak = 0
        if self.encoding == "amplitude":
            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)
        else:
            self.preprocessor = nn.Sequential(
                nn.Linear(self.classic_network[0], self.classic_network[-2]).to(
                    self.device
                ),
                nn.Tanh(),
                nn.Linear(self.classic_network[-2], self.num_qubits).to(self.device),
            ).to(self.device)
        self.postprocessor = nn.Sequential(
            nn.Linear(self.num_qubits, self.classic_network[-2]).to(self.device),
            nn.Tanh(),
            nn.Linear(self.classic_network[-2], self.classic_network[-1]).to(
                self.device
            ),
        ).to(self.device)
        self.activation = nn.Tanh()
        self.num_qubits = args["num_qubits"]
        self.quantum_layer = DVQuantumLayer(self.args)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args["lr"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.9, patience=1000
        )
        self.loss_fn = torch.nn.MSELoss()
        self._initialize_logging()
        self._initialize_weights()
    def _initialize_weights(self):
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(
                    layer.weight
                )  
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  
    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            preprocessed = self.preprocessor(x)
            if self.draw_quantum_circuit_flag:
                self.draw_quantum_circuit(preprocessed)
                self.draw_quantum_circuit_flag = False
            quantum_out = self.quantum_layer(preprocessed).to(
                dtype=torch.float32, device=self.device
            )
            quantum_features = quantum_out.view( self.num_qubits , -1).T
            classical_out = self.postprocessor(quantum_features)
            return classical_out
        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise
    def save_state(self , path=None):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "quantum_params": self.quantum_layer.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "quantum_layer": self.quantum_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }
        if path is None:
            model_path = os.path.join(self.log_path, "model.pth")
        else:    
            model_path = path
        with open(model_path, "wb") as f:
            torch.save(state, f)
        self.logger.print(f"Model state saved to {model_path}")
    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            state = torch.load(f, map_location=map_location)
        return state
    def draw_quantum_circuit(self, x):
        if self.draw_quantum_circuit_flag:
            try:
                self.logger.print("The circuit used in the study:")
                if self.quantum_layer.params is not None:
                    fig, ax = qml.draw_mpl(self.quantum_layer.circuit)(x[0])
                    plt.savefig(os.path.join(self.log_path, "circuit.pdf"))
                    plt.close()
                    print(f"The circuit is saved in {self.log_path}")
            except Exception as e:
                self.logger.print(f"Failed to draw quantum circuit: {str(e)}")