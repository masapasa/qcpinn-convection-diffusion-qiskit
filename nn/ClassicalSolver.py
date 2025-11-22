import torch
import torch.nn as nn
import os
from utils.logger import Logging
from nn.hopfield_layer import HopfieldLayer

class ClassicalSolver(nn.Module):
    def __init__(self, args, logger: Logging, data=None, device=None):
        super().__init__()
        self.logger = logger
        self.device = device
        self.args = args
        self.data = data
        self.batch_size = self.args["batch_size"]
        self.epochs = self.args["epochs"]
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.classic_network = self.args["classic_network"]
        
        input_dim = self.classic_network[0]
        hidden_dim = self.classic_network[1]
        output_dim = self.classic_network[2]

        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        ).to(self.device)

        self.hopfield_layer = HopfieldLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            beta=1.0  # beta can be tuned
        ).to(self.device)

        self.postprocessor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
        
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"checkpoint path: {self.log_path=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            
            preprocessed_out = self.preprocessor(x)
            hopfield_out = self.hopfield_layer(preprocessed_out)
            
            # Add a residual connection for stable training
            residual_out = preprocessed_out + hopfield_out
            
            classical_out = self.postprocessor(residual_out)
            return classical_out
        except Exception as e:
            self.logger.print(f"Forward pass failed: {str(e)}")
            raise

    def save_state(self):
        state = {
            "args": self.args,
            "classic_network": self.classic_network,
            "preprocessor": self.preprocessor.state_dict(),
            "hopfield_layer": self.hopfield_layer.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "optimizer": (self.optimizer.state_dict() if self.optimizer is not None else None),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            "loss_history": self.loss_history,
            "log_path": self.log_path,
        }
        model_path = os.path.join(self.log_path, "model.pth")
        with open(model_path, "wb") as f:
            torch.save(state, f)
        self.logger.print(f"Model state saved to {model_path}")

    @classmethod
    def load_state(cls, file_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cpu")
        with open(file_path, "rb") as f:
            state = torch.load(f, map_location=map_location)
        print(f"Model state loaded from {file_path}")
        return state