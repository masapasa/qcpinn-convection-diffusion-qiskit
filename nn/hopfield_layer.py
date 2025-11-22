import torch
import torch.nn as nn

class HopfieldLayer(nn.Module):
    """
    Modern Hopfield Layer for processing a batch of inputs.
    It performs self-association on the batch.
    """

    def __init__(self, input_dim, hidden_dim=None, beta=1.0):
        super(HopfieldLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.beta = beta

        # Mappings for Key, Query, Value (similar to attention)
        self.w_q = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.w_v = nn.Linear(input_dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, hidden_dim)
        """
        # The batch itself acts as the set of stored patterns for self-association.
        query = self.w_q(x)  # (batch_size, hidden_dim)
        key = self.w_k(x)    # (batch_size, hidden_dim)
        value = self.w_v(x)  # (batch_size, hidden_dim)

        # Calculate attention scores (energies)
        # (batch_size, hidden_dim) @ (hidden_dim, batch_size) -> (batch_size, batch_size)
        attention_scores = torch.matmul(query, key.t())

        # Scale by beta for controlling sharpness
        scaled_attention = attention_scores * self.beta

        # Compute attention weights via softmax
        attention_weights = torch.softmax(scaled_attention, dim=-1)

        # Compute weighted sum of values
        # (batch_size, batch_size) @ (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        output = torch.matmul(attention_weights, value)

        return output