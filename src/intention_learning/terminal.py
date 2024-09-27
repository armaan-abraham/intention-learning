import torch
import torch.nn as nn
from intention_learning.judge import Judge
from intention_learning.data import DataHandler

class TerminalModel:
    """Model of the terminal valuation of states."""
    def __init__(self, network: nn.Module):
        self.network = network

    def train(self, judge: Judge, data_handler: DataHandler):
        pass





class TerminalNetwork(nn.Module):
    """Network for the terminal value function."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

