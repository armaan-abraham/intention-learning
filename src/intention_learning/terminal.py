import torch
import torch.nn as nn
from intention_learning.data import DataHandler

class TerminalModel:
    """Model of the terminal valuation of states."""
    def __init__(self, network: nn.Module):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, amsgrad=True)

    def train(self, data_handler: DataHandler, n_iter: int = 10, batch_size: int = 500):
        for _ in range(n_iter):
            # sample judged pairs
            samples1, samples2, judgments = data_handler.sample_judgments(n_samples=batch_size)
            rewards1, rewards2 = self.predict(samples1), self.predict(samples2)
            # reproduce judgments based on the predicted terminals
            p_2_better = torch.sigmoid(rewards2 - rewards1)
            loss = nn.functional.binary_cross_entropy(p_2_better, judgments)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

class TerminalNetwork(nn.Module):
    """Network for the terminal value function."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

