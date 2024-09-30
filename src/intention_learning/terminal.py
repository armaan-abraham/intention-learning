import torch
import torch.nn as nn
from intention_learning.data import DataHandler, MODELS_DIR


class TerminalNetwork(nn.Module):
    """Network for the terminal value function."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class TerminalModel:
    """Model of the terminal (instrinsic) value of states."""

    def __init__(self, data_handler: DataHandler, device: torch.device, lr: float = 5e-3, network: nn.Module = None):
        self.network = network if network is not None else TerminalNetwork().to(device)
        self.data_handler = data_handler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, amsgrad=True
        )

    def train(self, n_iter: int = 10, batch_size: int = 100):
        for _ in range(n_iter):
            # sample judged pairs
            states1, states2, judgments = self.data_handler.sample_past_judgments(
                n_samples=batch_size
            )
            p_2_better = self.regress_states(states1, states2)
            loss = self.loss(p_2_better, judgments)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def loss(self, predictions: torch.Tensor, judgments: torch.Tensor) -> torch.Tensor:
        judgments = judgments.to(torch.float32)
        return nn.functional.binary_cross_entropy(predictions, judgments)

    def regress_states(
        self, states1: torch.Tensor, states2: torch.Tensor
    ) -> torch.Tensor:
        """Returns probability of state 2 being more terminally valuable than state 1."""
        rewards1, rewards2 = self.predict(states1), self.predict(states2)
        return torch.sigmoid(rewards2 - rewards1)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    @torch.no_grad()
    def sample_and_evaluate_loss(self, n_samples: int = 100):
        samples1, samples2, judgments = self.data_handler.sample_past_judgments(
            n_samples=n_samples
        )
        p_2_better = self.regress_states(samples1, samples2)
        return self.loss(p_2_better, judgments)

    def save(self):
        self.data_handler.save_model(self.network, "terminal")
