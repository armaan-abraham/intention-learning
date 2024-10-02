import torch
import torch.nn as nn

from data import DataHandler


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, device):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.to(device)

    def forward(self, state):
        return self.action_high * self.layers(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.layers(x)


class Agent:
    def __init__(self, state_dim, action_dim, action_low, action_high, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, action_low, action_high, device)
        self.actor.name = "actor"
        self.actor_target = Actor(
            state_dim, action_dim, action_low, action_high, device
        )
        self.actor_target.name = "actor_target"
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim, device)
        self.critic1.name = "critic1"
        self.critic2 = Critic(state_dim, action_dim, device)
        self.critic2.name = "critic2"
        self.critic1_target = Critic(state_dim, action_dim, device)
        self.critic1_target.name = "critic1_target"
        self.critic2_target = Critic(state_dim, action_dim, device)
        self.critic2_target.name = "critic2_target"
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)

        self.noise_scale = 0.5
        self.noise_decay = 0.95

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def validate_states(self, states: torch.Tensor):
        assert torch.all(states <= 1)
        assert torch.all(states >= -1)

    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        # TODO: remove validation
        self.validate_states(states)
        return self.actor(states)

    def select_action_with_noise(self, states: torch.Tensor) -> torch.Tensor:
        self.validate_states(states)
        actions = self.select_action(states)
        noise = torch.normal(
            0, self.noise_scale, size=actions.shape, device=self.device
        )
        return torch.clamp(actions + noise, self.action_low, self.action_high)

    def decay_noise(self):
        self.noise_scale *= self.noise_decay

    def train(
        self,
        data_handler: DataHandler,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.001,
    ):
        print("Initial loss")
        self.print_loss(data_handler, batch_size, discount)

        for i in range(iterations):
            self.total_it += 1
            # Sample from the replay buffer
            state, action, next_state, reward, done = (
                data_handler.sample_past_experiences(batch_size)
            )
            # TODO: remove validation
            self.validate_states(state)
            self.validate_states(next_state)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.actor_target(next_state) + noise).clamp(
                    self.action_low, self.action_high
                )

                # Compute the target Q value
                target_Q1 = self.critic1_target(next_state, next_action)
                target_Q2 = self.critic2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (~done) * discount * target_Q

            # Get current Q estimates
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            # Compute critic loss
            critic_loss = nn.functional.mse_loss(
                current_Q1, target_Q
            ) + nn.functional.mse_loss(current_Q2, target_Q)

            # Optimize the critics
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                    self.critic1.parameters(), self.critic1_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic2.parameters(), self.critic2_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        print("Final loss")
        self.print_loss(data_handler, batch_size, discount)
        print("-" * 50)

    def print_loss(self, data_handler: DataHandler, batch_size: int, discount: float):
        with torch.no_grad():
            state, action, next_state, reward, done = (
                data_handler.sample_past_experiences(batch_size)
            )

            target_Q1 = self.critic1_target(next_state, self.actor_target(next_state))
            target_Q2 = self.critic2_target(next_state, self.actor_target(next_state))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (~done) * discount * target_Q
            current_Q1 = self.critic1_target(state, action)
            current_Q2 = self.critic2_target(state, action)
            critic1_loss = nn.functional.mse_loss(current_Q1, target_Q).item()
            critic2_loss = nn.functional.mse_loss(current_Q2, target_Q).item()
            critic_loss = critic1_loss + critic2_loss

            actor_loss = (
                -self.critic1_target(state, self.actor_target(state)).mean().item()
            )

            print(
                f"Average actor prediction magnitude: {self.actor_target(state).abs().mean().item()}"
            )
            print(f"Critic1 loss: {critic1_loss:.6f}")
            print(f"Critic2 loss: {critic2_loss:.6f}")
            print(f"Critic loss: {critic_loss:.6f}")
            print(f"Actor loss: {actor_loss:.6f}")
