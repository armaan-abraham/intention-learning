# Data handling and environment management
import torch
from typing import Mapping, Sequence, Dict, Tuple, Type
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"

MODELS_DIR = DATA_DIR / "models"
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True)

BUFFERS_DIR = DATA_DIR / "buffers"
if not BUFFERS_DIR.exists():
    BUFFERS_DIR.mkdir(parents=True)

IMAGES_DIR = DATA_DIR / "images"
if not IMAGES_DIR.exists():
    IMAGES_DIR.mkdir(parents=True)


# TODO: allow initilaization iwth an id
class Buffer:
    name = "generic"

    def __init__(
        self,
        max_size: int,
        device: torch.device,
        var_dtypes: Mapping[str, type],
        var_shapes: Mapping[str, Sequence[int]],
    ):
        assert (
            isinstance(max_size, int) and max_size > 0
        ), f"Invalid max size: {max_size}"
        self.var_names = list(var_dtypes.keys())
        self.var_dtypes = var_dtypes
        self.var_shapes = var_shapes
        for (var_name, dtype), (_, shape) in zip(
            var_dtypes.items(), var_shapes.items()
        ):
            setattr(
                self,
                var_name,
                torch.zeros(max_size, *shape, dtype=dtype, device=device),
            )
        self.size = 0
        self.max_size = max_size
        self.ptr = 0

    def add(self, **new_elems: Mapping[str, torch.Tensor]):
        assert set(new_elems.keys()) == set(
            self.var_names
        ), "Items do not match buffer variables."
        for name, tensor in new_elems.items():
            assert (
                tensor.shape[1:] == self.var_shapes[name]
            ), f"Variable shapes do not match for {name}: {tensor.shape[1:]} != {self.var_shapes[name]}"
            assert (
                tensor.dtype == self.var_dtypes[name]
            ), f"Variable dtypes do not match for {name}."
        # make sure the number of new elements is the same for each variable
        n_new_elems = [tensor.shape[0] for tensor in new_elems.values()]
        assert (
            len(set(n_new_elems)) == 1
        ), "All variables must have the same number of new elements."
        n_new_elems = n_new_elems[0]
        if hasattr(self, "validate"):
            self.validate(new_elems)
        for var_name, tensor in new_elems.items():
            getattr(self, var_name)[self.ptr : self.ptr + n_new_elems] = tensor
        self.ptr = (self.ptr + n_new_elems) % self.max_size
        self.size = min(self.size + n_new_elems, self.max_size)

    def sample(self, n_samples: int) -> Dict[str, torch.Tensor]:
        assert isinstance(n_samples, int) and n_samples > 0
        assert (
            n_samples <= self.size
        ), f"Sampling more elements than stored is not allowed."
        indices = torch.randperm(self.size)[:n_samples]
        return {
            var_name: getattr(self, var_name)[indices] for var_name in self.var_names
        }
    
    def all(self) -> Dict[str, torch.Tensor]:
        return {var_name: getattr(self, var_name)[: self.size] for var_name in self.var_names}


def validate_states(states: torch.Tensor):
    assert torch.all(torch.abs(states[:, 0]) <= 1), f"Invalid state at dim 0"
    assert torch.all(torch.abs(states[:, 1]) <= 1), f"Invalid state at dim 1"
    assert torch.all(torch.abs(states[:, 2]) <= 8), f"Invalid state at dim 3"


class StateBuffer(Buffer):
    name = "state"

    def __init__(self, max_size: int, device: torch.device):
        super().__init__(
            max_size,
            device,
            var_dtypes={"states": torch.float32},
            var_shapes={"states": (3,)},
        )

    def validate(self, new_elems: Dict[str, torch.Tensor]):
        states = new_elems["states"]
        validate_states(states)


class JudgmentBuffer(Buffer):
    name = "judgment"

    def __init__(self, max_size: int, device: torch.device):
        super().__init__(
            max_size,
            device,
            var_dtypes={
                "states1": torch.float32,
                "states2": torch.float32,
                "judgments": torch.float32,
            },
            var_shapes={"states1": (3,), "states2": (3,), "judgments": (1,)},
        )

    def validate(self, new_elems: Dict[str, torch.Tensor]):
        states1, states2, judgments = (
            new_elems["states1"],
            new_elems["states2"],
            new_elems["judgments"],
        )
        validate_states(states1)
        validate_states(states2)
        assert torch.all((judgments == 0) | (judgments == 1)), f"Invalid judgment value"


class DataHandler:
    """Handles environment initialization, data generation, and storage."""

    def __init__(self, state_buffer: StateBuffer, judgment_buffer: JudgmentBuffer):
        """Initialize the environments and relevant parameters.

        Args:
            env_name (str): The name of the environment to instantiate.
            num_envs (int): The number of parallel environments to run.
            seed (int): The random seed for reproducibility.
        """
        # ...
        # ...
        self.state_buffer = state_buffer
        self.judgment_buffer = judgment_buffer
        # set id as current timestamp
        self.id = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"Data handler initialized with id {self.id}")

    def sample_past_states(self, n_states: int) -> torch.Tensor:
        return self.state_buffer.sample(n_states)["states"]

    def store_states(self, states: torch.Tensor):
        self.state_buffer.add(states=states)

    def save_judgments(
        self, states1: torch.Tensor, states2: torch.Tensor, judgments: torch.Tensor
    ):
        """Save judgments to storage.

        Args:
            states1 (torch.Tensor): The first states.
            states2 (torch.Tensor): The second states.
            judgments (torch.Tensor): The judgments.
        """
        self.judgment_buffer.add(states1=states1, states2=states2, judgments=judgments)

    def sample_past_judgments(
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.judgment_buffer.sample(n_samples)
        return result["states1"], result["states2"], result["judgments"]

    def save_model(self, model: torch.nn.Module):
        # make directory if it doesn't exist
        model_dir = MODELS_DIR / self.id
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        model_name = model.__class__.__name__
        # save the model
        torch.save(model.state_dict(), model_dir / f"{model_name}.pth")
    
    @classmethod
    def load_model(cls, id: str, model_cls: Type[torch.nn.Module], model_name: str = None) -> torch.nn.Module:
        model_dir = MODELS_DIR / id
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        # get string name of class
        model_name = model_name if model_name is not None else model_cls.__name__
        model_path = model_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        model = model_cls()
        model.load_state_dict(torch.load(model_path))
        return model
    
    @classmethod
    def load_buffer(cls, id: str, buffer_cls: Type[Buffer], device: torch.device) -> Buffer:
        # TODO: this is broken when dtypes are not float32
        """Load a buffer from disk.

        Args:
            buffer_name (str): The name of the buffer file to load.
            id (str): The identifier associated with the DataHandler.
            buffer_cls: The class of the buffer to instantiate.

        Returns:
            Buffer: The loaded buffer instance.
        """
        buffer_dir = BUFFERS_DIR / id
        if not buffer_dir.exists():
            raise FileNotFoundError(f"Buffer directory {buffer_dir} does not exist.")

        buffer_path = buffer_dir / f"{buffer_cls.name}.pt"
        if not buffer_path.exists():
            raise FileNotFoundError(f"Buffer file {buffer_path} does not exist.")

        # Load the data dictionary from the .pt file
        data = torch.load(buffer_path)
        var_names = data['var_names']
        var_dtypes = data['var_dtypes']
        var_shapes = data['var_shapes']
        size = data['size']
        max_size = data['max_size']
        ptr = data['ptr']

        # Instantiate the buffer
        buffer = buffer_cls(max_size, device)
        buffer.size = size
        buffer.ptr = ptr

        # Reconstruct the buffer variables
        for var_name in var_names:
            var_data = data[var_name]
            var_tensor = torch.zeros(
                max_size,
                *var_shapes[var_name],
                dtype=var_dtypes[var_name],
                device=device
            )
            var_tensor[:size] = var_data.to(device)
            setattr(buffer, var_name, var_tensor)

        return buffer

    def save_buffer(self, buffer: Buffer):
        """Save the contents of a buffer to disk.

        Args:
            buffer (Buffer): The buffer whose contents are to be saved.
            buffer_name (str): The name to use when saving the buffer.
        """
        # Create directory for buffers if it doesn't exist
        buffer_dir = BUFFERS_DIR / self.id
        if not buffer_dir.exists():
            buffer_dir.mkdir(parents=True)

        # Prepare the data to be saved
        data = {
            'var_names': buffer.var_names,
            'var_dtypes': buffer.var_dtypes,
            'var_shapes': buffer.var_shapes,
            'size': buffer.size,
            'max_size': buffer.max_size,
            'ptr': buffer.ptr,
        }
        # Save each variable tensor up to 'size'
        for var_name in buffer.var_names:
            var_data = getattr(buffer, var_name)[:buffer.size].cpu()
            data[var_name] = var_data

        # Save the data dictionary as a .pt file
        torch.save(data, buffer_dir / f"{buffer.name}.pt")

    def get_environment_specs(self):
        """Retrieve environment specifications such as state dimensions and action ranges.

        Returns:
            Tuple:
                - state_dim (int): Dimension of the state space.
                - action_dim (int): Dimension of the action space.
                - action_low (np.ndarray): Lower bound of action values.
                - action_high (np.ndarray): Upper bound of action values.
        """
        pass

    def reset(self):
        """Reset the environments and return the initial preprocessed states.

        Returns:
            np.ndarray: The initial preprocessed states after reset.
        """
        pass

    def step_environments(self, actions):
        """Take a step in all environments with the provided actions.

        Args:
            actions (np.ndarray): Actions to take in each environment.

        Returns:
            Tuple:
                - next_states (np.ndarray): The next preprocessed states.
                - rewards_ground_truth (np.ndarray): Rewards received.
                - dones (np.ndarray): Whether each environment is done.
        """
        # ...
        dones = self._reset_done_environments(dones)
        # ...
        # STILL TODO
        pass

    def _preprocess_state(self, state):
        """Preprocess the state before feeding it to the agent.

        Args:
            state (np.ndarray): The raw state from the environment.

        Returns:
            np.ndarray: The preprocessed state.
        """
        pass  # Internal method for state preprocessing

    def render_state(self, state):
        """Render the state into an image.

        Args:
            state (np.ndarray): The state to render.

        Returns:
            PIL.Image.Image: The rendered image.
        """
        pass

    def create_labeled_pair(self, img1, img2, target_width=150):
        """Create a labeled image pair from two images.

        Args:
            img1 (PIL.Image.Image): First image.
            img2 (PIL.Image.Image): Second image.
            target_width (int, optional): The width to which images are resized.

        Returns:
            PIL.Image.Image: Combined labeled image pair.
        """
        pass

    def _initialize_replay_buffer(self, max_size):
        """Initialize the replay buffer.

        Args:
            max_size (int): Maximum size of the replay buffer.
        """
        pass  # Initialize ReplayBuffer instance

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        """Add experiences to the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            next_state (np.ndarray): Next state.
            reward (float): Reward received.
            done (bool): Whether the episode ended.
        """
        pass

    def sample_from_replay_buffer(self, batch_size):
        """Sample a batch from the replay buffer.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            Tuple[torch.Tensor]: Batches of states, actions, next_states, rewards, dones.
        """
        pass

    def _reset_done_environments(self, dones):
        """Reset environments that are done and update internal state.

        Args:
            dones (np.ndarray): Boolean array indicating which environments are done.
        """
        pass

    def get_num_envs(self):
        """Get the number of parallel environments.

        Returns:
            int: Number of environments.
        """
        pass

    def save_experiences(
        self, states, actions, next_states, rewards, gt_rewards, dones
    ):
        """
        Save experiences to relevant files and replay buffer.

        Args:
            states (np.ndarray): The states.
            actions (np.ndarray): The actions.
            next_states (np.ndarray): The next states.
            rewards (np.ndarray): The rewards.
            gt_rewards (np.ndarray): The ground truth rewards.
            dones (np.ndarray): The dones.
        """
        pass

    def close(self):
        """Close all environments properly."""
        pass
