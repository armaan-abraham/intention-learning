# Data handling and environment management
import torch
from typing import Mapping, Sequence, Dict, Tuple, Type
from pathlib import Path
from datetime import datetime
from gymnasium.vector import AsyncVectorEnv

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
    assert torch.all(torch.abs(states[:, 2]) <= 1), f"Invalid state at dim 3"


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

class ExperienceBuffer(Buffer):
    def __init__(self, max_size: int, device: torch.device):
        super().__init__(max_size, device, var_dtypes={"states": torch.float32, "actions": torch.float32, "next_states": torch.float32, "rewards": torch.float32, "dones": torch.bool}, var_shapes={"states": (3,), "actions": (3,), "next_states": (3,), "rewards": (1,), "dones": (1,)})

    def validate(self, new_elems: Dict[str, torch.Tensor]):
        states, actions, next_states, rewards, dones = (
            new_elems["states"],
            new_elems["actions"],
            new_elems["next_states"],
            new_elems["rewards"],
            new_elems["dones"],
        )
        validate_states(states)
        validate_states(next_states)


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

class CyclingEnvHandler:
    def __init__(self, num_envs: int, device: torch.device):
        self.envs = AsyncVectorEnv([lambda: gym.make("Pendulum-v1") for i in range(NUM_ENVS)])
        self.device = device
    
    def normalize_states(self, states: torch.Tensor):
        states[:, 2] /= 8
    
    def reset(self, seed: int = None) -> torch.Tensor:
        kwargs = {"seed": seed} if seed is not None else {}
        states = torch.tensor(self.envs.reset(**kwargs)[0]["observation"], device=self.device)
        self.normalize_states(states)
        return states
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = actions.cpu().numpy()
        next_states, rewards, terminated, truncated, info = self.envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        if np.any(dones):
            for i in range(self.num_envs):
                if dones[i]:
                    next_states[i] = info["final_observation"][i]["observation"]
        next_states = torch.tensor(next_states, device=self.device)
        self.normalize_states(next_states)
        return next_states, torch.tensor(dones, device=self.device)
    
    def select_random_actions(self) -> torch.Tensor:
        # TODO: test this
        return torch.tensor(self.envs.action_space.sample(), device=self.device)
    
    def get_environment_specs(self):
        observation_space = self.envs.single_observation_space["observation"]

        state_dim = observation_space.shape[0]
        action_dim = self.envs.single_action_space.shape[0]
        action_low = self.envs.single_action_space.low
        action_high = self.envs.single_action_space.high

        assert action_low == -action_high
        return state_dim, action_dim, action_low, action_high
        

class DataHandler:
    """Handles environment initialization, data generation, and storage."""

    def __init__(self, state_buffer: StateBuffer = None, judgment_buffer: JudgmentBuffer = None, experience_buffer: ExperienceBuffer = None, env_handler: CyclingEnvHandler = None):
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
        self.experience_buffer = experience_buffer
        self.env_handler = env_handler
        self.id = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"Data handler initialized with id {self.id}")

    def sample_past_states(self, n_states: int) -> torch.Tensor:
        return self.state_buffer.sample(n_states)["states"]

    def save_states(self, states: torch.Tensor):
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
    
    def save_experiences(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        self.experience_buffer.add(states=states, actions=actions, next_states=next_states, rewards=rewards, dones=dones)
    
    def step_environments(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_states, dones = self.env_handler.step(actions)
        return next_states, dones

    def sample_past_judgments(
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.judgment_buffer.sample(n_samples)
        return result["states1"], result["states2"], result["judgments"]
    
    def sample_past_experiences(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = self.experience_buffer.sample(n_samples)
        return sampled["states"], sampled["actions"], sampled["next_states"], sampled["rewards"], sampled["dones"]
    
    def reset_envs(self, seed: int = None) -> torch.Tensor:
        return self.env_handler.reset(seed)
    
    def select_random_actions(self) -> torch.Tensor:
        return self.env_handler.select_random_actions()
    
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

    def save_model(self, model: torch.nn.Module):
        # make directory if it doesn't exist
        model_dir = MODELS_DIR / self.id
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        model_name = model.__class__.__name__
        # save the model
        torch.save(model.state_dict(), model_dir / f"{model_name}.pth")

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
        return self.env_handler.get_environment_specs()
        
