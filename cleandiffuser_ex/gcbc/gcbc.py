from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser_ex.gciql.gciql import GCActor_torch


class GCBC(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        actor_hidden_dims: Sequence[int] = (512, 512, 512),
        const_std_actor: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim if goal_dim is not None else self.obs_dim
        self.device = device

        self.actor = GCActor_torch(
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            use_layer_norm=False,
            const_std=const_std_actor,
            state_dependent_std=False,
            tanh_squash=True,
            activations=nn.GELU,
        ).to(device)

    @torch.no_grad()
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)

        obs_tensor = torch.from_numpy(observations).to(self.device).float()
        goal_tensor = torch.from_numpy(goals).to(self.device).float()

        is_single_input = obs_tensor.ndim == 1
        if is_single_input:
            obs_tensor = obs_tensor.unsqueeze(0)
            goal_tensor = goal_tensor.unsqueeze(0)

        actor_temp_for_dist = temperature if temperature > 0.0 else 1e-6
        actor_dist = self.actor(obs_tensor, goal_tensor, temperature=actor_temp_for_dist)

        if temperature == 0.0:
            actions = actor_dist.mean
            if self.actor.tanh_squash:
                actions = torch.tanh(actions)
        else:
            actions = actor_dist.sample()

        actions_np = np.clip(actions.cpu().numpy(), -1.0, 1.0)
        return actions_np.squeeze(0) if is_single_input else actions_np

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
