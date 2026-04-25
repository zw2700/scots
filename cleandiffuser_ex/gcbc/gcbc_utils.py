import pickle
from typing import Any, Dict

import flax.core
import flax.serialization
import jax
import jax.numpy
import numpy as np
import torch

from cleandiffuser_ex.gciql.gciql_utils import convert_jax_mlp_to_torch_mlp_modulelist, convert_flax_dense_to_torch_linear


def unfreeze_and_npify(pytree: Any) -> Any:
    def map_leaf(leaf):
        if isinstance(leaf, (jax.Array, jax.numpy.ndarray)):
            return np.asarray(jax.device_get(leaf))
        return leaf

    np_pytree = jax.tree_util.tree_map(map_leaf, pytree)
    try:
        unfrozen_pytree = flax.serialization.from_state_dict(np_pytree, np_pytree)
        unfrozen_pytree = jax.tree_util.tree_map(
            lambda x: dict(x) if isinstance(x, flax.core.FrozenDict) else x,
            unfrozen_pytree,
            is_leaf=lambda x: isinstance(x, flax.core.FrozenDict),
        )
    except Exception:
        unfrozen_pytree = jax.tree_util.tree_map(
            lambda x: dict(x) if isinstance(x, flax.core.FrozenDict) else x,
            np_pytree,
            is_leaf=lambda x: isinstance(x, flax.core.FrozenDict),
        )
    return unfrozen_pytree


def load_gcbc_jax_checkpoint_to_pytorch(
    jax_checkpoint_path: str,
    pytorch_agent: Any,
):
    with open(jax_checkpoint_path, 'rb') as f:
        raw_loaded_dict = pickle.load(f)

    if not (
        'agent' in raw_loaded_dict
        and isinstance(raw_loaded_dict['agent'], (dict, flax.core.FrozenDict))
        and 'network' in raw_loaded_dict['agent']
        and isinstance(raw_loaded_dict['agent']['network'], (dict, flax.core.FrozenDict))
        and 'params' in raw_loaded_dict['agent']['network']
    ):
        raise ValueError("JAX checkpoint structure does not match expected 'agent.network.params' path.")

    jax_params = unfreeze_and_npify(raw_loaded_dict['agent']['network']['params'])
    actor_module_params = jax_params.get('modules_actor')
    if actor_module_params is None:
        raise KeyError("Missing 'modules_actor' in JAX params")

    actor_net_params = actor_module_params.get('actor_net')
    mean_net_params = actor_module_params.get('mean_net')
    if actor_net_params is None or mean_net_params is None:
        raise KeyError("Expected 'actor_net' and 'mean_net' within 'modules_actor'")

    final_pytorch_state_dict: Dict[str, Any] = {}
    final_pytorch_state_dict.update(
        convert_jax_mlp_to_torch_mlp_modulelist(
            actor_net_params,
            pytorch_agent.actor.actor_net.layers,
            "actor.actor_net.layers.",
        )
    )

    for name, tensor in convert_flax_dense_to_torch_linear(mean_net_params).items():
        final_pytorch_state_dict[f"actor.mean_net.{name}"] = tensor

    if 'log_stds' in actor_module_params and hasattr(pytorch_agent.actor, 'log_stds_param'):
        final_pytorch_state_dict["actor.log_stds_param"] = torch.from_numpy(
            np.array(actor_module_params['log_stds'])
        ).float()

    pytorch_agent.load_state_dict(final_pytorch_state_dict, strict=False)
