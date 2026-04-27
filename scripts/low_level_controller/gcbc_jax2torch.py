import os
os.environ['MUJOCO_GL'] = 'egl'
import glob
import dataclasses

import ogbench
import pyrallis

from cleandiffuser_ex.gcbc import GCBC, load_gcbc_jax_checkpoint_to_pytorch


@dataclasses.dataclass
class Config:
    env_name: str = "pointmaze-large-navigate-v0"
    save_dir: str = 'exp/'
    restore_epoch: int = 1000000


@pyrallis.wrap()
def main(config: Config):
    if 'checker' in config.env_name:
        import ogbench_ex
        env, dataset, _ = ogbench_ex.make_env_and_datasets(
            config.env_name,
            compact_dataset=False,
        )
    else:
        env, dataset, _ = ogbench.make_env_and_datasets(
            config.env_name,
            compact_dataset=False,
        )

    del dataset
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    gcbc = GCBC(obs_dim, act_dim, device="cuda")

    candidates = glob.glob(f'./exp/gcbc-{config.env_name}')
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{config.restore_epoch}.pkl'
    load_gcbc_jax_checkpoint_to_pytorch(restore_path, gcbc)

    if os.path.exists(config.save_dir) is False:
        os.makedirs(config.save_dir)

    gcbc.save(os.path.join(config.save_dir, f"gcbc_ckpt_{config.restore_epoch}.pt"))
    gcbc.save(os.path.join(config.save_dir, "gcbc_ckpt_latest.pt"))


if __name__ == '__main__':
    main()
