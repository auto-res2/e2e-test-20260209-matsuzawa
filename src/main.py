import os
import subprocess
import sys

import hydra


def apply_mode_overrides(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


@hydra.main(config_path="../config", config_name="config")
def main(cfg) -> None:
    apply_mode_overrides(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
