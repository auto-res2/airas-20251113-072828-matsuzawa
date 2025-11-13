import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

################################################################################
#                         Experiment orchestrator                              #
################################################################################

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # Handle case where cfg.run might be a string or an object with run_id
    run_id = cfg.run if isinstance(cfg.run, str) else cfg.run.run_id

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    if cfg.mode == "trial":
        cmd += ["wandb.mode=disabled", "optuna.n_trials=0", "training.max_steps=2"]
    else:
        cmd += ["wandb.mode=online"]

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()