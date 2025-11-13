import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

################################################################################
#                         Experiment orchestrator                              #
################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run.run_id}",
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