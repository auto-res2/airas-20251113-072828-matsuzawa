import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# -----------------------------------------------------------------------------
# Project-local imports (Hydra changes CWD → use absolute paths!)
# -----------------------------------------------------------------------------
from src.model import build_model, switch_quantisation
from src.preprocess import build_dataset, build_tokenizer, make_dataloader

CACHE_DIR = ".cache/"
PRIMARY_METRIC = "val_accuracy"  # 100 % identical name across train/evaluate
Path(CACHE_DIR).mkdir(exist_ok=True)

################################################################################
#                                Determinism                                   #
################################################################################

def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

################################################################################
#                           Carbon / Cost metering                             #
################################################################################

class CarbonMonitor:
    """Tiny GPU power → energy/CO₂/euro estimator (5 s sampling)."""

    def __init__(
        self,
        sample_interval_s: int = 5,
        power_meter: str = "nvidia_smi",
        carbon_intensity_g_per_kwh: float = 400.0,
        price_per_kwh_eur: float = 0.25,
    ):
        self.interval = sample_interval_s
        self.carbon_intensity = carbon_intensity_g_per_kwh
        self.price_kwh = price_per_kwh_eur

        self._energy_kwh = 0.0
        self._last = time.time()

        self._nvml_ready = False
        if power_meter == "nvidia_smi":
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False

    def _power(self) -> float:
        """Return instantaneous board power in Watt (falls back to constant)."""
        if self._nvml_ready:
            import pynvml  # type: ignore

            return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW → W
        return 300.0  # pessimistic CPU-only default

    def update(self):
        now = time.time()
        if now - self._last < self.interval:
            return
        dt = now - self._last
        self._energy_kwh += self._power() * dt / 3_600_000.0  # W·s → Wh → kWh
        self._last = now

    # ------------------ read-only properties ---------------------------------
    @property
    def energy_kwh(self):
        return self._energy_kwh

    @property
    def carbon_g(self):
        return self._energy_kwh * self.carbon_intensity

    @property
    def euro(self):
        return self._energy_kwh * self.price_kwh

################################################################################
#                   Gaussian-Process surrogate (tokens, lr, bs)                #
################################################################################

class TokenAccSurrogate:
    """GP surrogate used by C3PO-LRS to predict accuracy progression."""

    def __init__(self, beta: float = 1.0):
        kernel = 1.0 * RBF([1e4, 3e-4, 16.0]) + WhiteKernel(1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.X, self.y = [], []
        self.beta = beta
        self._fitted = False

    def add(self, tokens: int, lr: float, bs: int, acc: float):
        self.X.append([tokens, lr, bs])
        self.y.append(acc)
        self.gp.fit(np.asarray(self.X), np.asarray(self.y))
        self._fitted = True

    def predict_ucb(self, tokens: int, lr: float, bs: int):
        if not self._fitted:
            return 0.0
        mu, sigma = self.gp.predict([[tokens, lr, bs]], return_std=True)
        return float(mu + self.beta * sigma)

################################################################################
#                           Particle-Swarm MPC search                          #
################################################################################

def _pso_mpc(cfg: DictConfig, surrogate: TokenAccSurrogate, tokens_seen: int, carbon_left: float, euro_left: float, base_lr: float):
    """Return (lr_scaling, micro_batch, bits). Very light-weight PSO (≤10 ms)."""
    mb_choices = cfg.training.controller.micro_batch_choices
    bit_choices = cfg.training.controller.quantisation_choices
    euro_per_kwh = cfg.training.controller.get("euro_per_kwh", 0.25)

    # Objective under remaining-budget constraints ---------------------------
    def _objective(p: np.ndarray):
        lr_scal = float(p[0])
        mb = mb_choices[int(round(p[1]))]
        bits = bit_choices[int(round(p[2]))]

        future_tokens = mb * cfg.dataset.max_length * cfg.training.controller.solve_every_n_steps
        pred_acc = surrogate.predict_ucb(tokens_seen + future_tokens, base_lr * lr_scal, mb)

        # Simple linear power model ~8/bits ---------------------------------
        step_seconds = 0.5 * (8 / bits)
        energy_kwh = 0.25 * step_seconds * cfg.training.controller.solve_every_n_steps / 3600.0
        carbon_need = energy_kwh * 400.0
        euro_need = energy_kwh * euro_per_kwh
        if carbon_need > carbon_left or euro_need > euro_left:
            return -1.0  # infeasible
        return pred_acc

    bounds = np.asarray([(0.05, 1.0), (0, len(mb_choices) - 1), (0, len(bit_choices) - 1)])
    n_particles, n_iter = 16, 25
    part = np.vstack([[np.random.uniform(l, h) for l, h in bounds] for _ in range(n_particles)])
    vel = np.zeros_like(part)
    p_best = part.copy(); p_best_val = np.full(n_particles, -np.inf)

    for _ in range(n_iter):
        vals = np.array([_objective(p) for p in part])
        better = vals > p_best_val
        p_best[better] = part[better]; p_best_val[better] = vals[better]
        g_best = p_best[p_best_val.argmax()]
        w, c1, c2 = 0.5, 0.8, 1.2
        r1, r2 = np.random.rand(*part.shape), np.random.rand(*part.shape)
        vel = w * vel + c1 * r1 * (p_best - part) + c2 * r2 * (g_best - part)
        part = np.clip(part + vel, bounds[:, 0], bounds[:, 1])

    lr_scal, mb_idx, bit_idx = p_best[p_best_val.argmax()]
    return float(lr_scal), mb_choices[int(round(mb_idx))], bit_choices[int(round(bit_idx))]

################################################################################
#                                LR controllers                                #
################################################################################

class _LRController:
    def __init__(self, optimiser: torch.optim.Optimizer, base_lr: float):
        self.opt = optimiser
        self.base_lr = base_lr
        self._scal = 1.0

    @property
    def lr_scaling(self):
        return self._scal

    def set_scaling(self, scal: float):
        self._scal = max(scal, 1e-8)
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * self._scal

class C3POController(_LRController):
    def __init__(self, cfg: DictConfig, optimiser: torch.optim.Optimizer, surrogate: TokenAccSurrogate, carbon: CarbonMonitor, dataloaders: Dict[int, DataLoader], model_state: Dict[str, Any]):
        super().__init__(optimiser, cfg.training.base_learning_rate)
        self.cfg, self.surr, self.carbon = cfg, surrogate, carbon
        self.dataloaders = dataloaders
        self.state = model_state  # current quant bits
        self._last = 0

    def maybe_solve(self, step: int, tokens_seen: int):
        if step < self.cfg.training.controller.warm_start_steps:
            return None, self.state["bits"]
        if (step - self._last) < self.cfg.training.controller.solve_every_n_steps:
            return None, self.state["bits"]
        self._last = step

        lr_scal, mb, bits = _pso_mpc(
            self.cfg,
            self.surr,
            tokens_seen,
            self.cfg.training.controller.carbon_budget_g - self.carbon.carbon_g,
            self.cfg.training.controller.money_budget_eur - self.carbon.euro,
            self.base_lr,
        )
        self.set_scaling(lr_scal)
        return self.dataloaders[mb], bits

class BLuRCANAController(_LRController):
    def __init__(self, cfg: DictConfig, optimiser: torch.optim.Optimizer, carbon: CarbonMonitor):
        base_lr = cfg.training.get("learning_rate", cfg.training.base_learning_rate)
        super().__init__(optimiser, base_lr)
        self.cfg, self.carbon = cfg, carbon

    def maybe_step(self):
        limit = self.cfg.training.reactive_controller.carbon_budget_g - self.cfg.training.reactive_controller.overshoot_margin_g
        if self.carbon.carbon_g > limit:
            new_scal = max(self.lr_scaling * self.cfg.training.lr_decay_gamma, self.cfg.training.min_lr_scaling)
            self.set_scaling(new_scal)

################################################################################
#                          Validation / evaluation helper                      #
################################################################################

def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, tokenizer):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            gen = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=32)
            for p_ids, t_ids in zip(gen, batch["labels"]):
                p = tokenizer.decode(p_ids.tolist(), skip_special_tokens=True).split("Answer:")[-1].strip()
                t = tokenizer.decode(t_ids.tolist(), skip_special_tokens=True).split("Answer:")[-1].strip()
                preds.append(p); labels.append(t)
    acc = float(np.mean([int(p == t) for p, t in zip(preds, labels)]))
    model.train(); return acc, preds, labels

################################################################################
#                            Optuna utilities                                  #
################################################################################

def _inject(cfg: DictConfig, dotted: str, value: Any):
    node = cfg; *parents, last = dotted.split(".")
    for p in parents:
        node = node[p]
    node[last] = value

################################################################################
#                              MAIN experiment                                 #
################################################################################

def run_experiment(cfg: DictConfig):
    _set_seed(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------ WandB initialisation --------------------------
    wb_run = None
    if cfg.wandb.mode != "disabled":
        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ------------------------ Data pipeline ---------------------------------
    tokenizer = build_tokenizer(cfg)
    train_ds, val_ds = build_dataset(cfg, tokenizer, mode=cfg.mode)

    # Dataloaders for candidate micro-batch sizes ----------------------------
    if cfg.method.startswith("C3PO"):
        loaders = {bs: make_dataloader(train_ds, bs, tokenizer) for bs in cfg.training.controller.micro_batch_choices}
        cur_bs = cfg.training.controller.micro_batch_choices[0]
    else:
        bs = cfg.training.micro_batch_size
        loaders = {bs: make_dataloader(train_ds, bs, tokenizer)}
        cur_bs = bs
    train_loader = loaders[cur_bs]; train_iter = iter(train_loader)
    val_loader = make_dataloader(val_ds, cfg.training.validation.get("batch_size", 8), tokenizer, shuffle=False)

    # ------------------------ Model / optimiser / scheduler -----------------
    init_bits = cfg.model.quantisation.get("default_bits", cfg.model.quantisation.get("bits", 8))
    model = build_model(cfg, bits=init_bits).to(device)

    base_lr = cfg.training.get("base_learning_rate", cfg.training.get("learning_rate"))
    optimiser = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(0.05 * cfg.training.max_steps),
        num_training_steps=cfg.training.max_steps,
    )

    # ------------------------ Aux objects -----------------------------------
    surrogate = TokenAccSurrogate(beta=cfg.training.controller.get("ucb_beta", 1.0) if cfg.method.startswith("C3PO") else 1.0)
    carbon = CarbonMonitor(**cfg.training.carbon_monitor)
    model_state = {"bits": init_bits}

    if cfg.method.startswith("C3PO"):
        controller = C3POController(cfg, optimiser, surrogate, carbon, loaders, model_state)
    else:
        controller = BLuRCANAController(cfg, optimiser, carbon)

    # ------------------------ Training loop ---------------------------------
    best_acc = 0.0; tokens_seen, tokens_since_val = 0, 0
    for step in range(1, cfg.training.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
        optimiser.step(); scheduler.step(); optimiser.zero_grad(set_to_none=True)

        batch_tokens = batch["input_ids"].numel()
        tokens_seen += batch_tokens; tokens_since_val += batch_tokens
        carbon.update()

        # -------- logging ----------------------------------------------------
        if wb_run:
            wandb.log({"train_loss": loss.item(), "lr": optimiser.param_groups[0]["lr"], "carbon_used_g": carbon.carbon_g, "euro_spent": carbon.euro}, step=step)

        # -------- controller interaction ------------------------------------
        if isinstance(controller, C3POController):
            maybe_loader, maybe_bits = controller.maybe_solve(step, tokens_seen)
            if maybe_loader and maybe_loader.batch_size != train_loader.batch_size:
                train_loader, train_iter, cur_bs = maybe_loader, iter(maybe_loader), maybe_loader.batch_size
            if maybe_bits != model_state["bits"]:
                model, optimiser, scheduler = switch_quantisation(
                    model,
                    cfg,
                    maybe_bits,
                    optimiser,
                    scheduler,
                    base_lr,
                    device,
                )
                controller.opt = optimiser  # sync controller
                model_state["bits"] = maybe_bits
        else:
            controller.maybe_step()

        # -------- periodic validation ---------------------------------------
        if tokens_since_val >= cfg.training.validation.every_n_tokens or step == cfg.training.max_steps:
            val_acc, v_preds, v_lbls = _evaluate(model, val_loader, device, tokenizer)
            best_acc = max(best_acc, val_acc)
            if wb_run:
                wandb.log({PRIMARY_METRIC: val_acc}, step=step)
                wandb.summary[PRIMARY_METRIC] = best_acc
                wandb.summary["val_predictions"], wandb.summary["val_labels"] = v_preds, v_lbls
                labels_unique = sorted(set(v_preds + v_lbls))[:100]
                cm = confusion_matrix(v_lbls, v_preds, labels=labels_unique)
                wandb.summary["confusion_matrix"] = {"labels": labels_unique, "matrix": cm.tolist()}
            surrogate.add(tokens_seen, optimiser.param_groups[0]["lr"], cur_bs, val_acc)
            tokens_since_val = 0

        # -------- ultra-light trial mode ------------------------------------
        if cfg.mode == "trial" and step >= 2:
            break

    # ------------------------ wrap-up ---------------------------------------
    if wb_run:
        wandb.summary["carbon_used_g"] = carbon.carbon_g
        wandb.summary["euro_spent"] = carbon.euro
        print(f"WandB URL → {wb_run.url}")
        wandb.finish()

    return {PRIMARY_METRIC: best_acc}

################################################################################
#                               Optuna driver                                  #
################################################################################

def _run_optuna(cfg: DictConfig):
    if cfg.optuna.n_trials == 0:
        return

    space = cfg.optuna.search_space

    def _suggest(trial: optuna.Trial, name: str, spec: Dict[str, Any]):
        t = spec["type"].lower()
        if t == "uniform":
            return trial.suggest_float(name, spec["low"], spec["high"], log=False)
        if t == "loguniform":
            return trial.suggest_float(name, spec["low"], spec["high"], log=True)
        if t == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], log=False)
        raise ValueError(f"Unknown Optuna type {t}")

    def objective(trial: optuna.Trial):
        params = {k: _suggest(trial, k, v) for k, v in space.items()}
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.wandb.mode = "disabled"
        sub_cfg.optuna.n_trials = 0
        OmegaConf.update(sub_cfg, "training.max_steps", max(50, int(cfg.training.max_steps * 0.1)), merge=False)
        OmegaConf.update(sub_cfg, "run.run_id", f"{cfg.run.run_id}-optuna-{trial.number}", merge=False)
        for k, v in params.items():
            _inject(sub_cfg, k, v)
        res = run_experiment(sub_cfg)
        return res[PRIMARY_METRIC]

    direction = cfg.optuna.direction.lower()
    study = optuna.create_study(direction="maximize" if direction.startswith("max") else "minimize")
    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    for k, v in study.best_params.items():
        _inject(cfg, k, v)

################################################################################
#                                     CLI                                      #
################################################################################

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig):
    # ----------------- mode-specific overrides -----------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        # Temporarily disable struct mode to allow updates
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, "training.max_steps", 2, merge=False)
        OmegaConf.update(cfg, "training.validation.every_n_tokens", cfg.dataset.max_length * 2, merge=False)
        OmegaConf.set_struct(cfg, True)
    else:
        cfg.wandb.mode = "online"

    _run_optuna(cfg)
    run_experiment(cfg)

if __name__ == "__main__":
    hydra_entry()