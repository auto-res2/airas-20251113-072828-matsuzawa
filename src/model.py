from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup

CACHE_DIR = ".cache/"

################################################################################
#                        BitsAndBytes helper                                   #
################################################################################

def _bnb(bits: int | None):
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

################################################################################
#                              Model builder                                   #
################################################################################

def build_model(cfg, *, bits: int = 8):
    q_cfg = _bnb(bits)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=CACHE_DIR,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=q_cfg,
    )
    # ----------------------------- PEFT / LoRA ------------------------------
    if cfg.model.peft.technique == "lora":
        lora_cfg = LoraConfig(r=cfg.model.peft.r, lora_alpha=cfg.model.peft.alpha, lora_dropout=cfg.model.peft.dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
    return model

################################################################################
#                         Quantisation switcher                                #
################################################################################

def switch_quantisation(
    current_model: torch.nn.Module,
    cfg,
    bits: int,
    optimiser: torch.optim.Optimizer,
    scheduler,
    base_lr: float,
    device,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
    """Change quantisation *without* losing progress.

    1. Save weights to CPU.
    2. Re-instantiate model with new quant level and load weights.
    3. Re-create optimiser (AdamW) and LR-scheduler (linear warm-up/decay).
    """

    # ---------------- 1) Preserve weights -----------------------------------
    state_dict = {k: v.detach().cpu() for k, v in current_model.state_dict().items()}
    del current_model; torch.cuda.empty_cache()

    # ---------------- 2) Build new model ------------------------------------
    new_model = build_model(cfg, bits=bits)
    missing, unexpected = new_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[switch_quantisation] missing keys: {len(missing)}")
    if unexpected:
        print(f"[switch_quantisation] unexpected keys: {len(unexpected)}")
    new_model.to(device)

    # ---------------- 3) Optimiser & scheduler ------------------------------
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=base_lr, weight_decay=cfg.training.weight_decay)

    if scheduler is not None:
        last_epoch = scheduler.last_epoch
        warm = int(0.05 * cfg.training.max_steps)
        total = cfg.training.max_steps
        new_sch = get_linear_schedule_with_warmup(new_opt, num_warmup_steps=warm, num_training_steps=total, last_epoch=last_epoch)
    else:
        new_sch = None

    return new_model, new_opt, new_sch