import re
from typing import Tuple

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

CACHE_DIR = ".cache/"

################################################################################
#                            Tokeniser builder                                 #
################################################################################

def build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

################################################################################
#                       Dataset preprocessing (GSM8K)                          #
################################################################################

def _strip_calc(ans: str) -> str:
    m = re.search(r"#### ([-+]?\d[\d,]*)", ans)
    return m.group(1).replace(",", "") if m else ans.strip()


def _format_ex(example, remove_calc: bool = False):
    q, a = example["question"].strip(), example["answer"].strip()
    if remove_calc:
        a = _strip_calc(a)
    return {"text": f"Question: {q}\nAnswer: {a}"}


def build_dataset(cfg, tokenizer, *, mode="full") -> Tuple[Dataset, Dataset]:
    raw = load_dataset(cfg.dataset.name, cfg.dataset.config, cache_dir=CACHE_DIR)
    splits = raw["train"].train_test_split(test_size=1 - cfg.dataset.split_ratio.train, seed=42)
    train, val = splits["train"], splits["test"]

    train = train.map(lambda ex: _format_ex(ex, cfg.dataset.preprocessing.remove_calculation_annotations))
    val = val.map(lambda ex: _format_ex(ex, cfg.dataset.preprocessing.remove_calculation_annotations))

    # trial-mode subsampling --------------------------------------------------
    if mode == "trial":
        train = train.select(range(min(40, len(train))))
        val = val.select(range(min(40, len(val))))

    def _tokenise(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=cfg.dataset.max_length)
        enc["labels"] = enc["input_ids"].copy(); return enc

    train = train.map(_tokenise, batched=True, remove_columns=train.column_names).with_format("torch")
    val = val.map(_tokenise, batched=True, remove_columns=val.column_names).with_format("torch")
    return train, val

################################################################################
#                               DataLoader util                                #
################################################################################

def make_dataloader(ds: Dataset, batch_size: int, tokenizer, shuffle: bool = True):
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)