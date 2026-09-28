"""Minimal lm-eval on a Megatron run directory.

    python playground/eval/lm_eval_megatron.py --load <run_dir> [--ckpt-step N] --bf16

Loads the latest checkpoint (or iter N), runs a fixed zero-shot task list, prints a table.
"""

import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lm_eval
from lm_eval.api.model import TemplateLM
from lm_eval.utils import make_table

from gpt_builders import gpt_builder
from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
from megatron.training.global_vars import set_global_variables
from model_provider import model_provider

# Everything we might ever run (download script pre-fetches all of these).
ALL_TASKS = [
    "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande",
    "boolq", "openbookqa", "sciq", "copa", "lambada_openai",
    "mmlu",      # slow, ~chance below 1B
    "wikitext",  # needs loglikelihood_rolling (not implemented yet)
]
# Default subset actually evaluated.
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]

# Architecture flags that --use-checkpoint-args does not restore; taken from run_config.json.
ARCH_KEYS = {"qk_layernorm", "attention_residuals", "vocab_size"}
ARCH_PREFIXES = ("attn_res_",)


class MegatronLM(TemplateLM):
    def __init__(self, model, tokenizer, max_length):
        super().__init__()
        self.model, self.tokenizer, self._max_length = model, tokenizer, max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eod

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.tokenize(string)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, **kwargs):
        out = []
        for _, ctx, cont in requests:
            toks = (ctx + cont)[-(self._max_length + 1):]
            inp = torch.tensor([toks[:-1]], device="cuda")
            pos = torch.arange(inp.shape[1], device="cuda").unsqueeze(0)
            logits = self.model(inp, pos, None)[0, -len(cont):, : self.tokenizer.vocab_size].float()
            logp = F.log_softmax(logits, -1)
            tgt = torch.tensor(toks[-len(cont):], device="cuda")
            out.append((logp.gather(1, tgt[:, None]).sum().item(), bool((logp.argmax(-1) == tgt).all())))
        return out

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    args, _ = load_args_from_checkpoint(args)  # standard arch + tokenizer from the checkpoint
    cfg_path = Path(args.load) / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found; cannot recover model flags")
    for k, v in json.load(open(cfg_path)).items():
        if k in ARCH_KEYS or k.startswith(ARCH_PREFIXES):
            setattr(args, k, v)
    validate_args(args, {"no_load_rng": True, "no_load_optim": True})
    set_global_variables(args)
    initialize_megatron()

    model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model, None, None)
    model = model[0].eval()

    lm = MegatronLM(model, get_tokenizer(), args.seq_length)
    results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0)
    print(f"iteration {iteration}")
    print(make_table(results))
