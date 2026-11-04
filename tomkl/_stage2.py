"""Stage 2: fit TCPSA with the shared foundation frozen."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

from . import _adapter_training as readout_train
from .tcpsa import (
    METHOD, from_base_checkpoint, export_adaptation_checkpoint, load_adaptation_checkpoint,
)
from ._stage1 import temporary_bindings


LOGGER = readout_train.LOGGER
check_data_dimensions = readout_train.check_data_dimensions
state_sha256 = readout_train.state_sha256
_READOUT_PARSE_ARGS = readout_train.parse_args
_LOAD_BASE_MODEL = readout_train.load_base_model
_LOAD_READOUT_CHECKPOINT = readout_train.load_readout_checkpoint
source_hashes = readout_train.source_hashes
_JSON_CONFIG = readout_train.json_config
FIXED_ARGUMENTS = ("--readout_mode", "query", "--query_source", "purchase-representation",
                   "--attention_weight", "0", "--residual_penalty", "0")


def parse_args(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    # Expose only meaningful controls. No abbreviations or alternate lambda/mu
    # settings can silently turn this finalized method into another experiment.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base_checkpoint", required=True, type=Path)
    for name in ("rank", "num_epochs", "patience", "batch_size", "eval_batch_size",
                 "random_seed", "purchase_min_history", "purchase_hard_topk"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("lr", "aug_weight", "purchase_hard_ratio"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("device", "data_path", "save_dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--save_extra_validation_checkpoints", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.parse_args(values)
    args = _READOUT_PARSE_ARGS([*values, *FIXED_ARGUMENTS])
    args.method = METHOD
    if args.save_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        args.save_dir = Path(__file__).resolve().parent / "results" / f"PDA_s{args.random_seed}_{stamp}"
    return args


def load_base_model(args, device):
    with temporary_bindings(readout_train, from_base_checkpoint=from_base_checkpoint, source_hashes=source_hashes):
        model, checkpoint, metadata, base_args = _LOAD_BASE_MODEL(args, device)
    expected = {"aug_objective": "purchase", "purchase_loss": "bpr", "purchase_history": "remove-target",
                "memory_calibration": "evidence", "memory_scope": "purchase-only"}
    if any(checkpoint["config"].get(name) != value for name, value in expected.items()):
        raise ValueError("The finalized three-module foundation requires purchase BPR/remove-target and evidence/purchase-only")
    metadata.update(
        method=METHOD, readout_config=model.readout_config,
        foundation_components=expected,
        fixed_direction_design={"query_source": "purchase-representation", "branch_pooling": "latest_valid",
                                "attention_weight": 0.0, "residual_penalty": 0.0, "attention_computed": False},
        training_objective="aug_weight * purchase_BPR; adapter_only",
        training_budget="completed_source_checkpoint + fresh_adapter_epochs; no_BASE_resume",
        direction_diagnostics="latest-state weights; no_attention_logits; legacy_logit_sd_zero_is_inapplicable",
    )
    return model, checkpoint, metadata, base_args


def load_readout_checkpoint(path, device, data_path=None):
    with temporary_bindings(readout_train, load_adaptation_checkpoint=load_adaptation_checkpoint):
        return _LOAD_READOUT_CHECKPOINT(path, device, data_path)


class LatestStateDiagnostics(readout_train.ReadoutDiagnostics):
    """Retain history columns, without computing unused query/key logits.

The legacy attention_* columns describe the actual one-hot latest-state
pool. Entropy is zero; logit_sd is zero and explicitly marked inapplicable.
    """

    @torch.no_grad()
    def update(self, record):
        valid = record["valid_query"].bool()
        if not valid.any():
            return
        query, context = record["query"].detach()[valid], record["context"].detach()[valid]
        projected, masks = record["projected_states"].detach()[valid], record["masks"].bool()[valid]
        weights = record["weights"].detach()[valid]
        self.queries += query.size(0)
        self.query_norm += float(query.norm(dim=-1).sum())
        self.centered_norm += float((query - query.mean(0)).norm(dim=-1).sum())
        self.condition_norm += float(context[:, 2 * (context.size(-1) // 3):].norm(dim=-1).sum())
        counts, norms = masks.sum(-1), projected.norm(dim=-1)
        for index, branch in enumerate(readout_train.READOUT_BRANCHES):
            aggregate = self.branches[branch]
            multi = counts[:, index].gt(1)
            aggregate["zero"] += int(counts[:, index].eq(0).sum())
            aggregate["single"] += int(counts[:, index].eq(1).sum())
            aggregate["multi"] += int(multi.sum())
            aggregate["latest"] += float(weights[:, index].sum(-1)[multi].sum())
            aggregate["key_norm"] += float(norms[:, index][masks[:, index]].sum())
            aggregate["states"] += int(counts[:, index].sum())


def main(argv=None):
    args = parse_args(argv)
    run_metadata = {}

    def load_with_metadata(*values, **kwargs):
        result = load_base_model(*values, **kwargs)
        run_metadata["value"] = result[2]
        return result

    def direction_json_config(options):
        # The shared trainer adds attention-era descriptions after loading.
        # Normalize them before its first log/config/checkpoint serialization.
        # This reference is local to this run and disappears with the scope.
        metadata = run_metadata["value"]
        metadata.update(
            training_objective="aug_weight * purchase_BPR; adapter_only",
            readout_diagnostics_scope="training_auxiliary_queries; latest-state pool; no_extra_encoder_forward",
            readout_diagnostics_conditioning="latest_mass requires behavior_count>1; query centering is per batch",
            readout_diagnostics_weights="one-hot last valid state per behavior; zero for absent behavior; entropy=0",
            readout_diagnostics_logits="inapplicable; no query-key logits computed; legacy logit_sd column is zero",
        )
        return _JSON_CONFIG(options)

    with temporary_bindings(
        readout_train, parse_args=lambda argv=None: args,
        from_base_checkpoint=from_base_checkpoint,
        export_adaptation_checkpoint=export_adaptation_checkpoint,
        load_adaptation_checkpoint=load_adaptation_checkpoint,
        load_base_model=load_with_metadata, load_readout_checkpoint=load_readout_checkpoint,
        source_hashes=source_hashes, ReadoutDiagnostics=LatestStateDiagnostics,
        json_config=direction_json_config,
    ):
        return readout_train.main()
