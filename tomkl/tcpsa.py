"""Target-Conditioned Preference Space Adaptation (TCPSA)."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from . import _readout as readout


METHOD = "frozen_purchase_conditioned_direction_adapter"
CHECKPOINT_FORMAT = "frozen_purchase_direction_adapter_v1"
FIXED_READOUT = {"mode": "query", "query_source": "purchase-representation", "attention_weight": 0.0}


class TCPSA(readout.ReadoutAdapter):
    """Purchase-conditioned fusion of the latest valid state in each branch.

The query projection conditions the residual MLP, rather than producing
attention weights. A missing behavior contributes an exactly zero readout.
Projection names/shapes/initialization match the previous lambda=0 head.
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__(dim, mode="query", rank=rank, attention_weight=0.0)

    def pool(self, projected_states, masks, query):
        if (projected_states.ndim != 4 or projected_states.size(1) != 3
                or projected_states.size(-1) != self.rank or projected_states.size(2) < 1):
            raise ValueError("Projected states must have shape [batch, 3, nonempty length, rank]")
        if masks.shape != projected_states.shape[:3] or query.shape != (projected_states.size(0), self.rank):
            raise ValueError("Readout masks/query must match projected state dimensions")
        masks = masks.bool()
        positions = torch.arange(1, masks.size(-1) + 1, device=masks.device)
        latest = (positions * masks).amax(-1, keepdim=True)
        weights = (masks & positions.eq(latest)).to(projected_states.dtype)
        return (projected_states * weights.unsqueeze(-1)).sum(-2), weights


class TOMKL(readout.PurchaseReadoutModel):
    """Freeze the entire BASE+BPR+purchase-only gate; train only the adapter."""

    def __init__(self, base, rank: int = 8):
        super().__init__(base, mode="query", rank=rank,
                         query_source="purchase-representation", attention_weight=0.0)
        # Preserve the old head's exact initialization and the sampling RNG.
        previous_state = self.adapter.state_dict()
        with torch.random.fork_rng(devices=[]):
            adapter = TCPSA(base.num_units, rank)
        adapter.to(device=base.item_embedding.weight.device, dtype=base.item_embedding.weight.dtype)
        adapter.load_state_dict(previous_state, strict=True)
        self.adapter = adapter

    @property
    def readout_config(self):
        config = super().readout_config
        config.update(
            method=METHOD,
            branch_pooling="last_valid_original_behavior_state",
            query_pooling="latest_state",
            attention_computed=False,
            query_role="purchase_condition_for_residual_fusion",
            tied_key_value_projection=False,
            shared_behavior_state_projection=True,
            residual_penalty=0.0,
            residual_type="additive_vector; no_norm_preserving_rotation_constraint",
        )
        return config


def _check_fixed(mode, query_source, attention_weight):
    if (mode != FIXED_READOUT["mode"] or query_source != FIXED_READOUT["query_source"]
            or attention_weight != FIXED_READOUT["attention_weight"]):
        raise ValueError("Direction adaptation fixes purchase-representation, latest states (lambda=0), and query fusion")


def from_base_checkpoint(checkpoint_or_path, mode="query", rank=8, device="cpu",
                         query_source="purchase-representation", attention_weight=0.0):
    _check_fixed(mode, query_source, attention_weight)
    previous = readout.from_base_checkpoint(checkpoint_or_path, mode=mode, rank=rank, device=device,
                                            query_source=query_source, attention_weight=attention_weight)
    model = TOMKL(previous.base, rank=rank)
    model.adapter.load_state_dict(previous.adapter.state_dict(), strict=True)
    for name in ("base_config", "base_dimensions", "source_checkpoint_path",
                 "source_checkpoint_metadata", "source_base_sha256"):
        setattr(model, name, getattr(previous, name))
    return model


def export_adaptation_checkpoint(model, base_checkpoint, epoch, metrics, config,
                                 base_checkpoint_path=None, stage="best_validation"):
    if not isinstance(model, TOMKL):
        raise TypeError("A direction checkpoint requires TOMKL")
    _check_fixed(model.mode, model.query_source, model.attention_weight)
    if config.get("residual_penalty", 0.0) != 0.0:
        raise ValueError("The finalized direction adapter fixes residual_penalty=0")
    checkpoint = readout.export_adaptation_checkpoint(
        model, base_checkpoint, epoch, metrics, config,
        base_checkpoint_path=base_checkpoint_path, stage=stage,
    )
    checkpoint["format"] = CHECKPOINT_FORMAT
    checkpoint["method"] = METHOD
    checkpoint["adapter_state_sha256"] = readout.state_dict_sha256(model.adapter.state_dict())
    return checkpoint


def load_adaptation_checkpoint(checkpoint_or_path, device="cpu"):
    checkpoint, _ = readout._checkpoint_object(checkpoint_or_path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("method") != METHOD:
        raise ValueError("Unsupported purchase direction adapter checkpoint format/method")
    required = ("model_state_dict", "base_config", "model_dimensions", "readout_config",
                "base_state_sha256", "adapter_state_sha256", "config")
    if any(name not in checkpoint for name in required):
        raise ValueError("Direction checkpoint is missing self-contained model metadata")
    for name in ("model_state_dict", "base_config", "model_dimensions", "readout_config", "config"):
        if not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"Direction checkpoint {name} must be a mapping")
    config = checkpoint["readout_config"]
    _check_fixed(config.get("mode"), config.get("query_source"), config.get("attention_weight"))
    if checkpoint["config"].get("residual_penalty", 0.0) != 0.0:
        raise ValueError("Direction checkpoint must use residual_penalty=0")
    state = checkpoint["model_state_dict"]
    if any(not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all()
           for tensor in state.values()):
        raise ValueError("Direction checkpoint contains invalid/non-finite model state")
    base_state = {name[5:]: value for name, value in state.items() if name.startswith("base.")}
    adapter_state = {name[8:]: value for name, value in state.items() if name.startswith("adapter.")}
    if (readout.state_dict_sha256(base_state) != checkpoint["base_state_sha256"]
            or readout.state_dict_sha256(adapter_state) != checkpoint["adapter_state_sha256"]):
        raise ValueError("Direction checkpoint base/adapter fingerprint does not match")
    source = {"model_state_dict": base_state, "config": checkpoint["base_config"],
              "model_dimensions": checkpoint["model_dimensions"],
              **{key: checkpoint.get("base_checkpoint", {}).get(key) for key in
                 ("epoch", "metrics", "stage", "selection_metric")}}
    model = from_base_checkpoint(source, rank=config.get("rank"), device=device)
    if dict(config) != model.readout_config:
        raise ValueError("Direction checkpoint readout configuration does not match the fixed implementation")
    model.load_state_dict(state, strict=True)
    model.source_checkpoint_path = checkpoint.get("base_checkpoint", {}).get("path")
    model.eval()
    return model
