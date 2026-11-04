"""Shared preference readout and checkpoint utilities for TCPSA."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .tebc import Model as MemoryModel


READOUT_MODES = ("none", "residual", "uniform", "query")
QUERY_SOURCES = ("purchase-embedding", "purchase-representation")
CHECKPOINT_FORMAT = "frozen_purchase_readout_v1"


def validate_attention_weight(attention_weight, mode):
    """A fixed convex mixture is available only for query readouts."""
    if (not isinstance(attention_weight, (int, float)) or isinstance(attention_weight, bool)
            or not math.isfinite(attention_weight) or not 0 <= attention_weight <= 1):
        raise ValueError("attention_weight must be finite and in [0,1]")
    if mode != "query" and attention_weight != 1:
        raise ValueError("Nondefault attention_weight requires query readout mode")


def state_dict_sha256(state) -> str:
    """Fingerprint exact names, shapes, dtypes and values, independent of device."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("A model state fingerprint requires tensor values")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def relative_residual_penalty(delta, reference, valid_query=None, epsilon=1e-8):
    """Mean query-wise squared residual/base norm ratio; the base is detached."""
    if delta.ndim != 2 or reference.shape != delta.shape:
        raise ValueError("Residual and reference must be matching [batch, dim] tensors")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Residual penalty epsilon must be finite and positive")
    ratios = delta.square().sum(-1) / reference.detach().square().sum(-1).clamp_min(epsilon)
    if valid_query is None:
        valid_query = torch.ones(delta.size(0), dtype=torch.bool, device=delta.device)
    if valid_query.shape != delta.shape[:1]:
        raise ValueError("Residual query mask must have shape [batch]")
    valid = valid_query.to(device=delta.device, dtype=delta.dtype)
    return (ratios * valid).sum() / valid.sum().clamp_min(1)


class ReadoutAdapter(nn.Module):
    """Identical capacity controls differing only in historical pooling weights.

Projected states are tied keys and values. In residual/uniform mode the same
query projection is still used in the output fusion; no trainable parameter
is added merely to equalize the count. Query mode optionally anchors its
pool to each behavior's latest valid state with a fixed convex mixture.
    """

    def __init__(self, dim: int, mode: str = "query", rank: int = 8,
                 attention_weight: float = 1.0):
        super().__init__()
        if mode not in READOUT_MODES[1:]:
            raise ValueError("A readout head requires residual, uniform or query mode")
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Readout dimension must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("Readout rank must be a positive integer")
        validate_attention_weight(attention_weight, mode)
        self.dim, self.mode, self.rank = dim, mode, rank
        self.attention_weight = float(attention_weight)
        self.query_proj = nn.Linear(3 * dim, rank)
        self.state_proj = nn.Linear(dim, rank)
        self.fusion = nn.Sequential(nn.Linear(4 * rank, rank), nn.SiLU(), nn.Linear(rank, dim))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def pool(self, projected_states, masks, query):
        """Return [B,3,R] readouts and [B,3,L] actual pooling weights."""
        if (projected_states.ndim != 4 or projected_states.size(1) != 3
                or projected_states.size(-1) != self.rank or projected_states.size(2) < 1):
            raise ValueError("Projected states must have shape [batch, 3, nonempty length, rank]")
        if masks.shape != projected_states.shape[:3] or query.shape != (projected_states.size(0), self.rank):
            raise ValueError("Readout masks/query must match projected state dimensions")
        masks = masks.bool()
        if self.mode == "query":
            scores = (projected_states * query[:, None, None, :]).sum(-1) / math.sqrt(self.rank)
            available = masks.any(-1, keepdim=True)
            scores = scores.masked_fill(~masks, -torch.inf)
            # Do not softmax an all-masked row: unavailable memories are zero.
            scores = torch.where(available, scores, torch.zeros_like(scores))
            weights = F.softmax(scores, dim=-1) * masks.to(scores.dtype)
            if self.attention_weight < 1:
                # Anchor each branch to its last *valid* causal state, not
                # necessarily the final event of the mixed-behavior history.
                positions = torch.arange(1, masks.size(-1) + 1, device=masks.device)
                latest = (positions * masks).amax(-1, keepdim=True)
                latest_weights = (masks & positions.eq(latest)).to(weights.dtype)
                weights = ((1 - self.attention_weight) * latest_weights
                           + self.attention_weight * weights)
        elif self.mode == "uniform":
            weights = masks.to(projected_states.dtype) / masks.sum(-1, keepdim=True).clamp_min(1)
        else:
            positions = torch.arange(1, masks.size(-1) + 1, device=masks.device)
            latest = (positions * masks).amax(-1, keepdim=True)
            weights = (masks & positions.eq(latest)).to(projected_states.dtype)
        pooled = (projected_states * weights.unsqueeze(-1)).sum(-2)
        return pooled, weights

    def read(self, context, states, masks):
        """Inspect the three causal readouts before the zero-initialized output."""
        if context.ndim != 2 or context.size(-1) != 3 * self.dim:
            raise ValueError("Query context must have shape [batch, 3*dim]")
        if (states.ndim != 4 or states.size(0) != context.size(0)
                or states.size(1) != 3 or states.size(-1) != self.dim):
            raise ValueError("Behavior states must have shape [batch, 3, length, dim]")
        query = self.query_proj(context)
        projected = self.state_proj(states)
        pooled, weights = self.pool(projected, masks, query)
        return {"query": query, "pooled": pooled, "weights": weights,
                "projected_states": projected,
                "available": masks.bool().any(-1)}

    def forward(self, context, states, masks, valid_query=None):
        readout = self.read(context, states, masks)
        fused = torch.cat([readout["query"], readout["pooled"].flatten(1)], dim=-1)
        delta = self.fusion(fused)
        if valid_query is not None:
            if valid_query.shape != delta.shape[:1]:
                raise ValueError("Readout valid_query must have shape [batch]")
            delta = delta * valid_query.to(delta.dtype).unsqueeze(-1)
        return delta, readout


class PurchaseReadoutModel(nn.Module):
    """Wrap the current memory model; only ``adapter`` is trainable."""

    def __init__(self, base: MemoryModel, mode: str = "query", rank: int = 8,
                 query_source: str = "purchase-embedding", attention_weight: float = 1.0):
        super().__init__()
        if mode not in READOUT_MODES:
            raise ValueError("Readout mode must be none, residual, uniform or query")
        if query_source not in QUERY_SOURCES:
            raise ValueError("Unknown purchase query source")
        if query_source == "purchase-representation" and mode != "query":
            raise ValueError("purchase-representation query source requires query readout mode")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("Readout rank must be a positive integer")
        validate_attention_weight(attention_weight, mode)
        if getattr(base, "memory_calibration", None) != "evidence" or getattr(base, "memory_scope", None) != "purchase-only":
            raise ValueError("The frozen base must be an evidence/purchase-only memory model")
        self.base, self.mode, self.rank = base, mode, rank
        self.query_source = query_source
        self.attention_weight = float(attention_weight)
        self.base.requires_grad_(False)
        self.base.eval()
        self.base._collect_memory_statistics = False
        if mode == "none":
            self.adapter = None
        else:
            # Head creation must not shift the training sampler's global RNG.
            with torch.random.fork_rng(devices=[]):
                self.adapter = ReadoutAdapter(base.num_units, mode, rank,
                                              attention_weight=self.attention_weight)
            self.adapter.to(device=base.item_embedding.weight.device, dtype=base.item_embedding.weight.dtype)
        self._readout_sink = None
        self.base_config = None
        self.base_dimensions = {key: int(getattr(base, key)) for key in
                                ("num_user", "num_item", "num_cat", "num_seller", "num_brand")}
        self.source_checkpoint_path = None
        self.source_checkpoint_metadata = {}
        self.source_base_sha256 = state_dict_sha256(self.base.state_dict())

    @property
    def item_embedding(self):
        return self.base.item_embedding

    @property
    def num_units(self):
        return self.base.num_units

    @property
    def maxlen(self):
        return self.base.maxlen

    @property
    def readout_config(self):
        return {"mode": self.mode, "rank": self.rank, "dimension": self.num_units,
                "query_source": self.query_source,
                "attention_weight": self.attention_weight,
                "latest_anchor_weight": 1 - self.attention_weight if self.mode == "query" else None,
                "query_pooling": "latest_anchored_attention" if self.mode == "query" and self.attention_weight < 1
                else "attention" if self.mode == "query" else None,
                "latest_anchor_selection": "last_valid_state_per_behavior"
                if self.mode == "query" and self.attention_weight < 1 else None,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "location": "last_purchase_sequence_after_original_fusion",
                "query_conditioning": ["local", "attribute",
                                       "frozen_purchase_representation" if self.query_source == "purchase-representation"
                                       else "known_purchase_embedding"],
                "query_purchase_representation_location": "original_fused_sequence_before_readout_residual"
                if self.query_source == "purchase-representation" else None,
                "state_source": "original_behavior_encoder_outputs_before_memory_calibration",
                "tied_key_value_projection": True, "output_initialization": "zeros",
                "active_paths": ["auxiliary_purchase", "validation_purchase", "test_purchase"]
                if self.mode != "none" else [],
                "base_frozen": True, "base_mode": "eval", "extra_encoder_forwards": 0,
                "head_dropout": 0.0}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @contextmanager
    def capture_readout(self):
        """Capture current forward tensors for a query-scoped residual penalty."""
        previous = self._readout_sink
        captured = []
        self._readout_sink = captured
        try:
            yield captured
        finally:
            self._readout_sink = previous

    def _scaled_embedding(self, embedding, ids):
        with torch.no_grad():
            return self.base._scaled_embedding(embedding, ids)

    def forward(self, batch):
        # Retain the complete original main task as a read-only diagnostic.
        self.base.eval()
        with torch.no_grad():
            return self.base(batch)

    def _frozen_encode(self, batch, purchase_view, include_contrastive, capture_states):
        self.base.eval()
        captured, handles = {}, []

        def capture_branch(index):
            def capture(module, inputs, output):
                captured[index] = output
            return capture

        def capture_inputs(module, inputs, output):
            captured["inputs"] = output

        try:
            if capture_states:
                if any(not len(blocks) for blocks in self.base.behavior_blocks):
                    handles.append(self.base.item_input_norm.register_forward_hook(capture_inputs))
                for index, blocks in enumerate(self.base.behavior_blocks):
                    if len(blocks):
                        handles.append(blocks[-1].register_forward_hook(capture_branch(index)))
            with torch.no_grad():
                encoded = self.base._encode(batch, purchase_view, include_contrastive)
                if not capture_states:
                    return encoded, None, None, None
                masks = torch.stack([encoded[name] for name in ("purchase", "favorite", "view")], dim=1)
                states = torch.stack([
                    captured[index] if index in captured else
                    captured["inputs"] * masks[:, index].unsqueeze(-1)
                    for index in range(3)
                ], dim=1)
                # Validity masking also protects capture in the zero-block case.
                states = states * masks.unsqueeze(-1)
                local = encoded["local"][:, -1]
                attribute = encoded.get("attribute_context", torch.zeros_like(encoded["local"]))[:, -1]
                if self.query_source == "purchase-representation":
                    # The frozen BASE purchase view has already fused local,
                    # attribute and calibrated purchase-TBRE representations.
                    # Read it before adding this adapter's residual, never a
                    # held-out item or any recursively adapted representation.
                    purchase = encoded["sequence"][:, -1].detach()
                else:
                    purchase_ids = torch.zeros(local.size(0), dtype=torch.long, device=local.device)
                    purchase = self.base.behavior_embedding(purchase_ids)
                context = torch.cat([local, attribute, purchase], dim=-1)
            return encoded, states, masks, context
        finally:
            for handle in reversed(handles):
                handle.remove()

    def _encode(self, batch, purchase_view=False, include_contrastive=True):
        active = purchase_view and self.adapter is not None
        encoded, states, masks, context = self._frozen_encode(
            batch, purchase_view, include_contrastive, capture_states=active,
        )
        if not purchase_view:
            return encoded
        reference = encoded["sequence"][:, -1]
        valid_query = batch["input_id"][:, -1].ne(0)
        if active:
            delta, readout = self.adapter(context, states, masks, valid_query)
            result = dict(encoded)
            result["sequence"] = encoded["sequence"] + F.pad(delta.unsqueeze(1), (0, 0, encoded["sequence"].size(1) - 1, 0))
        else:
            delta = torch.zeros_like(reference)
            readout = {"weights": reference.new_zeros(reference.size(0), 3, batch["input_id"].size(1)),
                       "pooled": reference.new_zeros(reference.size(0), 3, self.rank)}
            result = encoded
        if self._readout_sink is not None:
            record = {"delta": delta, "reference": reference,
                      "valid_query": valid_query, "weights": readout["weights"],
                      "pooled": readout["pooled"]}
            if active:
                record.update(query=readout["query"], context=context,
                              projected_states=readout["projected_states"], masks=masks)
            self._readout_sink.append(record)
        return result

    def score_candidates(self, batch, candidate_ids):
        encoded = self._encode(batch, purchase_view=True, include_contrastive=False)
        if candidate_ids.ndim != 2 or candidate_ids.size(0) != encoded["sequence"].size(0):
            raise ValueError("candidate_ids must be [batch, candidates]")
        final = encoded["sequence"][:, -1]
        all_scores = final @ self.item_embedding.weight.T * math.sqrt(self.num_units)
        return all_scores.gather(1, candidate_ids.long())


def _checkpoint_object(checkpoint_or_path):
    if isinstance(checkpoint_or_path, Mapping):
        return dict(checkpoint_or_path), None
    path = Path(checkpoint_or_path).resolve()
    return torch.load(path, map_location="cpu", weights_only=True), str(path)


def from_base_checkpoint(checkpoint_or_path, mode="query", rank=8, device="cpu",
                         query_source="purchase-embedding", attention_weight=1.0):
    """Strictly load the entire current base before constructing a fresh head."""
    checkpoint, path = _checkpoint_object(checkpoint_or_path)
    for field in ("model_state_dict", "config", "model_dimensions"):
        if field not in checkpoint:
            raise ValueError(f"Base checkpoint is missing {field}")
    base = MemoryModel(**checkpoint["model_dimensions"], args=SimpleNamespace(**checkpoint["config"]))
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = PurchaseReadoutModel(base, mode, rank, query_source=query_source,
                                 attention_weight=attention_weight).to(device)
    model.base_config = dict(checkpoint["config"])
    model.base_dimensions = dict(checkpoint["model_dimensions"])
    model.source_checkpoint_path = path
    model.source_checkpoint_metadata = {key: checkpoint.get(key) for key in
                                        ("epoch", "metrics", "stage", "selection_metric")}
    model.source_base_sha256 = state_dict_sha256(checkpoint["model_state_dict"])
    return model


def export_adaptation_checkpoint(model, base_checkpoint, epoch, metrics, config,
                                 base_checkpoint_path=None, stage="best_validation"):
    """Embed all frozen and trainable weights; reload never needs the source file."""
    source, detected_path = _checkpoint_object(base_checkpoint)
    expected_hash = state_dict_sha256(source["model_state_dict"])
    current_hash = state_dict_sha256(model.base.state_dict())
    if current_hash != expected_hash or any(p.requires_grad for p in model.base.parameters()):
        raise RuntimeError("The pretrained base changed during frozen adaptation")
    if dict(source["model_dimensions"]) != model.base_dimensions:
        raise ValueError("Adaptation source has different model dimensions")
    source_path = str(Path(base_checkpoint_path).resolve()) if base_checkpoint_path is not None else (
        detected_path or model.source_checkpoint_path
    )
    return {"format": CHECKPOINT_FORMAT,
            "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "base_config": dict(source["config"]), "model_dimensions": dict(source["model_dimensions"]),
            "readout_config": model.readout_config, "base_state_sha256": current_hash,
            "base_checkpoint": {"path": source_path, **{key: source.get(key) for key in
                                 ("epoch", "metrics", "stage", "selection_metric")}},
            "epoch": int(epoch), "metrics": dict(metrics), "config": dict(config), "stage": stage}


def load_adaptation_checkpoint(checkpoint_or_path, device="cpu"):
    """Restore an exported adapter with strict shape/key/hash validation."""
    checkpoint, _ = _checkpoint_object(checkpoint_or_path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported purchase readout checkpoint format")
    required = ("model_state_dict", "base_config", "model_dimensions", "readout_config", "base_state_sha256")
    if any(field not in checkpoint for field in required):
        raise ValueError("Adaptation checkpoint is missing its self-contained model metadata")
    state = checkpoint["model_state_dict"]
    base_state = {name[len("base."):]: value for name, value in state.items() if name.startswith("base.")}
    source = {"model_state_dict": base_state, "config": checkpoint["base_config"],
              "model_dimensions": checkpoint["model_dimensions"],
              **{key: checkpoint.get("base_checkpoint", {}).get(key) for key in
                 ("epoch", "metrics", "stage", "selection_metric")}}
    readout = checkpoint["readout_config"]
    # Older v1 snapshots omitted query_source and/or attention_weight. Their
    # original fixed embedding and pure attention remain the loading defaults.
    model = from_base_checkpoint(source, mode=readout["mode"], rank=readout["rank"], device=device,
                                 query_source=readout.get("query_source", "purchase-embedding"),
                                 attention_weight=readout.get("attention_weight", 1.0))
    model.load_state_dict(state, strict=True)
    if state_dict_sha256(model.base.state_dict()) != checkpoint["base_state_sha256"]:
        raise ValueError("Adaptation checkpoint base fingerprint does not match")
    model.source_checkpoint_path = checkpoint.get("base_checkpoint", {}).get("path")
    model.eval()
    return model
