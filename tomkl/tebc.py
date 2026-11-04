"""Target-Evidence-Adaptive Behavioral Context Modeling (TEBC)."""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn

from .encoder import Model as BaseModel


BEHAVIOR_CODES = (0, 2, 3)
FEATURE_NAMES = (
    "availability", "log_count", "density", "age",
    "to_purchase", "to_favorite", "to_view",
    "category_coverage", "seller_coverage", "brand_coverage",
)


def masked_memory_anchor(scales, evidence, query_mask):
    """Mean squared gain deviation over supervised, available memories only."""
    if evidence.shape[:3] != scales.shape or query_mask.shape != scales.shape[:2]:
        raise ValueError("Anchor evidence/query mask must match [B,L,3] scales")
    available = evidence[..., 0].bool() & query_mask.bool().unsqueeze(-1)
    return ((scales - 1).square() * available).sum() / available.sum().clamp_min(1)


def history_evidence(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [B,L,3,10] statistics from each position's observed prefix only.

    Distances use event steps, not time intervals. Transition frequencies are
    counts divided by all consecutive-valid-event pairs observed so far.
    Padding, target labels and target attributes never supply evidence.
    """
    items, behaviors = batch["input_id"], batch["behavior_id"]
    if items.ndim != 2 or behaviors.shape != items.shape or items.size(1) < 1:
        raise ValueError("Memory evidence requires matching nonempty [B,L] inputs")
    valid = items.ne(0)
    steps = valid.long().cumsum(1)
    denominator = steps.clamp_min(1).float()
    masks = torch.stack([valid & behaviors.eq(code) for code in BEHAVIOR_CODES], dim=-1)
    counts = masks.float().cumsum(1)
    availability = counts.gt(0).float()
    latest_steps = (steps.unsqueeze(-1) * masks).cummax(1).values
    ages = (steps.unsqueeze(-1) - latest_steps).float() / denominator.unsqueeze(-1)
    ages = torch.where(availability.bool(), ages, torch.ones_like(ages))

    # Find the previous real event, also supporting internal padding holes.
    positions = torch.arange(1, items.size(1) + 1, device=items.device)
    latest_positions = (positions.unsqueeze(0) * valid).cummax(1).values
    previous_positions = torch.cat([torch.zeros_like(latest_positions[:, :1]), latest_positions[:, :-1]], dim=1)
    padded_behaviors = torch.cat([torch.full_like(behaviors[:, :1], 4), behaviors], dim=1)
    previous_behaviors = padded_behaviors.gather(1, previous_positions)
    pairs = valid & previous_positions.gt(0)
    pair_denominator = pairs.float().cumsum(1).clamp_min(1)
    transitions = []
    for destination in BEHAVIOR_CODES:
        values = torch.stack([
            pairs & previous_behaviors.eq(origin) & behaviors.eq(destination)
            for origin in BEHAVIOR_CODES
        ], dim=-1).float().cumsum(1)
        transitions.append(values / pair_denominator.unsqueeze(-1))

    coverage = []
    for name in ("cat_id", "seller_id", "brand_id"):
        ids = batch.get(name)
        if ids is not None and ids.shape != items.shape:
            raise ValueError(f"Memory evidence {name} must match input_id")
        observed = torch.zeros_like(valid) if ids is None else valid & ids.ne(0)
        fraction = observed.float().cumsum(1) / denominator
        coverage.append(fraction.unsqueeze(-1).expand_as(counts))
    features = torch.stack([
        availability, counts.log1p(), counts / denominator.unsqueeze(-1), ages,
        *transitions, *coverage,
    ], dim=-1)
    return features * valid[:, :, None, None]


class TEBC(nn.Module):
    """Small, zero-initialized residual gain on three existing memories."""

    def __init__(self, dim: int, mode: str = "evidence", gain: float = 0.1, hidden: int = 16):
        super().__init__()
        if mode not in ("evidence", "plain"):
            raise ValueError("Calibrator mode must be evidence or plain")
        if not math.isfinite(gain) or not 0 <= gain < 1:
            raise ValueError("memory_gain must be finite and in [0,1)")
        if hidden < 1 or dim < 1:
            raise ValueError("memory_hidden and dimension must be positive")
        self.dim, self.mode, self.gain = int(dim), mode, float(gain)
        # Plain uses identical capacity, with explicit history statistics zeroed.
        self.network = nn.Sequential(
            nn.Linear(6 * dim + 3 * len(FEATURE_NAMES), hidden),
            nn.SiLU(), nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def compute_scales(self, stacked, evidence, local, attribute, target_context):
        return self.compute_scales_and_logits(stacked, evidence, local, attribute, target_context)[0]

    def compute_scales_and_logits(self, stacked, evidence, local, attribute, target_context):
        if stacked.shape[-1] != 3 * self.dim:
            raise ValueError("Calibrator needs [B,L,3D] stacked memories")
        if evidence.shape != (*stacked.shape[:2], 3, len(FEATURE_NAMES)):
            raise ValueError("Calibrator evidence must have shape [B,L,3,10]")
        for value in (local, attribute, target_context):
            if value.shape != (*stacked.shape[:2], self.dim):
                raise ValueError("Calibrator context must have shape [B,L,D]")
        statistics = evidence if self.mode == "evidence" else torch.zeros_like(evidence)
        condition = torch.cat([
            stacked.detach(), local.detach(), attribute.detach(),
            target_context.detach(), statistics.detach().flatten(-2),
        ], dim=-1)
        logits = self.network(condition)
        available = evidence[..., FEATURE_NAMES.index("availability")].to(logits.dtype)
        return 1 + self.gain * torch.tanh(logits) * available, logits

    def forward(self, stacked, evidence, local, attribute, target_context):
        scales = self.compute_scales(stacked, evidence, local, attribute, target_context)
        return (stacked.reshape(*stacked.shape[:2], 3, self.dim)
                * scales.unsqueeze(-1)).reshape_as(stacked)


class Model(BaseModel):
    """A BASE subclass using scoped hooks at the existing memory/TBRE boundary."""

    def __init__(self, num_user, num_item, num_cat, num_seller, num_brand, args):
        mode = getattr(args, "memory_calibration", "none")
        scope = getattr(args, "memory_scope", "all")
        gain = float(getattr(args, "memory_gain", 0.1))
        hidden = int(getattr(args, "memory_hidden", 16))
        anchor_weight = float(getattr(args, "memory_anchor_weight", 0.0))
        if mode not in ("none", "evidence", "plain"):
            raise ValueError("memory_calibration must be none, evidence or plain")
        if scope not in ("all", "purchase-only"):
            raise ValueError("memory_scope must be all or purchase-only")
        if not math.isfinite(gain) or not 0 <= gain < 1 or hidden < 1:
            raise ValueError("memory_gain must be in [0,1) and memory_hidden positive")
        if not math.isfinite(anchor_weight) or anchor_weight < 0:
            raise ValueError("memory_anchor_weight must be finite and nonnegative")
        if mode != "none" and "b" not in args.ext_modules:
            raise ValueError("Behavior memory calibration requires BASE behavior module b")
        super().__init__(num_user, num_item, num_cat, num_seller, num_brand, args)
        self.memory_calibration, self.memory_gain, self.memory_hidden = mode, gain, hidden
        self.memory_scope = scope
        self.memory_anchor_weight = anchor_weight
        if mode != "none":
            # New CPU parameter initialization must not advance BASE's RNG.
            with torch.random.fork_rng(devices=[]):
                self.memory_calibrator = TEBC(self.num_units, mode, gain, hidden)
        self._collect_memory_statistics = False
        self._memory_statistics = {}
        self._memory_forward_counts = {}
        self._memory_anchor_statistics = {}
        self._anchor_query_mask = None
        self._anchor_sink = None

    @contextmanager
    def capture_memory_anchor(self, query_mask):
        """Capture one forward's anchor without retaining its graph afterward."""
        previous = self._anchor_query_mask, self._anchor_sink
        anchors = []
        self._anchor_query_mask, self._anchor_sink = query_mask, anchors
        try:
            yield anchors
        finally:
            self._anchor_query_mask, self._anchor_sink = previous

    def forward(self, batch):
        if (not self.training or self.memory_calibration == "none"
                or self.memory_scope == "purchase-only" or self.memory_anchor_weight == 0):
            return super().forward(batch)
        query_mask = batch["input_id"].ne(0) & batch["pos_id"].ne(0)
        with self.capture_memory_anchor(query_mask) as anchors:
            outputs = super().forward(batch)
        if len(anchors) != 1:
            raise RuntimeError("BASE main forward must produce one memory anchor")
        raw = anchors[0]
        weighted = self.memory_anchor_weight * raw
        outputs["memory_anchor_loss"] = raw
        outputs["memory_anchor_weighted_loss"] = weighted
        outputs["loss"] = outputs["loss"] + weighted
        return outputs

    def reset_memory_statistics(self):
        self._memory_statistics = {}
        self._memory_forward_counts = {}
        self._memory_anchor_statistics = {}
        self._collect_memory_statistics = self.memory_calibration != "none"

    def pop_memory_statistics(self):
        result = {}
        for source, values in self._memory_statistics.items():
            counts, sums, deviations, minimum, maximum, squares, boundaries, absolute_logits, derivatives = [
                value.cpu().tolist() for value in values
            ]
            for index, behavior in enumerate(("purchase", "favorite", "view")):
                prefix = f"{source}_{behavior}"
                count = int(counts[index])
                result[f"{prefix}_positions"] = count
                result[f"{prefix}_scale_mean"] = sums[index] / count if count else 1.0
                result[f"{prefix}_absolute_change_mean"] = deviations[index] / count if count else 0.0
                result[f"{prefix}_scale_min"] = minimum[index] if count else 1.0
                result[f"{prefix}_scale_max"] = maximum[index] if count else 1.0
                result[f"{prefix}_square_change_mean"] = squares[index] / count if count else 0.0
                result[f"{prefix}_near_boundary_fraction"] = boundaries[index] / count if count else 0.0
                result[f"{prefix}_absolute_logit_mean"] = absolute_logits[index] / count if count else 0.0
                result[f"{prefix}_tanh_derivative_mean"] = derivatives[index] / count if count else 1.0
        # Epoch losses in the original trainer are means over main batches;
        # skipped auxiliary forwards contribute zero under the same denominator.
        batches = max(self._memory_forward_counts.get("main", 0), 1)
        for source, (total, positions, calls) in self._memory_anchor_statistics.items():
            raw = float(total.cpu().item()) / batches
            result[f"{source}_anchor_loss"] = raw
            result[f"{source}_anchor_weighted_loss"] = self.memory_anchor_weight * raw
            result[f"{source}_anchor_positions"] = int(positions.cpu().item())
            result[f"{source}_anchor_forwards"] = calls
        self._collect_memory_statistics = False
        self._memory_statistics = {}
        self._memory_forward_counts = {}
        self._memory_anchor_statistics = {}
        return result

    def _record_scales(self, source, scales, evidence, logits, query_mask=None):
        if not self.training or not self._collect_memory_statistics:
            return
        scales = scales.detach()
        mask = evidence[..., 0].bool()
        if query_mask is not None:
            mask = mask & query_mask.bool().unsqueeze(-1)
        logits = logits.detach()
        dimensions = (0, 1)
        values = [
            mask.sum(dimensions), (scales * mask).sum(dimensions),
            ((scales - 1).abs() * mask).sum(dimensions),
            scales.masked_fill(~mask, torch.inf).amin(dimensions),
            scales.masked_fill(~mask, -torch.inf).amax(dimensions),
            ((scales - 1).square() * mask).sum(dimensions),
            (((scales - 1).abs() >= 0.99 * self.memory_gain) * mask).sum(dimensions)
            if self.memory_gain > 0 else torch.zeros_like(mask.sum(dimensions)),
            (logits.abs() * mask).sum(dimensions),
            ((1 - torch.tanh(logits).square()) * mask).sum(dimensions),
        ]
        self._memory_forward_counts[source] = self._memory_forward_counts.get(source, 0) + 1
        if source not in self._memory_statistics:
            self._memory_statistics[source] = values
        else:
            old = self._memory_statistics[source]
            self._memory_statistics[source] = [
                *(left + right for left, right in zip(old[:3], values[:3])),
                torch.minimum(old[3], values[3]), torch.maximum(old[4], values[4]),
                *(left + right for left, right in zip(old[5:], values[5:])),
            ]

    def _record_anchor(self, source, raw, evidence, query_mask):
        if not self.training or not self._collect_memory_statistics:
            return
        positions = (evidence[..., 0].bool() & query_mask.bool().unsqueeze(-1)).sum()
        if source not in self._memory_anchor_statistics:
            self._memory_anchor_statistics[source] = [raw.detach(), positions, 1]
        else:
            old = self._memory_anchor_statistics[source]
            self._memory_anchor_statistics[source] = [old[0] + raw.detach(), old[1] + positions, old[2] + 1]

    def _encode(self, batch, purchase_view=False, include_contrastive=True):
        if (self.memory_calibration == "none"
                or (self.memory_scope == "purchase-only" and not purchase_view)):
            if self.training and self._collect_memory_statistics and not purchase_view:
                # Anchor epoch means use all main batches as their denominator,
                # including batches whose auxiliary query is skipped. Counting
                # this bypass must not compute scales or change BASE's RNG.
                self._memory_forward_counts["main"] = self._memory_forward_counts.get("main", 0) + 1
            return super()._encode(batch, purchase_view, include_contrastive)
        valid = batch["input_id"].ne(0)
        evidence = history_evidence(batch)
        captured = {}
        handles = []

        def capture_local(module, inputs, output):
            captured["local"] = output * valid.unsqueeze(-1)

        def capture_attribute(module, inputs, output):
            captured["attribute"] = output * valid.unsqueeze(-1)

        def calibrate(module, inputs):
            stacked = inputs[0]
            local = captured["local"]
            attribute = captured.get("attribute", torch.zeros_like(local))
            if purchase_view:
                target_ids = torch.zeros_like(batch["input_id"])
            else:
                target_ids = batch["target_behavior"].long()
            target_context = self.behavior_embedding(target_ids)
            scales, logits = self.memory_calibrator.compute_scales_and_logits(
                stacked, evidence, local, attribute, target_context,
            )
            calibrated = (stacked.reshape(*stacked.shape[:2], 3, self.num_units)
                          * scales.unsqueeze(-1)).reshape_as(stacked)
            source = "auxiliary" if purchase_view else "main"
            self._record_scales(source, scales, evidence, logits)
            if purchase_view:
                last_query = torch.zeros_like(valid)
                last_query[:, -1] = valid[:, -1]
                self._record_scales("auxiliary_query", scales, evidence, logits, last_query)
            if self.training and self._anchor_sink is not None:
                raw = masked_memory_anchor(scales, evidence, self._anchor_query_mask)
                self._anchor_sink.append(raw)
                self._record_anchor(source, raw, evidence, self._anchor_query_mask)
            return (calibrated, *inputs[1:])

        try:
            local_module = self.local_blocks[-1] if len(self.local_blocks) else self.item_input_norm
            handles.append(local_module.register_forward_hook(capture_local))
            if "a" in self.ext_modules:
                handles.append(self.dynamic_attribute_attention.register_forward_hook(capture_attribute))
            handles.append(self.tbre.register_forward_pre_hook(calibrate))
            return super()._encode(batch, purchase_view, include_contrastive)
        finally:
            for handle in reversed(handles):
                handle.remove()


def calibration_config(args) -> dict:
    mode = getattr(args, "memory_calibration", "none")
    scope = getattr(args, "memory_scope", "all")
    anchor_weight = float(getattr(args, "memory_anchor_weight", 0.0))
    memory_lr = getattr(args, "memory_lr", None)
    active_paths = (["main"] if scope == "all" else []) + [
        "auxiliary_purchase", "validation_purchase", "test_purchase",
    ]
    anchor_scope = (["main_supervised_positions"] if scope == "all" else []) + ["auxiliary_last_query"]
    if mode == "none":
        training_objective = "BASE + aug_weight*purchase_loss"
    elif scope == "purchase-only":
        training_objective = "BASE_original + aug_weight*(purchase_loss_calibrated + lambda*P_aux)"
    else:
        training_objective = "BASE + lambda*P_main + aug_weight*(purchase_loss + lambda*P_aux)"
    return {
        "mode": mode, "gain": float(getattr(args, "memory_gain", 0.1)),
        "scope": scope,
        "hidden_units": int(getattr(args, "memory_hidden", 16)),
        "location": "existing_behavior_memories_before_TBRE",
        "active_stages": ["train", "validation", "test"] if mode != "none" else [],
        "active_paths": active_paths if mode != "none" else [],
        "main_view": "original_BASE" if mode == "none" or scope == "purchase-only" else "memory_calibrated",
        "conditioning": "detached_content_target_behavior_and_causal_history_statistics",
        "statistic_features": list(FEATURE_NAMES) if mode == "evidence" else [],
        "gain_rule": "1 + memory_gain * tanh(logit) * availability",
        "last_layer_initialization": "zeros", "extra_encoder_forwards": 0,
        "extra_losses": ["memory_scale_anchor"] if mode != "none" and anchor_weight > 0 else [],
        "memory_learning_rate": float(memory_lr if memory_lr is not None else getattr(args, "lr", 0.001)),
        "memory_anchor_weight": anchor_weight,
        "memory_anchor_definition": "mean((scale-1)^2) over available supervised memories; not normalized by gain",
        "memory_anchor_scope": anchor_scope if mode != "none" and anchor_weight > 0 else [],
        "training_objective": training_objective,
    }
