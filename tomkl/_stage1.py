"""Stage 1: joint training of the shared encoder, TDPL and TEBC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

from . import _training as base_train
from .tebc import Model, calibration_config
from .tdpl import purchase_recommendation_loss as original_purchase_recommendation_loss


@contextmanager
def temporary_bindings(module, **bindings):
    original = {name: getattr(module, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


class _AttributeOverlay:
    """Delegate module attributes while overriding only local entry bindings."""

    def __init__(self, wrapped, **overrides):
        self._wrapped, self._overrides = wrapped, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._wrapped, name)


def build_memory_optimizer(parameters, *, lr, memory_parameters=(), memory_lr=None, **kwargs):
    """Adam with an optional, disjoint lower-LR group for the calibrator only."""
    parameters, memory_parameters = list(parameters), list(memory_parameters)
    ids, memory_ids = [id(p) for p in parameters], [id(p) for p in memory_parameters]
    if len(set(ids)) != len(ids) or len(set(memory_ids)) != len(memory_ids):
        raise ValueError("Optimizer parameters must be unique")
    if not set(memory_ids).issubset(ids):
        raise ValueError("Memory parameters must belong to the model")
    if memory_lr is not None and (not math.isfinite(memory_lr) or memory_lr <= 0):
        raise ValueError("memory_lr must be finite and positive")
    if not memory_parameters or memory_lr is None or memory_lr == lr:
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    memory_ids = set(memory_ids)
    base_parameters = [p for p in parameters if id(p) not in memory_ids]
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": lr, "name": "base"})
    groups.append({"params": [p for p in parameters if id(p) in memory_ids],
                   "lr": memory_lr, "name": "memory"})
    return torch.optim.Adam(groups, lr=lr, **kwargs)


@contextmanager
def memory_optimizer_context(model, args):
    """Configure original train_model/refit without changing torch.optim.Adam."""
    memory_parameters = list(model.memory_calibrator.parameters()) if hasattr(model, "memory_calibrator") else []

    def adam(parameters, lr=0.001, **kwargs):
        optimizer = build_memory_optimizer(
            parameters, lr=lr, memory_parameters=memory_parameters,
            memory_lr=getattr(args, "memory_lr", None), **kwargs,
        )
        base_train.LOGGER.info("behavior_memory_optimizer=%s", json.dumps([
            {"name": group.get("name", "all"), "lr": group["lr"],
             "parameters": sum(p.numel() for p in group["params"])}
            for group in optimizer.param_groups
        ], sort_keys=True))
        return optimizer

    original_torch = base_train.torch
    overlay = _AttributeOverlay(original_torch, optim=_AttributeOverlay(original_torch.optim, Adam=adam))
    with temporary_bindings(base_train, torch=overlay):
        yield


def anchored_purchase_recommendation_loss(
    model, batch, negative_sampler=None, statistics=None, loss_type="bce",
):
    """Original purchase loss plus this forward's last-query memory anchor."""
    options = {"negative_sampler": negative_sampler, "statistics": statistics, "loss_type": loss_type}
    if not model.training or model.memory_calibration == "none" or model.memory_anchor_weight == 0:
        return original_purchase_recommendation_loss(model, batch, **options)
    query_mask = torch.zeros_like(batch["input_id"], dtype=torch.bool)
    query_mask[:, -1] = batch["input_id"][:, -1].ne(0) & batch["pos_id"][:, -1].ne(0)
    with model.capture_memory_anchor(query_mask) as anchors:
        loss = original_purchase_recommendation_loss(model, batch, **options)
    if len(anchors) != 1:
        raise RuntimeError("Purchase forward must produce one memory anchor")
    return loss + model.memory_anchor_weight * anchors[0]


def parse_args(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, description="Optional behavior-memory experiment settings")
    parser.add_argument("--memory_calibration", choices=("none", "evidence", "plain"), default="none")
    parser.add_argument("--memory_scope", choices=("all", "purchase-only"), default="all",
                        help="all preserves prior calibration; purchase-only keeps the full BASE main path unchanged")
    parser.add_argument("--memory_gain", type=float, default=0.1, help="Maximum absolute residual gain; use [0,1)")
    parser.add_argument("--memory_hidden", type=int, default=16)
    parser.add_argument("--memory_lr", type=float, default=None,
                        help="Calibrator Adam learning rate; default uses the original BASE lr")
    parser.add_argument("--memory_anchor_weight", type=float, default=0.0,
                        help="Weight of mean((memory scale - 1)^2); default disables the anchor")
    if "--help" in values or "-h" in values:
        parser.print_help()
        print("\nOriginal BASE training arguments:", flush=True)
        base_train.parse_args(["--help"])
    extension, remaining = parser.parse_known_args(values)
    args = base_train.parse_args(remaining)
    vars(args).update(vars(extension))
    if not math.isfinite(args.memory_gain) or not 0 <= args.memory_gain < 1:
        raise ValueError("memory_gain must be finite and in [0,1)")
    if args.memory_hidden < 1:
        raise ValueError("memory_hidden must be positive")
    if args.memory_lr is not None and (not math.isfinite(args.memory_lr) or args.memory_lr <= 0):
        raise ValueError("memory_lr must be finite and positive")
    if not math.isfinite(args.memory_anchor_weight) or args.memory_anchor_weight < 0:
        raise ValueError("memory_anchor_weight must be finite and nonnegative")
    if args.memory_calibration != "none" and "b" not in args.ext_modules:
        raise ValueError("memory calibration requires ext_modules containing b")
    if args.save_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        args.save_dir = str(Path(__file__).resolve().parent / "results" /
                            f"{args.dataset}_BehaviorMemory_{args.memory_calibration}_seed{args.random_seed}_{stamp}")
    return args


def source_hashes():
    root = Path(__file__).resolve().parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.py"))
    }


def main(argv=None):
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    original_epoch = base_train._train_epoch
    original_train_model = base_train.train_model
    original_refit = base_train._refit
    original_logging = base_train._setup_logging
    hashes_before = source_hashes()
    diagnostics_jsonl = save_dir / "behavior_memory_history.jsonl"
    diagnostics_csv = save_dir / "behavior_memory_history.csv"
    statistic_defaults = {}
    if args.memory_calibration != "none":
        for source in ("main", "auxiliary", "auxiliary_query"):
            for behavior in ("purchase", "favorite", "view"):
                prefix = f"{source}_{behavior}"
                statistic_defaults.update({
                    f"{prefix}_positions": 0, f"{prefix}_scale_mean": 1.0,
                    f"{prefix}_absolute_change_mean": 0.0,
                    f"{prefix}_scale_min": 1.0, f"{prefix}_scale_max": 1.0,
                    f"{prefix}_square_change_mean": 0.0,
                    f"{prefix}_near_boundary_fraction": 0.0,
                    f"{prefix}_absolute_logit_mean": 0.0,
                    f"{prefix}_tanh_derivative_mean": 1.0,
                })
        for source in ("main", "auxiliary"):
            statistic_defaults.update({
                f"{source}_anchor_loss": 0.0, f"{source}_anchor_weighted_loss": 0.0,
                f"{source}_anchor_positions": 0, f"{source}_anchor_forwards": 0,
                f"{source}_anchor_total_contribution": 0.0,
            })
        statistic_defaults["memory_anchor_total_contribution"] = 0.0
    csv_fields = ["stage", "epoch", "mode", *statistic_defaults]

    def setup_logging(directory, name):
        original_logging(directory, name)
        metadata = {"behavior_memory_calibration": calibration_config(args), "original_source_sha256": hashes_before}
        (directory / "behavior_memory_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
        )
        base_train.LOGGER.info("behavior_memory_calibration=%s", json.dumps(metadata["behavior_memory_calibration"], sort_keys=True))
        diagnostics_jsonl.write_text("", encoding="utf-8")
        diagnostics_csv.write_text("", encoding="utf-8")

    def train_epoch(*values, **kwargs):
        model = values[0] if values else kwargs["model"]
        epoch = values[5] if len(values) > 5 else kwargs["epoch"]
        stage = values[7] if len(values) > 7 else kwargs["stage"]
        model.reset_memory_statistics()
        try:
            losses = original_epoch(*values, **kwargs)
        finally:
            statistics = model.pop_memory_statistics()
        main_penalty = statistics.get("main_anchor_weighted_loss", 0.0)
        auxiliary_penalty = statistics.get("auxiliary_anchor_weighted_loss", 0.0)
        auxiliary_weight = kwargs.get("aug_weight", values[11] if len(values) > 11 else 0.1)
        if args.memory_calibration != "none":
            statistics.update({
                "main_anchor_total_contribution": main_penalty,
                "auxiliary_anchor_total_contribution": auxiliary_weight * auxiliary_penalty,
                "memory_anchor_total_contribution": main_penalty + auxiliary_weight * auxiliary_penalty,
            })
        if args.memory_anchor_weight > 0 and args.memory_calibration != "none":
            # Keep the existing component-history columns semantically pure;
            # combined loss still includes both new regularization terms.
            if "base_loss" in losses:
                losses["base_loss"] -= main_penalty
            if "aux_item_loss" in losses:
                losses["aux_item_loss"] -= auxiliary_penalty
            if "aux_weighted_loss" in losses:
                losses["aux_weighted_loss"] -= auxiliary_weight * auxiliary_penalty
        record = {"stage": stage, "epoch": epoch, "mode": args.memory_calibration,
                  **statistic_defaults, **statistics}
        with diagnostics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        # Keep the schema stable even when an epoch has no auxiliary forward.
        # None mode has only the three identifier columns.
        has_header = diagnostics_csv.stat().st_size > 0
        with diagnostics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            if not has_header:
                writer.writeheader()
            writer.writerow(record)
        base_train.LOGGER.info("behavior_memory_epoch=%s", json.dumps(record, sort_keys=True, allow_nan=False))
        return losses

    def train_model(*values, **kwargs):
        model = values[0] if values else kwargs["model"]
        with memory_optimizer_context(model, args):
            result = original_train_model(*values, **kwargs)
        result["behavior_memory_calibration"] = calibration_config(args)
        result["behavior_memory_history_jsonl"] = str(diagnostics_jsonl)
        result["behavior_memory_history_csv"] = str(diagnostics_csv)
        return result

    def refit(*values, **kwargs):
        model = values[0] if values else kwargs["model"]
        with memory_optimizer_context(model, args):
            return original_refit(*values, **kwargs)

    with temporary_bindings(base_train, Model=Model, parse_args=lambda argv=None: args,
                            _setup_logging=setup_logging, _train_epoch=train_epoch, train_model=train_model,
                            _refit=refit, purchase_recommendation_loss=anchored_purchase_recommendation_loss):
        report = base_train.main()
    if source_hashes() != hashes_before:
        raise RuntimeError("An original BASE source file changed during this run")
    # The selection wrapper adds metadata before the original main report save.
    # Update only this run's direct-test artifact, never other files in save_dir.
    if report.get("direct_test_report"):
        path = Path(report["direct_test_report"])
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["behavior_memory_calibration"] = calibration_config(args)
        path.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")
    return report
