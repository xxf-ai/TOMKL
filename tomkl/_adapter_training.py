"""Training and checkpoint utilities for preference adaptation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from .data import Dataset
from ._readout import (
    from_base_checkpoint, export_adaptation_checkpoint, load_adaptation_checkpoint,
    state_dict_sha256 as state_sha256, QUERY_SOURCES, validate_attention_weight,
    relative_residual_penalty,
)
from ._training import _device, _prefix, _seed_everything, _training_batch, evaluate_model
from .tdpl import (
    PURCHASE_HISTORY_FIELDS, PURCHASE_NEGATIVE_HISTORY_FIELDS,
    PURCHASE_NEGATIVE_SCORE_DENOMINATORS, augmentation_config,
    make_training_augmenter, purchase_recommendation_loss,
)
from ._stage1 import source_hashes


LOGGER = logging.getLogger("BASE.PurchaseReadout")
METRICS = tuple(f"{metric}@{k}" for k in (1, 5, 10) for metric in ("HR", "NDCG", "MRR"))
READOUT_BRANCHES = ("purchase", "favorite", "view")
READOUT_DIAGNOSTIC_FIELDS = (
    "query_norm_mean", "query_batch_centered_norm_ratio", "purchase_query_condition_norm_mean",
    *(f"attention_{branch}_{name}" for branch in READOUT_BRANCHES for name in (
        "zero_queries", "single_queries", "multi_queries", "normalized_entropy_mean",
        "logit_sd_mean", "latest_mass_mean", "key_norm_mean",
    )),
)
HISTORY_FIELDS = (
    "stage", "epoch", "train_loss", "bpr_loss", "residual_penalty_loss",
    "residual_relative_norm_mean", "residual_relative_norm_max", "train_queries",
    "train_batches", "seconds", "is_best", "users", *METRICS,
    *PURCHASE_HISTORY_FIELDS, *PURCHASE_NEGATIVE_HISTORY_FIELDS, *READOUT_DIAGNOSTIC_FIELDS,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True, type=Path)
    parser.add_argument("--readout_mode", choices=("none", "residual", "uniform", "query"), default="query")
    parser.add_argument("--query_source", choices=QUERY_SOURCES, default="purchase-embedding",
                        help="Third query block: fixed task embedding or frozen pre-residual purchase representation")
    parser.add_argument("--attention_weight", type=float, default=1.0,
                        help="Query pooling: (1-weight)*last valid state + weight*attention; default keeps pure attention")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--aug_weight", type=float, default=0.05)
    parser.add_argument("--purchase_min_history", type=int, default=2)
    parser.add_argument("--purchase_hard_ratio", type=float, default=0.5)
    parser.add_argument("--purchase_hard_topk", type=int, default=20)
    parser.add_argument("--residual_penalty", type=float, default=0.0,
                        help="Soft relative squared-norm penalty coefficient inside aug_weight*(BPR+coefficient*penalty)")
    parser.add_argument("--save_extra_validation_checkpoints", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args(argv)
    for key in ("rank", "num_epochs", "patience", "batch_size", "eval_batch_size", "purchase_min_history", "purchase_hard_topk"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("lr", "aug_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(args.residual_penalty) or args.residual_penalty < 0:
        raise ValueError("residual_penalty must be finite and nonnegative")
    if args.readout_mode == "none" and args.residual_penalty > 0:
        raise ValueError("Positive residual_penalty requires a trainable readout head")
    if not math.isfinite(args.purchase_hard_ratio) or not 0 <= args.purchase_hard_ratio <= 1:
        raise ValueError("purchase_hard_ratio must be finite and in [0,1]")
    if not 0 <= args.random_seed < 2**31:
        raise ValueError("random_seed must be in [0, 2**31)")
    if args.query_source == "purchase-representation" and args.readout_mode != "query":
        raise ValueError("purchase-representation query source requires --readout_mode query")
    validate_attention_weight(args.attention_weight, args.readout_mode)
    args.base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_output_directory(path):
    directory = Path(path).expanduser().resolve()
    # Reject long Windows paths before expensive validation/training starts.
    if len(str(directory / "best_valid_ndcg10.pt")) >= 260:
        raise ValueError("Output checkpoint path must be shorter than 260 characters")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logging(directory):
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "train.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        LOGGER.addHandler(handler)


def json_config(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def check_data_dimensions(data, dimensions):
    for key, data_key in (("num_user", "user_maxid"), ("num_item", "eval_item_maxid"),
                          ("num_cat", "cat_maxid"), ("num_seller", "seller_maxid"), ("num_brand", "brand_maxid")):
        if int(getattr(data, data_key)) > int(dimensions[key]):
            raise ValueError(f"Data {data_key} exceeds frozen checkpoint {key}")


def load_base_model(args, device):
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "best_validation":
        raise ValueError("base_checkpoint must be a best_validation checkpoint")
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("model_dimensions"), dict):
        raise ValueError("Missing original BASE checkpoint configuration/dimensions")
    base_args = SimpleNamespace(**checkpoint["config"])
    if args.data_path is not None:
        base_args.path = args.data_path
    if args.purchase_min_history > int(base_args.maxlen):
        raise ValueError("purchase_min_history cannot exceed the frozen maxlen")
    # Initializing BASE and a mode-specific head cannot advance sampling RNG.
    cuda_devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.random_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.random_seed)
        model = from_base_checkpoint(checkpoint, mode=args.readout_mode, rank=args.rank, device=device,
                                     query_source=args.query_source, attention_weight=args.attention_weight)
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("All original BASE and memory-gate parameters must be frozen")
    if model.base.training:
        raise RuntimeError("Frozen BASE must stay in evaluation mode")
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise FloatingPointError("Non-finite pretrained or initial readout parameters")
    metadata = {
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "base_checkpoint_epoch": int(checkpoint["epoch"]),
        "base_checkpoint_validation": checkpoint["metrics"],
        "frozen_base_sha256_before": state_sha256(model.base.state_dict()),
        "original_source_sha256": source_hashes(),
        "original_base_config": checkpoint["config"],
        "model_dimensions": checkpoint["model_dimensions"],
        "original_parameters": sum(p.numel() for p in model.base.parameters()),
        "trainable_readout_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adaptation": "fresh_adapter_from_completed_checkpoint; not_optimizer_resume",
        "frozen_base_eval": True,
        "training_splits": ["train"],
        "evaluation_inputs": "historical_item_behavior_attributes_only; purchase_view; no_heldout_labels",
    }
    return model, checkpoint, metadata, base_args


def load_readout_checkpoint(path, device, data_path=None):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "best_validation":
        raise ValueError("Direct test requires a best_validation readout checkpoint")
    model = load_adaptation_checkpoint(checkpoint, device=device)
    base_args = SimpleNamespace(**checkpoint["base_config"])
    if data_path is not None:
        base_args.path = data_path
    elif checkpoint.get("config", {}).get("data_path") is not None:
        base_args.path = checkpoint["config"]["data_path"]
    return model, checkpoint, base_args


def save_checkpoint(path, model, base_checkpoint, args, epoch, metrics, metadata, selection_metric="HR@10"):
    checkpoint = export_adaptation_checkpoint(
        model, base_checkpoint, epoch, metrics, json_config(args),
        base_checkpoint_path=str(args.base_checkpoint), stage="best_validation",
    )
    metadata["readout_config"] = checkpoint["readout_config"]
    checkpoint["selection_metric"] = selection_metric
    checkpoint["frozen_adaptation_metadata"] = metadata
    torch.save(checkpoint, path)
    return checkpoint


def sampling_args(args):
    # Query/negative seeds depend only on the run seed, never on head shape/mode.
    return SimpleNamespace(
        random_seed=args.random_seed, aug_seed=args.random_seed, train_aug="none",
        aug_objective="purchase", aug_weight=args.aug_weight,
        purchase_history="remove-target", purchase_min_history=args.purchase_min_history,
        purchase_negative="mixed-hard", purchase_hard_ratio=args.purchase_hard_ratio,
        purchase_hard_topk=args.purchase_hard_topk, purchase_loss="bpr",
    )


def residual_constraint_config(args):
    """Training-only soft penalty; it does not clamp the inference residual."""
    return {
        "enabled": args.residual_penalty > 0,
        "type": "soft_relative_squared_norm",
        "coefficient": args.residual_penalty,
        "effective_coefficient": args.aug_weight * args.residual_penalty,
        "definition": "mean_valid(||delta||^2 / max(||frozen_purchase_sequence||^2,1e-8))",
        "location": "last_purchase_sequence_after_original_fusion",
        "reference_detached": True, "valid_query_only": True,
        "inference_penalty": False,
    }


def loss_components(training, args):
    """Explain the actual objective without changing the historic CSV schema."""
    return {
        "weighted_bpr_loss": args.aug_weight * training["bpr_loss"],
        "weighted_residual_loss": args.aug_weight * args.residual_penalty * training["residual_penalty_loss"],
    }


class ReadoutDiagnostics:
    """Detached training-query sums; no extra encoder forward or RNG use.

    Entropy/latest/logit statistics require at least two states of the given
    behavior. Query centering is per batch, not a whole-validation statistic.
    """

    def __init__(self):
        self.queries = 0
        self.query_norm = self.centered_norm = self.condition_norm = 0.0
        self.branches = {branch: {name: 0.0 for name in (
            "zero", "single", "multi", "entropy", "logit_sd", "latest", "key_norm", "states",
        )} for branch in READOUT_BRANCHES}

    @torch.no_grad()
    def update(self, record):
        valid = record["valid_query"].bool()
        if not valid.any():
            return
        query = record["query"].detach()[valid]
        context = record["context"].detach()[valid]
        projected = record["projected_states"].detach()[valid]
        masks = record["masks"].bool()[valid]
        weights = record["weights"].detach()[valid]
        self.queries += query.size(0)
        self.query_norm += float(query.norm(dim=-1).sum())
        self.centered_norm += float((query - query.mean(0)).norm(dim=-1).sum())
        self.condition_norm += float(context[:, 2 * (context.size(-1) // 3):].norm(dim=-1).sum())
        counts = masks.sum(-1)
        scores = (projected * query[:, None, None, :]).sum(-1) / math.sqrt(query.size(-1))
        score_mean = (scores * masks).sum(-1) / counts.clamp_min(1)
        score_sd = (((scores - score_mean.unsqueeze(-1)).square() * masks).sum(-1)
                    / counts.clamp_min(1)).sqrt()
        entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
        positions = torch.arange(1, masks.size(-1) + 1, device=masks.device)
        latest = masks & positions.eq((positions * masks).amax(-1, keepdim=True))
        latest_mass = (weights * latest).sum(-1)
        key_norms = projected.norm(dim=-1)
        for index, branch in enumerate(READOUT_BRANCHES):
            aggregate = self.branches[branch]
            multi = counts[:, index].gt(1)
            aggregate["zero"] += int(counts[:, index].eq(0).sum())
            aggregate["single"] += int(counts[:, index].eq(1).sum())
            aggregate["multi"] += int(multi.sum())
            aggregate["entropy"] += float((entropy[:, index] /
                counts[:, index].clamp_min(2).float().log())[multi].sum())
            aggregate["logit_sd"] += float(score_sd[:, index][multi].sum())
            aggregate["latest"] += float(latest_mass[:, index][multi].sum())
            aggregate["key_norm"] += float(key_norms[:, index][masks[:, index]].sum())
            aggregate["states"] += int(counts[:, index].sum())

    def values(self):
        values = {"query_norm_mean": self.query_norm / max(self.queries, 1),
                  "query_batch_centered_norm_ratio": self.centered_norm / max(self.query_norm, 1e-8),
                  "purchase_query_condition_norm_mean": self.condition_norm / max(self.queries, 1)}
        for branch, aggregate in self.branches.items():
            prefix = f"attention_{branch}_"
            values.update({prefix + "zero_queries": int(aggregate["zero"]),
                           prefix + "single_queries": int(aggregate["single"]),
                           prefix + "multi_queries": int(aggregate["multi"]),
                           prefix + "normalized_entropy_mean": aggregate["entropy"] / max(aggregate["multi"], 1),
                           prefix + "logit_sd_mean": aggregate["logit_sd"] / max(aggregate["multi"], 1),
                           prefix + "latest_mass_mean": aggregate["latest"] / max(aggregate["multi"], 1),
                           prefix + "key_norm_mean": aggregate["key_norm"] / max(aggregate["states"], 1)})
        return values


def run_epoch(model, dataset, optimizer, sampler, args, device, epoch):
    model.train()
    if model.base.training or any(p.requires_grad for p in model.base.parameters()):
        raise RuntimeError("Frozen BASE training state changed")
    users = dataset.user_list.copy()
    random.shuffle(users)
    batch_total = math.ceil(len(users) / args.batch_size)
    if batch_total < 1:
        raise ValueError("No training users")
    totals = {key: 0 for key in (*PURCHASE_HISTORY_FIELDS, *PURCHASE_NEGATIVE_HISTORY_FIELDS)}
    bpr_total = penalty_total = objective_total = relative_total = 0.0
    query_total = relative_count = 0
    relative_max = 0.0
    diagnostics = ReadoutDiagnostics()
    iterator = tqdm(range(0, len(users), args.batch_size), total=batch_total,
                    desc=f"adapter {epoch}/{args.num_epochs}", disable=args.no_progress,
                    dynamic_ncols=True, ascii=True, file=sys.stderr)
    for offset in iterator:
        batch = _training_batch(dataset.sample_batch(user_ids=users[offset:offset + args.batch_size]), device)
        purchase_batch, statistics = sampler(batch)
        for key, value in statistics.items():
            totals[key] += value
        if purchase_batch["input_id"].size(0) == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        negative_statistics = {}
        with model.capture_readout() as readouts:
            bpr = purchase_recommendation_loss(model, purchase_batch, negative_sampler=sampler.negative_sampler,
                                              statistics=negative_statistics, loss_type="bpr")
        if len(readouts) != 1:
            raise RuntimeError("Purchase BPR must produce exactly one readout")
        record = readouts[0]
        if model.mode == "query":
            diagnostics.update(record)
        valid = record["valid_query"].bool()
        reference, delta = record["reference"].detach(), record["delta"]
        square_ratio = delta.square().sum(-1) / reference.square().sum(-1).clamp_min(1e-8)
        penalty = relative_residual_penalty(delta, reference, valid)
        objective = args.aug_weight * (bpr + args.residual_penalty * penalty)
        if not all(torch.isfinite(value).all() for value in (bpr, penalty, objective)):
            raise FloatingPointError("Non-finite readout loss")
        objective.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise RuntimeError("A frozen BASE parameter received gradients")
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite readout gradients")
        optimizer.step()
        for key, value in negative_statistics.items():
            totals[key] += value
        bpr_total += float(bpr.detach())
        penalty_total += float(penalty.detach())
        objective_total += float(objective.detach())
        query_total += int(purchase_batch["input_id"].size(0))
        relative = square_ratio[valid].detach().sqrt()
        if relative.numel():
            relative_total += float(relative.sum())
            relative_count += relative.numel()
            relative_max = max(relative_max, float(relative.max()))
    for score, denominator in PURCHASE_NEGATIVE_SCORE_DENOMINATORS.items():
        totals[score] /= max(totals[denominator], 1)
    return {**totals, **diagnostics.values(), "train_loss": objective_total / batch_total,
            "bpr_loss": bpr_total / batch_total, "residual_penalty_loss": penalty_total / batch_total,
            "residual_relative_norm_mean": relative_total / max(relative_count, 1),
            "residual_relative_norm_max": relative_max, "train_queries": query_total, "train_batches": batch_total}


def append_history(jsonl, csv_path, record):
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"Non-finite history value: {key}")
    with jsonl.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(record)


def main(argv=None):
    args = parse_args(argv)
    device = _device(args.device)
    model, base_checkpoint, metadata, base_args = load_base_model(args, device)
    prefix = _prefix(base_args)
    validation = Dataset(prefix, is_valid=1)
    validation.fix_length(int(base_args.maxlen))
    check_data_dimensions(validation, base_checkpoint["model_dimensions"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    mode_tag = args.readout_mode + ("_pstate" if args.query_source == "purchase-representation" else "")
    if args.attention_weight < 1:
        mixture_tag = format(args.attention_weight, ".9g").replace(".", "p").replace("-", "m")
        mode_tag += f"_anchor{mixture_tag}"
    if args.residual_penalty > 0:
        penalty_tag = format(args.residual_penalty, ".9g").replace(".", "p").replace("-", "m")
        mode_tag += f"_rrn{penalty_tag}"
    directory = checked_output_directory(args.save_dir or Path(__file__).parent / "results" /
                                         f"{base_args.dataset}_PRA_{mode_tag}_s{args.random_seed}_{stamp}")
    args.save_dir = directory
    setup_logging(directory)
    sampler_options = sampling_args(args)
    metadata.update(
        data_prefix=str(prefix.resolve()), train_augmentation=augmentation_config(sampler_options),
        training_objective="aug_weight * (purchase_BPR + residual_penalty * relative_residual_squared)",
        primary_selection_metric="HR@10", full_validation_every_epoch=True,
        epoch0_is_candidate=True, original_main_loss_backward=False,
        aug_weight_interpretation="only_auxiliary_is_optimized; this weight scales the whole adapter gradient",
        readout_diagnostics_scope="training_auxiliary_queries; query_mode_only; no_extra_encoder_forward",
        readout_diagnostics_conditioning="entropy/logit_sd/latest require behavior_count>1; query centering is per batch",
        readout_diagnostics_weights="actual pooled mixture weights; entropy/latest include the fixed anchor",
        readout_diagnostics_logits="unmixed query-key scores; logit_sd describes the attention component only",
        residual_constraint=residual_constraint_config(args),
    )
    LOGGER.info("config=%s", json.dumps(json_config(args), sort_keys=True))
    LOGGER.info("frozen_base=%s", json.dumps(metadata, sort_keys=True))
    (directory / "config.json").write_text(json.dumps({"config": json_config(args), **metadata}, indent=2, sort_keys=True), encoding="utf-8")
    history_jsonl, history_csv = directory / "history.jsonl", directory / "history.csv"
    history_jsonl.write_text("", encoding="utf-8")
    with history_csv.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
    # Reset AFTER constructing every mode. Dataset user shuffling and original
    # random negatives therefore have the same RNG stream across controls.
    _seed_everything(args.random_seed)
    started = time.perf_counter()
    epoch0 = evaluate_model(model, validation, args.eval_batch_size, device,
                            show_progress=not args.no_progress, description="epoch 0 validation")
    epoch0_differences = {metric: float(epoch0[metric]) - float(base_checkpoint["metrics"][metric])
                         for metric in METRICS}
    metadata["epoch0_validation_difference_from_source"] = epoch0_differences
    metadata["epoch0_validation_matches_source"] = all(abs(value) <= 1e-9 for value in epoch0_differences.values())
    if not metadata["epoch0_validation_matches_source"]:
        LOGGER.warning("Epoch-zero validation differs from source metrics: %s", json.dumps(epoch0_differences, sort_keys=True))
    best_epoch, best_metrics = 0, epoch0
    checkpoint_path = directory / "best_valid.pt"
    save_checkpoint(checkpoint_path, model, base_checkpoint, args, 0, epoch0, metadata)
    epoch0_path = directory / "epoch0.pt"
    save_checkpoint(epoch0_path, model, base_checkpoint, args, 0, epoch0, metadata)
    (directory / "config.json").write_text(json.dumps({"config": json_config(args), **metadata}, indent=2, sort_keys=True), encoding="utf-8")
    extra = {}
    if args.save_extra_validation_checkpoints:
        for metric, filename in (("NDCG@10", "best_valid_ndcg10.pt"), ("MRR@10", "best_valid_mrr10.pt")):
            path = directory / filename
            save_checkpoint(path, model, base_checkpoint, args, 0, epoch0, metadata, metric)
            extra[metric] = {"best_epoch": 0, "checkpoint": str(path), "validation": epoch0}
    zero_record = {key: 0 for key in HISTORY_FIELDS if key not in METRICS}
    append_history(history_jsonl, history_csv, {**zero_record, **epoch0, "stage": "epoch0", "epoch": 0,
                                               "is_best": True, "seconds": time.perf_counter() - started})
    LOGGER.info("epoch=0 validation=%s", json.dumps(epoch0, sort_keys=True))
    last_epoch = 0
    stopped_early = False
    patience_reached = False
    if args.readout_mode != "none":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Selected readout mode has no trainable parameters")
        adapter_ids = {id(p) for p in model.adapter.parameters()}
        if {id(p) for p in parameters} != adapter_ids:
            raise RuntimeError("Only the new readout adapter may be optimized")
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        sampler = make_training_augmenter(sampler_options, validation)
        for epoch in range(1, args.num_epochs + 1):
            started = time.perf_counter()
            training = run_epoch(model, validation, optimizer, sampler, args, device, epoch)
            metrics = evaluate_model(model, validation, args.eval_batch_size, device,
                                     show_progress=not args.no_progress, description=f"validation {epoch}")
            improved = metrics["HR@10"] > best_metrics["HR@10"]
            if improved:
                best_epoch, best_metrics = epoch, metrics
                save_checkpoint(checkpoint_path, model, base_checkpoint, args, epoch, metrics, metadata)
            for metric, selection in extra.items():
                if metrics[metric] > selection["validation"][metric]:
                    save_checkpoint(Path(selection["checkpoint"]), model, base_checkpoint, args, epoch, metrics, metadata, metric)
                    extra[metric] = {"best_epoch": epoch, "checkpoint": selection["checkpoint"], "validation": metrics}
            append_history(history_jsonl, history_csv, {"stage": "adapter", "epoch": epoch, **training, **metrics,
                                                       "is_best": improved, "seconds": time.perf_counter() - started})
            LOGGER.info("epoch=%s train_loss=%.8f loss_components=%s validation=%s best_epoch=%s readout_diagnostics=%s",
                        epoch, training["train_loss"], json.dumps(loss_components(training, args), sort_keys=True),
                        json.dumps(metrics, sort_keys=True), best_epoch,
                        json.dumps({key: training[key] for key in READOUT_DIAGNOSTIC_FIELDS}, sort_keys=True))
            last_epoch = epoch
            if epoch - best_epoch >= args.patience:
                patience_reached = True
                stopped_early = epoch < args.num_epochs
                break
    frozen_after = state_sha256(model.base.state_dict())
    if frozen_after != metadata["frozen_base_sha256_before"]:
        raise RuntimeError("Frozen original parameters/buffers changed")
    if file_sha256(args.base_checkpoint) != metadata["base_checkpoint_sha256"] or source_hashes() != metadata["original_source_sha256"]:
        raise RuntimeError("Original checkpoint or BASE source files changed during adapter fit")
    # Load the selected self-contained checkpoint, including epoch-zero fallback.
    model, selected, selected_args = load_readout_checkpoint(checkpoint_path, device, args.data_path)
    if state_sha256(model.base.state_dict()) != frozen_after:
        raise RuntimeError("Selected checkpoint contains different frozen BASE weights")
    final = Dataset(_prefix(selected_args), is_valid=0)
    final.fix_length(int(selected_args.maxlen))
    check_data_dimensions(final, selected["model_dimensions"])
    test = evaluate_model(model, final, args.eval_batch_size, device,
                          show_progress=not args.no_progress, description="direct test")
    direct_report = {
        "mode": "direct_best_validation_checkpoint_without_refit", "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": best_epoch, "validation": best_metrics, "test": test,
        "data_prefix": str(prefix.resolve()), "readout_config": selected["readout_config"],
        "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_epoch": metadata["base_checkpoint_epoch"],
    }
    direct_path = directory / "direct_test.json"
    direct_path.write_text(json.dumps(direct_report, indent=2, sort_keys=True), encoding="utf-8")
    report = {**metadata, **direct_report, "best_epoch": best_epoch, "last_epoch": last_epoch,
              "epochs_completed": last_epoch, "stopped_early": stopped_early,
              "patience_reached": patience_reached,
              "stop_reason": "reference_only" if args.readout_mode == "none" else
                             ("patience" if stopped_early else "epoch_budget"),
              "epoch0_validation": epoch0, "fell_back_to_epoch0": best_epoch == 0,
              "epoch0_checkpoint": str(epoch0_path),
              "frozen_base_verified": True, "frozen_base_sha256_after": frozen_after,
              "optimizer_parameters": "adapter_only", "base_gradients_absent": True,
              "history_jsonl": str(history_jsonl), "history_csv": str(history_csv),
              "direct_test_report": str(direct_path), "extra_validation_checkpoints": extra,
              "config": json_config(args)}
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("report=%s", report_path)
    LOGGER.info("direct_test=%s", json.dumps(test, sort_keys=True))
    return report
