"""PyTorch training entry point for BASE.

The validation set chooses an epoch using the original HR@10 criterion. With
the default ``--valid 0``, the model is then refit for that many epochs on the
original final-test training history (train + valid + between_val_test) and
evaluated on test exactly once. ``--valid 1`` stops after validation.
``--test_mode direct``
tests the selected validation checkpoint without refitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import Dataset
from .encoder import Model
from .tdpl import (
    AUGMENTATION_HISTORY_FIELDS, AUXILIARY_HISTORY_FIELDS, PURCHASE_HISTORY_FIELDS,
    PURCHASE_NEGATIVE_HISTORY_FIELDS, PURCHASE_NEGATIVE_SCORE_DENOMINATORS,
    PurchasePrefixSampler,
    augmentation_config, make_training_augmenter,
    purchase_recommendation_loss,
)


LOGGER = logging.getLogger("BASE")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the PyTorch BASE model")
    parser.add_argument(
        "--path", type=str, default=None,
        help="Directory containing <dataset>_*.csv, or its parent; default uses the bundled ZIP",
    )
    parser.add_argument("--dataset", type=str, default="Tmall", choices=["Tmall", "kaggle", "JD"])
    parser.add_argument(
        "--valid", type=int, default=0, choices=[0, 1],
        help="1: validation only; 0: validate, refit on the final training history, then test",
    )
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--num_behavior", type=int, default=4)
    parser.add_argument("--hidden_units", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_blocks_behavior", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--l2_reg", type=float, default=0.0)
    parser.add_argument("--ext_modules", type=str, default="bcatl")
    parser.add_argument("--eva_interval", type=int, default=1)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--num_expert_sh", type=int, default=3)
    parser.add_argument("--num_expert_sp", type=int, default=1)
    parser.add_argument("--mmoe_hidden_units", type=int, default=50)
    parser.add_argument("--tower_hidden_units", type=int, default=50)
    parser.add_argument("--att_cl_rate", type=float, default=1.0)
    parser.add_argument("--bsmmoe_cl_rate", type=float, default=0.1)
    parser.add_argument("--copy", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--save_extra_validation_checkpoints", action="store_true",
        help="Also save NDCG@10/MRR@10 validation peaks; HR@10 still controls selection, early stopping and automatic test",
    )
    parser.add_argument(
        "--train_aug", choices=("none",), default="none",
        help="The full TOMKL profile uses TDPL purchase histories",
    )
    parser.add_argument(
        "--aug_sigma", type=float, default=0.2,
        help="Legacy checkpoint metadata",
    )
    parser.add_argument(
        "--aug_seed", type=int, default=None,
        help="Independent augmentation RNG seed; defaults to random_seed",
    )
    parser.add_argument(
        "--aug_objective", choices=("purchase",), default="purchase",
        help="Shared recommendation plus causal purchase-prefix ranking",
    )
    parser.add_argument(
        "--aug_weight", type=float, default=0.1,
        help="Nonnegative auxiliary recommendation weight; 0 skips the auxiliary forward pass",
    )
    parser.add_argument(
        "--purchase_history", choices=("keep", "remove-target"), default="remove-target",
        help="Purchase objective only: retain all prefix events, or delete the target item's nonpurchase history",
    )
    parser.add_argument(
        "--purchase_min_history", type=int, default=2,
        help="Both purchase controls require this many events AFTER target nonpurchase removal",
    )
    parser.add_argument(
        "--purchase_negative", choices=("original", "mixed-hard"), default="original",
        help="Purchase auxiliary only: reuse the original negative, or mix it with one high-score negative",
    )
    parser.add_argument(
        "--purchase_loss", choices=("bce", "bpr"), default="bce",
        help="Purchase auxiliary only: original positive/negative BCE or pairwise softplus(negative-positive); BASE main loss is unchanged",
    )
    parser.add_argument(
        "--purchase_hard_ratio", type=float, default=0.5,
        help="Probability of using a hard negative per purchase query; still exactly one negative",
    )
    parser.add_argument(
        "--purchase_hard_topk", type=int, default=20,
        help="Uniformly sample the hard negative among this many highest-scoring eligible training items",
    )
    parser.add_argument(
        "--test_mode", choices=("refit", "direct"), default="refit",
        help="With valid=0: refit as before, or directly test the best validation checkpoint",
    )
    parser.add_argument(
        "--no_progress", action="store_true",
        help="Disable terminal progress bars for training and evaluation",
    )
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _prefix(args) -> Path:
    if args.path is None:
        return (
            Path(__file__).resolve().parent.parent
            / "data_process" / args.dataset / "processed_data_att" / args.dataset
        )
    path = Path(args.path).expanduser()
    if path.name.lower() == args.dataset.lower():
        return path
    return path / args.dataset


def _setup_logging(save_dir: Path, name: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(save_dir / f"{name}.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        LOGGER.addHandler(handler)


_HISTORY_FIELDS = (
    "stage", "epoch", "train_loss", "item_loss", "att_loss", "bsm_loss",
    "seconds", "is_best", "users",
    *(f"{metric}@{k}" for k in (1, 5, 10) for metric in ("HR", "NDCG", "MRR")),
)


def _initialize_history(
    jsonl_path: Path, csv_path: Path, augmentation: bool = False, auxiliary: bool = False,
    purchase: bool = False,
    purchase_negative: bool = False,
) -> None:
    jsonl_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = _HISTORY_FIELDS + (AUGMENTATION_HISTORY_FIELDS if augmentation else ())
        fields += AUXILIARY_HISTORY_FIELDS if auxiliary else ()
        fields += PURCHASE_HISTORY_FIELDS if purchase else ()
        fields += PURCHASE_NEGATIVE_HISTORY_FIELDS if purchase_negative else ()
        csv.DictWriter(stream, fieldnames=fields).writeheader()


def _append_history(jsonl_path: Path, csv_path: Path, record: dict) -> None:
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        extra_fields = (AUGMENTATION_HISTORY_FIELDS + AUXILIARY_HISTORY_FIELDS + PURCHASE_HISTORY_FIELDS
                        + PURCHASE_NEGATIVE_HISTORY_FIELDS)
        fields = _HISTORY_FIELDS + tuple(key for key in extra_fields if key in record)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerow(record)


_SAMPLE_FIELDS = (
    "user_id", "input_id", "behavior_id", "pos_id", "neg_id",
    "target_behavior", "mask_pos", "cat_id", "seller_id", "brand_id",
    "target_cat", "target_seller", "target_brand",
)


def _training_batch(sample, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32 if name == "mask_pos" else torch.long, device=device)
        for name, value in zip(_SAMPLE_FIELDS, sample)
    }


def _history_batch(dataset: Dataset, users: list[int], device: torch.device):
    """Build model inputs from the prefix before each held-out purchase."""
    def column(sequences):
        return torch.as_tensor(
            [sequences[user][:-1] for user in users], dtype=torch.long, device=device
        )

    return {
        "user_id": torch.as_tensor(users, dtype=torch.long, device=device),
        "input_id": column(dataset.valid_seq),
        "behavior_id": column(dataset.valid_behavior_seq),
        "cat_id": column(dataset.valid_cat_seq),
        "seller_id": column(dataset.valid_seller_seq),
        "brand_id": column(dataset.valid_brand_seq),
    }


def _evaluation_batch(dataset: Dataset, users: list[int], device: torch.device):
    batch = _history_batch(dataset, users, device)
    target = [[dataset.valid_seq[user][-1]] for user in users]
    candidates = torch.as_tensor(
        dataset.gen_batch_candidate(target), dtype=torch.long, device=device
    )
    return batch, candidates


@torch.no_grad()
def evaluate_model(
    model: Model,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    show_progress: bool = False,
    description: str = "evaluate",
) -> dict[str, float]:
    """Full-catalog HR, NDCG and MRR at 1/5/10, target at column zero."""
    model.eval()
    users = list(dataset.valid_seq)
    if not users:
        raise ValueError("No evaluation users have a training history")
    sums = {f"{metric}@{k}": 0.0 for k in (1, 5, 10) for metric in ("HR", "NDCG", "MRR")}
    batches = range(0, len(users), batch_size)
    for offset in tqdm(
        batches, total=len(batches), desc=description, unit="batch",
        disable=not show_progress, dynamic_ncols=True, ascii=True, file=sys.stderr,
    ):
        batch_users = users[offset:offset + batch_size]
        batch, candidates = _evaluation_batch(dataset, batch_users, device)
        scores = model.score_candidates(batch, candidates)
        if scores.shape != candidates.shape:
            raise ValueError(f"score_candidates returned {tuple(scores.shape)}, expected {tuple(candidates.shape)}")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite candidate scores during evaluation")
        # Column zero is the positive. A deterministic first-column tie break
        # avoids sorting the entire catalog a second time just to find ranks.
        ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        ranks = ranks.cpu().numpy()
        for k in (1, 5, 10):
            selected = ranks[ranks <= k]
            sums[f"HR@{k}"] += float(len(selected))
            sums[f"NDCG@{k}"] += float(np.sum(1.0 / np.log2(selected + 1)))
            sums[f"MRR@{k}"] += float(np.sum(1.0 / selected))
    return {metric: value / len(users) for metric, value in sums.items()} | {"users": len(users)}


def _train_epoch(
    model: Model,
    dataset: Dataset,
    optimizer,
    batch_size: int,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    stage: str,
    show_progress: bool,
    augmenter: PurchasePrefixSampler | None = None,
    aug_objective: str = "replace",
    aug_weight: float = 0.1,
    purchase_loss: str = "bce",
) -> dict[str, float]:
    model.train()
    users = dataset.user_list.copy()
    random.shuffle(users)
    loss_names = ("loss", "item_loss", "att_loss", "bsm_loss")
    totals = {name: 0.0 for name in loss_names}
    purchase = augmenter is not None and aug_objective == "purchase"
    negative_sampler = getattr(augmenter, "negative_sampler", None) if purchase else None
    statistics_fields = PURCHASE_HISTORY_FIELDS if purchase else AUGMENTATION_HISTORY_FIELDS
    augmentation_totals = {key: 0 for key in statistics_fields} if augmenter is not None else {}
    auxiliary = augmenter is not None and aug_objective in ("auxiliary", "purchase")
    auxiliary_totals = {key: 0 for key in AUXILIARY_HISTORY_FIELDS} if auxiliary else {}
    negative_totals = {key: 0 for key in PURCHASE_NEGATIVE_HISTORY_FIELDS} if negative_sampler is not None else {}
    batches = range(0, len(users), batch_size)
    progress = tqdm(
        batches, total=len(batches), desc=f"{stage} {epoch}/{total_epochs}",
        unit="batch", disable=not show_progress, dynamic_ncols=True, ascii=True,
        file=sys.stderr,
    )
    for batch_number, offset in enumerate(progress, start=1):
        sample = dataset.sample_batch(user_ids=users[offset:offset + batch_size])
        batch = _training_batch(sample, device)
        if augmenter is not None:
            augmented, statistics = augmenter(batch)
            for key, value in statistics.items():
                augmentation_totals[key] += value
            if not auxiliary:
                batch = augmented
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BASE training loss")
        loss.backward()
        if auxiliary:
            # Accumulate both gradients before one optimizer step. Backward on
            # the clean view first releases its activations before the second
            # forward, without detaching or reweighting the BASE main loss.
            aux_loss = loss.new_zeros(())
            weighted = aux_loss
            has_auxiliary = (augmented["input_id"].size(0) > 0 if purchase
                             else statistics["aug_removed_positions"] > 0)
            if aug_weight > 0 and has_auxiliary:
                if purchase:
                    with augmenter.forward_rng.fork(device):
                        if negative_sampler is None:
                            aux_loss = purchase_recommendation_loss(model, augmented, loss_type=purchase_loss)
                        else:
                            negative_statistics = {}
                            aux_loss = purchase_recommendation_loss(
                                model, augmented, negative_sampler=negative_sampler,
                                statistics=negative_statistics,
                                loss_type=purchase_loss,
                            )
                            for key, value in negative_statistics.items():
                                negative_totals[key] += value
                weighted = aug_weight * aux_loss
                if not torch.isfinite(aux_loss) or not torch.isfinite(weighted):
                    raise FloatingPointError("Non-finite auxiliary recommendation loss")
                weighted.backward()
            auxiliary_totals["base_loss"] += float(loss.detach().item())
            auxiliary_totals["aux_item_loss"] += float(aux_loss.detach().item())
            auxiliary_totals["aux_weighted_loss"] += float(weighted.detach().item())
            auxiliary_totals["base_targets"] += (int(batch["pos_id"].ne(0).sum().item()) if purchase
                                                 else statistics["aug_targets_before"])
            loss = loss.detach() + weighted.detach()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite combined training loss")
        optimizer.step()
        for name in loss_names:
            totals[name] += float((loss if name == "loss" else outputs[name]).detach().item())
        if show_progress:
            progress.set_postfix(loss=f"{totals['loss'] / batch_number:.4f}")
    auxiliary_means = {
        name: value if name == "base_targets" else value / len(batches)
        for name, value in auxiliary_totals.items()
    }
    for score, denominator in PURCHASE_NEGATIVE_SCORE_DENOMINATORS.items():
        if score in negative_totals:
            negative_totals[score] /= max(negative_totals[denominator], 1)
    return {**{name: value / len(batches) for name, value in totals.items()},
            **augmentation_totals, **auxiliary_means, **negative_totals}


def _save_checkpoint(
    path: Path, model: Model, args, epoch: int, metrics: dict[str, float], stage: str,
    selection_metric: str | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "stage": stage,
        "config": vars(args),
        "model_dimensions": {
            key: int(getattr(model, key))
            for key in ("num_user", "num_item", "num_cat", "num_seller", "num_brand")
        },
    }
    if selection_metric is not None:
        checkpoint["selection_metric"] = selection_metric
    torch.save(checkpoint, path)


def train_model(
    model: Model,
    dataset: Dataset,
    args,
    device: torch.device,
    checkpoint_path: Path,
    history_jsonl: Path | None = None,
    history_csv: Path | None = None,
) -> dict:
    """Fit on train and choose the best epoch using validation HR@10."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    augmenter = make_training_augmenter(args, dataset)
    best = None
    best_epoch = 0
    save_extra = getattr(args, "save_extra_validation_checkpoints", False)
    extra_best = {}
    for epoch in range(1, args.num_epochs + 1):
        start = time.monotonic()
        losses = _train_epoch(
            model, dataset, optimizer, args.batch_size, device,
            epoch, args.num_epochs, "train", not getattr(args, "no_progress", False),
            augmenter=augmenter,
            aug_objective=getattr(args, "aug_objective", "replace"),
            aug_weight=getattr(args, "aug_weight", 0.1),
            purchase_loss=getattr(args, "purchase_loss", "bce"),
        )
        record = {"stage": "selection", "epoch": epoch, "train_loss": losses["loss"],
                  "item_loss": losses["item_loss"], "att_loss": losses["att_loss"],
                  "bsm_loss": losses["bsm_loss"], "is_best": False}
        statistics = {key: losses[key] for key in (AUGMENTATION_HISTORY_FIELDS + PURCHASE_HISTORY_FIELDS
                                                 + PURCHASE_NEGATIVE_HISTORY_FIELDS) if key in losses}
        record.update(statistics)
        auxiliary_record = {key: losses[key] for key in AUXILIARY_HISTORY_FIELDS if key in losses}
        record.update(auxiliary_record)
        if auxiliary_record:
            LOGGER.info("epoch=%d auxiliary=%s", epoch, json.dumps(auxiliary_record, sort_keys=True))
        if statistics:
            LOGGER.info("epoch=%d augmentation=%s", epoch, json.dumps(statistics, sort_keys=True))
        LOGGER.info(
            "epoch=%d train_loss=%.6f item_loss=%.6f att_loss=%.6f bsm_loss=%.6f",
            epoch, losses["loss"], losses["item_loss"], losses["att_loss"], losses["bsm_loss"],
        )
        if epoch % args.eva_interval == 0 or epoch == args.num_epochs:
            metrics = evaluate_model(
                model, dataset, args.eval_batch_size or args.batch_size, device,
                show_progress=not getattr(args, "no_progress", False),
                description=f"valid epoch {epoch}",
            )
            LOGGER.info("epoch=%d valid=%s", epoch, json.dumps(metrics, sort_keys=True))
            record.update(metrics)
            if best is None or metrics["HR@10"] > best["HR@10"]:
                best, best_epoch = metrics, epoch
                record["is_best"] = True
                _save_checkpoint(checkpoint_path, model, args, epoch, metrics, "best_validation")
            if save_extra:
                # Reuse this validation pass; these snapshots never reset the
                # HR@10 patience counter or change the automatic test model.
                for metric in ("NDCG@10", "MRR@10"):
                    previous = extra_best.get(metric)
                    if previous is None or metrics[metric] > previous["validation"][metric]:
                        suffix = metric.replace("@", "").lower()
                        extra_path = checkpoint_path.with_name(
                            f"{checkpoint_path.stem}_{suffix}{checkpoint_path.suffix}"
                        )
                        _save_checkpoint(extra_path, model, args, epoch, metrics,
                                         "best_validation", selection_metric=metric)
                        extra_best[metric] = {
                            "best_epoch": epoch, "validation": metrics,
                            "checkpoint": str(extra_path),
                        }
                        LOGGER.info("extra_validation_best metric=%s epoch=%d value=%.9f checkpoint=%s",
                                    metric, epoch, metrics[metric], extra_path)
        record["seconds"] = time.monotonic() - start
        if history_jsonl is not None and history_csv is not None:
            _append_history(history_jsonl, history_csv, record)
        LOGGER.info("epoch=%d seconds=%.1f", epoch, record["seconds"])
        if best is not None and epoch - best_epoch >= args.patience:
            LOGGER.info("early_stop epoch=%d best_epoch=%d patience=%d", epoch, best_epoch, args.patience)
            break
    if best is None:
        raise RuntimeError("No validation evaluation was run")
    selection = {"best_epoch": best_epoch, "validation": best, "checkpoint": str(checkpoint_path)}
    if save_extra:
        selection.update(primary_selection_metric="HR@10", extra_validation_checkpoints=extra_best)
    return selection


def _refit(
    model: Model, dataset: Dataset, args, device: torch.device, epochs: int,
    history_jsonl: Path | None = None, history_csv: Path | None = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    augmenter = make_training_augmenter(args, dataset)
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        losses = _train_epoch(
            model, dataset, optimizer, args.batch_size, device,
            epoch, epochs, "refit", not getattr(args, "no_progress", False),
            augmenter=augmenter,
            aug_objective=getattr(args, "aug_objective", "replace"),
            aug_weight=getattr(args, "aug_weight", 0.1),
            purchase_loss=getattr(args, "purchase_loss", "bce"),
        )
        record = {"stage": "refit", "epoch": epoch, "train_loss": losses["loss"],
                  "item_loss": losses["item_loss"], "att_loss": losses["att_loss"],
                  "bsm_loss": losses["bsm_loss"], "seconds": time.monotonic() - start,
                  "is_best": False}
        statistics = {key: losses[key] for key in (AUGMENTATION_HISTORY_FIELDS + PURCHASE_HISTORY_FIELDS
                                                 + PURCHASE_NEGATIVE_HISTORY_FIELDS) if key in losses}
        record.update(statistics)
        auxiliary_record = {key: losses[key] for key in AUXILIARY_HISTORY_FIELDS if key in losses}
        record.update(auxiliary_record)
        if auxiliary_record:
            LOGGER.info("refit_epoch=%d auxiliary=%s", epoch, json.dumps(auxiliary_record, sort_keys=True))
        if statistics:
            LOGGER.info("refit_epoch=%d augmentation=%s", epoch, json.dumps(statistics, sort_keys=True))
        if history_jsonl is not None and history_csv is not None:
            _append_history(history_jsonl, history_csv, record)
        LOGGER.info("refit_epoch=%d train_loss=%.6f seconds=%.1f",
                    epoch, losses["loss"], record["seconds"])


def main(argv=None):
    args = parse_args(argv)
    if args.num_epochs < 1 or args.batch_size < 1 or args.eva_interval < 1 or args.patience < 1:
        raise ValueError("num_epochs, batch_size, eva_interval and patience must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if args.hidden_units % args.num_heads:
        raise ValueError("hidden_units must be divisible by num_heads")
    if not 0.0 <= args.aug_sigma <= 1.0:
        raise ValueError("aug_sigma must be finite and between 0 and 1")
    if not math.isfinite(args.aug_weight) or args.aug_weight < 0:
        raise ValueError("aug_weight must be finite and nonnegative")
    if args.purchase_min_history < 1:
        raise ValueError("purchase_min_history must be positive")
    if not math.isfinite(args.purchase_hard_ratio) or not 0 <= args.purchase_hard_ratio <= 1:
        raise ValueError("purchase_hard_ratio must be finite and between 0 and 1")
    if args.purchase_hard_topk < 1:
        raise ValueError("purchase_hard_topk must be positive")
    if args.purchase_negative != "original" and args.aug_objective != "purchase":
        raise ValueError("purchase_negative mixed-hard requires aug_objective=purchase")
    if args.purchase_loss != "bce" and args.aug_objective != "purchase":
        raise ValueError("purchase_loss bpr requires aug_objective=purchase")
    if args.aug_objective == "purchase":
        if args.train_aug != "none":
            raise ValueError("TOMKL requires train_aug=none")
        if args.purchase_min_history > args.maxlen:
            raise ValueError("purchase_min_history cannot exceed maxlen")
    if args.test_mode == "direct" and args.valid != 0:
        raise ValueError("test_mode direct requires valid=0")
    device = _device(args.device)
    prefix = _prefix(args)
    save_dir = Path(args.save_dir) if args.save_dir else Path(__file__).resolve().parent / "results"
    training_augmentation = args.train_aug != "none" or args.aug_objective == "purchase"
    name = "tomkl_stage1"
    _setup_logging(save_dir, name)
    history_jsonl = save_dir / f"{name}_history.jsonl"
    history_csv = save_dir / f"{name}_history.csv"
    _initialize_history(
        history_jsonl, history_csv, augmentation=args.train_aug != "none",
        auxiliary=training_augmentation and args.aug_objective in ("auxiliary", "purchase"),
        purchase=args.aug_objective == "purchase",
        purchase_negative=args.aug_objective == "purchase" and args.purchase_negative == "mixed-hard",
    )
    LOGGER.info("config=%s", json.dumps(vars(args), sort_keys=True))
    LOGGER.info("device=%s data_prefix=%s", device, prefix)
    LOGGER.info("history_jsonl=%s history_csv=%s", history_jsonl, history_csv)
    if training_augmentation:
        LOGGER.info("train_augmentation=%s", json.dumps(augmentation_config(args), sort_keys=True))

    validation = Dataset(prefix, is_valid=1)
    validation.fix_length(args.maxlen)
    final = Dataset(prefix, is_valid=0) if args.valid == 0 else None
    if final is not None:
        final.fix_length(args.maxlen)
    vocabularies = [dataset for dataset in (validation, final) if dataset is not None]
    dimensions = [
        max(getattr(dataset, key) for dataset in vocabularies)
        for key in ("user_maxid", "eval_item_maxid", "cat_maxid", "seller_maxid", "brand_maxid")
    ]

    _seed_everything(args.random_seed)
    model = Model(*dimensions, args).to(device)
    selection = train_model(
        model, validation, args, device, save_dir / f"{name}_best_valid.pt",
        history_jsonl, history_csv,
    )
    LOGGER.info("selected=%s", json.dumps(selection, sort_keys=True))
    report = {**selection, "history_jsonl": str(history_jsonl),
              "history_csv": str(history_csv)}
    if training_augmentation:
        report["train_augmentation"] = augmentation_config(args)

    if final is not None and args.test_mode == "direct":
        checkpoint = torch.load(selection["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        test_metrics = evaluate_model(
            model, final, args.eval_batch_size or args.batch_size, device,
            show_progress=not args.no_progress, description="direct test",
        )
        direct_report = {
            "mode": "direct_best_validation_checkpoint_without_refit",
            "checkpoint": str(Path(selection["checkpoint"]).resolve()),
            "checkpoint_epoch": selection["best_epoch"],
            "validation": selection["validation"], "test": test_metrics,
            "data_prefix": str(prefix.resolve()),
        }
        if training_augmentation:
            direct_report["train_augmentation"] = augmentation_config(args)
        direct_path = save_dir / f"{name}_direct_test.json"
        direct_path.write_text(json.dumps(direct_report, indent=2, sort_keys=True), encoding="utf-8")
        report.update(mode=direct_report["mode"], test=test_metrics, direct_test_report=str(direct_path))
        LOGGER.info("direct_test=%s", json.dumps(test_metrics, sort_keys=True))
        LOGGER.info("direct_test_report=%s", direct_path)
    elif final is not None:
        # Original --valid 0 trains on train + valid + between_val_test. Refit
        # after epoch selection keeps that training history without using test
        # feedback to choose a checkpoint.
        _seed_everything(args.random_seed)
        final_model = Model(*dimensions, args).to(device)
        _refit(final_model, final, args, device, selection["best_epoch"],
               history_jsonl, history_csv)
        final_checkpoint = save_dir / f"{name}_refit_test.pt"
        test_metrics = evaluate_model(
            final_model, final, args.eval_batch_size or args.batch_size, device,
            show_progress=not args.no_progress, description="test",
        )
        _save_checkpoint(final_checkpoint, final_model, args, selection["best_epoch"], test_metrics, "refit_final")
        LOGGER.info("test=%s", json.dumps(test_metrics, sort_keys=True))
        report["test"] = test_metrics
        report["final_checkpoint"] = str(final_checkpoint)

    report_path = save_dir / f"{name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("report=%s", report_path)
    return report
