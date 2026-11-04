"""Train, evaluate and rank items with the complete TOMKL model."""

import argparse
import json
from pathlib import Path

from . import profiles


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _seed(value):
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be in [0, 2^32 - 1]")
    return number


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train the foundation, then fit TCPSA")
    train.add_argument("--dataset", choices=profiles.DATASET_NAMES, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=_seed, default=1)
    train.add_argument("--stage1-epochs", type=_positive_int, default=profiles.FOUNDATION_EPOCHS)
    train.add_argument("--stage2-epochs", type=_positive_int, default=profiles.ADAPTATION_EPOCHS)
    train.add_argument("--batch-size", type=_positive_int, default=128)
    evaluate = commands.add_parser("evaluate", help="evaluate a validation-selected TOMKL checkpoint")
    predict = commands.add_parser("predict", help="rank the catalog for a held-out user's history")
    predict.add_argument("--user-id", type=_positive_int, required=True)
    predict.add_argument("--top-k", type=_positive_int, default=10)
    for command in (train, evaluate, predict):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
        command.add_argument("--data-path", required=True, type=Path,
                             help="processed CSV directory, dataset prefix or processed_data_att.zip")
    for command in (train, evaluate):
        command.add_argument("--eval-batch-size", type=_positive_int, default=32)
        command.add_argument("--no-progress", action="store_true")
    for command in (evaluate, predict):
        command.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def _data_prefix(path, dataset):
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "processed_data_att.zip":
        return path.parent / "processed_data_att" / dataset
    if path.name.lower() == dataset.lower() and Path(f"{path}_train.csv").is_file():
        return path
    if (path / f"{dataset}_train.csv").is_file():
        return path / dataset
    if (path / "processed_data_att.zip").is_file():
        return path / "processed_data_att" / dataset
    if (path.parent / "processed_data_att.zip").is_file() and path.name == "processed_data_att":
        return path / dataset
    if path.name.lower() == dataset.lower() and (path.parent.parent / "processed_data_att.zip").is_file():
        return path
    raise FileNotFoundError(f"No processed {dataset} CSVs or processed_data_att.zip at {path}")


def _options(values):
    return [part for name, value in values.items() for part in (f"--{name}", str(value))]


def _train(args):
    from . import _stage1, _stage2

    dataset = profiles.DATASET_NAMES[args.dataset]
    prefix = _data_prefix(args.data_path, dataset)
    output = args.output_dir.expanduser().resolve()
    # A run owns its output directory. Use a new name to retry a failed run.
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "device": args.device,
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
    }
    progress = ["--no_progress"] if args.no_progress else []
    foundation = _stage1.main(_options({
        **profiles.FOUNDATION, **common, "dataset": dataset, "path": prefix,
        "num_epochs": args.stage1_epochs, "patience": args.stage1_epochs, "save_dir": output / "stage1",
    }) + progress)
    checkpoint = Path(foundation["checkpoint"]).resolve()
    adaptation = _stage2.main(_options({
        **profiles.ADAPTATION, **common, "base_checkpoint": checkpoint, "data_path": prefix,
        "num_epochs": args.stage2_epochs, "patience": args.stage2_epochs, "save_dir": output / "stage2",
    }) + progress)
    return {
        "model": "TOMKL",
        "dataset": args.dataset,
        "stage1_checkpoint": str(checkpoint),
        "checkpoint": adaptation["checkpoint"],
        "stage2_epoch": adaptation["best_epoch"],
        "frozen_foundation_verified": adaptation["frozen_base_verified"],
        "test": adaptation["test"],
    }


def _load(args):
    import torch

    from ._adapter_training import check_data_dimensions
    from ._training import _device
    from .data import Dataset
    from .tcpsa import load_adaptation_checkpoint

    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "best_validation":
        raise ValueError("Use the complete TOMKL checkpoint selected on validation")
    config = checkpoint.get("base_config", {})
    dataset = config.get("dataset")
    if dataset not in profiles.DATASET_NAMES.values():
        raise ValueError("Checkpoint does not identify a supported TOMKL dataset")
    model = load_adaptation_checkpoint(checkpoint, device=device)
    data = Dataset(_data_prefix(args.data_path, dataset), is_valid=0)
    data.fix_length(int(config["maxlen"]))
    check_data_dimensions(data, checkpoint["model_dimensions"])
    return model, data, device, checkpoint


def _evaluate(args):
    from ._training import evaluate_model

    model, data, device, checkpoint = _load(args)
    return {
        "model": "TOMKL",
        "checkpoint_epoch": checkpoint["epoch"],
        "test": evaluate_model(model, data, args.eval_batch_size, device,
                               show_progress=not args.no_progress, description="TOMKL test"),
    }


def _predict(args):
    import torch

    from ._training import _history_batch

    model, data, device, _ = _load(args)
    if args.user_id not in data.valid_seq:
        raise ValueError(f"No eligible test history for user {args.user_id}")
    batch = _history_batch(data, [args.user_id], device)
    candidates = torch.arange(1, data.eval_item_maxid + 1, device=device).unsqueeze(0)
    with torch.inference_mode():
        scores = model.score_candidates(batch, candidates)[0]
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite candidate scores")
        order = torch.argsort(scores, descending=True, stable=True)[:args.top_k]
    return {
        "model": "TOMKL",
        "user_id": args.user_id,
        "recommendations": [
            {"item_id": item, "score": score}
            for item, score in zip((order + 1).tolist(), scores[order].tolist())
        ],
    }


def main():
    args = _args()
    action = {"train": _train, "evaluate": _evaluate, "predict": _predict}[args.command]
    try:
        result = action(args)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        raise SystemExit(f"TOMKL: {error}") from None
    print(json.dumps(result, indent=2, allow_nan=False))
