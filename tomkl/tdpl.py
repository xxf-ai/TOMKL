"""Training operators used by TDPL to form target queries and negatives."""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


AUGMENTATION_HISTORY_FIELDS = (
    "aug_input_positions", "aug_removed_positions",
    "aug_targets_before", "aug_targets_after",
)

AUXILIARY_HISTORY_FIELDS = (
    "base_loss", "aux_item_loss", "aux_weighted_loss", "base_targets",
)

PURCHASE_HISTORY_FIELDS = (
    "purchase_candidates", "purchase_users_selected", "purchase_users_skipped",
    "purchase_original_positions", "purchase_prefix_positions", "purchase_removed_positions",
    "purchase_samples_with_removal", "purchase_last_event_removed",
    "purchase_target_seen_before", "purchase_target_seen_after",
)

PURCHASE_NEGATIVE_HISTORY_FIELDS = (
    "purchase_negative_queries", "purchase_hard_requested", "purchase_hard_used",
    "purchase_hard_fallback", "purchase_hard_seen_history",
    "purchase_positive_score", "purchase_original_negative_score",
    "purchase_negative_score", "purchase_hard_negative_score",
)

# Sampler returns score sums; the trainer divides by the appropriate epoch
# count, avoiding bias from a smaller final batch or hard-negative fallback.
PURCHASE_NEGATIVE_SCORE_DENOMINATORS = {
    "purchase_positive_score": "purchase_negative_queries",
    "purchase_original_negative_score": "purchase_negative_queries",
    "purchase_negative_score": "purchase_negative_queries",
    "purchase_hard_negative_score": "purchase_hard_used",
}

# These are all twelve position-aligned fields returned by sample_batch().
# Behavior padding is 4 (0 is the purchase behavior); other IDs pad with 0.
_POSITION_PADDING = {
    "input_id": 0,
    "behavior_id": 4,
    "pos_id": 0,
    "neg_id": 0,
    "target_behavior": 4,
    "mask_pos": 0.0,
    "cat_id": 0,
    "seller_id": 0,
    "brand_id": 0,
    "target_cat": 0,
    "target_seller": 0,
    "target_brand": 0,
}


def augmentation_config(args) -> dict:
    seed = getattr(args, "aug_seed", None)
    seed = int(args.random_seed if seed is None else seed)
    if getattr(args, "aug_objective", "replace") == "purchase":
        config = {
            "mode": "purchase-prefix", "objective": "purchase",
            "weight": float(getattr(args, "aug_weight", 0.1)),
            "history_policy": getattr(args, "purchase_history", "remove-target"),
            "minimum_history": int(getattr(args, "purchase_min_history", 2)),
            "seed": seed, "auxiliary_dropout_seed": (seed + 1) % (2 ** 64),
            "training_only": True, "main_view": "original",
            "auxiliary_loss": "purchase_end_recommendation_only",
            "contrastive_view": "original", "regularization_view": "original",
            "target_policy": "original_purchase_label_from_causal_prefix",
            "sampling": "one_eligible_purchase_per_user_per_batch",
            "candidate_policy": "minimum_history_after_target_nonpurchase_removal_for_both_controls",
            "negative_policy": "reuse_original_query_negative",
            "independent_auxiliary_dropout_rng": True,
        }
        if getattr(args, "purchase_loss", "bce") == "bpr":
            config.update(purchase_loss="bpr", auxiliary_loss="purchase_end_pairwise_bpr")
        if getattr(args, "purchase_negative", "original") == "mixed-hard":
            config.update(
                negative_policy="one_negative_original_random_or_uniform_topk_high_score",
                negative_mode="mixed-hard",
                hard_ratio=float(getattr(args, "purchase_hard_ratio", 0.5)),
                hard_topk=int(getattr(args, "purchase_hard_topk", 20)),
                negative_sampling_seed=(seed + 2) % (2 ** 64),
                hard_candidate_pool="observed_items_in_current_training_stage",
                hard_exclusions="current_target_and_all_known_purchases_in_current_training_stage",
                known_purchase_source="untruncated_train_csv; valid_and_between_added_only_for_refit",
                hard_scoring="same_auxiliary_forward_detached_final_and_item_table",
                auxiliary_negatives_per_query=1,
                hard_fallback="reuse_original_query_negative",
            )
        return config
    raise ValueError("TOMKL requires the TDPL purchase objective")


def make_training_augmenter(args, dataset=None) -> PurchasePrefixSampler:
    config = augmentation_config(args)
    negative_sampler = None
    if config.get("negative_mode") == "mixed-hard":
        if dataset is None:
            raise ValueError("mixed-hard purchase negatives require the training dataset")
        negative_sampler = PurchaseNegativeSampler(
            config["negative_sampling_seed"], dataset.item_pool,
            _known_training_purchases(dataset), config["hard_ratio"], config["hard_topk"],
        )
    return PurchasePrefixSampler(
        config["seed"], config["minimum_history"], config["history_policy"],
        negative_sampler=negative_sampler,
    )


def _known_training_purchases(dataset) -> dict[int, set[int]]:
    """Negative filtering reads only the current stage's training labels.

    fix_length() truncates dataset.train, so reload the original training CSV
    to protect known purchases outside the window too. In direct/selection
    this is strictly train.csv. Refit already trains on valid/between; include
    those here only in that stage. The held-out test labels are never read.
    This filtering information never enters the causal prefix encoder.
    """
    purchased = {}
    splits = ["train"] if dataset.is_valid else ["train", "valid", "between_val_test"]
    for split in splits:
        frame = dataset._read_split(split)
        pairs = frame.loc[frame.action_type.eq(0), ["user_id", "item_id"]]
        for user, item in pairs.itertuples(index=False, name=None):
            purchased.setdefault(int(user), set()).add(int(item))
    return purchased


class PurchaseNegativeSampler:
    """Choose one original random or high-score negative per purchase query.

    Mining has no gradient or extra encoder forward. The selected ID is later
    scored again with the live auxiliary representation and embedding, so BCE
    still trains both. Keep a separate CPU RNG from query sampling/dropout.
    """

    def __init__(
        self, seed: int, item_pool: list[int], purchases_by_user: dict[int, set[int]],
        hard_ratio: float = 0.5, topk: int = 20,
    ):
        if not math.isfinite(hard_ratio) or not 0 <= hard_ratio <= 1:
            raise ValueError("purchase_hard_ratio must be finite and between 0 and 1")
        if topk < 1:
            raise ValueError("purchase_hard_topk must be positive")
        self.item_pool = sorted(set(map(int, item_pool)))
        if not self.item_pool or self.item_pool[0] < 1:
            raise ValueError("Hard-negative item_pool must contain positive item IDs")
        self.column_by_item = {item: column for column, item in enumerate(self.item_pool)}
        self.purchase_columns = {
            int(user): [self.column_by_item[int(item)] for item in purchased
                        if int(item) in self.column_by_item]
            for user, purchased in purchases_by_user.items()
        }
        self.hard_ratio = float(hard_ratio)
        self.topk = int(topk)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._device_pools = {}

    @torch.no_grad()
    def select(self, model, sequence: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        positives = batch["pos_id"][:, -1].long()
        original = batch["neg_id"][:, -1].long()
        if sequence.ndim != 2 or sequence.size(0) != positives.numel():
            raise ValueError("Purchase negative mining needs one final representation per query")
        if self.item_pool[-1] >= model.item_embedding.num_embeddings:
            raise ValueError("Hard-negative item_pool exceeds the model item vocabulary")
        selected = original.clone()
        count, device = original.numel(), sequence.device
        stats = {name: 0 for name in PURCHASE_NEGATIVE_HISTORY_FIELDS}
        stats["purchase_negative_queries"] = count
        if count == 0:
            return selected, stats

        if self.hard_ratio == 0:
            requests = torch.zeros(count, dtype=torch.bool)
        elif self.hard_ratio == 1:
            requests = torch.ones(count, dtype=torch.bool)
        else:
            requests = torch.rand(count, generator=self.generator) < self.hard_ratio
        requested_rows = requests.nonzero(as_tuple=True)[0]
        stats["purchase_hard_requested"] = requested_rows.numel()
        used_rows = []
        if requested_rows.numel():
            key = str(device)
            if key not in self._device_pools:
                self._device_pools[key] = torch.tensor(self.item_pool, dtype=torch.long, device=device)
            pool = self._device_pools[key]
            rows = requested_rows.to(device)
            scores = sequence.detach().index_select(0, rows) @ model.item_embedding.weight.detach().index_select(0, pool).T
            scores *= math.sqrt(model.num_units)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite auxiliary hard-negative mining scores")
            scores.masked_fill_(pool.unsqueeze(0).eq(positives.index_select(0, rows).unsqueeze(1)), -torch.inf)
            users = batch["user_id"].index_select(0, rows).detach().cpu().tolist()
            masked_rows, masked_columns = [], []
            for row, user in enumerate(users):
                columns = self.purchase_columns.get(user, [])
                masked_rows.extend([row] * len(columns))
                masked_columns.extend(columns)
            if masked_rows:
                scores[
                    torch.tensor(masked_rows, dtype=torch.long, device=device),
                    torch.tensor(masked_columns, dtype=torch.long, device=device),
                ] = -torch.inf
            top = scores.topk(min(self.topk, pool.numel()), dim=1)
            eligible_counts = torch.isfinite(top.values).sum(1).cpu().tolist()
            choices, local_rows = [], []
            for local_row, eligible in enumerate(eligible_counts):
                if eligible:
                    choices.append(int(torch.randint(eligible, (), generator=self.generator).item()))
                    local_rows.append(local_row)
            if local_rows:
                local = torch.tensor(local_rows, dtype=torch.long, device=device)
                choices = torch.tensor(choices, dtype=torch.long, device=device)
                actual_rows = rows.index_select(0, local)
                chosen_columns = top.indices[local, choices]
                selected[actual_rows] = pool[chosen_columns]
                used_rows = actual_rows
                stats["purchase_hard_used"] = len(local_rows)
                seen = (batch["input_id"].index_select(0, actual_rows).eq(selected[actual_rows].unsqueeze(1))
                        & batch["input_id"].index_select(0, actual_rows).ne(0)).any(1)
                stats["purchase_hard_seen_history"] = int(seen.sum().item())
        stats["purchase_hard_fallback"] = stats["purchase_hard_requested"] - stats["purchase_hard_used"]

        def score(ids):
            return (sequence.detach() * model._scaled_embedding(model.item_embedding, ids).detach()).sum(-1)

        positive_score, original_score, selected_score = score(positives), score(original), score(selected)
        if not all(torch.isfinite(value).all() for value in (positive_score, original_score, selected_score)):
            raise FloatingPointError("Non-finite auxiliary positive/negative scores")
        stats["purchase_positive_score"] = float(positive_score.sum().item())
        stats["purchase_original_negative_score"] = float(original_score.sum().item())
        stats["purchase_negative_score"] = float(selected_score.sum().item())
        if stats["purchase_hard_used"]:
            stats["purchase_hard_negative_score"] = float(selected_score[used_rows].sum().item())
        return selected, stats


class AuxiliaryForwardRNG:
    """Keep auxiliary dropout reproducible without advancing the main RNG."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.cpu_state = torch.Generator(device="cpu").manual_seed(self.seed).get_state()
        self.cuda_states = {}

    @contextmanager
    def fork(self, device: torch.device):
        device = torch.device(device)
        cuda_index = None
        if device.type == "cuda":
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            if cuda_index not in self.cuda_states:
                self.cuda_states[cuda_index] = torch.Generator(
                    device=torch.device("cuda", cuda_index)
                ).manual_seed(self.seed).get_state()
        with torch.random.fork_rng(devices=[] if cuda_index is None else [cuda_index]):
            torch.random.set_rng_state(self.cpu_state)
            if cuda_index is not None:
                torch.cuda.set_rng_state(self.cuda_states[cuda_index], cuda_index)
            try:
                yield
            finally:
                self.cpu_state = torch.random.get_rng_state()
                if cuda_index is not None:
                    self.cuda_states[cuda_index] = torch.cuda.get_rng_state(cuda_index)


class PurchasePrefixSampler:
    """Select one causal purchase task per user, with matched keep/remove controls.

    A query at t predicts pos_id[t], so its history ends at input_id[t].
    Both controls select from tasks with sufficient history AFTER removing the
    target's nonpurchase events. This holds target/negative sampling constant
    when comparing the two history policies. Original main-batch tensors are
    never modified, and no global RNG is consumed.
    """

    def __init__(
        self, seed: int, min_history: int = 2, history_policy: str = "remove-target",
        negative_sampler: PurchaseNegativeSampler | None = None,
    ):
        if min_history < 1:
            raise ValueError("purchase_min_history must be positive")
        if history_policy not in ("keep", "remove-target"):
            raise ValueError("purchase_history must be keep or remove-target")
        self.min_history = int(min_history)
        self.history_policy = history_policy
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.forward_rng = AuxiliaryForwardRNG((int(seed) + 1) % (2 ** 64))
        self.negative_sampler = negative_sampler

    def __call__(self, batch: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        inputs = batch["input_id"]
        if inputs.ndim != 2 or inputs.size(1) == 0:
            raise ValueError("Training input_id must be a nonempty [batch, length] tensor")
        for name in ("behavior_id", "pos_id", "neg_id", "target_behavior"):
            if name not in batch:
                raise ValueError(f"Purchase training requires {name}")
        for name in _POSITION_PADDING:
            if name in batch and batch[name].shape != inputs.shape:
                raise ValueError(f"Training field {name} must match input_id shape")
        if batch["user_id"].shape != (inputs.size(0),):
            raise ValueError("Training user_id must have one value per input row")

        # Transfer the four selection fields once; subsequent work is CPU-only.
        items, behaviors, positives, actions = torch.stack([
            inputs, batch["behavior_id"], batch["pos_id"], batch["target_behavior"]
        ]).detach().cpu().tolist()
        stats = {name: 0 for name in PURCHASE_HISTORY_FIELDS}
        rows, queries, selected_positions = [], [], []
        for row, (row_items, row_behaviors, row_positive, row_actions) in enumerate(
            zip(items, behaviors, positives, actions)
        ):
            candidates = []
            for query, (item, target, action) in enumerate(zip(row_items, row_positive, row_actions)):
                if item == 0 or target == 0 or action != 0:
                    continue
                prefix = [index for index in range(query + 1) if row_items[index] != 0]
                retained = [index for index in prefix
                            if row_items[index] != target or row_behaviors[index] == 0]
                if len(retained) >= self.min_history:
                    candidates.append((query, prefix, retained))
            stats["purchase_candidates"] += len(candidates)
            if not candidates:
                stats["purchase_users_skipped"] += 1
                continue
            index = int(torch.randint(len(candidates), (), generator=self.generator).item())
            query, prefix, retained = candidates[index]
            positions = prefix if self.history_policy == "keep" else retained
            rows.append(row)
            queries.append(query)
            selected_positions.append(positions)
            target = row_positive[query]
            removed = len(prefix) - len(positions)
            stats["purchase_users_selected"] += 1
            stats["purchase_original_positions"] += len(prefix)
            stats["purchase_prefix_positions"] += len(positions)
            stats["purchase_removed_positions"] += removed
            stats["purchase_samples_with_removal"] += int(removed > 0)
            stats["purchase_last_event_removed"] += int(prefix[-1] not in positions)
            stats["purchase_target_seen_before"] += int(any(row_items[i] == target for i in prefix))
            stats["purchase_target_seen_after"] += int(any(row_items[i] == target for i in positions))

        device, length = inputs.device, inputs.size(1)
        row_ids = torch.tensor(rows, dtype=torch.long, device=device)
        query_ids = torch.tensor(queries, dtype=torch.long, device=device)
        source = torch.tensor(
            [[0] * (length - len(positions)) + positions for positions in selected_positions],
            dtype=torch.long, device=device,
        ).reshape(len(rows), length)
        counts = torch.tensor([len(positions) for positions in selected_positions], device=device)
        valid = torch.arange(length, device=device).unsqueeze(0) >= (length - counts).unsqueeze(1)
        view = {
            "user_id": batch["user_id"].index_select(0, row_ids),
            "query_position": query_ids,
        }
        # Only historical item/behavior/attributes are copied into the view.
        for name in ("input_id", "behavior_id", "cat_id", "seller_id", "brand_id"):
            if name in batch:
                view[name] = batch[name].index_select(0, row_ids).gather(1, source).masked_fill(
                    ~valid, _POSITION_PADDING[name]
                )
        if "mask_pos" in batch:
            view["mask_pos"] = valid.to(dtype=batch["mask_pos"].dtype)
        for name in ("pos_id", "neg_id", "target_behavior", "target_cat", "target_seller", "target_brand"):
            if name in batch:
                view[name] = batch[name].new_full((len(rows), length), _POSITION_PADDING[name])
        for name in ("pos_id", "neg_id"):
            view[name][:, -1] = batch[name][row_ids, query_ids]
        view["target_behavior"][:, -1] = 0
        return view, stats


def purchase_recommendation_loss(
    model, batch: dict[str, torch.Tensor], negative_sampler: PurchaseNegativeSampler | None = None,
    statistics: dict | None = None,
    loss_type: str = "bce",
) -> torch.Tensor:
    """Predict the explicit purchase label at the new end of a causal prefix.

    Use BASE's purchase scoring tower with original BCE or optional pairwise
    BPR, without CL, regularization, or intermediate next-event supervision.
    """
    if loss_type not in ("bce", "bpr"):
        raise ValueError("purchase loss_type must be bce or bpr")
    history = {name: batch[name] for name in (
        "input_id", "behavior_id", "cat_id", "seller_id", "brand_id"
    ) if name in batch}
    sequence = model._encode(history, purchase_view=True, include_contrastive=False)["sequence"][:, -1, :]
    pos_ids, neg_ids = batch["pos_id"][:, -1].long(), batch["neg_id"][:, -1].long()
    if negative_sampler is not None:
        neg_ids, negative_statistics = negative_sampler.select(model, sequence, batch)
        if statistics is not None:
            statistics.update(negative_statistics)
    positive = (sequence * model._scaled_embedding(model.item_embedding, pos_ids)).sum(dim=-1)
    negative = (sequence * model._scaled_embedding(model.item_embedding, neg_ids)).sum(dim=-1)
    if loss_type == "bpr":
        return F.softplus(negative - positive).mean()
    return (F.softplus(-positive) + F.softplus(negative)).mean()
