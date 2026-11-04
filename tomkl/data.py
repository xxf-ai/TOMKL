"""Data loader for the original BASE Tmall/Kaggle/JD interaction protocol.

The published archives contain CSV files under ``processed_data_att/``. This
loader accepts either those archives or an already extracted directory. Item
and attribute IDs remain exactly as stored in the data; zero is padding.
"""

from __future__ import annotations

import logging
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class Dataset:
    """Original per-user BASE sequences and full-catalog evaluation candidates.

    ``is_valid=1`` uses train for history/training and valid for evaluation.
    ``is_valid=0`` uses train + valid + between_val_test for history/training
    and test for evaluation, matching the original final-test protocol.
    """

    def __init__(self, file_prefix: str | Path, is_valid: int = 1):
        self.prefix = self._resolve_prefix(file_prefix)
        self.data = str(self.prefix)
        self.dataset_name = self.prefix.name.lower()
        if self.dataset_name not in {"tmall", "kaggle", "jd"}:
            raise ValueError(
                f"Unsupported BASE dataset {self.prefix.name!r}; expected Tmall, kaggle or JD"
            )
        self.is_valid = bool(is_valid)
        self.columns = ["user_id", "item_id", "cat_id"]
        if self.dataset_name != "kaggle":
            self.columns.append("seller_id")
        self.columns += ["brand_id", "timestamp", "action_type"]

        self.train = self._read_split("train")
        validation = self._read_split("valid")
        if self.is_valid:
            self.valid = validation
        else:
            self.bet_val_test = self._read_split("between_val_test")
            self.valid = self._read_split("test")
            self.train = pd.concat(
                [self.train, validation, self.bet_val_test], ignore_index=True
            )
        # The published negative file is retained for compatibility with the
        # original Dataset API, although gen_batch_candidate ranks the full
        # catalog rather than sampling from it.
        try:
            self.candidate = self._read_split("negative", columns=None)
        except FileNotFoundError:
            self.candidate = pd.DataFrame()

        if self.dataset_name == "kaggle":
            self.train["seller_id"] = 0
            self.valid["seller_id"] = 0

    @staticmethod
    def _resolve_prefix(file_prefix: str | Path) -> Path:
        prefix = Path(file_prefix).expanduser()
        if prefix.is_absolute():
            return prefix
        code_dir = Path(__file__).resolve().parent
        candidates = [Path.cwd() / prefix, code_dir / prefix]
        for candidate in candidates:
            if Path(f"{candidate}_train.csv").is_file():
                return candidate.resolve()
            if (candidate.parent.parent / "processed_data_att.zip").is_file():
                return candidate.resolve()
        return candidates[0].resolve()

    def _read_split(self, name: str, columns: list[str] | None = ...) -> pd.DataFrame:
        if columns is ...:
            columns = self.columns
        csv_path = Path(f"{self.prefix}_{name}.csv")
        if csv_path.is_file():
            return pd.read_csv(csv_path, header=None, names=columns)

        archive_path = self.prefix.parent.parent / "processed_data_att.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Missing {csv_path} and {archive_path}; extract the BASE data "
                "archive or pass --path to a directory with processed CSVs"
            )
        with zipfile.ZipFile(archive_path) as archive:
            suffix = f"/{self.prefix.name}_{name}.csv"
            members = [member for member in archive.namelist() if ("/" + member).endswith(suffix)]
            if len(members) != 1:
                raise FileNotFoundError(f"Expected one *{suffix} in {archive_path}")
            with archive.open(members[0]) as stream:
                return pd.read_csv(stream, header=None, names=columns)

    @staticmethod
    def _pad(values: np.ndarray, length: int, pad_value: int = 0) -> list[int]:
        values = values[-length:]
        return np.pad(
            values.astype(np.int64, copy=False),
            (length - len(values), 0),
            constant_values=pad_value,
        ).tolist()

    def fix_length(self, maxlen: int = 50) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        self.maxlen = maxlen
        self.train = self.train.sort_values(
            ["user_id", "timestamp"], kind="mergesort"
        )
        if self.train.empty:
            raise ValueError("Training data is empty")

        # Preserve the original cardinalities (computed before truncation).
        self.user_maxid = int(self.train.user_id.max())
        self.item_maxid = int(self.train.item_id.max())
        self.cat_maxid = int(self.train.cat_id.max())
        self.seller_maxid = int(self.train.seller_id.max())
        self.brand_maxid = int(self.train.brand_id.max())
        self.user_set = set(map(int, self.train.user_id.unique()))
        self.user_list = sorted(self.user_set)
        self.item_set = set(map(int, self.train.item_id.unique()))
        self.item_pool = sorted(self.item_set)
        self.item_list = list(range(1, self.item_maxid + 1))
        self.cat_set = set(map(int, self.train.cat_id.unique()))
        self.seller_set = set(map(int, self.train.seller_id.unique()))
        self.brand_set = set(map(int, self.train.brand_id.unique()))
        self.eval_item_maxid = max(self.item_maxid, int(self.valid.item_id.max()))
        LOGGER.info(
            "dataset=%s users=%d items=%d catalog_max=%d evaluation_users=%d",
            self.dataset_name,
            len(self.user_set),
            len(self.item_set),
            self.eval_item_maxid,
            self.valid.user_id.nunique(),
        )

        self.train = self.train.groupby("user_id", sort=False).tail(maxlen + 1)
        self.train_seq: dict[int, list[int]] = {}
        self.train_behavior_seq: dict[int, list[int]] = {}
        self.train_cat_seq: dict[int, list[int]] = {}
        self.train_seller_seq: dict[int, list[int]] = {}
        self.train_brand_seq: dict[int, list[int]] = {}
        self.train_mask_real_pos_items: dict[int, list[float]] = {}

        for user, rows in self.train.groupby("user_id", sort=False):
            uid = int(user)
            self.train_seq[uid] = self._pad(rows.item_id.to_numpy(), maxlen + 1)
            self.train_behavior_seq[uid] = self._pad(
                rows.action_type.to_numpy(), maxlen + 1, pad_value=4
            )
            self.train_cat_seq[uid] = self._pad(rows.cat_id.to_numpy(), maxlen + 1)
            self.train_seller_seq[uid] = self._pad(rows.seller_id.to_numpy(), maxlen + 1)
            self.train_brand_seq[uid] = self._pad(rows.brand_id.to_numpy(), maxlen + 1)
            purchases = {
                item for item, behavior in zip(
                    self.train_seq[uid], self.train_behavior_seq[uid]
                ) if behavior == 0
            }
            self.train_mask_real_pos_items[uid] = [
                0.0 if item in purchases else 1.0 for item in self.train_seq[uid]
            ]

        self.valid_seq: dict[int, list[int]] = {}
        self.valid_behavior_seq: dict[int, list[int]] = {}
        self.valid_cat_seq: dict[int, list[int]] = {}
        self.valid_seller_seq: dict[int, list[int]] = {}
        self.valid_brand_seq: dict[int, list[int]] = {}
        self.valid_neg_cand: dict[int, list[int]] = {}
        negative_by_user = {}
        if not self.candidate.empty:
            negative_by_user = {
                int(row[0]): [int(value) for value in row[1:]]
                for row in self.candidate.itertuples(index=False, name=None)
            }
        # The original loader takes the first valid/test row per user, without
        # reordering those rows. Histories are shifted by one before appending
        # the held-out item so each evaluation input has length maxlen.
        for row in self.valid.drop_duplicates("user_id").itertuples(index=False):
            user = int(row.user_id)
            if user not in self.train_seq:
                continue
            self.valid_seq[user] = self.train_seq[user][1:] + [int(row.item_id)]
            self.valid_behavior_seq[user] = (
                self.train_behavior_seq[user][1:] + [int(row.action_type)]
            )
            self.valid_cat_seq[user] = self.train_cat_seq[user][1:] + [int(row.cat_id)]
            self.valid_seller_seq[user] = (
                self.train_seller_seq[user][1:] + [int(row.seller_id)]
            )
            self.valid_brand_seq[user] = (
                self.train_brand_seq[user][1:] + [int(row.brand_id)]
            )
            self.valid_neg_cand[user] = negative_by_user.get(user, [])

    def gen_batch_candidate(self, batch_target: list[list[int]] | np.ndarray) -> np.ndarray:
        """Return [positive, all other catalog IDs] for each row, as before."""
        targets = np.asarray(batch_target, dtype=np.int64).reshape(-1)
        catalog = np.arange(1, self.eval_item_maxid + 1, dtype=np.int64)
        candidates = np.empty((len(targets), len(catalog)), dtype=np.int64)
        for row, target in enumerate(targets):
            if target < 1 or target > self.eval_item_maxid:
                raise ValueError(f"Invalid held-out item ID {target}")
            candidates[row, 0] = target
            candidates[row, 1:target] = catalog[:target - 1]
            candidates[row, target:] = catalog[target:]
        return candidates

    def sample_batch(self, batch_size: int = 128, user_ids=None):
        """Return the original 13 fields in their original order.

        Passing ``user_ids`` lets the trainer visit each user once per epoch;
        omitting it retains random user sampling for external callers.
        """
        users = (
            list(map(int, user_ids))
            if user_ids is not None
            else random.sample(self.user_list, batch_size)
        )
        if not users:
            raise ValueError("Cannot sample an empty batch")
        fields = [[] for _ in range(12)]
        for user in users:
            sequence = self.train_seq[user]
            seen = set(sequence)
            available_count = len(self.item_set) - len(self.item_set.intersection(seen))
            if available_count >= self.maxlen:
                # Rejection sampling is uniform without replacement and avoids
                # copying the whole item catalog for every user in every epoch.
                selected = set()
                negatives = []
                while len(negatives) < self.maxlen:
                    item = random.choice(self.item_pool)
                    if item not in seen and item not in selected:
                        selected.add(item)
                        negatives.append(item)
            elif available_count:
                negative_pool = [item for item in self.item_pool if item not in seen]
                negatives = random.choices(negative_pool, k=self.maxlen)
            else:
                # Degenerate tiny datasets have no unseen training item. Zero
                # is padding and contributes no meaningful negative signal.
                negatives = [0] * self.maxlen
            values = (
                sequence[:-1],
                self.train_behavior_seq[user][:-1],
                sequence[1:],
                negatives,
                self.train_behavior_seq[user][1:],
                self.train_mask_real_pos_items[user][:-1],
                self.train_cat_seq[user][:-1],
                self.train_seller_seq[user][:-1],
                self.train_brand_seq[user][:-1],
                self.train_cat_seq[user][1:],
                self.train_seller_seq[user][1:],
                self.train_brand_seq[user][1:],
            )
            for field, value in zip(fields, values):
                field.append(value)
        return (users, *fields)
