"""Full TOMKL training settings from the JD, Tmall and Cosmetics runs."""

DATASET_NAMES = {"JD": "JD", "Tmall": "Tmall", "Cosmetics": "kaggle"}

FOUNDATION = {
    "valid": 0,
    "test_mode": "direct",
    "copy": 0,
    "eva_interval": 1,
    "lr": 0.001,
    "maxlen": 50,
    "hidden_units": 50,
    "num_behavior": 4,
    "num_blocks": 2,
    "num_blocks_behavior": 1,
    "num_heads": 1,
    "dropout_rate": 0.5,
    "l2_reg": 0,
    "ext_modules": "bcatl",
    "L": 10,
    "num_expert_sh": 3,
    "num_expert_sp": 1,
    "mmoe_hidden_units": 50,
    "tower_hidden_units": 50,
    "att_cl_rate": 1,
    "bsmmoe_cl_rate": 0.1,
    "train_aug": "none",
    "aug_objective": "purchase",
    "aug_weight": 0.05,
    "purchase_history": "remove-target",
    "purchase_min_history": 2,
    "purchase_negative": "mixed-hard",
    "purchase_hard_ratio": 0.5,
    "purchase_hard_topk": 20,
    "purchase_loss": "bpr",
    "memory_calibration": "evidence",
    "memory_scope": "purchase-only",
    "memory_gain": 0.1,
    "memory_hidden": 16,
    "memory_lr": 0.0001,
    "memory_anchor_weight": 1,
}

ADAPTATION = {
    "rank": 8,
    "lr": 0.0001,
    "aug_weight": 0.05,
    "purchase_min_history": 2,
    "purchase_hard_ratio": 0.5,
    "purchase_hard_topk": 20,
}

FOUNDATION_EPOCHS = 500
ADAPTATION_EPOCHS = 30
