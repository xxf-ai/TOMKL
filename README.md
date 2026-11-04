# TOMKL

**Target-Oriented Multi-Behavior Knowledge Learning** is a PyTorch implementation for purchase recommendation from sequences of browsing, auxiliary actions, and purchases.

| Module | Role |
| --- | --- |
| TDPL — Target-Decoupled Multi-Behavior Preference Learning | Learns a purchase ranking from training histories with the target item's auxiliary events removed. |
| TEBC — Target-Evidence-Adaptive Behavioral Context Modeling | Adjusts the contribution of behavior memories using the observed history. |
| TCPSA — Target-Conditioned Preference Space Adaptation | Learns a residual preference from the latest behavior states while the shared foundation stays frozen. |

The shared sequence, behavior and attribute encoders are based on the [BASE method and data protocol](https://github.com/Erin-Gr/BASE). See [NOTICE](NOTICE) for attribution.

## Install

Use Python 3.10 or newer and install PyTorch for your CPU or CUDA environment. From the cloned repository, run:

```sh
python -m pip install .
python -m tomkl --help
```

## Data

Obtain the processed JD, Tmall or Cosmetics archive from the corresponding directory in [BASE's data repository](https://github.com/Erin-Gr/BASE/tree/main/data_process). Cosmetics is stored under `kaggle` upstream. Give the CLI the original `processed_data_att.zip` file, an extracted CSV directory, or a CSV prefix such as `data/JD/processed_data_att/JD`.

The loader uses the train, valid, between-val-and-test, and test splits. Item and attribute IDs follow the source data. Behavior ID 0 is purchase, 2 is favorite or cart, and 3 is browse. Cosmetics has no seller field. The model learns from training histories and selects checkpoints by validation HR@10. Test histories include the observed interactions before the held-out purchase.

## Train, evaluate and predict

```sh
python -m tomkl train --dataset JD --data-path data/JD/processed_data_att.zip --output-dir runs/jd --device auto
python -m tomkl evaluate --checkpoint runs/jd/stage2/best_valid.pt --data-path data/JD/processed_data_att.zip
python -m tomkl predict --checkpoint runs/jd/stage2/best_valid.pt --data-path data/JD/processed_data_att.zip --user-id 1 --top-k 10
```

Use `--dataset Tmall` or `--dataset Cosmetics` with the corresponding archive. Use a fresh output directory per training run. `--stage1-epochs 1 --stage2-epochs 1 --device cpu` is useful for testing the workflow on a small dataset. The complete checkpoint is self-contained. The stage-two trainer considers the unadapted epoch-zero model when choosing the earliest validation HR@10 maximum.

`evaluate` reports full-catalog HR, NDCG, and MRR. `predict` uses the history of an eligible user in the processed test split and ranks the catalog using the model's scores. Pass an eligible user ID from that split. Scores are ranking scores, and item IDs refer to the processed catalog.

## Source layout

- `tomkl/encoder.py` and the expert layers implement the shared encoders.
- `tomkl/tdpl.py`, `tomkl/tebc.py`, and `tomkl/tcpsa.py` implement the three modules.
- `tomkl/data.py`, `tomkl/profiles.py`, and `tomkl/cli.py` handle data, run settings and commands.
- Private modules implement the shared training, evaluation and checkpoint routines.
