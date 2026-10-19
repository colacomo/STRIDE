# STRIDE

**Source Code of "Reconstructing cloud-free Sentinel-2 time series under complex degradations with state-driven spatio-temporal modeling"**


## Requirements

- Python >= 3.9
- PyTorch, torchvision
- CUDA-capable GPU (recommended)

Core dependencies: `timm`, `einops`, `diffusers`, `omegaconf`, `prodict`, `tqdm`, `matplotlib`, `seaborn`, `tensorboard` (or `wandb`), `torchinfo`, `torchgeometry`, `pandas`, `geopandas`, `opencv-python`, `scipy`, `nestargs`.

Install via pip:
```bash
pip install torch torchvision timm einops diffusers omegaconf prodict tqdm matplotlib seaborn tensorboard wandb torchinfo torchgeometry pandas geopandas opencv-python scipy nestargs
```

## Data Preparation

The dataset expects the following structure under a root directory:

```
<root>/
├── DATA_S2/<patch_id>.npy          # Sentinel-2: (T, C, H, W)
├── DATA_S1A/<patch_id>.npy         # Sentinel-1: (T, C, H, W)
├── REAL_MASKS/<patch_id>.npy       # Cloud mask: (T, 1, H, W)
├── metadata.geojson                # Patch metadata (dates, folds, site IDs)
└── bad_frames.json                 # Bad frame indices per patch
```

Set `data.root` in the config to point to your data directory. 

PASTIS-R：https://zenodo.org/records/5735646

Global sampled data：Baiduyun: <img width="1279" height="1706" alt="391410276712d70c7f01e5ecda69dfdf" src="https://github.com/user-attachments/assets/cd9f82d3-970a-4de4-9508-e2154d6183a0" /> code:vtty 
 . Examples and metadata can be found in the "data" folder. 

## Training

```bash
python run.py
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--config_file` | `./train/config_train.yaml` | YAML config (merged over `train/default_train.yaml`) |
| `--save_dir` | `./results/` | Output root directory |
| `--resume_from` | `None` | Resume from a checkpoint path |


The experiment folder is created at `<save_dir>/<model_type>/<YYYY-MM-DD_HH-MM>/` and contains checkpoints and logs.

## Evaluation

```bash
python test.py  
```

Key arguments:

| Argument | Description |
|---|---|
| `--config_file` | Training config YAML |
| `--test_config` | Test-specific data config (e.g. `./train/config_test.yaml`) |
| `--checkpoint` | Path to trained model checkpoint |

Metrics reported: MAE, RMSE, SSIM, PSNR, SAM — over all pixels, masked pixels, and observed pixels.


## Models

- **STRIDE** (`lib/models/STRIDE.py`)
- **STRIDE_SAR** (`lib/models/STRIDE_SAR.py`)


