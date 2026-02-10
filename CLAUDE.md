# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for a master thesis investigating **why an EEGNet model achieves unexpectedly strong cross-subject motor imagery (MI) classification** on the Apple-Catcher paradigm. The primary analysis target is `all_subjects_interpretation/models/model.pt` — an EEGNet trained on all 40 subjects (76.6% validation accuracy, binary left/right MI, -1 to 2s epochs at 250 Hz, 32 channels, raw/unfiltered).

The research question: the model classifies better pre-cue than post-cue. Is it learning visual/ocular artifacts, preparatory motor activity, MRCPs, ERD/ERS, or some unknown confound? See `SPEC.md` for the full analysis plan.

## Development Environment

- Python 3.11 via conda (no requirements.txt/pyproject.toml — dependencies managed externally)
- Core deps: `torch`, `numpy`, `scipy`, `scikit-learn`, `mne`, `pyriemann`, `matplotlib`, `seaborn`
- Planned addition: `captum` (for DeepLIFT)
- No test suite, linter, or CI configured — this is a research-only codebase
- Data directory (`data/`) is gitignored

## Package Architecture (`eeg_mi/`)

### Data flow

```
Raw EEG (NPZ/FIF) → load_all_subjects() → prepare_loso_fold()
  → normalize_*_split() [AFTER split] → trainer.train() → trainer.evaluate()
  → interpretability analysis
```

### Key modules

- **`data/loaders.py`** — `load_subject_data_npz()` loads per-subject NPZ trials (shape: `n_trials × n_channels × n_timepoints`). `load_all_subjects()` batch-loads into `Dict[subject_id, (data, labels)]`. `prepare_loso_fold()` creates LOSO train/test splits.

- **`data/normalization.py`** — Z-score normalization with strict data leakage prevention. Training stats never computed from test data. `normalize_train_test_split()` for single-stage, `normalize_calibration_test_split()` for two-stage transfer learning.

- **`models/eegnet.py`** — EEGNet (PyTorch `nn.Module`). Input: `(batch, 1, channels, samples)` NCHW format. Block 1: temporal conv (1×64) → depthwise spatial conv → ELU → AvgPool(1×8). Block 2: separable conv (1×31) → AvgPool(1×16). Total downsampling: 128×. Supports `classifier` and `feature_extractor` modes.

- **`training/nested_loso.py`** — `nested_loso_cv()`: outer loop holds out test subject, inner loop does K-fold HP search on remaining subjects. Uses trainer factory pattern — detects trainer type via `inspect.signature()` to handle CNN vs DANN trainers.

- **`training/eegnet_trainer.py`** — Supports single-stage and two-stage (pretrain on source → finetune on target calibration) training with optional feature layer freezing.

- **`interpretability/`** — Three classes exported: `SaliencyMapGenerator` (vanilla/integrated gradients/gradient×input), `TemporalAblationStudy` (sliding/fixed/progressive window masking), `TemporalImportanceAnalyzer` (unified pre/post-cue analysis combining both methods).

### Critical conventions

- **Normalization MUST happen after train/test split** — the codebase enforces this pattern and logs warnings if violated
- **EEGNet input format**: always `(batch, 1, channels, samples)` — the extra dim is added by `EEGNetTrainer._create_dataloader()`
- **NPZ trial format**: each file is `trial_XXX.npz` with keys `'data'` (channels × timepoints) and `'label'`
- **Cue onset at t=0**: epochs span -1.0 to 2.0s, so sample index 250 is cue onset at 250 Hz

## Data Layout

```
data/raw/apple_catcher/s01-s40/   # 40 subjects, NPZ trial files
data/raw/BCICIV-2b/               # Benchmark dataset
all_subjects_interpretation/
  models/model.pt                 # Primary model under analysis
  *_results_*.json                # Training config and metrics
```

## Running Analysis

```python
from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability import TemporalImportanceAnalyzer
import torch

# Load model (F1=8, D=2, F2=16, kernel_length=64, 32 channels, 750 samples)
model = EEGNet(nb_classes=2, Chans=32, Samples=750,
               kernLength=64, F1=8, D=2, F2=16, dropoutRate=0.3)
model.load_state_dict(torch.load('all_subjects_interpretation/models/model.pt'))

# Load data
from eeg_mi.data import load_subject_data
data, labels = load_subject_data('data/raw/apple_catcher/s01', tmin=-1.0, tmax=2.0)
```
