# Handoff Document

## Goal

Master thesis analysis investigating **why an EEGNet model achieves unexpectedly strong cross-subject motor imagery classification** on the Apple-Catcher paradigm. Previous results showed better classification pre-cue than post-cue, raising concerns about visual artifacts vs genuine motor signals. The full analysis plan is in `SPEC.md` (8 phases).

## Current Progress

### Completed Phases

**Phase 1: Filter & Weight Visualization** (PR #2, merged)
- `eeg_mi/interpretability/filters.py` — `FilterVisualizer` class
- Extracts/plots temporal filters (8 × 64 kernels), spatial filters (16 × 32), separable conv, classifier weights
- Frequency response with mu/beta band annotations
- Results in `results/interpretability/phase1_filters/`

**Phase 2: Gradient-Based Attribution** (existing tools, run via script)
- Pre-cue vs post-cue analysis, temporal profiles, channel importance, saliency heatmaps, class-conditional saliency (left vs right)
- Results in `results/interpretability/phase2_saliency/`

**Phase 3: DeepLIFT Integration** (PR #4, merged)
- `deeplift()` method added to `SaliencyMapGenerator` (uses captum, lazy import)
- DeepLIFT wired into all `TemporalImportanceAnalyzer` dispatch blocks
- `compare_methods()` and `plot_method_comparison()` for cross-method validation
- NOT yet run on real data (only code integrated)

**Phase 6: Artifact Investigation** (PR #3, merged)
- `eeg_mi/interpretability/artifacts.py` — `ArtifactAnalyzer` class
- Channel group saliency, channel group ablation, artifact trial detection, clean-vs-artifact comparison, single channel ablation
- Results in `results/interpretability/phase6_artifacts/`

### Key Findings (5 subjects, 200 trials)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Pre/post gradient importance ratio | 1.01 | Nearly equal (inconclusive alone) |
| Pre-cue accuracy drop (ablation) | **-0.080** | Model IMPROVES without pre-cue |
| Post-cue accuracy drop (ablation) | **+0.245** | Model needs post-cue period |
| Frontal/Central saliency ratio | **0.60** | Model focuses on central, not frontal |
| Central channel ablation drop | 0.065 | Central channels are critical |
| Frontal channel ablation drop | 0.015 | Frontal channels barely matter |
| Top single channels | C4, CP4, CP2, C6 | All right sensorimotor |
| Artifact trials detected | 1/200 (0.5%) | Very few frontal artifacts |
| Baseline accuracy (5 subjects) | 0.670 | Lower than full 40-subject model (0.766) |

**Initial conclusion**: The model primarily uses sensorimotor signals, not frontal/ocular artifacts. Pre-cue information is noise, not a confound.

### Uncommitted Changes

- `scripts/run_analysis.py` — main analysis script (loads model, data, runs phases 1/2/6)
- `eeg_mi/interpretability/ablation.py` — off-by-one bugfix on line 209 (`times[min(end_idx, n_samples - 1)]`)
- `results/` directory — all generated figures and text results
- `__pycache__` files

## What Worked

- **Model loading**: checkpoint is a dict with key `model_state_dict` (not raw state_dict). Use `weights_only=False`.
- **Data is FIF format** (not NPZ): `data/raw/apple_catcher/s01-s40/` contains `*_epo.fif` files. Load with `mne.read_epochs()`, crop to `-1.0` to `2.0` s, take first 750 samples.
- **Channel montage**: 32 channels, no Fp1/Fp2 (most frontal is AFz, F1-F4). No occipital or temporal electrodes in this montage. The artifact detector's frontal detection works but finds very few artifacts.
- **EEGNet params**: `nb_classes=2, Chans=32, Samples=750, kernLength=64, F1=8, D=2, F2=16, dropoutRate=0.3`
- Running analyses on 200-trial subset is fast (~2 min total). Full 1040 trials from 5 subjects loads in seconds.

## What Didn't Work

- `torch.load(..., weights_only=True)` fails because model.pt contains config metadata. Must use `weights_only=False`.
- `ablation.py` line 209 had an off-by-one bug: `times[end_idx]` when `end_idx == n_samples`. Fixed to `times[min(end_idx, n_samples - 1)]`. **This fix is uncommitted — commit it.**
- The `find_critical_time_window()` returned `0.50s to -1.00s` which seems inverted/buggy. Investigate the progressive ablation logic if this method is important.
- Parallel agents couldn't run `Bash` commands for git — had to handle git workflow in the main conversation.

## Next Steps

### Immediate (uncommitted work)
1. Commit the ablation bugfix and `scripts/run_analysis.py`
2. Add `results/` and `__pycache__/` to `.gitignore`

### Phase 2 follow-up
3. Run analysis on **all 40 subjects** (not just 5) for publication-quality numbers
4. Investigate the `find_critical_time_window()` bug (returns inverted start/end)

### Phase 3: DeepLIFT (code exists, not yet run)
5. Run `compare_methods()` on real data to cross-validate integrated gradients vs DeepLIFT
6. Generate `plot_method_comparison()` figure

### Phase 4: ERD/ERS Analysis (not started)
7. Bandpass filter input into sub-bands (mu 8-13 Hz, beta 13-30 Hz), run saliency on each
8. Time-frequency saliency analysis
9. Model-derived ERD/ERS curves from Block 1 activations at C3/C4
10. Compare with classical ERD/ERS from raw EEG

### Phase 5: MRCP & Preparatory Potentials (not started)
11. Low-frequency (<5 Hz) saliency to check for MRCPs
12. Check temporal filter frequency responses for DC/low-freq passbands
13. Overlay model importance on grand-average ERP at Cz/FCz

### Phase 7: Cross-Subject Consistency (not started)
14. Per-subject saliency patterns and inter-subject correlation
15. Subject clustering by saliency patterns

### Phase 8: Publication Figures (not started)
16. Topographic scalp maps at key time points (needs MNE topomap)
17. Multi-panel publication figures
18. Statistical tests (permutation/bootstrap)

## Key Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Full 8-phase analysis plan |
| `CLAUDE.md` | Claude Code project guidance |
| `scripts/run_analysis.py` | Main analysis script |
| `all_subjects_interpretation/models/model.pt` | Trained EEGNet model |
| `eeg_mi/interpretability/filters.py` | Filter visualization (Phase 1) |
| `eeg_mi/interpretability/artifacts.py` | Artifact analysis (Phase 6) |
| `eeg_mi/interpretability/saliency.py` | Saliency + DeepLIFT (Phase 2/3) |
| `eeg_mi/interpretability/temporal_analysis.py` | Temporal analysis |
| `eeg_mi/interpretability/visualization.py` | All plotting functions |
| `results/interpretability/` | Generated analysis outputs |
