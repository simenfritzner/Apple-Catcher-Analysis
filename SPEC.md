# EEGNet Model Interpretability Analysis — Specification

## Research Question

The EEGNet model achieves unexpectedly strong cross-subject performance on the
Apple-Catcher motor imagery (MI) dataset compared to other MI paradigms. **What
is the model actually learning?** Possible explanations:

1. **Visual/ocular artifacts** — The visual stimulus (apple) is present
   pre-cue, and the model may rely on eye movements or visual evoked potentials
2. **Preparatory motor activity** — Legitimate pre-cue motor preparation
   (readiness potentials, Bereitschaftspotential)
3. **Movement-related cortical potentials (MRCPs)** — Genuine lateralized motor
   signals
4. **Event-related desynchronization/synchronization (ERD/ERS)** — Mu/beta
   suppression/rebound patterns
5. **Unknown confound** — Experimental artifact, data leakage, or paradigm-specific
   confound

### Key Observation

Previous results show **better classification pre-cue (t < 0) than post-cue
(t > 0)**. The epoch window is -1 to 2 s (cue at t = 0). Pre-cue contains
visual stimuli; post-cue is the MI period but the visual cue remains on screen.

---

## Model Under Analysis

| Property        | Value                                   |
|-----------------|-----------------------------------------|
| Architecture    | EEGNet (PyTorch)                        |
| Location        | `all_subjects_interpretation/models/model.pt` |
| Training        | All 40 subjects pooled                  |
| Validation acc  | 76.6%                                   |
| Classes         | MI_left (0) vs MI_right (1)             |
| Epoch           | -1.0 to 2.0 s (cue at t = 0)           |
| Sampling rate   | 250 Hz → 750 samples per trial          |
| Channels        | 32 EEG channels                         |
| Preprocessing   | Raw (no filtering, no normalization)    |
| Hyperparams     | F1=8, D=2, F2=16, kernel_length=64, dropout=0.3 |

---

## Analysis Plan

### Phase 1: Weight & Filter Inspection

Direct examination of what the trained network has learned at each layer.

- [ ] **1.1 Temporal filter visualization** — Plot the 8 learned temporal
  convolution kernels (Block 1, Conv2d: 1→F1, kernel size 1×64). Inspect their
  frequency response (FFT) to see which frequency bands the model attends to
  (mu ~8-13 Hz, beta ~13-30 Hz, or lower frequencies associated with artifacts)
- [ ] **1.2 Spatial filter visualization** — Plot the 16 depthwise spatial
  filters (Block 1, Conv2d: F1→F1*D, kernel size Chans×1) as topographic scalp
  maps. Check if filters correspond to sensorimotor areas (C3/C4) vs frontal/
  occipital areas (artifact-related)
- [ ] **1.3 Separable convolution filters** — Inspect Block 2 filters (1×31
  depthwise + 1×1 pointwise) for secondary temporal/spectral patterns
- [ ] **1.4 Classifier weights** — Examine the final linear layer weights to
  understand which feature dimensions drive left vs right classification

### Phase 2: Gradient-Based Attribution (Existing Tools)

Leverage the existing `eeg_mi.interpretability` module.

- [ ] **2.1 Run pre-cue vs post-cue analysis** — Use
  `TemporalImportanceAnalyzer.analyze_pre_vs_post_cue()` with integrated
  gradients on the all-subjects model and representative test data
- [ ] **2.2 Temporal importance profile** — Use
  `analyze_temporal_profile()` with sliding window to pinpoint exactly when
  importance peaks relative to cue onset
- [ ] **2.3 Channel importance ranking** — Use
  `analyze_channel_importance()` to identify top channels; map to 10-20
  positions. Key question: are top channels over sensorimotor cortex (C3/C4/Cz)
  or over frontal/occipital areas?
- [ ] **2.4 Saliency heatmaps** — Generate time × channel saliency maps
  averaged across trials, separately for left-MI and right-MI. Look for
  lateralized patterns (contralateral dominance = good sign)
- [ ] **2.5 Critical time window** — Use `find_critical_time_window()` to
  identify the minimal epoch the model needs

### Phase 3: DeepLIFT Attribution

Add DeepLIFT as a complementary attribution method (addresses limitations of
gradient-based methods for ReLU/ELU networks).

- [ ] **3.1 Implement DeepLIFT wrapper** — Use `captum.attr.DeepLift` or
  implement custom. Integrate into `SaliencyMapGenerator` as a new method
- [ ] **3.2 Compare DeepLIFT vs integrated gradients** — Run same pre/post-cue
  analysis. Agreement between methods strengthens conclusions; disagreement
  indicates method-specific artifacts
- [ ] **3.3 DeepLIFT temporal profile** — Same sliding-window temporal
  importance as Phase 2 but with DeepLIFT attributions

### Phase 4: ERD/ERS Analysis

Investigate whether the model captures event-related desynchronization (mu/beta
suppression during MI) and event-related synchronization (beta rebound).

- [ ] **4.1 Frequency-band saliency** — Bandpass filter input data into
  sub-bands (delta 1-4, theta 4-8, alpha/mu 8-13, beta 13-30, gamma 30-45),
  run saliency on each. Determine which bands drive classification
- [ ] **4.2 Time-frequency saliency** — Compute saliency in time-frequency
  domain (short-time FFT or wavelet transform of saliency maps). Look for
  mu/beta ERD patterns at sensorimotor channels
- [ ] **4.3 Model-derived ERD/ERS curves** — Extract temporal filter outputs
  from Block 1, compute power in mu/beta bands over time. Compare left-MI vs
  right-MI for contralateral ERD at C3/C4
- [ ] **4.4 Compare with classical ERD/ERS** — Compute traditional ERD/ERS
  from raw EEG (no model) and correlate with model-derived patterns

### Phase 5: MRCP & Preparatory Potential Analysis

Investigate movement-related cortical potentials and readiness potentials.

- [ ] **5.1 Low-frequency saliency** — Focus saliency analysis on < 5 Hz
  components where MRCPs live. Check if pre-cue importance comes from slow
  cortical potentials
- [ ] **5.2 Temporal filter frequency response** — From Phase 1.1, determine
  if any learned filters have low-frequency (< 5 Hz) passbands indicative of
  MRCP sensitivity
- [ ] **5.3 Grand-average comparison** — Overlay model temporal importance
  curve on grand-average ERP at Cz/FCz. Check alignment with
  Bereitschaftspotential components (BP, NS')
- [ ] **5.4 Lateralized readiness potential (LRP)** — Compute LRP from C3/C4
  channels (classical method) and compare timing with model's lateralized
  saliency patterns

### Phase 6: Artifact Investigation

Specifically test whether the model exploits ocular or muscular artifacts.

- [ ] **6.1 Frontal channel analysis** — Compare saliency at frontal channels
  (Fp1, Fp2, F3, F4) vs central channels (C3, C4, Cz). High frontal importance
  suggests EOG contamination
- [ ] **6.2 EOG correlation** — If EOG channels exist in the data, correlate
  saliency-weighted input with EOG signal. High correlation = artifact reliance
- [ ] **6.3 Blink/saccade epoch analysis** — Identify trials with high
  frontal amplitude (likely blinks), run model separately on clean vs
  artifact-contaminated trials. If accuracy drops on clean trials → artifact
  dependence
- [ ] **6.4 ICA-cleaned comparison** — Run ICA on raw data, remove ocular
  components, re-evaluate model on cleaned data. Accuracy drop quantifies
  artifact contribution
- [ ] **6.5 Peripheral channel ablation** — Ablate peripheral/frontal channels
  and measure accuracy impact vs ablating central channels

### Phase 7: Cross-Subject Consistency

Understand why cross-subject performance is unusually high.

- [ ] **7.1 Per-subject saliency patterns** — Run attribution for each of the
  40 subjects individually. Compute inter-subject correlation of saliency maps.
  High consistency = shared signal (could be artifact or genuine)
- [ ] **7.2 Subject clustering** — Cluster subjects by their saliency patterns.
  Check if clusters correspond to performance tiers, demographics, or recording
  sessions
- [ ] **7.3 Left-out subject analysis** — For a few subjects, compare model
  attributions when the subject was in vs out of training data (requires
  retraining or using existing LOSO models if available)

### Phase 8: Visualization & Reporting

Generate publication-quality figures and summary statistics.

- [ ] **8.1 Topographic saliency maps** — MNE-style topomap at key time
  points (pre-cue peak, cue onset, post-cue peak) showing spatial distribution
  of model attention
- [ ] **8.2 Temporal importance figure** — Multi-panel figure: (top) temporal
  importance curve with pre/post-cue shading, (bottom) time-frequency saliency
- [ ] **8.3 Filter gallery** — Grid of all learned filters with frequency
  responses and spatial topomaps
- [ ] **8.4 Comparison table** — Summary table: pre/post-cue importance ratio,
  critical time window, top channels, frequency bands, artifact indicators
- [ ] **8.5 Statistical tests** — Permutation tests or bootstrap CIs for
  pre-vs-post importance difference, channel importance rankings

---

## Existing Infrastructure

Already implemented in `eeg_mi/interpretability/`:

| Module               | Status | What it does                                    |
|----------------------|--------|-------------------------------------------------|
| `saliency.py`        | Done   | Vanilla grad, integrated grad, gradient × input |
| `ablation.py`        | Done   | Sliding window, fixed window, progressive       |
| `temporal_analysis.py`| Done  | Pre/post-cue comparison, temporal profile, critical window |
| `visualization.py`   | Done   | Plotting utilities for saliency/ablation        |

### Still Needed

| Component                        | Priority | Complexity |
|----------------------------------|----------|------------|
| Weight/filter visualization      | High     | Low        |
| DeepLIFT integration             | High     | Medium     |
| Topographic scalp maps           | High     | Medium     |
| Frequency-band saliency          | High     | Medium     |
| ERD/ERS model-derived curves     | High     | Medium     |
| MRCP/LRP analysis                | Medium   | Medium     |
| Artifact-specific tests (6.1-6.5)| High     | Medium-High|
| ICA-cleaned comparison           | Medium   | High       |
| Cross-subject consistency        | Medium   | Medium     |
| Statistical testing              | Medium   | Low        |
| Publication figures              | Low*     | Medium     |

*Low priority now, high priority later

---

## Data Access

- **Raw data**: `data/raw/apple_catcher/s01-s40/` (NPZ format)
- **Channel layout**: Needs verification — check NPZ files for channel names/montage
- **Loading**: `eeg_mi.data.loaders.load_subject_data_npz()`

## Dependencies to Add

- `captum` — For DeepLIFT (and potentially other attribution methods like
  GradCAM, SHAP)
- `mne` — Already used, needed for topographic maps
- Existing: `torch`, `numpy`, `scipy`, `matplotlib`

---

## Execution Order (Recommended)

1. **Phase 1** (filters) — Quick wins, immediate insight into what frequency
   bands and spatial regions the model targets
2. **Phase 2** (existing tools) — Run what's already built on the all-subjects
   model
3. **Phase 6** (artifacts) — Address the most pressing question: is it artifacts?
4. **Phase 4** (ERD/ERS) — If not pure artifacts, is it legitimate ERD?
5. **Phase 5** (MRCP) — Check preparatory potentials
6. **Phase 3** (DeepLIFT) — Cross-validate findings with different method
7. **Phase 7** (cross-subject) — Understand the surprisingly good transfer
8. **Phase 8** (figures) — Final reporting
