# EEG Classification Interpretability Tools

Tools for understanding what CNN models learn from EEG data, specifically designed to investigate pre-cue vs post-cue information usage.

## Problem Statement

When classifying motor imagery from EEG data, if the visual stimulus (apple) appears before the cue onset, the model might learn to rely on visual artifacts rather than genuine motor-related brain activity. These tools help distinguish between:

1. **Early motor activation** - Legitimate preparatory motor signals
2. **Visual artifacts** - Confounding signals from seeing the stimulus

## Available Methods

### 1. Saliency Maps (Gradient-Based)

Shows which input features (time × channel) the model finds important:

- **Vanilla Gradients**: Simple gradient of output w.r.t. input
- **Integrated Gradients**: More robust attribution along path from baseline to input
- **Gradient × Input**: Element-wise multiplication for highlighting

### 2. Temporal Ablation (Perturbation-Based)

Measures performance degradation when temporal windows are masked:

- **Sliding Window**: Systematically mask different time windows
- **Fixed Windows**: Test specific hypotheses (pre-cue vs post-cue)
- **Progressive Ablation**: Find minimal critical time window

### 3. Channel Importance

Identifies which EEG channels are most important for predictions.

## Usage

### Single Subject Analysis

```bash
python experiments/analysis/analyze_temporal_importance.py \
    --model_path results/cnn_nested_loso_raw_window_m1.5_2.5/models/best_model_s01.pt \
    --config configs/cnn_nested_loso_raw_window1.yaml \
    --subject s01 \
    --output_dir results/interpretability/window1_s01 \
    --cue_onset 0.0
```

### Batch Analysis (All Subjects)

```bash
python experiments/analysis/run_interpretability_batch.py \
    --results_dir results/cnn_nested_loso_raw_window_m1.5_2.5 \
    --config configs/cnn_nested_loso_raw_window1.yaml \
    --output_dir results/interpretability/window1 \
    --cue_onset 0.0
```

## Output Files

Each analysis generates:

1. **pre_post_comparison.png** - Bar charts comparing pre-cue vs post-cue importance
2. **temporal_profile.png** - Time course showing when model focuses attention
3. **saliency_map.png** - Heatmap of time × channel importance
4. **channel_importance.png** - Ranking of most important EEG channels
5. **pre_post_analysis.txt** - Numerical results

## Interpreting Results

### Key Metrics

**Gradient-based importance ratio**:
- Ratio > 1: Model relies more on pre-cue information (concerning!)
- Ratio < 1: Model relies more on post-cue information (expected)

**Accuracy drop when masked**:
- High pre-cue drop: Model depends on pre-cue information
- High post-cue drop: Model depends on post-cue information (expected)

### Example Interpretation

```
Pre-cue importance: 0.0234
Post-cue importance: 0.0567
Ratio (pre/post): 0.41

Pre-cue accuracy drop: 0.023
Post-cue accuracy drop: 0.187
```

**Interpretation**: Model primarily uses post-cue information (good!). When pre-cue is masked, accuracy drops minimally (2.3%), but when post-cue is masked, accuracy drops substantially (18.7%). This suggests the model learns genuine motor imagery patterns, not visual artifacts.

### Warning Signs

- Pre/post importance ratio > 1.5: Model heavily relies on pre-cue
- Pre-cue accuracy drop > 0.10: Performance depends on pre-cue information
- Temporal profile peaks before cue onset: Suspicious pre-cue learning

## Programmatic Usage

```python
from eeg_mi.models.eegnet import EEGNet
from eeg_mi.interpretability import TemporalImportanceAnalyzer
import torch

# Load model and data
model = EEGNet(...)
model.load_state_dict(torch.load('model.pt'))

# Create analyzer
analyzer = TemporalImportanceAnalyzer(model, device='cpu')

# Analyze pre vs post cue
results = analyzer.analyze_pre_vs_post_cue(
    x=test_data,
    y=test_labels,
    cue_onset=0.0,
    sfreq=250.0,
    tmin=-1.0
)

print(f"Pre-cue importance: {results['pre_cue_importance']:.4f}")
print(f"Post-cue importance: {results['post_cue_importance']:.4f}")
print(f"Ratio: {results['importance_ratio']:.2f}")
```

## Methods Details

### Integrated Gradients

More reliable than vanilla gradients because it:
- Integrates gradients along path from baseline (zeros) to actual input
- Satisfies sensitivity and implementation invariance axioms
- Reduces noise and spurious attributions

### Temporal Ablation

Complements gradient methods by:
- Actually measuring performance impact (not just gradients)
- Robust to gradient saturation
- Provides direct evidence of necessity

## Recommended Workflow

1. **Run comprehensive analysis** for one subject to understand patterns
2. **Check pre vs post-cue metrics** - Is pre-cue reliance high?
3. **Examine temporal profile** - When does importance peak?
4. **Review saliency maps** - Which channels show pre-cue activity?
5. **Run batch analysis** - Are patterns consistent across subjects?
6. **Compare different time windows** - How does changing the window affect reliance?

## Technical Notes

- Saliency maps are averaged across trials for stability
- Ablation uses zero-masking by default (can use mean or noise)
- All visualizations include cue onset marker (red dashed line)
- Integrated gradients uses 50 interpolation steps (adjustable)
