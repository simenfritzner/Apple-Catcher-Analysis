# Phase 3: DeepLIFT Attribution Analysis Summary

## Channel Importance

DeepLIFT identifies **FC6** (0.0058) as the dominant channel, nearly twice the importance of the second-ranked FC5 (0.0034). The top-5 channels are FC6, FC5, C4, C6, and CP3. This pattern is bilateral but right-lateralized, with fronto-central and central channels dominating. Notably, classic motor cortex channels (C3, C5) rank lower (9th-10th), while lateral fronto-central sites lead.

Channel rankings between DeepLIFT and Integrated Gradients are virtually identical: Pearson r = 0.9999, Spearman rho = 0.9989, with perfect 5/5 top-channel overlap (CP3, FC5, C4, C6, FC6).

## Pre-Cue vs Post-Cue Attribution

DeepLIFT gradient-based importance is nearly equal between pre-cue (0.001346) and post-cue (0.001329), ratio 1.01. However, ablation-based analysis reveals a stark asymmetry: masking the post-cue window drops accuracy by 24.5 percentage points, while masking the pre-cue window *increases* accuracy by 8.0 points (from a 67.0% baseline). This indicates the model's classification depends heavily on post-cue information, and pre-cue signals may introduce noise or confounding features.

## Temporal Profile

The gradient-based temporal profile peaks sharply around cue onset (t=0) and the first 0.5s post-cue, then decays steadily. The saliency heatmap confirms this, with the brightest attributions concentrated on FC5/FC6 in the peri-cue window (-0.25s to +0.5s). The ablation-based temporal profile corroborates this: the largest accuracy drops occur in the 0 to 0.5s post-cue window (up to 19 percentage points).

## Method Correlation

DeepLIFT and Integrated Gradients are near-identical (Pearson r = 0.9997), forming one cluster. Vanilla saliency and Gradient x Input form another (r = 0.9968). Cross-cluster correlations remain high (r > 0.95). All four methods agree on the overall temporal shape, differing mainly in absolute scale.

## Key Takeaway

The model relies most on fronto-central channels (FC5/FC6) in the peri-cue to early post-cue window. The right-lateralized FC6 dominance and the ablation result showing pre-cue masking *improves* accuracy suggest the model is not primarily leveraging preparatory motor activity. Instead, the critical features appear in the early post-cue period, consistent with either stimulus-evoked processing or early motor response initiation rather than pre-cue MRCPs.
