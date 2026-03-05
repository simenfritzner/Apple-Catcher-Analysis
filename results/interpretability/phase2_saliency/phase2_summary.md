# Phase 2: Saliency Map Analysis Summary

## Channel Importance

The model relies most heavily on fronto-central channels. FC6 dominates (0.0051), nearly double FC5 (0.0030). Central channels C4 (0.0021) and C6 (0.0017) rank next, followed by centro-parietal CP3 (0.0016) and parietal P6 (0.0015). The strong FC6/FC5 lateralization and fronto-central focus is atypical for pure sensorimotor mu/beta activity, which would be expected to peak at C3/C4.

## Temporal Profile

Gradient-based saliency shows a broad peak from approximately -0.5s to +0.5s centered on cue onset. Ablation-based importance reveals a sharp functional peak at +0.1s to +0.5s post-cue, where masking causes up to 0.19 accuracy drop. Late-epoch regions (>1.0s) contribute minimally.

## Pre-Cue vs Post-Cue

Gradient saliency is nearly equal across periods (ratio 1.01). However, ablation testing reveals a strong dissociation: masking post-cue data drops accuracy by 24.5 percentage points (from 67.0% baseline), while masking pre-cue data actually improves accuracy by 8.0 points. The model functionally depends on post-cue information despite distributing gradient attention broadly.

## Left vs Right MI Differences

Saliency heatmaps for left and right MI are strikingly similar. Both show sustained FC6 activation and a transient spike at cue onset across fronto-central channels. Left MI shows marginally stronger saliency at FC5 near cue onset; right MI has a slightly more diffuse temporal distribution. The minimal class differentiation in spatial patterns suggests the model may be leveraging a shared visual/cognitive evoked response rather than lateralized sensorimotor features.

## Key Implications

1. The dominant FC6/FC5 channels suggest possible ocular or visual artifact contribution rather than canonical motor cortex activity.
2. The sharp cue-locked transient at t=0 across many channels points to a visual evoked potential (VEP) from the cue stimulus.
3. Pre-cue saliency does not functionally contribute to classification -- ablating it improves accuracy -- arguing against preparatory motor activity (MRCPs) as the primary learned feature.
4. The post-cue dependence and fronto-central topography are most consistent with the model exploiting cue-evoked visual/cognitive responses rather than sustained motor imagery patterns.
