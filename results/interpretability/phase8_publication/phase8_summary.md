# Phase 8: Publication-Ready Statistical Analysis Summary

## Temporal Importance

The model relies on pre-cue activity more than post-cue activity. Mean saliency is highest in the late pre-cue window (approximately -0.5 to 0 s), peaking near cue onset, and declines steadily throughout the post-cue period. Pre-cue importance (0.001178) slightly exceeds post-cue importance (0.001165), yielding a pre/post ratio of 1.010. Critically, ablating the pre-cue window causes an accuracy *increase* of 8 percentage points (from 67.0% baseline), while ablating the post-cue window causes a 24.5 percentage point accuracy *drop*. This dissociation suggests the model extracts discriminative features from both windows, but the post-cue period carries more task-relevant classification information despite lower saliency magnitude.

## Statistical Tests

A two-sided permutation test (n=10,000) confirms the pre-cue vs. post-cue saliency difference is statistically significant (observed difference = 1.2e-5, p = 0.0029). However, the 95% bootstrap confidence interval for this difference spans zero ([-1.6e-5, 4.1e-5]), indicating that while the effect is reliable under permutation, its magnitude is small and uncertain in direction at the individual-sample level.

## Channel Importance

FC6 dominates all other channels (mean importance 0.00509, 95% CI [0.00473, 0.00546]), followed by FC5 (0.00301) and C4 (0.00206). The top-5 channels are FC6, FC5, C4, C6, and CP3. This fronto-central and right-lateralized pattern is consistent across bootstrap resamples with non-overlapping CIs for the top three channels. Midline and posterior channels (Fz, C1, C2) contribute least.

## Topographic Saliency

Topographic maps at -0.5 s, 0 s, +0.5 s, and +1.0 s show saliency concentrated over right fronto-central regions (FC5/FC6) at cue onset, shifting to bilateral fronto-central coverage post-cue. The strong FC6/FC5 activation at and around cue onset is spatially inconsistent with pure sensorimotor mu/beta ERD/ERS and instead overlaps with regions associated with ocular/frontal artifacts or preparatory frontal activity.

## Learned Filters

The eight temporal filters (T1-T8) show varied frequency tuning. Several filters have dominant low-frequency content (below 10 Hz), while others capture broader spectral ranges. The corresponding spatial filters (S1-S8) show lateralized and frontal topographies, with some filters clearly targeting fronto-central regions.

## Key Conclusions

The model's strongest features are fronto-central (FC6/FC5), not classically sensorimotor (C3/C4). The statistically significant pre-cue reliance and the frontal spatial distribution raise concern that the model may be exploiting preparatory or artifact-related signals rather than canonical motor imagery patterns.
