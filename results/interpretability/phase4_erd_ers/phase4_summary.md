# Phase 4: ERD/ERS Analysis Summary

## Band Importance

The model prioritizes low-frequency activity. Band saliency ranking: delta (0.000572) > theta (0.000482) > mu (0.000478) > beta (0.000415) > gamma (0.000202). Delta and theta together account for more saliency than mu and beta combined, despite mu/beta being the canonical MI-discriminative bands.

## Time-Frequency Saliency

The TF saliency map is dominated by the lowest frequencies (1-3 Hz), with peak power at 1.0 Hz. Saliency in the mu (8-13 Hz) and beta (13-30 Hz) ranges is negligible by comparison. All frequency bands show peak saliency in the pre-cue window (approximately -0.5 to 0.0s), declining steadily after cue onset. This pre-cue concentration aligns with the model's previously observed stronger classification performance before the cue.

## Model-Derived vs Classical ERD/ERS

Classical ERD/ERS at C3 and C4 shows expected contralateral desynchronization patterns post-cue: beta ERD at C3 is stronger for left MI, and mu/beta patterns at C4 differentiate right MI. The model-derived ERD curves are noisier and do not clearly replicate these sustained post-cue desynchronization patterns.

## Correlation Between Model and Classical ERD

Beta-band correlations are moderate for Class 0 (left MI), strongest at C3 (r=0.360), F2 (r=0.421), and FC2 (r=0.408). Mu-band correlations are weak: C3 r=0.154, C4 r=-0.009 for Class 0. For Class 1, mu correlations are predominantly negative (e.g., P6 r=-0.357, CPz r=-0.318), suggesting the model's mu-band attention inversely tracks classical ERD for right MI.

## Interpretation

The model is not primarily relying on classical ERD/ERS for classification. Its dominant sensitivity to delta/theta frequencies and pre-cue timing points toward slow cortical potentials (MRCPs, preparatory activity) or low-frequency artifacts rather than the canonical mu/beta desynchronization expected in motor imagery. The moderate beta correlation at C3 for left MI suggests partial sensitivity to genuine sensorimotor rhythms, but this is secondary to the low-frequency, pre-cue signal the model predominantly uses.
