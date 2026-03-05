# Phase 6: Artifact Analysis Summary

## Channel Group Ablation

Baseline accuracy: 67.0%. Ablating the **central group** (14 channels) caused the largest accuracy drop (6.5%, to 60.5%), followed by **parietal** (12 channels, 5.5% drop, to 61.5%). Frontal channels (6 channels) produced only a 1.5% drop. Occipital and temporal groups had no channels in this montage and thus zero impact. Saliency importance confirms the same ranking: central (0.0014) > parietal (0.0010) > frontal (0.0009), with a frontal/central ratio of 0.60.

## Clean vs Artifact Trials

Only 1 of 200 trials was flagged as artifact-prone (100% accuracy on that single trial). With n=1, this comparison is statistically uninformative. Clean trial accuracy (66.8%, n=199) is virtually identical to overall accuracy (67.0%), indicating the model's performance does not depend on artifact-contaminated trials.

## Single Channel Ablation

The channels whose removal causes the largest accuracy decrease (i.e., most critical to the model) are **P6, CP6, FC4, C3, P2, CP3, FCz, and FC6** (3-6% drops each). These are predominantly central and centro-parietal channels consistent with sensorimotor areas. Conversely, removing **C4, CP4, CP2, and C6** actually improved accuracy by 2-4%, suggesting these channels may introduce noise or conflicting information.

Frontal/ocular channels (Fz, F4, AFz, F3, F2) show small, mixed effects (under 1.5%), arguing against ocular artifact reliance.

## Interpretation

The model relies primarily on **central and parietal channels** over sensorimotor cortex, not frontal/ocular sites. The frontal/central saliency ratio of 0.60 and the negligible frontal ablation impact (1.5%) indicate that ocular artifacts are not a major driver of classification. The spatial pattern of critical channels (FC4, C3, CP6, FCz) is consistent with lateralized sensorimotor processing relevant to left/right motor imagery. The near-absence of artifact-flagged trials further supports that the model operates on neural signals rather than systematic artifacts.
