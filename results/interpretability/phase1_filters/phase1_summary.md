# Phase 1: Filter Analysis Summary

## Temporal Filters (Frequency Response)

The 8 temporal convolution filters (kernel length 64, at 250 Hz) show a strong bias toward low-frequency activity. The frequency response overlay reveals that the dominant spectral energy across nearly all filters concentrates in the **delta (1-4 Hz) and theta (4-8 Hz)** bands, with most filters peaking below 10 Hz. Several filters (notably Filters 1, 2, and 5) show their highest magnitude responses at or below 4 Hz. A smaller subset of filters capture energy in the **mu (8-13 Hz)** range, while beta-band (13-30 Hz) sensitivity is minimal. There is negligible energy captured above 30 Hz. The time-domain waveforms confirm this: most kernels are smooth, slowly varying signals consistent with low-frequency bandpass behavior, rather than the oscillatory patterns expected for mu/beta extraction.

## Spatial Filters (Depthwise Convolution)

The 16 depthwise spatial filters (F1=8, D=2) show **distributed weighting across the full scalp** rather than focal sensorimotor patterns. Sensorimotor channels (C3, C4, Cz, FC3, FC4, highlighted in red) do carry moderate weights in several filters (e.g., Filters 5, 7, 12), but they are not consistently dominant. Many filters place their largest absolute weights on non-sensorimotor channels, including frontal and peripheral electrode sites. Several filters show broad bipolar patterns spanning frontal-to-posterior channels, consistent with capturing slow spatial gradients or common-mode/reference-related signals rather than lateralized sensorimotor sources.

## Classifier Weights

The classifier weight heatmap (2 classes x 80 features) shows a clear **anti-symmetric pattern** between MI_left and MI_right: features with strong positive weights for one class tend to have strong negative weights for the other, and vice versa. This is expected for binary classification. The discriminative features are spread across the feature space rather than concentrated in a narrow range, suggesting the model uses information from multiple filter-spatial combinations.

## Key Takeaway

The model's temporal filters overwhelmingly target delta/theta frequencies (< 8 Hz), with limited mu-band and negligible beta-band sensitivity. This frequency profile is more consistent with **slow cortical potentials, MRCPs, or ocular/movement artifacts** than with classical sensorimotor rhythms (mu/beta ERD/ERS). The spatial filters do not show preferential weighting of sensorimotor channels. Together, these findings suggest the model's pre-cue advantage may be driven by slow preparatory potentials or low-frequency artifacts rather than canonical motor imagery oscillatory signatures.
