"""Subject selection for transfer learning.

Selects the most relevant source subjects for a target subject by training
simple classifiers on each source subject and testing on target calibration data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def select_subjects_lda(
    source_subjects: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target_cal_data: np.ndarray,
    target_cal_labels: np.ndarray,
    train_size: int = 100,
    selection_method: str = 'threshold',
    threshold: float = 0.55,
    top_k: int = 9,
    min_subjects: int = 5,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[Tuple[str, float]]]:
    """Select most relevant source subjects using LDA.

    For each source subject:
    1. Train LDA on that subject's data
    2. Test on target calibration set
    3. Record accuracy

    Then select subjects based on method:
    - 'threshold': Select subjects with accuracy > threshold
    - 'top_k': Select top K subjects by accuracy
    - 'adaptive': Select subjects above threshold, but at least min_subjects

    Note: Data should already be normalized (e.g., z-score per subject) before calling
    this function. LDA is relatively scale-invariant, but normalization helps ensure
    consistent feature scales across subjects.

    Args:
        source_subjects: Dict mapping subject_id -> (data, labels).
                        Data should be normalized (n_trials, channels, timepoints)
        target_cal_data: Target calibration data (n_cal, channels, timepoints).
                        Should be normalized using same method as source subjects
        target_cal_labels: Target calibration labels (n_cal,)
        train_size: Number of trials to use from each source subject
        selection_method: 'threshold', 'top_k', or 'adaptive'
        threshold: Accuracy threshold for selection (default: 0.55 = better than chance)
        top_k: Number of subjects to select if method='top_k'
        min_subjects: Minimum subjects to select if method='adaptive'

    Returns:
        selected_subjects: Dict of selected source subjects
        rankings: List of (subject_id, accuracy) sorted by accuracy
    """
    print(f"\n{'='*80}")
    print("SUBJECT SELECTION WITH LDA")
    print(f"{'='*80}")
    print(f"Method: {selection_method}")
    if selection_method == 'threshold':
        print(f"Threshold: {threshold:.1%}")
    elif selection_method == 'top_k':
        print(f"Top-K: {top_k}")
    elif selection_method == 'adaptive':
        print(f"Threshold: {threshold:.1%}, Min subjects: {min_subjects}")
    print(f"Source subjects: {len(source_subjects)}")
    print(f"Target calibration: {len(target_cal_data)} trials")
    print(f"{'='*80}\n")

    # Flatten target calibration data for LDA
    n_cal = target_cal_data.shape[0]
    n_features = target_cal_data.shape[1] * target_cal_data.shape[2]
    target_cal_flat = target_cal_data.reshape(n_cal, n_features)

    # Evaluate each source subject
    subject_scores = []
    total_subjects = len(source_subjects)

    print(f"Evaluating {total_subjects} source subjects...")
    print(f"Features per sample: {n_features} (this may be slow with high-dimensional data)")
    import time

    for idx, (subject_id, (data, labels)) in enumerate(source_subjects.items(), 1):
        start_time = time.time()

        # More frequent progress updates
        if idx == 1 or idx % 2 == 0 or idx == total_subjects:
            print(f"  [{idx}/{total_subjects}] Training LDA for {subject_id}...", flush=True)

        # Use first train_size trials from this subject
        train_data = data[:train_size]
        train_labels = labels[:train_size]

        # Flatten training data
        train_data_flat = train_data.reshape(len(train_data), n_features)

        # Train LDA with SVD solver (fastest for high-dimensional data)
        # With 16k+ features and only 100 samples, both lsqr solvers are too slow
        # SVD solver is much faster and doesn't require shrinkage
        lda = LinearDiscriminantAnalysis(solver='svd')
        try:
            lda.fit(train_data_flat, train_labels)

            # Test on target calibration
            predictions = lda.predict(target_cal_flat)
            accuracy = accuracy_score(target_cal_labels, predictions)

            subject_scores.append((subject_id, accuracy))

            elapsed = time.time() - start_time
            if idx <= 3:  # Show timing for first few
                print(f"      → {subject_id}: {accuracy:.1%} (took {elapsed:.1f}s)", flush=True)

        except Exception as e:
            # LDA can fail with singular matrices
            print(f"  Warning: LDA failed for {subject_id}: {e}", flush=True)
            subject_scores.append((subject_id, 0.5))  # Assign chance level

    # Sort by accuracy (highest first)
    subject_scores.sort(key=lambda x: x[1], reverse=True)

    # Display rankings
    print("Subject rankings (top 15):")
    print(f"{'Rank':<6} {'Subject':<10} {'Accuracy':>10}")
    print("─" * 30)
    for rank, (subject_id, acc) in enumerate(subject_scores[:15], 1):
        marker = "✓" if acc > threshold else " "
        print(f"{marker} {rank:<4} {subject_id:<10} {acc:>9.1%}")

    # Select subjects based on method
    if selection_method == 'threshold':
        # Select all subjects above threshold
        selected_ids = [sid for sid, acc in subject_scores if acc > threshold]
        print(f"\n→ Selected {len(selected_ids)} subjects above {threshold:.1%} threshold")

    elif selection_method == 'top_k':
        # Select top K subjects
        selected_ids = [sid for sid, _ in subject_scores[:top_k]]
        print(f"\n→ Selected top-{top_k} subjects")

    elif selection_method == 'adaptive':
        # Select subjects above threshold, but at least min_subjects
        selected_ids = [sid for sid, acc in subject_scores if acc > threshold]
        if len(selected_ids) < min_subjects:
            print(f"\n  Warning: Only {len(selected_ids)} subjects above threshold")
            print(f"  Selecting top-{min_subjects} instead")
            selected_ids = [sid for sid, _ in subject_scores[:min_subjects]]
        print(f"\n→ Selected {len(selected_ids)} subjects (adaptive)")

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    # Create dict of selected subjects
    selected_subjects = {
        subject_id: source_subjects[subject_id]
        for subject_id in selected_ids
    }

    # Print summary
    if len(selected_ids) > 0:
        selected_accs = [acc for sid, acc in subject_scores if sid in selected_ids]
        print(f"\nSelected subjects accuracy: {np.mean(selected_accs):.1%} ± {np.std(selected_accs):.1%}")
        print(f"Range: {np.min(selected_accs):.1%} to {np.max(selected_accs):.1%}")
    else:
        print("\n⚠️  WARNING: No subjects selected! Using all subjects as fallback.")
        selected_subjects = source_subjects

    print(f"{'='*80}\n")

    return selected_subjects, subject_scores


def print_selection_summary(rankings: List[Tuple[str, float]], selected_ids: List[str]) -> None:
    """Print a summary of subject selection results.

    Args:
        rankings: List of (subject_id, accuracy) sorted by accuracy
        selected_ids: List of selected subject IDs
    """
    print(f"\nSubject Selection Summary:")
    print(f"{'─'*50}")
    print(f"Total source subjects: {len(rankings)}")
    print(f"Selected subjects: {len(selected_ids)}")
    print(f"Selection rate: {len(selected_ids)/len(rankings):.1%}")

    if len(selected_ids) > 0:
        selected_accs = [acc for sid, acc in rankings if sid in selected_ids]
        excluded_accs = [acc for sid, acc in rankings if sid not in selected_ids]

        print(f"\nSelected subjects:")
        print(f"  Mean accuracy: {np.mean(selected_accs):.1%}")
        print(f"  Std accuracy: {np.std(selected_accs):.1%}")

        if len(excluded_accs) > 0:
            print(f"\nExcluded subjects:")
            print(f"  Mean accuracy: {np.mean(excluded_accs):.1%}")
            print(f"  Improvement: {np.mean(selected_accs) - np.mean(excluded_accs):+.1%}")


def load_subject_rankings(
    rankings_file: Union[str, Path],
) -> List[Tuple[str, float]]:
    """Load pre-computed subject rankings from JSON file.

    Args:
        rankings_file: Path to rankings JSON file (e.g., rankings_s01.json)

    Returns:
        List of (subject_id, accuracy) tuples sorted by accuracy (descending)
    """
    rankings_file = Path(rankings_file)

    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_file}")

    with open(rankings_file, 'r') as f:
        data = json.load(f)

    # Extract rankings
    rankings = [
        (item['subject_id'], item['accuracy'])
        for item in data['rankings']
    ]

    return rankings


def filter_subjects_by_rankings(
    source_subjects: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rankings: List[Tuple[str, float]],
    selection_method: str = 'adaptive',
    threshold: float = 0.55,
    top_k: int = 9,
    min_subjects: int = 5,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Filter source subjects based on pre-computed rankings.

    This is much faster than select_subjects_lda() because it doesn't
    need to train LDA classifiers.

    Args:
        source_subjects: Dict mapping subject_id -> (data, labels)
        rankings: Pre-computed rankings from load_subject_rankings()
        selection_method: 'threshold', 'top_k', or 'adaptive'
        threshold: Accuracy threshold for selection
        top_k: Number of subjects if method='top_k'
        min_subjects: Minimum subjects if method='adaptive'

    Returns:
        Dict of selected source subjects
    """
    print(f"\n{'='*80}")
    print("SUBJECT SELECTION (using pre-computed rankings)")
    print(f"{'='*80}")
    print(f"Method: {selection_method}")
    if selection_method == 'threshold':
        print(f"Threshold: {threshold:.1%}")
    elif selection_method == 'top_k':
        print(f"Top-K: {top_k}")
    elif selection_method == 'adaptive':
        print(f"Threshold: {threshold:.1%}, Min subjects: {min_subjects}")
    print(f"Source subjects available: {len(source_subjects)}")
    print(f"{'='*80}\n")

    # Display top rankings
    print("Subject rankings (top 15):")
    print(f"{'Rank':<6} {'Subject':<10} {'Accuracy':>10}")
    print("─" * 30)
    for rank, (subject_id, acc) in enumerate(rankings[:15], 1):
        marker = "✓" if acc > threshold else " "
        print(f"{marker} {rank:<4} {subject_id:<10} {acc:>9.1%}")

    # Select subjects based on method
    if selection_method == 'threshold':
        selected_ids = [sid for sid, acc in rankings if acc > threshold]
        print(f"\n→ Selected {len(selected_ids)} subjects above {threshold:.1%} threshold")

    elif selection_method == 'top_k':
        selected_ids = [sid for sid, _ in rankings[:top_k]]
        print(f"\n→ Selected top-{top_k} subjects")

    elif selection_method == 'adaptive':
        selected_ids = [sid for sid, acc in rankings if acc > threshold]
        if len(selected_ids) < min_subjects:
            print(f"\n  Warning: Only {len(selected_ids)} subjects above threshold")
            print(f"  Selecting top-{min_subjects} instead")
            selected_ids = [sid for sid, _ in rankings[:min_subjects]]
        print(f"\n→ Selected {len(selected_ids)} subjects (adaptive)")

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    # Filter source_subjects dict
    selected_subjects = {
        subject_id: source_subjects[subject_id]
        for subject_id in selected_ids
        if subject_id in source_subjects
    }

    # Check for missing subjects
    missing = set(selected_ids) - set(selected_subjects.keys())
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} selected subjects not found in source_subjects:")
        for sid in missing:
            print(f"     {sid}")

    # Print summary
    if len(selected_subjects) > 0:
        selected_accs = [acc for sid, acc in rankings if sid in selected_subjects]
        print(f"\nSelected subjects accuracy: {np.mean(selected_accs):.1%} ± {np.std(selected_accs):.1%}")
        print(f"Range: {np.min(selected_accs):.1%} to {np.max(selected_accs):.1%}")
    else:
        print("\n⚠️  WARNING: No subjects selected! Using all subjects as fallback.")
        selected_subjects = source_subjects

    print(f"{'='*80}\n")

    return selected_subjects
