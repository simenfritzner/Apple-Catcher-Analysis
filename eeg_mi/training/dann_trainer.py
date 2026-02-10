"""Trainer for Domain Adversarial Neural Networks (DANN).

Implements unsupervised domain adaptation training with:
- Dual loss (classification + domain adversarial)
- Progressive lambda scheduling (Ganin et al. 2016)
- Mixed source/target batching (50/50 split)
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Sampler
from sklearn.metrics import accuracy_score

from eeg_mi.training.base_trainer import BaseTrainer


class MixedDomainSampler(Sampler):
    """Custom sampler for 50/50 source/target batching.
    
    Ensures each batch contains equal numbers of source and target domain samples.
    """

    def __init__(
        self,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        batch_size: int,
    ):
        """Initialize mixed domain sampler.
        
        Args:
            source_indices: Indices of source domain samples
            target_indices: Indices of target domain samples
            batch_size: Total batch size (must be even for 50/50 split)
        """
        assert batch_size % 2 == 0, "Batch size must be even for 50/50 split"
        
        self.source_indices = source_indices
        self.target_indices = target_indices
        self.batch_size = batch_size
        self.samples_per_domain = batch_size // 2

    def __iter__(self):
        """Generate batches with 50/50 source/target split."""
        # Shuffle indices
        source_shuffled = np.random.permutation(self.source_indices)
        target_shuffled = np.random.permutation(self.target_indices)

        # The number of batches must be determined by the LARGER dataset (source) 
        # to ensure the classification task is properly trained.
        n_source_batches = len(source_shuffled) // self.samples_per_domain

        # Cycle through the target indices to match the number of source batches
        # by repeating the shuffled target indices as many times as needed.
        n_target_repeats = int(np.ceil(n_source_batches / (len(target_shuffled) / self.samples_per_domain)))
        target_cycled = np.tile(target_shuffled, n_target_repeats)

        # Now use the source batch count as the number of batches
        n_batches = n_source_batches

        # Generate mixed batches
        for i in range(n_batches):
            source_batch = source_shuffled[
                i * self.samples_per_domain : (i + 1) * self.samples_per_domain
            ]

            # Use the cycled target indices for the target batch
            target_batch = target_cycled[
                i * self.samples_per_domain : (i + 1) * self.samples_per_domain
            ]

            # Interleave source and target
            batch = np.empty(self.batch_size, dtype=np.int64)
            # You might want to consider shuffling the source and target batches before interleaving
            # to make the overall batch order more random, but interleaving is fine for now.
            batch[0::2] = source_batch
            batch[1::2] = target_batch

            yield from batch

    def __len__(self):
        """Number of samples (limited by the number of source batches)."""
        # Length is based on the source dataset, as it determines the total number of iterations
        n_source = len(self.source_indices)
        n_source_batches = n_source // self.samples_per_domain
        return n_source_batches * self.batch_size


class DANNTrainer(BaseTrainer):
    """Trainer for DANN with unsupervised domain adaptation.
    
    Key features:
    - Dual loss: L_total = L_label + λ * L_domain
    - Progressive lambda scheduling from Ganin et al.
    - Mixed source/target batching (50/50)
    - Unsupervised: target labels used only for evaluation, not training
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        dropout: float = 0.5,
        gamma: float = 10.0,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize DANN trainer.
        
        Args:
            learning_rate: Learning rate for optimizer
            batch_size: Batch size (must be even for 50/50 split)
            epochs: Number of training epochs
            dropout: Dropout rate for model
            gamma: Gamma parameter for lambda scheduling (default: 10.0)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional config parameters
        """
        super().__init__(**kwargs)
        
        assert batch_size % 2 == 0, "Batch size must be even for 50/50 source/target split"
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.gamma = gamma
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def compute_lambda(self, epoch: int, total_epochs: int) -> float:
        """Compute progressive lambda schedule from Ganin et al.
        
        λ_p = (2 / (1 + exp(-γ * p))) - 1
        
        where p ∈ [0, 1] is training progress.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Lambda value for gradient reversal
        """
        p = epoch / total_epochs
        lambda_p = (2.0 / (1.0 + math.exp(-self.gamma * p))) - 1.0
        return lambda_p

    def train(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        source_domains: Optional[np.ndarray] = None,
        target_data: np.ndarray = None,
        target_labels: np.ndarray = None,
        target_domains: Optional[np.ndarray] = None,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        two_stage: bool = False,
        pretrain_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        freeze_features: bool = False,
    ) -> Any:
        """Train DANN with domain adaptation.

        Args:
            source_data: Source domain data (n_source, n_channels, n_timepoints)
            source_labels: Source domain MI labels (used for training)
            source_domains: Source domain labels (all 0 for binary DANN)
            target_data: Target calibration data (n_target, n_channels, n_timepoints)
            target_labels: Target domain MI labels (used for supervised training)
            target_domains: Target domain labels (all 1 for binary DANN)
            val_data: Validation data (held-out test set)
            val_labels: Validation labels
            two_stage: If True, use two-stage training (pretrain + finetune)
            pretrain_epochs: Epochs for pretraining on source (default: self.epochs)
            finetune_epochs: Epochs for finetuning on calibration (default: self.epochs // 2)
            freeze_features: If True, freeze feature extractor during fine-tuning (only train classifier)

        Returns:
            Trained DANN model
        """
        if two_stage:
            return self._train_two_stage(
                source_data, source_labels, source_domains,
                target_data, target_labels, target_domains,
                val_data, val_labels,
                pretrain_epochs or self.epochs,
                finetune_epochs or (self.epochs // 2),
                freeze_features,
            )
        else:
            return self._train_joint(
                source_data, source_labels, source_domains,
                target_data, target_labels, target_domains,
                val_data, val_labels,
            )

    def extract_features(
        self,
        model: nn.Module,
        data: np.ndarray,
    ) -> np.ndarray:
        """Extract features from the feature extractor.

        Args:
            model: Trained DANN model
            data: Input data (n_samples, n_channels, n_timepoints)

        Returns:
            Features (n_samples, flattened_dim)
        """
        model.eval()
        data_tensor = torch.FloatTensor(data).unsqueeze(1).to(self.device)

        with torch.no_grad():
            features = model.feature_extractor(data_tensor)
            flat_features = model.pooling(features)

        return flat_features.cpu().numpy()

    def save_visualizations(
        self,
        model: nn.Module,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        target_data: np.ndarray,
        target_labels: np.ndarray,
        output_dir: str,
        subject_id: Optional[str] = None,
    ) -> None:
        """Save DANN training visualizations.

        Args:
            model: Trained DANN model
            source_data: Source domain data
            source_labels: Source domain labels
            target_data: Target domain data
            target_labels: Target domain labels
            output_dir: Directory to save visualizations
            subject_id: Subject identifier for filenames
        """
        from pathlib import Path
        from eeg_mi.interpretability.visualization import plot_dann_dynamics, plot_dann_feature_space

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = f"{subject_id}_" if subject_id else ""

        # 1. Plot training dynamics
        print("Saving DANN training dynamics plot...")
        plot_dann_dynamics(
            self.history,
            save_path=output_path / f"{prefix}dann_dynamics.png"
        )

        # 2. Extract and visualize feature space
        print("Extracting features and creating feature space visualization...")
        source_features = self.extract_features(model, source_data)
        target_features = self.extract_features(model, target_data)

        plot_dann_feature_space(
            source_features,
            target_features,
            source_labels,
            target_labels,
            method="tsne",
            save_path=output_path / f"{prefix}dann_feature_space.png"
        )

        print(f"Visualizations saved to {output_path}")

    def _train_joint(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        source_domains: Optional[np.ndarray],
        target_data: np.ndarray,
        target_labels: Optional[np.ndarray],
        target_domains: Optional[np.ndarray],
        val_data: Optional[np.ndarray],
        val_labels: Optional[np.ndarray],
    ) -> Any:
        """Joint training on source + calibration data (original method)."""
        from eeg_mi.models.dann import DANN
        from eeg_mi.models.eegnet import EEGNet

        # Get data dimensions
        n_channels = source_data.shape[1]
        n_timepoints = source_data.shape[2]
        n_classes = len(np.unique(source_labels))

        # Get model params
        model_params = self.config.get('model_params', {})
        F1 = model_params.get('F1', 8)
        D = model_params.get('D', 2)
        F2 = model_params.get('F2', 16)
        kernel_length = model_params.get('kernel_length', 64)
        predictor_hidden = model_params.get('predictor_hidden', 256)
        predictor_layers = model_params.get('predictor_layers', 2)
        discriminator_hidden = model_params.get('discriminator_hidden', 256)
        discriminator_layers = model_params.get('discriminator_layers', 2)

        # Create EEGNet feature extractor
        feature_extractor = EEGNet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_timepoints,
            F1=F1,
            D=D,
            F2=F2,
            kernLength=kernel_length,
            dropoutRate=self.dropout,
            mode='feature_extractor',
        )

        # Create DANN model with binary domain classification (n_domains=2)
        model = DANN(
            feature_extractor=feature_extractor,
            feature_dim=F2,
            n_classes=n_classes,
            samples=n_timepoints,  # Total timepoints for flatten calculation
            n_domains=2,  # Binary: source (0) vs target (1)
            predictor_hidden=predictor_hidden,
            predictor_layers=predictor_layers,
            discriminator_hidden=discriminator_hidden,
            discriminator_layers=discriminator_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Use provided domain labels or create binary labels
        if source_domains is None:
            source_domains = np.zeros(len(source_data), dtype=np.int64)  # Source = 0
        if target_domains is None:
            target_domains = np.ones(len(target_data), dtype=np.int64)   # Target = 1

        # Create dataloader with mixed batching
        train_loader = self._create_mixed_dataloader(
            source_data, source_labels, source_domains,
            target_data, target_labels, target_domains,
        )
        
        # Setup training
        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.history = {
            "train_loss": [],
            "class_loss": [],
            "domain_loss": [],
            "lambda": [],
            "source_acc": [],
            "target_acc_train": [],  # Target accuracy during training (only for supervised)
            "domain_acc": [],  # How well discriminator distinguishes domains
        }

        # Track whether we're in supervised or unsupervised mode
        self.supervised_mode = target_labels is not None

        # Always initialize target_acc_test if we have validation data
        # (can evaluate even in unsupervised mode)
        if val_data is not None and val_labels is not None:
            self.history["target_acc_test"] = []  # Separate test accuracy tracking
        
        for epoch in range(self.epochs):
            # Compute lambda for this epoch
            lambda_current = self.compute_lambda(epoch, self.epochs)

            # Train epoch
            losses = self._train_epoch(
                model, train_loader,
                criterion_class, criterion_domain,
                optimizer, lambda_current,
                len(source_data),
            )

            # Log
            self.history["train_loss"].append(losses["total"])
            self.history["class_loss"].append(losses["class"])
            self.history["domain_loss"].append(losses["domain"])
            self.history["lambda"].append(lambda_current)
            self.history["source_acc"].append(losses["source_acc"])
            self.history["target_acc_train"].append(losses["target_acc_train"])
            self.history["domain_acc"].append(losses["domain_acc"])

            # Print every 10 epochs (or first/last)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                log_msg = (f"Epoch {epoch:3d}/{self.epochs} | "
                          f"Loss: {losses['total']:.4f} (C:{losses['class']:.4f} D:{losses['domain']:.4f}) | "
                          f"λ: {lambda_current:.3f} | "
                          f"Src: {losses['source_acc']:.2%}")

                # Show target training accuracy if in supervised mode
                if self.supervised_mode and losses['target_acc_train'] > 0:
                    log_msg += f" | Tgt(train): {losses['target_acc_train']:.2%}"

                log_msg += f" | Dom: {losses['domain_acc']:.2%}"

                # Evaluate on separate validation set if provided
                if val_data is not None and val_labels is not None:
                    val_acc = self._evaluate_target(model, val_data, val_labels)
                    self.history["target_acc_test"].append(val_acc)
                    log_msg += f" | Val: {val_acc:.2%}"

                print(log_msg, flush=True)
        
        return model

    def _create_mixed_dataloader(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        source_domain: np.ndarray,
        target_data: np.ndarray,
        target_labels: Optional[np.ndarray],
        target_domain: np.ndarray,
    ) -> DataLoader:
        """Create DataLoader with mixed source/target batching.

        Args:
            source_data: Source domain data
            source_labels: Source domain labels
            source_domain: Source domain labels (all 0)
            target_data: Target domain data
            target_labels: Target domain labels (None for unsupervised, or array for supervised)
            target_domain: Target domain labels (all 1)

        Returns:
            DataLoader with MixedDomainSampler
        """
        # Combine all data
        all_data = np.concatenate([source_data, target_data], axis=0)

        # Labels: source has real labels, target has real labels OR dummy labels (-1)
        if target_labels is not None:
            # Supervised: use real target labels
            all_labels = np.concatenate([source_labels, target_labels])
        else:
            # Unsupervised: use dummy labels for target
            dummy_target_labels = np.full(len(target_data), -1, dtype=np.int64)
            all_labels = np.concatenate([source_labels, dummy_target_labels])
        
        # Domain labels
        all_domains = np.concatenate([source_domain, target_domain])
        
        # Convert to tensors (add channel dimension for EEGNet)
        data_tensor = torch.FloatTensor(all_data).unsqueeze(1)  # (N, 1, C, T)
        labels_tensor = torch.LongTensor(all_labels)
        domain_tensor = torch.LongTensor(all_domains)
        
        # Create dataset
        dataset = TensorDataset(data_tensor, labels_tensor, domain_tensor)
        
        # Create indices for source/target
        source_indices = np.arange(len(source_data))
        target_indices = np.arange(len(source_data), len(all_data))
        
        # Create sampler
        sampler = MixedDomainSampler(
            source_indices,
            target_indices,
            self.batch_size,
        )
        
        return DataLoader(dataset, batch_sampler=None, sampler=sampler, batch_size=self.batch_size)

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion_class: nn.Module,
        criterion_domain: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_current: float,
        n_source: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        class_loss_sum = 0.0
        domain_loss_sum = 0.0
        source_correct = 0
        source_total = 0
        target_correct = 0
        target_total = 0
        domain_correct = 0
        domain_total = 0
        n_batches = 0
        
        for data, labels, domains in dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_output, domain_output = model(data, alpha=lambda_current)
            
            # Classification loss (on samples with labels != -1)
            labeled_mask = labels != -1
            if labeled_mask.sum() > 0:
                class_loss = criterion_class(class_output[labeled_mask], labels[labeled_mask])

                # Track accuracy separately for source and target domains
                _, predicted = class_output[labeled_mask].max(1)
                labeled_domains = domains[labeled_mask]
                labeled_correct = predicted.eq(labels[labeled_mask])

                # Source accuracy (domain == 0)
                source_mask = labeled_domains == 0
                if source_mask.sum() > 0:
                    source_correct += labeled_correct[source_mask].sum().item()
                    source_total += source_mask.sum().item()

                # Target accuracy (domain == 1, only tracked in supervised mode)
                target_mask = labeled_domains == 1
                if target_mask.sum() > 0:
                    target_correct += labeled_correct[target_mask].sum().item()
                    target_total += target_mask.sum().item()
            else:
                class_loss = torch.tensor(0.0, device=self.device)
            
            # Domain loss (on all samples)
            domain_loss = criterion_domain(domain_output, domains)

            # Track domain discriminator accuracy
            _, domain_predicted = domain_output.max(1)
            domain_correct += domain_predicted.eq(domains).sum().item()
            domain_total += domains.size(0)

            # Total loss
            loss = class_loss + domain_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            class_loss_sum += class_loss.item()
            domain_loss_sum += domain_loss.item()
            n_batches += 1
        
        return {
            "total": total_loss / n_batches,
            "class": class_loss_sum / n_batches,
            "domain": domain_loss_sum / n_batches,
            "source_acc": source_correct / source_total if source_total > 0 else 0.0,
            "target_acc_train": target_correct / target_total if target_total > 0 else 0.0,
            "domain_acc": domain_correct / domain_total if domain_total > 0 else 0.0,
        }

    def _evaluate_target(
        self,
        model: nn.Module,
        target_data: np.ndarray,
        target_labels: np.ndarray,
    ) -> float:
        """Evaluate on target domain (for monitoring only - not used in training)."""
        model.eval()
        
        data_tensor = torch.FloatTensor(target_data).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            predictions = model.predict(data_tensor).cpu().numpy()
        
        return accuracy_score(target_labels, predictions)

    def _train_two_stage(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        source_domains: Optional[np.ndarray],
        target_data: np.ndarray,
        target_labels: np.ndarray,
        target_domains: Optional[np.ndarray],
        val_data: np.ndarray,
        val_labels: np.ndarray,
        pretrain_epochs: int,
        finetune_epochs: int,
        freeze_features: bool = False,
    ) -> Any:
        """Two-stage training: pretrain on source, finetune on calibration.

        Stage 1: Train on source with binary domain adaptation (source=0, target=1)
        Stage 2: Fine-tune on target calibration data (supervised learning)

        Args:
            source_domains: Source domain labels (all 0 for binary DANN)
            target_domains: Target domain labels (all 1 for binary DANN)
            freeze_features: If True, freeze feature extractor (EEGNet) during stage 2,
                           only train the class predictor head
        """
        from eeg_mi.models.dann import DANN
        from eeg_mi.models.eegnet import EEGNet
        import torch.nn as nn

        print(f"\n{'='*80}")
        print(f"TWO-STAGE TRAINING")
        print(f"Stage 1: Pretrain on source ({len(source_data)} trials) for {pretrain_epochs} epochs")
        print(f"Stage 2: Finetune on calibration ({len(target_data)} trials) for {finetune_epochs} epochs")
        print(f"{'='*80}\n")

        # Get data dimensions
        n_channels = source_data.shape[1]
        n_timepoints = source_data.shape[2]
        n_classes = len(np.unique(source_labels))

        # Get model params
        model_params = self.config.get('model_params', {})
        F1 = model_params.get('F1', 8)
        D = model_params.get('D', 2)
        F2 = model_params.get('F2', 16)
        kernel_length = model_params.get('kernel_length', 64)
        predictor_hidden = model_params.get('predictor_hidden', 256)
        predictor_layers = model_params.get('predictor_layers', 2)
        discriminator_hidden = model_params.get('discriminator_hidden', 256)
        discriminator_layers = model_params.get('discriminator_layers', 2)

        # Create EEGNet feature extractor
        feature_extractor = EEGNet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_timepoints,
            F1=F1,
            D=D,
            F2=F2,
            kernLength=kernel_length,
            dropoutRate=self.dropout,
            mode='feature_extractor',
        )

        # Use binary domain classification (source=0, target=1)
        n_domains = 2
        print(f"Binary DANN: source=0, target=1")

        # Create domain labels if not provided
        if source_domains is None:
            source_domains = np.zeros(len(source_data), dtype=np.int64)
        if target_domains is None:
            target_domains = np.ones(len(target_data), dtype=np.int64)

        # Create DANN model with binary domain classification
        model = DANN(
            feature_extractor=feature_extractor,
            feature_dim=F2,
            n_classes=n_classes,
            samples=n_timepoints,  # Total timepoints for flatten calculation
            n_domains=2,  # Binary: source (0) vs target (1)
            predictor_hidden=predictor_hidden,
            predictor_layers=predictor_layers,
            discriminator_hidden=discriminator_hidden,
            discriminator_layers=discriminator_layers,
            dropout=self.dropout,
        ).to(self.device)

        # ============================================================
        # STAGE 1: PRETRAIN WITH BINARY DOMAIN ADAPTATION
        # ============================================================
        print(f"\n{'─'*80}")
        print("STAGE 1: Binary DANN pretraining with source and target data")
        print(f"Domain discriminator: Binary classification (source=0, target=1)")
        print(f"Gradient reversal: Forces domain-invariant features")
        print(f"Target data: Unlabeled (dummy labels -1 for classification loss)")
        print(f"{'─'*80}")

        # Create mixed dataloader with both source and target for domain adaptation
        # Target labels are used only for domain discrimination, not for classification (set to -1)
        pretrain_loader = self._create_mixed_dataloader(
            source_data, source_labels, source_domains,
            target_data,
            np.full(len(target_data), -1, dtype=np.int64),  # Dummy labels for unsupervised target
            target_domains,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Stage 1 training loop - Binary DANN with domain discriminator
        for epoch in range(pretrain_epochs):
            model.train()
            train_class_loss = 0.0
            train_domain_loss = 0.0
            source_correct = 0
            source_total = 0
            domain_correct = 0
            domain_total = 0
            n_batches = 0

            # Compute lambda for gradient reversal
            lambda_p = self.compute_lambda(epoch, pretrain_epochs)

            for data_batch, labels_batch, domains_batch in pretrain_loader:
                data_batch = data_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                domains_batch = domains_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass through feature extractor
                features = model.feature_extractor(data_batch)
                flat_features = model.pooling(features)

                # Class prediction (only on labeled samples, i.e., source domain)
                class_output = model.label_predictor(flat_features)
                labeled_mask = labels_batch != -1
                if labeled_mask.sum() > 0:
                    class_loss = criterion(class_output[labeled_mask], labels_batch[labeled_mask])

                    # Track source accuracy
                    _, predicted = class_output[labeled_mask].max(1)
                    source_correct += predicted.eq(labels_batch[labeled_mask]).sum().item()
                    source_total += labeled_mask.sum().item()
                else:
                    class_loss = torch.tensor(0.0, device=self.device)

                # Domain prediction with gradient reversal (on ALL samples)
                domain_output = model.domain_discriminator(flat_features, lambda_p)
                domain_loss = criterion(domain_output, domains_batch)

                # Combined loss
                loss = class_loss + domain_loss
                loss.backward()
                optimizer.step()

                # Track metrics
                train_class_loss += class_loss.item()
                train_domain_loss += domain_loss.item()

                _, domain_pred = domain_output.max(1)
                domain_correct += domain_pred.eq(domains_batch).sum().item()
                domain_total += domains_batch.size(0)
                n_batches += 1

            source_acc = source_correct / source_total if source_total > 0 else 0.0
            domain_acc = domain_correct / domain_total if domain_total > 0 else 0.0
            val_acc = self._evaluate_target(model, val_data, val_labels)

            if epoch % 10 == 0 or epoch == pretrain_epochs - 1:
                log_msg = (f"[Stage 1] Epoch {epoch:3d}/{pretrain_epochs} | "
                          f"λ={lambda_p:.3f} | "
                          f"C_loss: {train_class_loss/n_batches:.4f} | "
                          f"D_loss: {train_domain_loss/n_batches:.4f} | "
                          f"Src: {source_acc:.2%} | "
                          f"Dom: {domain_acc:.2%} | "
                          f"Val: {val_acc:.2%}")
                print(log_msg, flush=True)

        # ============================================================
        # STAGE 2: FINE-TUNE ON CALIBRATION DATA
        # ============================================================
        print(f"\n{'─'*80}")
        print("STAGE 2: Fine-tuning on target calibration data")
        if freeze_features:
            print("Mode: Freeze feature extractor, train classifier only")
        else:
            print("Mode: Fine-tune all layers")
        print(f"{'─'*80}")

        # Optionally freeze feature extractor
        if freeze_features:
            # Freeze EEGNet feature extractor
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
            # Keep label predictor trainable
            for param in model.label_predictor.parameters():
                param.requires_grad = True

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        else:
            # Unfreeze all parameters (in case they were frozen)
            for param in model.parameters():
                param.requires_grad = True

        # Create dataloader for calibration data only
        cal_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(target_data).unsqueeze(1),
            torch.LongTensor(target_labels),
        )
        cal_loader = torch.utils.data.DataLoader(
            cal_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Lower learning rate for fine-tuning (or higher if only training classifier)
        if freeze_features:
            # Can use higher LR when only training classifier
            finetune_lr = self.learning_rate
        else:
            # Lower LR when fine-tuning all layers
            finetune_lr = self.learning_rate * 0.1

        # Only optimize trainable parameters
        optimizer_ft = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=finetune_lr
        )
        criterion_ft = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        # Stage 2 training loop
        for epoch in range(finetune_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data_batch, labels_batch in cal_loader:
                data_batch = data_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer_ft.zero_grad()

                # Forward pass through feature extractor and classifier
                features = model.feature_extractor(data_batch)
                flat_features = model.pooling(features)
                class_output = model.label_predictor(flat_features)  # Raw logits

                loss = criterion_ft(class_output, labels_batch)

                loss.backward()
                optimizer_ft.step()

                train_loss += loss.item()
                _, predicted = class_output.max(1)
                train_correct += predicted.eq(labels_batch).sum().item()
                train_total += labels_batch.size(0)

            train_acc = train_correct / train_total

            # Evaluate
            val_acc = self._evaluate_target(model, val_data, val_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 10 == 0 or epoch == finetune_epochs - 1:
                log_msg = (f"[Stage 2] Epoch {epoch:3d}/{finetune_epochs} | "
                          f"Loss: {train_loss/len(cal_loader):.4f} | "
                          f"Cal: {train_acc:.2%} | "
                          f"Val: {val_acc:.2%}")
                print(log_msg, flush=True)

        print(f"\n{'='*80}")
        print(f"Two-stage training complete!")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        print(f"{'='*80}\n")

        return model

    def evaluate(
        self,
        model: nn.Module,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate trained DANN model."""
        model.eval()
        
        data_tensor = torch.FloatTensor(test_data).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            predictions = model.predict(data_tensor).cpu().numpy()
        
        accuracy = accuracy_score(test_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "n_samples": len(test_labels),
        }

    def predict(
        self,
        model: nn.Module,
        data: np.ndarray,
    ) -> np.ndarray:
        """Make predictions with trained DANN."""
        model.eval()
        
        data_tensor = torch.FloatTensor(data).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            predictions = model.predict(data_tensor).cpu().numpy()
        
        return predictions
