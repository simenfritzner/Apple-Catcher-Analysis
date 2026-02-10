"""Trainer for PyTorch-based EEGNet model."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from eeg_mi.training.base_trainer import BaseTrainer


class EEGNetTrainer(BaseTrainer):
    """Trainer for EEGNet deep learning model.
    
    Handles PyTorch-specific training loop, optimization, and device management.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        dropout: float = 0.5,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize EEGNet trainer.

        Args:
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            dropout: Dropout rate
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional config parameters
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        two_stage: bool = False,
        pretrain_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        freeze_features: bool = False,
        cal_data: Optional[np.ndarray] = None,
        cal_labels: Optional[np.ndarray] = None,
    ) -> Any:
        """Train EEGNet model.

        Args:
            train_data: Training data (source subjects, n_trials, n_channels, n_timepoints)
            train_labels: Training labels
            val_data: Validation data (test subject held-out data)
            val_labels: Validation labels
            two_stage: If True, use two-stage training (pretrain + finetune)
            pretrain_epochs: Epochs for pretraining on source (default: self.epochs)
            finetune_epochs: Epochs for finetuning on calibration (default: self.epochs // 2)
            freeze_features: If True, freeze feature layers during fine-tuning
            cal_data: Calibration data (test subject first N trials)
            cal_labels: Calibration labels

        Returns:
            Trained PyTorch model
        """
        if two_stage and cal_data is not None:
            return self._train_two_stage(
                train_data, train_labels,
                cal_data, cal_labels,
                val_data, val_labels,
                pretrain_epochs or self.epochs,
                finetune_epochs or (self.epochs // 2),
                freeze_features,
            )
        else:
            return self._train_single_stage(
                train_data, train_labels,
                val_data, val_labels,
            )

    def _train_single_stage(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray],
        val_labels: Optional[np.ndarray],
    ) -> Any:
        """Standard single-stage training (original method)."""
        from eeg_mi.models import EEGNet

        # Get data dimensions
        n_channels = train_data.shape[1]
        n_timepoints = train_data.shape[2]
        n_classes = len(np.unique(train_labels))

        # Create model
        model = EEGNet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_timepoints,
            dropoutRate=self.dropout,
            kernLength=64,
            F1=8,
            D=2,
            F2=16,
        ).to(self.device)

        # Create data loaders
        train_loader = self._create_dataloader(train_data, train_labels, shuffle=True)
        val_loader = None
        if val_data is not None and val_labels is not None:
            val_loader = self._create_dataloader(val_data, val_labels, shuffle=False)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training loop
        self.history = {"train_loss": [], "train_acc": []}
        if val_loader is not None:
            self.history["val_loss"] = []
            self.history["val_acc"] = []

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

        return model

    def _create_dataloader(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        # Add dimension for EEGNet input: (batch, 1, channels, timepoints)
        data_tensor = torch.FloatTensor(data).unsqueeze(1)
        labels_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def evaluate(
        self,
        model: nn.Module,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        model.eval()
        predictions = self.predict(model, test_data)
        
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
        """Make predictions."""
        model.eval()
        
        # Create dataloader
        data_tensor = torch.FloatTensor(data).unsqueeze(1)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for (inputs,) in dataloader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def _train_two_stage(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        cal_data: np.ndarray,
        cal_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        pretrain_epochs: int,
        finetune_epochs: int,
        freeze_features: bool,
    ) -> Any:
        """Two-stage training: pretrain on source, finetune on calibration."""
        from eeg_mi.models import EEGNet

        # Initialize history (for compatibility with single-stage training)
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        print(f"\n{'='*80}")
        print(f"TWO-STAGE TRAINING")
        print(f"Stage 1: Pretrain on source ({len(source_data)} trials) for {pretrain_epochs} epochs")
        print(f"Stage 2: Finetune on calibration ({len(cal_data)} trials) for {finetune_epochs} epochs")
        print(f"{'='*80}\n")

        # Get data dimensions
        n_channels = source_data.shape[1]
        n_timepoints = source_data.shape[2]
        n_classes = len(np.unique(source_labels))

        # Create model
        model = EEGNet(
            nb_classes=n_classes,
            Chans=n_channels,
            Samples=n_timepoints,
            dropoutRate=self.dropout,
            kernLength=64,
            F1=8,
            D=2,
            F2=16,
        ).to(self.device)

        # ========================================================================
        # STAGE 1: PRETRAIN ON SOURCE DATA
        # ========================================================================
        print(f"\n{'─'*80}")
        print("STAGE 1: Pretraining on source data")
        print(f"{'─'*80}")

        source_loader = self._create_dataloader(source_data, source_labels, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(pretrain_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data_batch, labels_batch in source_loader:
                data_batch = data_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(data_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels_batch).sum().item()
                train_total += labels_batch.size(0)

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(source_loader)

            # Evaluate on validation
            val_acc = self._evaluate(model, val_data, val_labels)

            # Track history (store every epoch for stage 1)
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc * 100.0)  # Convert to percentage
            self.history["val_acc"].append(val_acc * 100.0)
            self.history["val_loss"].append(0.0)  # Placeholder (not computed for two-stage)

            if epoch % 10 == 0 or epoch == pretrain_epochs - 1:
                log_msg = (f"[Stage 1] Epoch {epoch:3d}/{pretrain_epochs} | "
                          f"Loss: {train_loss/len(source_loader):.4f} | "
                          f"Train: {train_acc:.2%} | "
                          f"Val: {val_acc:.2%}")
                print(log_msg, flush=True)

        # ========================================================================
        # STAGE 2: FINE-TUNE ON CALIBRATION DATA
        # ========================================================================
        print(f"\n{'─'*80}")
        print("STAGE 2: Fine-tuning on calibration data")
        if freeze_features:
            print("Mode: Freeze feature layers, train classifier only")
        else:
            print("Mode: Fine-tune all layers")
        print(f"{'─'*80}")

        # Optionally freeze feature layers
        if freeze_features:
            # Freeze all layers except the last (classifier)
            for name, param in model.named_parameters():
                # Keep classifier unfrozen (EEGNet uses 'classifier' not 'fc')
                if 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = False

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

            # Safety check: Ensure we have at least some trainable parameters
            if trainable_params == 0:
                raise ValueError(
                    "freeze_features=True resulted in NO trainable parameters! "
                    "Check that the model has a 'classifier' or 'fc' layer."
                )
        else:
            # Unfreeze all
            for param in model.parameters():
                param.requires_grad = True

        cal_loader = self._create_dataloader(cal_data, cal_labels, shuffle=True)

        # Lower learning rate for fine-tuning (or same if only training classifier)
        # Allow finetune_lr to be passed as a hyperparameter
        default_finetune_lr = self.learning_rate if freeze_features else self.learning_rate * 0.1
        finetune_lr = self.config.get('finetune_lr', default_finetune_lr)

        optimizer_ft = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=finetune_lr,
        )

        best_val_acc = 0.0

        for epoch in range(finetune_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data_batch, labels_batch in cal_loader:
                data_batch = data_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer_ft.zero_grad()
                outputs = model(data_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer_ft.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels_batch).sum().item()
                train_total += labels_batch.size(0)

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(cal_loader)
            val_acc = self._evaluate(model, val_data, val_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Track history (append stage 2 metrics)
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc * 100.0)
            self.history["val_acc"].append(val_acc * 100.0)
            self.history["val_loss"].append(0.0)  # Placeholder

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

    def _evaluate(self, model: nn.Module, data: np.ndarray, labels: np.ndarray) -> float:
        """Helper to evaluate model on data."""
        model.eval()
        data_tensor = torch.FloatTensor(data).unsqueeze(1).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)

        with torch.no_grad():
            outputs = model(data_tensor)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels_tensor).sum().item()
            accuracy = correct / len(labels)

        return accuracy
