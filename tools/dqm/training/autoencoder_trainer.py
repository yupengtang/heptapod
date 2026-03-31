"""
# autoencoder_trainer.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import datetime
import json
import os
from typing import Optional

import numpy as np

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from ..data_access.histogram_utils import (
    read_histograms_jsonl,
    records_to_numpy,
    generate_synthetic_histograms,
)
from .training_config import (
    TrainingRunCard,
    load_run_card,
    VALID_ARCHITECTURES,
)

# --------------------------------------------------------------------------- #
# Schema version for model output metadata
# --------------------------------------------------------------------------- #
MODEL_SCHEMA_VERSION = "dqmmodel-1.0"


# =========================================================================== #
# PyTorch model definitions
# =========================================================================== #

def _build_vanilla_ae(input_dim: int, latent_dim: int, dropout: float = 0.0):
    """Build a vanilla (fully-connected) autoencoder."""
    import torch
    import torch.nn as nn

    class VanillaAE(nn.Module):
        def __init__(self):
            super().__init__()
            mid = max(latent_dim * 2, input_dim // 4)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mid, latent_dim),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, mid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mid, input_dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    return VanillaAE()


def _build_conv_ae(input_shape: list, latent_dim: int,
                   encoder_channels: list, dropout: float = 0.0):
    """Build a 1-D convolutional autoencoder."""
    import torch
    import torch.nn as nn

    in_channels = input_shape[0] if len(input_shape) >= 2 else 1
    seq_len = input_shape[-1]

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            enc_layers = []
            prev_ch = in_channels
            for ch in encoder_channels:
                enc_layers.extend([
                    nn.Conv1d(prev_ch, ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                ])
                if dropout > 0:
                    enc_layers.append(nn.Dropout(dropout))
                prev_ch = ch

            self.encoder_conv = nn.Sequential(*enc_layers)

            # Compute flattened size after convolutions
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, seq_len)
                conv_out = self.encoder_conv(dummy)
                self._conv_out_shape = conv_out.shape[1:]  # (channels, length)
                flat_size = conv_out.numel()

            self.encoder_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, latent_dim),
                nn.ReLU(inplace=True),
            )

            # Decoder
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, flat_size),
                nn.ReLU(inplace=True),
            )

            dec_layers = []
            rev_channels = list(reversed(encoder_channels))
            for i, ch in enumerate(rev_channels[:-1]):
                next_ch = rev_channels[i + 1]
                dec_layers.extend([
                    nn.ConvTranspose1d(ch, next_ch, kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm1d(next_ch),
                    nn.ReLU(inplace=True),
                ])
            # Final layer back to input channels
            dec_layers.extend([
                nn.ConvTranspose1d(rev_channels[-1], in_channels,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.Sigmoid(),
            ])
            self.decoder_conv = nn.Sequential(*dec_layers)

        def forward(self, x):
            # Ensure 3-D input: (batch, channels, length)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            z_conv = self.encoder_conv(x)
            z = self.encoder_fc(z_conv)
            dec_flat = self.decoder_fc(z)
            dec_reshaped = dec_flat.view(-1, *self._conv_out_shape)
            recon = self.decoder_conv(dec_reshaped)
            # Trim or pad to match input length
            if recon.shape[-1] != x.shape[-1]:
                recon = recon[..., :x.shape[-1]]
            return recon.squeeze(1) if x.dim() == 2 else recon

    return ConvAE()


def _build_vae(input_dim: int, latent_dim: int, dropout: float = 0.0):
    """Build a variational autoencoder (VAE)."""
    import torch
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            mid = max(latent_dim * 2, input_dim // 4)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.fc_mu = nn.Linear(mid, latent_dim)
            self.fc_logvar = nn.Linear(mid, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, mid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mid, input_dim),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar

    return VAE()


# =========================================================================== #
# Loss functions
# =========================================================================== #

def _vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence."""
    import torch
    import torch.nn.functional as F

    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# =========================================================================== #
# AutoencoderTrainerTool
# =========================================================================== #

class AutoencoderTrainerTool(BaseTool):
    """
    Train an autoencoder model on DQM histogram data for anomaly
    detection.  Supports vanilla, convolutional, and variational
    autoencoders via run-card-driven configuration.

    The trained model is exported to ONNX format for deployment with
    ONNXInferenceTool.

    Inputs (runtime):
      - run_card_path: Relative path to YAML training run card.
        If omitted, ``input_data_path`` must be provided and default
        hyperparameters are used.
      - input_data_path: (optional) Relative path to histogram JSONL.
        Overrides the ``data.input_jsonl`` field in the run card.
      - output_dir: Relative output directory for model artifacts.
      - use_synthetic_data: (optional, default False) Generate and train
        on synthetic histogram data for demonstration / testing.
      - n_synthetic: (optional) Number of synthetic histograms (default 500).

    State:
      - base_directory: Sandbox root for file operations.

    Behavior:
      1. Loads run card (or uses defaults).
      2. Loads histogram data from JSONL or generates synthetic data.
      3. Normalizes and splits into train / validation sets.
      4. Trains the autoencoder with early stopping.
      5. Exports the model to ONNX.
      6. Saves training metrics and provenance metadata.

    Output (JSON):
      {
        "status": "ok",
        "model_path": "<relative path to .onnx>",
        "metrics": {
          "final_train_loss": <float>,
          "final_val_loss": <float>,
          "best_epoch": <int>,
          "total_epochs": <int>
        },
        "architecture": "<architecture name>",
        "input_dim": <int>,
        "latent_dim": <int>
      }

    Errors:
      Returns self.format_error() on failures including:
        - Missing input data
        - PyTorch not installed
        - Invalid run card
    """

    # ======================== Runtime fields ======================== #
    run_card_path: Optional[str] = RuntimeField(
        default=None,
        description="Relative path to YAML training run card"
    )
    input_data_path: Optional[str] = RuntimeField(
        default=None,
        description="Relative path to histogram JSONL (overrides run card data.input_jsonl)"
    )
    output_dir: str = RuntimeField(
        description="Relative output directory for model artifacts (e.g. 'models/ecal_ae_v1')"
    )
    use_synthetic_data: bool = RuntimeField(
        default=False,
        description="Generate and train on synthetic histogram data"
    )
    n_synthetic: int = RuntimeField(
        default=500,
        description="Number of synthetic histograms to generate (if use_synthetic_data=True)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        default=".", description="Base sandbox directory"
    )
    # ================================================================ #

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        if not rel_or_abs:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        if full.startswith(self.base_directory + os.sep) or full == self.base_directory:
            return full
        return None

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        # Check PyTorch availability
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason="PyTorch is not installed",
                suggestion="Install with: pip install torch",
                context=str(e)
            )

        # Resolve output path
        out_abs = self._safe_path(self.output_dir)
        if not out_abs:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
                suggestion="Use a relative path"
            )
        os.makedirs(out_abs, exist_ok=True)

        # ---- Load or create run card ---- #
        if self.run_card_path:
            rc_path = self._safe_path(self.run_card_path)
            if not rc_path or not os.path.exists(rc_path):
                return self.format_error(
                    error="File Not Found",
                    reason="Run card not found",
                    context=f"path={self.run_card_path}",
                    suggestion="Provide a valid YAML run card"
                )
            try:
                card = load_run_card(rc_path)
            except Exception as e:
                return self.format_error(
                    error="Config Error",
                    reason=f"Invalid run card: {e}",
                    suggestion="Check YAML syntax and required fields"
                )
        else:
            card = TrainingRunCard()

        # ---- Load data ---- #
        try:
            if self.use_synthetic_data:
                n_bins = card.model.input_shape[-1] if card.model.input_shape else 64
                data_array, _ = generate_synthetic_histograms(
                    n_histograms=self.n_synthetic,
                    n_bins=n_bins,
                    anomaly_fraction=0.0,  # Train only on "good" data
                    seed=card.training.seed,
                )
            else:
                data_path = self.input_data_path or (card.data.input_jsonl if card.data else None)
                if not data_path:
                    return self.format_error(
                        error="Missing Data",
                        reason="No input data path specified",
                        suggestion="Set input_data_path, run_card data.input_jsonl, or use_synthetic_data=True"
                    )
                dp_abs = self._safe_path(data_path)
                if not dp_abs or not os.path.exists(dp_abs):
                    return self.format_error(
                        error="File Not Found",
                        reason="Input data not found",
                        context=f"path={data_path}"
                    )
                records = read_histograms_jsonl(dp_abs)
                data_array = records_to_numpy(
                    records,
                    flatten=card.data.flatten if card.data else True,
                    normalization=card.data.normalization if card.data else "unit_area",
                )
        except Exception as e:
            return self.format_error(
                error="Data Error",
                reason=str(e),
                suggestion="Check input data format"
            )

        if data_array.shape[0] < 10:
            return self.format_error(
                error="Insufficient Data",
                reason=f"Only {data_array.shape[0]} histograms found (need >= 10)",
                suggestion="Provide more training data"
            )

        # ---- Prepare dataset ---- #
        torch.manual_seed(card.training.seed)
        np.random.seed(card.training.seed)

        input_dim = data_array.shape[1]
        n_total = data_array.shape[0]
        n_train = int(n_total * card.training.train_split)
        n_val = n_total - n_train

        perm = np.random.permutation(n_total)
        train_data = torch.tensor(data_array[perm[:n_train]], dtype=torch.float32)
        val_data = torch.tensor(data_array[perm[n_train:]], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=card.training.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=card.training.batch_size,
            shuffle=False,
        )

        # ---- Build model ---- #
        arch = card.model.architecture
        if arch not in VALID_ARCHITECTURES:
            return self.format_error(
                error="Config Error",
                reason=f"Unknown architecture: {arch}",
                suggestion=f"Choose from: {VALID_ARCHITECTURES}"
            )

        try:
            if arch == "vanilla_ae":
                model = _build_vanilla_ae(input_dim, card.model.latent_dim, card.model.dropout)
            elif arch == "convolutional_ae":
                shape = [1, input_dim] if len(card.model.input_shape) < 2 else card.model.input_shape
                model = _build_conv_ae(shape, card.model.latent_dim,
                                       card.model.encoder_channels, card.model.dropout)
            elif arch == "variational_ae":
                model = _build_vae(input_dim, card.model.latent_dim, card.model.dropout)
        except Exception as e:
            return self.format_error(
                error="Model Error",
                reason=f"Failed to build model: {e}",
                suggestion="Check model configuration in run card"
            )

        is_vae = arch == "variational_ae"

        # ---- Optimizer ---- #
        opt_name = card.training.optimizer.lower()
        if opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=card.training.learning_rate,
                                  weight_decay=card.training.weight_decay)
        elif opt_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=card.training.learning_rate,
                                   weight_decay=card.training.weight_decay)
        elif opt_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=card.training.learning_rate,
                                 weight_decay=card.training.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=card.training.learning_rate)

        criterion = nn.MSELoss()

        # ---- Training loop ---- #
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []

        try:
            for epoch in range(card.training.epochs):
                # --- Train ---
                model.train()
                epoch_train_loss = 0.0
                for (batch,) in train_loader:
                    optimizer.zero_grad()
                    if is_vae:
                        recon, mu, logvar = model(batch)
                        loss = _vae_loss(recon, batch, mu, logvar) / batch.size(0)
                    else:
                        recon = model(batch)
                        loss = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * batch.size(0)

                avg_train = epoch_train_loss / len(train_loader.dataset)
                train_losses.append(avg_train)

                # --- Validate ---
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        if is_vae:
                            recon, mu, logvar = model(batch)
                            loss = _vae_loss(recon, batch, mu, logvar) / batch.size(0)
                        else:
                            recon = model(batch)
                            loss = criterion(recon, batch)
                        epoch_val_loss += loss.item() * batch.size(0)

                avg_val = epoch_val_loss / max(len(val_loader.dataset), 1)
                val_losses.append(avg_val)

                # --- Early stopping ---
                if avg_val < best_val_loss - card.training.early_stopping.min_delta:
                    best_val_loss = avg_val
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model state
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= card.training.early_stopping.patience:
                        break

        except Exception as e:
            return self.format_error(
                error="Training Error",
                reason=str(e),
                suggestion="Check model architecture and data compatibility"
            )

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        # ---- Export to ONNX ---- #
        model_path = os.path.join(out_abs, "model.onnx")
        try:
            model.eval()
            dummy_input = torch.randn(1, input_dim)
            # For VAE, wrap to only output recon
            if is_vae:
                class VAEWrapper(torch.nn.Module):
                    def __init__(self, vae):
                        super().__init__()
                        self.vae = vae
                    def forward(self, x):
                        recon, _, _ = self.vae(x)
                        return recon
                export_model = VAEWrapper(model)
            else:
                export_model = model

            torch.onnx.export(
                export_model,
                dummy_input,
                model_path,
                opset_version=card.export.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        except Exception as e:
            return self.format_error(
                error="Export Error",
                reason=f"Failed to export ONNX: {e}",
                suggestion="Check PyTorch and ONNX compatibility"
            )

        # ---- Save metrics ---- #
        metrics = {
            "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
            "final_val_loss": float(val_losses[-1]) if val_losses else 0.0,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "total_epochs": len(train_losses),
            "train_loss_curve": [float(x) for x in train_losses],
            "val_loss_curve": [float(x) for x in val_losses],
        }

        metrics_path = os.path.join(out_abs, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ---- Save run card used ---- #
        try:
            from .training_config import save_run_card
            save_run_card(card, os.path.join(out_abs, "run_card.yaml"))
        except Exception:
            pass

        # ---- Save provenance metadata ---- #
        provenance = {
            "schema": MODEL_SCHEMA_VERSION,
            "created_utc": datetime.datetime.utcnow().replace(
                tzinfo=datetime.timezone.utc
            ).isoformat(),
            "tool": "AutoencoderTrainerTool",
            "architecture": arch,
            "input_dim": input_dim,
            "latent_dim": card.model.latent_dim,
            "n_train": n_train,
            "n_val": n_val,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
        }
        prov_path = os.path.join(out_abs, "model_metadata.json")
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2)

        # ---- Return result ---- #
        result = {
            "status": "ok",
            "model_path": os.path.relpath(model_path, self.base_directory),
            "metrics_path": os.path.relpath(metrics_path, self.base_directory),
            "metadata_path": os.path.relpath(prov_path, self.base_directory),
            "metrics": {
                "final_train_loss": metrics["final_train_loss"],
                "final_val_loss": metrics["final_val_loss"],
                "best_epoch": metrics["best_epoch"],
                "total_epochs": metrics["total_epochs"],
            },
            "architecture": arch,
            "input_dim": input_dim,
            "latent_dim": card.model.latent_dim,
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
