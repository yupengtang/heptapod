"""
# training_config.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Pydantic schema definitions and YAML helpers for DQM ML training run
cards.  These schemas enforce consistent, reproducible training
configurations across model architectures and datasets.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Schema version
# --------------------------------------------------------------------------- #
TRAINING_SCHEMA_VERSION = "dqm-training-1.0"

# --------------------------------------------------------------------------- #
# Valid choices
# --------------------------------------------------------------------------- #
VALID_ARCHITECTURES = ["vanilla_ae", "convolutional_ae", "variational_ae"]
VALID_OPTIMIZERS = ["adam", "adamw", "sgd"]
VALID_NORMALIZATIONS = ["unit_area", "max_bin", "z_score", "minmax"]
VALID_EXPORT_FORMATS = ["onnx", "torchscript", "both"]


# --------------------------------------------------------------------------- #
# Sub-schemas
# --------------------------------------------------------------------------- #

class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping during training."""
    patience: int = Field(default=10, ge=1, description="Epochs to wait for improvement")
    min_delta: float = Field(default=1e-4, ge=0, description="Minimum loss decrease to count as improvement")


class ModelConfig(BaseModel):
    """Neural network architecture specification."""
    architecture: str = Field(
        default="convolutional_ae",
        description=f"Architecture type. One of: {VALID_ARCHITECTURES}"
    )
    input_shape: List[int] = Field(
        default=[1, 64],
        description="Input tensor shape [channels, height] or [channels, height, width]"
    )
    latent_dim: int = Field(default=32, ge=1, description="Dimensionality of the latent representation")
    encoder_channels: List[int] = Field(
        default=[16, 32, 64],
        description="Number of channels in each encoder convolutional layer"
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")


class TrainingParams(BaseModel):
    """Training hyperparameters."""
    epochs: int = Field(default=100, ge=1, description="Maximum number of training epochs")
    batch_size: int = Field(default=64, ge=1, description="Mini-batch size")
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    optimizer: str = Field(
        default="adam",
        description=f"Optimizer. One of: {VALID_OPTIMIZERS}"
    )
    weight_decay: float = Field(default=0.0, ge=0, description="L2 regularization strength")
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description="Early stopping configuration"
    )
    train_split: float = Field(default=0.8, gt=0, lt=1, description="Fraction of data used for training")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class DataConfig(BaseModel):
    """Data source and preprocessing specification."""
    subsystem: Optional[str] = Field(default=None, description="CMS subsystem (e.g. 'ECAL', 'HCAL')")
    histogram_path: Optional[str] = Field(
        default=None,
        description="DQM histogram path pattern for training data"
    )
    input_jsonl: Optional[str] = Field(
        default=None,
        description="Path to pre-extracted histogram JSONL file"
    )
    normalization: str = Field(
        default="unit_area",
        description=f"Normalization method. One of: {VALID_NORMALIZATIONS}"
    )
    flatten: bool = Field(default=True, description="Flatten 2-D histograms to 1-D vectors")


class ExportConfig(BaseModel):
    """Model export specification."""
    format: str = Field(
        default="onnx",
        description=f"Export format. One of: {VALID_EXPORT_FORMATS}"
    )
    opset_version: int = Field(default=17, ge=9, description="ONNX opset version")


# --------------------------------------------------------------------------- #
# Top-level training run card
# --------------------------------------------------------------------------- #

class TrainingRunCard(BaseModel):
    """
    Complete training run card for a DQM autoencoder model.

    This schema can be serialized to/from YAML and fully specifies a
    reproducible training run.
    """
    schema_version: str = Field(default=TRAINING_SCHEMA_VERSION, description="Schema version identifier")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model architecture configuration")
    training: TrainingParams = Field(default_factory=TrainingParams, description="Training hyperparameters")
    data: DataConfig = Field(default_factory=DataConfig, description="Data source and preprocessing")
    export: ExportConfig = Field(default_factory=ExportConfig, description="Model export configuration")
    description: str = Field(default="", description="Human-readable description of this training run")


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #

def load_run_card(yaml_path: str) -> TrainingRunCard:
    """
    Load a training run card from a YAML file.

    Args:
        yaml_path: Path to the YAML run card.

    Returns:
        Validated TrainingRunCard instance.

    Raises:
        ImportError: If PyYAML is not installed.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    import yaml

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    return TrainingRunCard(**raw)


def save_run_card(card: TrainingRunCard, yaml_path: str) -> None:
    """
    Save a training run card to a YAML file.

    Args:
        card:      TrainingRunCard instance to serialize.
        yaml_path: Output file path.
    """
    import yaml
    import os

    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(
            card.model_dump(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def default_run_card() -> TrainingRunCard:
    """Return a TrainingRunCard populated with sensible defaults."""
    return TrainingRunCard()
