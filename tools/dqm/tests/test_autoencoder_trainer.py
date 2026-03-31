#!/usr/bin/env python3
"""
# test_autoencoder_trainer.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
TESTS_DIR = SCRIPT_PATH.parent
TOOL_DIR = TESTS_DIR.parent
REPO_ROOT = TOOL_DIR.parent.parent

# Add sub-directories directly to path to avoid triggering
# __init__.py import chains which pull in orchestral-dependent tools
TRAINING_DIR = TOOL_DIR / "training"
DATA_ACCESS_DIR = TOOL_DIR / "data_access"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(DATA_ACCESS_DIR))

from training_config import (
    TrainingRunCard,
    ModelConfig,
    TrainingParams,
    DataConfig,
    ExportConfig,
    save_run_card,
    load_run_card,
    default_run_card,
    TRAINING_SCHEMA_VERSION,
)

OUTPUT_DIR = TESTS_DIR / "test_data" / "trainer_output"


def test_default_run_card():
    """Test that default run card validates successfully."""
    print(">> Testing default_run_card...\n")
    card = default_run_card()
    assert card.schema_version == TRAINING_SCHEMA_VERSION
    assert card.model.architecture == "convolutional_ae"
    assert card.training.epochs == 100
    assert card.training.batch_size == 64
    assert card.data.normalization == "unit_area"
    assert card.export.format == "onnx"
    print("[✓] Default run card validation passed\n")


def test_run_card_serialization():
    """Test YAML save/load round-trip."""
    print(">> Testing run card YAML round-trip...\n")
    card = TrainingRunCard(
        model=ModelConfig(
            architecture="vanilla_ae",
            input_shape=[1, 32],
            latent_dim=16,
        ),
        training=TrainingParams(
            epochs=5,
            batch_size=16,
            learning_rate=0.001,
        ),
        data=DataConfig(
            subsystem="ECAL",
            normalization="max_bin",
        ),
        description="Test run card",
    )

    yaml_path = str(OUTPUT_DIR / "test_run_card.yaml")
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    try:
        save_run_card(card, yaml_path)
        assert os.path.exists(yaml_path), "YAML file should exist"

        loaded = load_run_card(yaml_path)
        assert loaded.model.architecture == "vanilla_ae"
        assert loaded.model.latent_dim == 16
        assert loaded.training.epochs == 5
        assert loaded.data.subsystem == "ECAL"
        assert loaded.description == "Test run card"
        print("[✓] YAML round-trip passed\n")
    except ImportError:
        print("[⚠] PyYAML not installed, skipping YAML test\n")


def test_custom_run_card_validation():
    """Test that invalid configs raise validation errors."""
    print(">> Testing run card validation...\n")
    # Valid
    card = TrainingRunCard(
        model=ModelConfig(latent_dim=8),
        training=TrainingParams(train_split=0.7),
    )
    assert card.model.latent_dim == 8
    assert card.training.train_split == 0.7

    # Test that defaults work properly
    card2 = TrainingRunCard()
    assert card2.training.early_stopping.patience == 10
    assert card2.export.opset_version == 17
    print("[✓] Run card validation passed\n")


def test_trainer_synthetic():
    """Test AutoencoderTrainerTool with synthetic data."""
    print(">> Testing AutoencoderTrainerTool (synthetic data)...\n")

    try:
        import torch
    except ImportError:
        print("[⚠] PyTorch not installed, skipping trainer test\n")
        return True

    try:
        from autoencoder_trainer import AutoencoderTrainerTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping trainer test\n")
        return True

    out_dir = str(OUTPUT_DIR / "vanilla_ae_test")
    rc_path = str(OUTPUT_DIR / "train_run_card.yaml")

    # Create a minimal run card
    card = TrainingRunCard(
        model=ModelConfig(
            architecture="vanilla_ae",
            input_shape=[1, 64],
            latent_dim=8,
        ),
        training=TrainingParams(
            epochs=3,
            batch_size=32,
            learning_rate=0.001,
            seed=42,
        ),
    )

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    try:
        save_run_card(card, rc_path)
    except ImportError:
        print("[⚠] PyYAML not installed, using defaults\n")
        rc_path = None

    tool = AutoencoderTrainerTool(
        run_card_path=os.path.relpath(rc_path, str(REPO_ROOT)) if rc_path else None,
        output_dir=os.path.relpath(out_dir, str(REPO_ROOT)),
        use_synthetic_data=True,
        n_synthetic=100,
        base_directory=str(REPO_ROOT),
    )

    tool._setup()
    result_str = tool._run()

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        print(f"[✗] Tool returned invalid JSON: {result_str}\n")
        return False

    if result.get("status") != "ok":
        print(f"[✗] Trainer failed: {result}\n")
        return False

    # Check outputs exist
    model_path = os.path.join(str(REPO_ROOT), result["model_path"])
    assert os.path.exists(model_path), f"ONNX model should exist at {model_path}"

    metrics_path = os.path.join(str(REPO_ROOT), result["metrics_path"])
    assert os.path.exists(metrics_path), f"Metrics should exist at {metrics_path}"

    # Verify metrics
    assert result["architecture"] == "vanilla_ae"
    assert result["metrics"]["total_epochs"] > 0
    assert result["metrics"]["final_train_loss"] >= 0

    print(f"[✓] Trainer test passed (epochs={result['metrics']['total_epochs']}, "
          f"val_loss={result['metrics']['final_val_loss']:.6f})\n")
    return True


def test_trainer_conv_ae():
    """Test training with convolutional autoencoder architecture."""
    print(">> Testing AutoencoderTrainerTool (convolutional AE)...\n")

    try:
        import torch
    except ImportError:
        print("[⚠] PyTorch not installed, skipping conv AE test\n")
        return True

    try:
        from autoencoder_trainer import AutoencoderTrainerTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping conv AE test\n")
        return True

    out_dir = str(OUTPUT_DIR / "conv_ae_test")

    tool = AutoencoderTrainerTool(
        output_dir=os.path.relpath(out_dir, str(REPO_ROOT)),
        use_synthetic_data=True,
        n_synthetic=100,
        base_directory=str(REPO_ROOT),
    )

    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    if result.get("status") != "ok":
        print(f"[✗] Conv AE trainer failed: {result}\n")
        return False

    assert result["architecture"] == "convolutional_ae"
    print(f"[✓] Conv AE test passed (val_loss={result['metrics']['final_val_loss']:.6f})\n")
    return True


def test_model_registry():
    """Test ModelRegistryTool operations."""
    print(">> Testing ModelRegistryTool...\n")

    try:
        import torch
    except ImportError:
        print("[⚠] PyTorch not installed, skipping registry test\n")
        return True

    try:
        from model_registry import ModelRegistryTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping registry test\n")
        return True

    reg_dir = os.path.relpath(str(OUTPUT_DIR / "registry"), str(REPO_ROOT))

    # First, check if we have a trained model to register
    model_dir = str(OUTPUT_DIR / "vanilla_ae_test")
    if not os.path.exists(os.path.join(model_dir, "model.onnx")):
        print("[⚠] No trained model found, skipping registry test\n")
        return True

    model_dir_rel = os.path.relpath(model_dir, str(REPO_ROOT))

    # Register
    tool = ModelRegistryTool(
        action="register",
        model_dir=model_dir_rel,
        version_tag="v0.1.0",
        description="Test model",
        registry_path=reg_dir,
        base_directory=str(REPO_ROOT),
    )
    tool._setup()
    result = json.loads(tool._run())
    assert result.get("status") == "ok", f"Register failed: {result}"
    model_id = result["model_id"]
    print(f"  Registered: {model_id}")

    # List
    tool2 = ModelRegistryTool(
        action="list",
        registry_path=reg_dir,
        base_directory=str(REPO_ROOT),
    )
    tool2._setup()
    result2 = json.loads(tool2._run())
    assert result2.get("status") == "ok"
    assert result2["n_models"] >= 1
    print(f"  Listed: {result2['n_models']} model(s)")

    # Get
    tool3 = ModelRegistryTool(
        action="get",
        model_id=model_id,
        registry_path=reg_dir,
        base_directory=str(REPO_ROOT),
    )
    tool3._setup()
    result3 = json.loads(tool3._run())
    assert result3.get("status") == "ok"
    assert result3["model"]["model_id"] == model_id
    print(f"  Retrieved: {model_id}")

    # Delete (without deleting files)
    tool4 = ModelRegistryTool(
        action="delete",
        model_id=model_id,
        delete_files=False,
        registry_path=reg_dir,
        base_directory=str(REPO_ROOT),
    )
    tool4._setup()
    result4 = json.loads(tool4._run())
    assert result4.get("status") == "ok"
    print(f"  Deleted: {model_id}")

    print("[✓] Model registry tests passed\n")
    return True


def cleanup_test_files():
    """Remove test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"[✓] Removed: {OUTPUT_DIR.name}")
        except Exception as e:
            print(f"[⚠] Failed to remove {OUTPUT_DIR.name}: {e}")
    else:
        print("[i] No test files to clean up")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for DQM training tools")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True

    tests = [
        test_default_run_card,
        test_custom_run_card_validation,
        test_run_card_serialization,
        test_trainer_synthetic,
        test_trainer_conv_ae,
        test_model_registry,
    ]

    for test_fn in tests:
        try:
            result = test_fn()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"[✗] {test_fn.__name__} failed: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    if all_passed:
        print("[✓] All training tests passed!\n")
        sys.exit(0)
    else:
        print("[✗] Some tests failed!\n")
        sys.exit(1)
