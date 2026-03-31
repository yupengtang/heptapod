"""
# model_registry.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import datetime
import json
import os
import shutil
from typing import Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField


MODEL_REGISTRY_SCHEMA = "dqm-registry-1.0"


class ModelRegistryTool(BaseTool):
    """
    Manage trained DQM ML model artifacts with versioning, metadata
    tracking, and querying.

    The registry is a JSON manifest file (``registry.json``) stored in
    the registry directory.  Each model entry contains paths to artifacts
    (.onnx, metadata, training config), plus searchable metadata.

    Inputs (runtime):
      - action: Operation to perform. One of:
          ``"register"``  – Register a new model from a training output dir.
          ``"list"``      – List all registered models.
          ``"query"``     – Search models by subsystem, architecture, or date.
          ``"get"``       – Get details for a specific model by ID.
          ``"delete"``    – Remove a model entry (optionally delete files).
      - model_dir: (for ``register``) Relative path to directory containing
        model.onnx and model_metadata.json.
      - model_id: (for ``get``/``delete``) Model identifier.
      - subsystem_filter: (for ``query``) Filter by CMS subsystem.
      - architecture_filter: (for ``query``) Filter by model architecture.
      - version_tag: (for ``register``) Semantic version tag (e.g. ``"v1.0.0"``).
      - description: (for ``register``) Human-readable model description.
      - delete_files: (for ``delete``) Also delete model files (default False).

    State:
      - registry_path: Relative path to registry directory.
      - base_directory: Sandbox root.

    Output (JSON):
      Varies by action, all include ``"status": "ok"`` on success.

    Errors:
      Returns self.format_error() on failures.
    """

    # ======================== Runtime fields ======================== #
    action: str = RuntimeField(
        description="Operation: 'register', 'list', 'query', 'get', or 'delete'"
    )
    model_dir: Optional[str] = RuntimeField(
        default=None,
        description="Relative path to model output directory (for register)"
    )
    model_id: Optional[str] = RuntimeField(
        default=None,
        description="Model identifier (for get/delete)"
    )
    subsystem_filter: Optional[str] = RuntimeField(
        default=None,
        description="Filter by CMS subsystem (for query)"
    )
    architecture_filter: Optional[str] = RuntimeField(
        default=None,
        description="Filter by architecture (for query)"
    )
    version_tag: Optional[str] = RuntimeField(
        default=None,
        description="Semantic version tag, e.g. 'v1.0.0' (for register)"
    )
    description: Optional[str] = RuntimeField(
        default=None,
        description="Human-readable model description (for register)"
    )
    delete_files: bool = RuntimeField(
        default=False,
        description="Also delete model files (for delete action)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    registry_path: str = StateField(
        default="dqm_model_registry",
        description="Relative path to registry directory"
    )
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

    def _registry_file(self) -> str:
        reg_dir = self._safe_path(self.registry_path)
        if reg_dir:
            os.makedirs(reg_dir, exist_ok=True)
            return os.path.join(reg_dir, "registry.json")
        return os.path.join(self.base_directory, "registry.json")

    def _load_registry(self) -> dict:
        path = self._registry_file()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {"schema": MODEL_REGISTRY_SCHEMA, "models": []}

    def _save_registry(self, registry: dict):
        path = self._registry_file()
        with open(path, "w") as f:
            json.dump(registry, f, indent=2)

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        valid_actions = {"register", "list", "query", "get", "delete"}
        if self.action not in valid_actions:
            return self.format_error(
                error="Invalid Action",
                reason=f"action must be one of {sorted(valid_actions)}",
                suggestion=f"Use one of: {', '.join(sorted(valid_actions))}"
            )

        try:
            if self.action == "register":
                return self._register()
            elif self.action == "list":
                return self._list_models()
            elif self.action == "query":
                return self._query()
            elif self.action == "get":
                return self._get_model()
            elif self.action == "delete":
                return self._delete_model()
        except Exception as e:
            return self.format_error(error="Registry Error", reason=str(e))

        return self.format_error(error="Unknown Error", reason="Unexpected code path")

    def _register(self) -> str:
        if not self.model_dir:
            return self.format_error(
                error="Missing Parameter",
                reason="model_dir is required for register",
                suggestion="Provide the path to the model output directory"
            )

        dir_abs = self._safe_path(self.model_dir)
        if not dir_abs or not os.path.isdir(dir_abs):
            return self.format_error(
                error="Directory Not Found",
                reason="Model directory does not exist",
                context=f"path={self.model_dir}"
            )

        # Look for model files
        onnx_path = os.path.join(dir_abs, "model.onnx")
        meta_path = os.path.join(dir_abs, "model_metadata.json")

        if not os.path.exists(onnx_path):
            return self.format_error(
                error="File Not Found",
                reason="model.onnx not found in model directory",
                suggestion="Train a model first using AutoencoderTrainerTool"
            )

        # Load metadata if available
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        # Generate model ID
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        subsys = metadata.get("subsystem", "unknown")
        arch = metadata.get("architecture", "unknown")
        model_id = f"{subsys}_{arch}_{timestamp}"

        # Build entry
        entry = {
            "model_id": model_id,
            "version": self.version_tag or "v1.0.0",
            "description": self.description or "",
            "registered_utc": datetime.datetime.utcnow().replace(
                tzinfo=datetime.timezone.utc
            ).isoformat(),
            "model_dir": os.path.relpath(dir_abs, self.base_directory),
            "onnx_path": os.path.relpath(onnx_path, self.base_directory),
            "architecture": arch,
            "subsystem": subsys,
            "input_dim": metadata.get("input_dim"),
            "latent_dim": metadata.get("latent_dim"),
            "best_val_loss": metadata.get("best_val_loss"),
            "metadata": metadata,
        }

        registry = self._load_registry()
        registry["models"].append(entry)
        self._save_registry(registry)

        result = {"status": "ok", "model_id": model_id, "entry": entry}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _list_models(self) -> str:
        registry = self._load_registry()
        models = registry.get("models", [])
        summaries = []
        for m in models:
            summaries.append({
                "model_id": m["model_id"],
                "version": m.get("version", ""),
                "architecture": m.get("architecture", ""),
                "subsystem": m.get("subsystem", ""),
                "registered_utc": m.get("registered_utc", ""),
                "best_val_loss": m.get("best_val_loss"),
            })
        result = {"status": "ok", "models": summaries, "n_models": len(summaries)}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _query(self) -> str:
        registry = self._load_registry()
        models = registry.get("models", [])

        filtered = models
        if self.subsystem_filter:
            filtered = [m for m in filtered
                        if m.get("subsystem", "").lower() == self.subsystem_filter.lower()]
        if self.architecture_filter:
            filtered = [m for m in filtered
                        if m.get("architecture", "").lower() == self.architecture_filter.lower()]

        result = {"status": "ok", "models": filtered, "n_matches": len(filtered)}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _get_model(self) -> str:
        if not self.model_id:
            return self.format_error(
                error="Missing Parameter",
                reason="model_id is required for get",
                suggestion="Provide the model identifier"
            )

        registry = self._load_registry()
        for m in registry.get("models", []):
            if m["model_id"] == self.model_id:
                result = {"status": "ok", "model": m}
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        return self.format_error(
            error="Not Found",
            reason=f"No model with id '{self.model_id}'",
            suggestion="Use 'list' action to see available models"
        )

    def _delete_model(self) -> str:
        if not self.model_id:
            return self.format_error(
                error="Missing Parameter",
                reason="model_id is required for delete"
            )

        registry = self._load_registry()
        models = registry.get("models", [])
        new_models = []
        deleted = None
        for m in models:
            if m["model_id"] == self.model_id:
                deleted = m
            else:
                new_models.append(m)

        if not deleted:
            return self.format_error(
                error="Not Found",
                reason=f"No model with id '{self.model_id}'"
            )

        # Optionally delete files
        if self.delete_files and deleted.get("model_dir"):
            dir_abs = self._safe_path(deleted["model_dir"])
            if dir_abs and os.path.isdir(dir_abs):
                shutil.rmtree(dir_abs)

        registry["models"] = new_models
        self._save_registry(registry)

        result = {"status": "ok", "deleted_model_id": self.model_id}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
