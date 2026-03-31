"""
# config.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

# ============================================================ #
# ================== Ollama LLM Configuration ================ #
# ============================================================ #

# For local Ollama, use None (default)
# For remote Ollama server, set to "http://SERVER_IP:11434"
ollama_host = None  # None = localhost:11434 (default)

# Default Ollama model to use
ollama_model = "gpt-oss:20b"  # Change to your preferred model

# ============================================================ #
# =================== External dependencies ================== #
# ============================================================ #

# FeynRules PATH.
# Example: "/path/to/FeynRules_v2.3.49"
feynrules_path = "/path/to/FeynRules"

# WolframScript executable PATH.
# Example (macOS): "/Applications/Mathematica.app/Contents/MacOS/wolframscript"
# Example (Linux): "/usr/local/bin/wolframscript"
wolframscript_path = "/path/to/wolframscript"

# MadGraph5_aMC PATH.
# Example: "/path/to/MG5_aMC_v3.6.6"
mg5_path = "/path/to/MG5_aMC"

# ============================================================ #
# ================ ML4DQM Configuration ====================== #
# ============================================================ #

# Local directory for cached DQM data files.
# Example: "/data/cms/dqm"
dqm_data_dir = "/path/to/dqm/data"

# CMS DQM GUI API endpoint.
# For offline DQM: "https://cmsweb.cern.ch/dqm/offline"
# For online DQM:  "https://cmsweb.cern.ch/dqm/online"
dqm_api_url = "https://cmsweb.cern.ch/dqm/offline"

# Directory for storing trained DQM ML model artifacts.
# Example: "/data/cms/dqm/models"
dqm_model_registry = "/path/to/dqm/models"

# ONNX Runtime execution providers (comma-separated).
# Options: "CPUExecutionProvider", "CUDAExecutionProvider"
onnx_runtime_providers = "CPUExecutionProvider"