# HEPTAPOD ML4DQM Tools

Machine Learning tools for CMS Data Quality Monitoring, designed to bring
anomaly detection capabilities to DQM workflows within the HEPTAPOD
orchestration framework.

---

## Overview

The ML4DQM module extends HEPTAPOD with three tool categories that mirror
the real CMS DQM machine learning pipeline:

```
Data Access  →  Training  →  Deployment
(ROOT/API)      (PyTorch)    (ONNX Runtime)
```

| Stage | Tool | Purpose |
|-------|------|---------|
| **Data Access** | `DQMIOReaderTool` | Read histograms from ROOT files |
| **Data Access** | `DQMAPIClientTool` | Query CMS DQM web service API |
| **Training** | `AutoencoderTrainerTool` | Train autoencoder models |
| **Training** | `ModelRegistryTool` | Manage model artifacts |
| **Deployment** | `ONNXInferenceTool` | Run ONNX model inference |
| **Deployment** | `AnomalyScorerTool` | Score and threshold anomalies |
| **Deployment** | `AlertManagerTool` | Generate reports and alerts |

---

## Data Format

### dqmhist-1.0 Schema

All histogram data uses the `dqmhist-1.0` JSONL schema:

```json
{
  "schema": "dqmhist-1.0",
  "name": "ECAL/EBOccupancy/occupancy_map",
  "shape": [64],
  "counts": [0.02, 0.05, 0.12, ...],
  "bin_edges": [0, 1, 2, ...],
  "run": 370000,
  "lumi_section": 1,
  "subsystem": "ECAL"
}
```

### dqmscore-1.0 Schema

Inference output uses the `dqmscore-1.0` schema:

```json
{
  "schema": "dqmscore-1.0",
  "histogram_index": 0,
  "histogram_name": "ECAL/EBOccupancy/occupancy_map",
  "anomaly_score": 0.00234,
  "run": 370000,
  "lumi_section": 1,
  "subsystem": "ECAL"
}
```

---

## Tool Documentation

### Data Access Tools

#### DQMIOReaderTool

**Purpose**: Read histogram data from ROOT files (DQMIO or legacy format)
using `uproot` for pure-Python access without CMSSW.

**Input Parameters:**
- `root_file_path` (str): Relative path to input ROOT file
- `output_path` (str): Relative path for output JSONL
- `subsystem_filter` (str, optional): Filter by subsystem prefix (e.g. `"ECAL/"`)
- `histogram_name_filter` (str, optional): Comma-separated histogram names
- `run_number` (int, optional): CMS run number for metadata
- `normalization` (str, optional): Normalization method
- `target_nbins` (int, optional): Rebin to this many bins

**Example:**
```python
from tools.dqm.data_access import DQMIOReaderTool

tool = DQMIOReaderTool(base_directory="./workspace")
result = tool.run(
    root_file_path="data/DQM_V0001.root",
    output_path="data/ecal_histograms.jsonl",
    subsystem_filter="ECAL/",
    normalization="unit_area",
    run_number=370000,
)
```

---

#### DQMAPIClientTool

**Purpose**: Query the CMS DQM GUI web API for histogram data and run
metadata.

**Actions:**
- `"list_runs"` — List available runs for a dataset
- `"list_histograms"` — List histogram paths for a run
- `"fetch_histogram"` — Download histogram data to JSONL
- `"run_summary"` — Get run quality status

**Example:**
```python
from tools.dqm.data_access import DQMAPIClientTool

tool = DQMAPIClientTool(
    base_directory="./workspace",
    dqm_api_url="https://cmsweb.cern.ch/dqm/offline",
)
result = tool.run(
    action="fetch_histogram",
    run_number=370000,
    histogram_path="ECAL/EBOccupancy/occupancy_map",
    output_path="data/fetched.jsonl",
)
```

---

### Training Tools

#### AutoencoderTrainerTool

**Purpose**: Train autoencoder models on DQM histogram data for anomaly
detection.  Supports vanilla, convolutional, and variational architectures
via YAML run cards.

**Supported Architectures:**
- `vanilla_ae` — Fully-connected autoencoder
- `convolutional_ae` — 1-D convolutional autoencoder with batch norm
- `variational_ae` — Variational autoencoder (VAE)

**Input Parameters:**
- `run_card_path` (str, optional): Path to YAML training run card
- `input_data_path` (str, optional): Path to histogram JSONL
- `output_dir` (str): Output directory for model artifacts
- `use_synthetic_data` (bool): Use synthetic data for testing
- `n_synthetic` (int): Number of synthetic histograms

**Output Artifacts:**
- `model.onnx` — Trained model in ONNX format
- `training_metrics.json` — Loss curves and performance
- `model_metadata.json` — Full provenance metadata
- `run_card.yaml` — Training configuration used

**Example Run Card** (`train_ecal.yaml`):
```yaml
schema: dqm-training-1.0
model:
  architecture: convolutional_ae
  input_shape: [1, 64]
  latent_dim: 32
  encoder_channels: [16, 32, 64]
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
  early_stopping:
    patience: 10
    min_delta: 0.0001
data:
  subsystem: ECAL
  input_jsonl: data/ecal_histograms.jsonl
  normalization: unit_area
export:
  format: onnx
  opset_version: 17
```

**Example:**
```python
from tools.dqm.training import AutoencoderTrainerTool

tool = AutoencoderTrainerTool(base_directory="./workspace")
result = tool.run(
    run_card_path="configs/train_ecal.yaml",
    output_dir="models/ecal_ae_v1",
)
```

---

#### ModelRegistryTool

**Purpose**: Manage trained model artifacts with versioning, metadata
tracking, and querying.

**Actions:**
- `"register"` — Register a new model
- `"list"` — List all registered models
- `"query"` — Search by subsystem or architecture
- `"get"` — Get details for a model by ID
- `"delete"` — Remove a model entry

**Example:**
```python
from tools.dqm.training import ModelRegistryTool

tool = ModelRegistryTool(base_directory="./workspace")
result = tool.run(
    action="register",
    model_dir="models/ecal_ae_v1",
    version_tag="v1.0.0",
    description="ECAL occupancy anomaly detector",
)
```

---

### Deployment Tools

#### ONNXInferenceTool

**Purpose**: Run ONNX model inference on DQM histograms to compute
per-histogram reconstruction errors.

**Input Parameters:**
- `model_path` (str): Path to ONNX model file
- `input_data_path` (str): Path to histogram JSONL
- `output_path` (str): Path for output anomaly scores JSONL
- `normalization` (str): Must match training normalization
- `batch_size` (int): Inference batch size

**Example:**
```python
from tools.dqm.deployment import ONNXInferenceTool

tool = ONNXInferenceTool(base_directory="./workspace")
result = tool.run(
    model_path="models/ecal_ae_v1/model.onnx",
    input_data_path="data/new_run.jsonl",
    output_path="results/scores.jsonl",
    normalization="unit_area",
)
```

---

#### AnomalyScorerTool

**Purpose**: Process raw reconstruction errors into human-interpretable
verdicts (GOOD / WARNING / BAD) with configurable thresholds.

**Threshold Methods:**
- `"static"` — Fixed absolute thresholds
- `"percentile"` — Distribution-based percentile thresholds
- `"sigma"` — Standard deviation multiples

**Example:**
```python
from tools.dqm.deployment import AnomalyScorerTool

tool = AnomalyScorerTool(base_directory="./workspace")
result = tool.run(
    scores_path="results/scores.jsonl",
    output_path="results/scored.jsonl",
    threshold_method="percentile",
    warning_threshold=95.0,
    bad_threshold=99.0,
)
```

---

#### AlertManagerTool

**Purpose**: Generate structured reports and alert payloads from scored
anomaly data.

**Output Artifacts:**
- `anomaly_report.md` — Markdown summary with top-N anomalies, subsystem
  breakdown, and threshold configuration
- `alert_payload.json` — JSON payload for downstream integration

**Example:**
```python
from tools.dqm.deployment import AlertManagerTool

tool = AlertManagerTool(base_directory="./workspace")
result = tool.run(
    scored_path="results/scored.jsonl",
    output_dir="reports/run_370000",
    top_n=10,
    report_title="Run 370000 DQM Report",
)
```

---

## Dependencies

**Required:**
- `numpy` (included with HEPTAPOD)
- `pydantic` (included with HEPTAPOD)

**For data access:**
- `uproot`, `awkward` — ROOT file reading (`pip install uproot awkward`)
- `requests` — DQM API access (included with most Python installations)

**For training:**
- `torch` — PyTorch for model training (`pip install torch`)
- `pyyaml` — YAML run card parsing (`pip install pyyaml`)

**For deployment:**
- `onnxruntime` — ONNX model inference (`pip install onnxruntime`)

---

## Testing

```bash
# Run all DQM tests
python test_runner.py --only dqm_data_access
python test_runner.py --only dqm_training
python test_runner.py --only dqm_deployment
```
