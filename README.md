# seizure-tgnn

Seizure prediction from wearable sensor data with a temporal graph neural network. This is my final project for the UoL MSc program.

## Layout

- run.py: shared pipeline helpers used by the trainers and scorers
- core/: config, recipes, and channel definitions
- data/: EDF loading, quality checks, windowing, labels, and features
- packs/: build the packed train, monitor, validation, and test splits
- graphs/: graph builders for pearson, mutual information, and attention
- model/: model glue, the EvoBrain loader, and the training loop
- train/: training entry points
- evaluation/: scoring, calibration, and report scripts
- tools/: screening, timing, and audit utilities
- tests/: the pytest suite
- results/: the run summaries behind the report numbers



## Setup

```
python -m pip install -r requirements.txt
python -m pytest
```

Training needs an NVIDIA GPU. The GPU setup uses Python 3.10, CUDA 12.1,
PyTorch 2.5.1, PyTorch Geometric 2.3.1 and mamba_ssm 2.2.4.

```
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
python -m pip install -r requirements-gpu.txt
python -m pip install mamba-ssm==2.2.4 --no-build-isolation
```

## Usage

Run everything as a module from the repo root, for example:

```
python -m packs.pack_sop sub-051 01 pearson 15
python -m train.train_sop sub-051 pearson 15 123
python -m evaluation.score_sop sub-051 pearson 15 123
```

Run one patient and then build the final summary:

```
python -m tools.run_patient_grid sub-051
python -m evaluation.experiment_analysis
```

Rebuild the saved results without a GPU:

```
python -m evaluation.experiment_analysis results/run-metrics.json
```

