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



## Usage

Run everything as a module from the repo root, for example:

```
python -m packs.pack_sop sub-051 01 pearson 15
python -m train.train_sop sub-051 pearson 15
python -m evaluation.report_sop
```

Also run the tests the same way:

```
python -m pytest
```

