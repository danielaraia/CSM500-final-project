# Stage 3 report: Building the training pipeline

Date: 12 September 2026

This stage builds the training pipeline and checks it is fast enough on a GPU. The model is EvoBrain, a temporal graph neural network cloned from its public repository. Training the patients comes in the next report. This one covers only the pipeline build and the timing gate.

## The model loads with locked arguments

The loader builds EvoBrain with locked arguments: 8 nodes, 60 time steps per window, 128 feature bins per node, 4 eigenvectors for the graph encoding, and data augmentation off. Locked means the config writer fails loudly if an extra value tries to change one of these. I locked them so every patient and every graph strategy trains under the same settings, because then runs can be compared later. The loader also fails loudly if the GPU packages, torch_geometric and mamba_ssm, are missing.

## The training pieces

One train step is Adam with binary cross entropy, and the loss is weighted by class counts: the weight is the number of negative windows divided by the number of positive ones. Pre seizure windows are rare, and this way the two classes pull equally on the loss. The loop watches the class weighted loss on the monitor slice, the last quarter of the train windows kept back for early stopping, and stops after 5 epochs without improvement, saving the best checkpoint. Each run starts in its own folder from a locked config, and the folder fails if it already exists, so nothing gets overwritten. After training, the run writes a metrics file with the class weight, the window counts, and the best monitor loss.

## The timing measurements on the RTX 4090

A CUDA script times the model on synthetic tensors of the locked shapes (60 time steps, 8 nodes, 128 bins) on an RTX 4090. One training sample costs about 1.1 milliseconds at batch 32. One scored second costs about 0.67 milliseconds at batch 256. Peak memory at batch 32 is about 1.7 GiB, and the model has about 284000 parameters. After this, one full patient run was timed end to end on subject 004 (sub 004), from the raw EDF files to a score. The wall clock was 77.17 seconds: about 74 for preprocessing, about 2.3 for training, and about 0.3 for scoring. Compiling the model took about 3 seconds and sits outside that number.

## The timing gate passed

The gate rule was plain. One full patient run, from EDF files to a score, must finish inside a measured and acceptable time before more patients get packed into training tensors or trained. The measured time is 77.17 seconds for the patient, and at 1.10 milliseconds per training sample the whole planned training campaign comes to about 94 GPU hours. That was acceptable, so the gate passed and the packing of more patients started.

## The first train ran on a laptop

The first train of packed windows was a smoke test on the laptop. EvoBrain needs CUDA packages the laptop does not have, so I ran a stand in model with 2 parameters instead. It reached a best monitor loss of 1.086 at epoch 97. The number means nothing as science. The point was to prove the loop, the checkpoints, and the metrics file all work on real packed windows. A real EvoBrain train step then ran on the 4090.
