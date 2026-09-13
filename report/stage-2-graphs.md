# Stage 2 report: Graph construction strategies

Date: 5 August to 3 September 2026

This stage builds the graph part of the model. The model reads 8 nodes from the wearable recorder: 2 behind the ear EEG channels, ECG, EMG, and 4 movement groups, accelerometer and gyroscope at the neck and at the chest. The nodes are the signals, and the edges say which signals relate to which and how strongly. Every 1 second step of the model gets its own 8 by 8 adjacency matrix, 28 node pairs. Three strategies for the edge weights were built and tested.

## Pearson correlation was built first

The first strategy sets each edge weight to the Pearson correlation between the two node signals. Pearson correlation measures linear coupling between two series. Unit tests cover an identical pair at 1, a hand checked known value, and a constant series that fails loudly. Pairs of electrical nodes use 1 second of signal at 256 Hz. Pairs that touch a movement node use 4 seconds at the native 25 Hz, for the reason below.

## Mutual information was built second

The second strategy sets each edge weight to the mutual information between the two node signals. Mutual information measures coupling without assuming linearity, so it can see relationships Pearson correlation misses. The estimator is the sklearn function mutual_info_regression with 3 neighbours and random state 123. The builder reuses the same pair series as Pearson, so the comparison is only the estimator, not the data.

## The movement upsample breaks mutual information

The movement sensors record at 25 Hz. The early design upsampled movement to 256 Hz inside each 1 second window, so 231 of the 256 points were fabricated by interpolation. The estimator treats every point as an independent draw, which is false for the fabricated points. A dedicated check measured the damage on 400 real windows from four patients. The estimate built on the upsampled points and an honest estimate on real points only correlate at 0.260, while Pearson correlation on the same interpolated pairs still tracks the honest estimate at 0.688. 22 of the 28 node pairs touch a movement node, so this is most of the graph, not a corner case. I did not expect the two estimators to differ this much. The fix: pairs touching a movement node are estimated on 4 seconds at the native 25 Hz, 100 real points on both sides, applied the same way to Pearson and to mutual information.

## Attention learns the edges by itself

The third strategy does not compute edges from the raw signals at all. A graph learner inside the network learns the edge weights from the same 128 bin FFT node features as the rest of the model. The module is self attention with hidden size 32 and adds 8192 parameters. Because the edges are learned at train time, the packer fails if asked to pack attention from EDF files. Attention reuses the Pearson window packs and replaces the stored adjacency during training and scoring.

## All three strategies share one pipeline

Each strategy was wired into the same pack and train pipeline, so the same patients run through all three. I kept Pearson as the default. Mutual information packs with one extra command argument into its own run folder, and attention reuses the Pearson packs. Subject 051 (sub 051) and subject 052 (sub 052) were trained through all three strategies with identical chronological cuts; only the adjacency differed. On the 30 minute prediction task, none of the three separated the pre seizure windows from windows far from any seizure on the test split.
