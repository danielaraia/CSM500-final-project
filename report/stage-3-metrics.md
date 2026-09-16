# Stage 3 report: Measuring what a clinician would ask

Date: 15 September 2026

## A loss number does not answer the clinical question

The training loop reports a loss number. A clinician asks two things instead: how many seizures did you catch, and how many false alarms per hour. So an alarm scorer was built. The model scores every 1 second step, a threshold turns scores into alarm seconds, and an alarm fires when at least 240 scored seconds inside the last 300 clock seconds sit over the threshold. A seizure is caught if an alarm fires inside its 30 minute pre seizure window, and false alarms are counted per hour on interictal seconds, the seconds far from any seizure. The study target, set in advance: catch at least half of the evaluable seizures at no more than 0.15 false alarms per hour. One honest detail: the scorer reloads the best checkpoint before scoring, since the train loop leaves the last epoch in memory, not the best one.

## A seizure without enough clean seconds is marked unevaluable

Quality control removes bad seconds, so some test seizures lack the 240 clean lookback seconds and the vote can never fire for them. Those seizures are marked unevaluable: reported separately, never counted as misses or silently dropped. For sub 004, the longest clean lookback inside the test pre seizure window is 145 seconds, under 240, so its test seizure is unevaluable.

## The threshold comes from validation, never from test

The threshold is picked on the validation slice, the slice of windows between train and test: the lowest threshold that keeps validation false alarms at or under 0.15 per hour. The test set is never used for this. When the validation slice is empty, the threshold falls back to the monitor slice, the last quarter of the train windows kept back for early stopping, at the same 0.15 per hour budget. The monitor slice was repacked at 1 second steps, since the 10 second train stride gives too few scored seconds for the vote.
