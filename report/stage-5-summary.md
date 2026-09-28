# Stage 5 report, the closing summary

Date: 27 September 2026

## What the project set out to do

I wanted to know if seizures can be predicted from wearable signals. It is a challenging problem. The recordings come from a wearable worn at home, not from a clinic EEG cap, seizures are rare events, and every patient is different. I tried three ways of building the graph that links the signals, Pearson correlation, mutual information and learned attention, and I tested each one at three warning windows, 5, 15 and 30 minutes. I fixed the target before I ran anything. A run had to catch at least half of the test seizures that could be scored, at no more than 0.15 false alarms per hour.

## What was built

I got the full pipeline working end to end. The packer turns the raw recordings into labelled windows, with data quality checks on the way. Three graph builders set the edge weights. The EvoBrain backbone trains on a rented GPU, always with seed 123 and the same training recipe. The alarm thresholds come from the validation data at the 0.15 per hour budget, never from the test data, and the scoring counts caught seizures and false alarms per hour. I also ran four controls. One retrained some runs to check the early stops. One compared against a model that always predicts the majority class. One was a logistic regression on the same windows. The last one rebuilt the labels from the raw event files and agreed with my labels on all 19 packs.

## What happened

A lot of it worked. The pipeline ran end to end on real wearable recordings, and all four controls passed, which tells me the machinery does what I designed it to do. The label audit agreed with my labels on all 19 packs. The sensor experiments gave a clear answer: the head sensors carry the signal. Dropping the chest unit kept 5 of the 6 control catches and brought the median false alarm rate down from 1.042 to 0.315 per hour. Pretraining across patients helped the one patient with a usable signal. For that patient the false alarm rate at the 30 minute window dropped from 1.279 to 0.640, and the catch was kept.

Even so, no run met the target I fixed at the start, at any window. Which seizures got caught depended on the patient, not on the graph strategy. No fixed threshold rule worked across days, because the recordings change from day to day.

## What it means

The answer I found is no, at least for this cohort and this budget. On most recordings the minutes before a seizure do not look different enough from quiet time, at any threshold I tried.

But this is good progress and a real start, not a dead end. I have a pipeline that runs end to end on wearable data, with controls that show the machinery works, so whatever comes next can be trusted. The results also point at where to go. The head sensors carry the signal, so a simpler rig is enough. Some patients look predictable even when most do not, and pretraining across patients already helped the one patient with a usable signal. With more patients, longer recordings and more pretraining data, I believe this could turn into a working warning for at least some of the people who need it.
