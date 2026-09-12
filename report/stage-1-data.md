# Stage 1 report: Data acquisition and preprocessing

Date: 14 July to 4 August 2026

This stage implements the data preparation. It turns the raw SeizeIT2 recordings into packed training tensors. SeizeIT2 is a public wearable seizure dataset on OpenNeuro, accession ds005873. Each run is four EDF files, for EEG, ECG, EMG and movement, plus an events.tsv with seizure times.

## Reading the EDF files

The loader then checks channel names in the EDF headers rather than guessing. It needs two behind the ear EEG channels and also fails loudly if they are missing. The events reader pulls seizure onsets and offsets from events.tsv. The converter rails, the physical minimum and maximum a channel can report, all come from the headers. One thing I did not expect: EDF allows silent gaps between records, and the Python library joins them together. So the loader reads record time stamps from the raw bytes and then it fails a run whose clock jumps.

## Preprocessing

Each signal is filtered with a band pass and a 50 Hz mains notch. Node features are the log amplitude of an FFT in 128 frequency bins per second. Movement channels arrive at 25 Hz and their magnitude is then upsampled up to 256 Hz to match the electrical channels.

Quality control works in one second steps. A second is bad if an electrical channel is flat, if a channel sits on its converter rail, or if EEG RMS falls outside 13 to 150 microvolts on both ears at once. I picked the both ears rule on purpose. One quiet ear used to remove almost all interictal seconds on the first patient. Impedance check spans and EDF gaps also drop seconds. On that patient about 40 thousand seconds were dropped, almost all by the RMS rule.

## Windows, labels and splits

Windows are 60 seconds long and never splice across a hole. A window covering a dropped second is thrown away. Train windows step every 10 seconds; monitor, validation and test every 1. Labels come from the window end second. The 30 minutes that end 5 minutes before a seizure are SOP and pack as 1. Seconds far from any seizure are interictal and pack as 0. Everything else is dropped.

Splits are cut from seizure times, not at random. The last third of usable seizures, rounded up, goes to test; the cut sits halfway between the last train and first test seizures. A validation slice grows backwards from there, and the last 25 percent of train becomes a monitor slice for early stopping. Per node z scoring is fit on gradient train windows only, then applied to the other splits, so no test statistics leak into training.

## A pack that looked fine but was useless

The packer fails to write a patient whose gradient train has no SOP windows. This gate exists because of subject 007, written sub 007 in the dataset. That patient packed without errors: 93 train windows, 31 monitor, 1208 validation, 132 test. But every train window was interictal. The patient had only two usable seizures, and the monitor cut had put all SOP windows in the monitor slice. The overall counts looked fine and the pack was useless. The folder is still on disk, untrained. I thought about deleting it, but leaving it keeps the failure visible.

## Screening more patients

A clock screen reads events.tsv and sidecar files to find candidates with enough usable seizures before any EDF download. It is cheap but blind: it cannot see quality control removing SOP seconds. Five candidates passed the screen, then failed the real pack with no SOP left where it was needed. Seven patients came out packed and usable: sub 004, sub 051, sub 052, sub 113, sub 073, sub 039 and sub 015.