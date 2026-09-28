# Stage 4 report, which sensors carry the signal

Date: 25 September 2026

## Why this test

The full rig has eight sensor nodes on two devices, a behind the ear headpiece and a chest unit. I wanted to know which of these sensors actually carry the seizure prediction signal. So I retrained the model with sensor groups dropped in steps, down to the smallest device I could test. This test is not about whether a smaller device can pass the budget. It is about which sensors matter. The earlier alarm checks had already shown the 0.15 false alarms per hour budget is out of reach with the full set at any threshold. The question here is a simpler one. What does each sensor group add?

## How I ran it

To drop a sensor group I zeroed its node features, in training and in every scoring pass. The graph comes from the pack and is never rebuilt, so the only thing that changes is the information the model sees. The alarm rule is unchanged. The threshold comes from the validation and monitor slices only, at the 0.15 false alarms per hour budget. The earlier full sensor runs are the controls, and each reduced run is compared against its own control, patient by patient. Fifteen patient and warning combinations had a trained control, and ten of those had a scored test seizure. The tiers follow the physical devices. The full rig is the headpiece plus the chest unit, eight nodes. Headset only drops the chest unit. EEG plus accelerometer keeps the two ear channels and the head motion sensor. EEG only is the two ear channels alone, the minimal hearable.

## What happened

Dropping sensors traded catches for silence. The median false alarm change was zero. Seven runs already over budget rose further, and none of them bought a catch. Headset only kept 5 of the 6 control catches and cut the median false alarm rate from 1.042 to 0.315 per hour, about two thirds down. EEG plus accelerometer kept 1 of the 10 scored catches, at a median of 0.000 false alarms per hour. EEG only kept 3 of 10, also at 0.000. Those 3 catches look like noise. Each is a single seizure, and the smallest tier caught more than the tier above it, which is not what a real signal would do.

## One extra test

On sub 051 at the 30 minute warning I ran one extra test. I trained on the other patients first, then fine tuned with EEG only. That run kept the catch that plain EEG only loses, 1 of 1, at the same false alarm rate as its control, 1.279 per hour. So the ear pair alone can carry this patient's signal, but only with help from the other patients' data.

## What failed, and what is left out

Sub 007 could not be trained in any tier. Its training windows fail the signal quality gate, the same gate that made it untrainable in the earlier stages. A single ear variant, one channel, is left out. The graph needs at least two nodes, so that variant needs a different graph setup. It is future work.

## Conclusion

The chest unit adds more false alarms than catches. Dropping it costs almost nothing and quiets the alarms by two thirds. Below the headset the picture flips. The ear pair alone goes quiet and misses nearly every seizure, so the extra head sensors do the real work. A small device would need those sensors, or help from the other patients' data. On sub 051 that help rescued even the minimal device.
