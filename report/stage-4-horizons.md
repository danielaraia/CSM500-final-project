# Stage 4 report, the 15 and 5 minute warnings and the threshold study

Date: 26 September 2026

## What I ran

The 30 minute warning did not separate for most patients, so I moved to the shorter windows. I ran the same recipe at the 15 minute and the 5 minute warning, for every patient with enough data, with all three ways of connecting the signals. That is 42 patient and warning combinations. Alongside this I ran a calibration study, asking whether any fixed rule for setting the alarm threshold works across patients.

## What happened at 15 and 5 minutes

Catches happen at 15 minutes. Sub 051 caught its one test seizure with one of the three ways of connecting the signals, and sub 113 caught its one with all three. But the false alarm rates land around 0.6 to 19 per hour across the scored runs, against the budget of 0.15 per hour. At 5 minutes only sub 113 caught, at 19.4 false alarms per hour. And 18 of the 42 combinations never ran at all. Their packs fail the data quality gate, because the shorter window can leave a training or monitor slice with no pre seizure windows. That is data geometry, recorded honestly, not worked around.

## The calibration study

The thresholds above come from the validation or monitor slices only, never from test. I checked whether any fixed rule for picking that threshold works across patients. Seven rules, applied blindly to all 34 scored runs across the three warning windows. None generalizes. On every run with selection data, the chosen threshold shows 0.000 false alarms on the slice it was picked on, then drifts over budget on test, on 22 of the 34 runs. These wearable recordings shift across days, so the quiet slices the threshold is picked on do not look like the test days.

## The check on test itself

To close the door I fitted the threshold on test itself, only to ask whether any threshold could work there. For 5 of the 8 runs that catch a seizure, no threshold at any value separates the caught seizures from the quiet windows. Calibration cannot fix those runs. The scores themselves do not separate on those recordings.

## Conclusion

The same conclusion as the first results, now with more evidence. The recipe is not broken. The training behaves, and no threshold rule was going to rescue the numbers. The task at this false alarm budget is hard on these wearable signals, at every warning window I tried.
