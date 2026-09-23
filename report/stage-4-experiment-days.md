# Stage 4 report, the experiment days

Date: 22 September 2026

## Why these days happened

The first results said the task is hard, not that the recipe is broken. So I spent two days trying three small changes to training. These were not in the original proposal. One change makes the loss focus more on hard examples (focal loss). One adds small random noise to the training windows. One trains the model on the other patients first, then fine tunes on the patient being tested (pretrain). Every run is scored the same way as the earlier runs, so the numbers read side by side. Alongside this I checked the labels themselves.

## The labels hold up

I rebuilt the labels from the raw event files, on my own, and compared them with what the packer recorded for every patient and warning window. The two agreed exactly on all 19 packs I could check: same split times, same seizure counts. The slices that looked empty are empty for honest reasons. The validation part has no pre seizure windows by design, since it is the slice used to set when the alarm fires. For sub 039 at the 10 and 5 minute warnings the pre seizure seconds exist, but the signal quality checks drop every one of them. That is a data quality fact, not a labelling bug. So the labels are sound.

## One change helped, on one patient

The clear result is the pretrain, and only on sub 051. It kept the one test seizure caught and cut the false alarms, at both the 15 minute warning (from 1.259 to 0.629 false alarms per hour) and the 30 minute warning (from 1.279 to 0.640). The 30 minute one is the repeat: the same gain twice, on two different windows. Nothing else did this. Focal loss alone and the noise alone each cost sub 051 its catch. All three together kept the catch but gave back the false alarm gain.

## The gain stops there

Pretrain helped nowhere else. On sub 051 at the 5 minute warning it made things worse, one false alarm becoming two. On sub 113 at 5 minutes it was worse too, four becoming five, and that recording stays far over budget no matter what; the earlier alarm checks proved no alarm setting can separate it. On sub 052 at 15 minutes it removed one false alarm but still missed the seizure, so nothing real moved. On sub 113 at 30 minutes and sub 039 at 15 minutes there was no scored test seizure to win. And on sub 022, the new patient added under the relaxed gate, pretrain missed the one scored seizure at all three warning windows. So the gain is specific to sub 051, not a general fix.

## Conclusion

Pretrain is a real but narrow gain: it helped the one patient where a usable signal already existed, and it does not rescue the patients where the task is too hard. That fits the wider picture from the alarm checks: on the hard recordings the scores themselves do not separate, so no training change can fix them. The label check closes the other door: the failures are not mislabelled data. What remains hard is the task itself on these wearable signals.
