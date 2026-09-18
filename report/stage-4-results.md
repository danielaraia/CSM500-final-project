# Stage 4 report, the first results and what they mean

Date: 17 September 2026

## The 30 minute warning did not separate for most patients

Five patients could be scored fairly at 30 minutes. They are subject 051 (sub 051), subject 052 (sub 052), subject 015 (sub 015), subject 039 (sub 039), and subject 073 (sub 073). For most, the model did not separate the pre seizure windows, the 30 minutes before a seizure, from other test windows. The study target was to catch at least half of the scored seizures at no more than 0.15 false alarms per hour, wrong alarms in an hour. Sub 039 met it, with caveats. Its one scored test seizure was caught with zero false alarms, but only 0.064 interictal hours, the hours far from any seizure, were usable, too few to judge the rate, and four of its five test seizures could not be scored. Sub 015 caught its one test seizure at 0.824 false alarms per hour, and sub 051 caught its one with two of the three ways of connecting the signals, both over budget. Sub 052 and sub 073 caught none. Subject 004 (sub 004) and subject 113 (sub 113) could not be scored fairly, and are reported, not dropped.

## The diagnosis points to a hard task, not a broken recipe

Three patients had the first training round as their best. An honest second training run showed the stops were real, not a bug. The training loss matched a model that always answers zero, and later rounds got worse. On sub 051 and sub 052, the test scores of the two window kinds overlapped fully for all three ways of connecting the signals. A simple baseline, a logistic regression trained on the same windows, also failed to separate them. The task looks hard at 30 minutes, not the recipe broken.

## What this means

The data suggests the 30 minute warning does not work on these wearable signals. The shorter windows, 15 and 5 minutes, may still have a usable signal. I will see how the progress goes.