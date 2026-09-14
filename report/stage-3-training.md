# Stage 3 report: Training the patients

Date: 13 September 2026

This stage trains the seven usable patient packs, in four batches. Every run uses the locked recipe from the pipeline stage: Pearson correlation graphs, a 30 minute pre seizure window as the positive class, seed 123, batch 32, and early stopping with patience 5 on a held back monitor slice. Train windows step every 10 seconds, and monitor, validation and test windows step every 1 second, so scoring sees every second. Training ran on rented GPUs through RunPod, mostly on an RTX 4090 card.

## The first batch trained two patients

Subject 004 (sub 004) and subject 051 (sub 051) trained first on the RTX 4090. Sub 051 had its best epoch at epoch 1. Sub 004 had its best epoch at epoch 2 of 7 epochs. Training took 1.1 seconds for sub 004 and 37.7 seconds for sub 051, and the full run from loading the pack to scoring took 10.3 and 136.5 seconds. The slow step was not the GPU. Uploading the 5 GB pack of sub 051 from home took about 28 minutes.

## The packs moved to Cloudflare R2

The slow home upload was the reason to change the data path. I uploaded the packed windows, about 20 GiB, once to Cloudflare R2, a cloud object store, and later pods pull the packs from there instead of from the laptop. The pull is fast: one later pod copied 7.6 GiB in 54 seconds.

## The second batch trained two more patients

Subject 052 (sub 052) and subject 113 (sub 113) trained next, on the same card and the same recipe. Their best epochs were 1 and 2, and the full runs took 25.1 and 5.9 seconds. Nothing notable happened in this batch.

## The third batch had the longest train

Subject 073 (sub 073) and subject 039 (sub 039) trained third. Sub 073 had its best epoch at 1 and trained in 10.6 seconds. Sub 039 was the long run of the campaign: 38 epochs, best at epoch 33, with 289.1 seconds of training. Its train loss stayed at 0.355 for all 38 epochs. The pod ran about 21 minutes.

## The last patient trained on a different card

Subject 015 (sub 015) trained last. The RTX 4090 was out of stock, so an L40S ran it. I built the pack on the pod itself, because the labelled tensors, about 22 GiB, were over the 14 GiB laptop memory limit, and the pack was then stored to R2. The best epoch was 9 of 14 epochs, with 84.0 seconds of training. The pod ran about 40 minutes. The whole training campaign cost under 100 US dollars, so the compute stayed affordable.

## Most runs stopped very early

The training logs showed one clear pattern, stated here only as an observation. Three of the seven patients had their best epoch at epoch 1, and with patience 5 that means the loop ran 6 epochs and stopped. Sub 004 and sub 113 had their best epoch at 2, sub 015 at 9, and sub 039 at 33. Why the model stops learning this early is not diagnosed in this report. Later reports look at why.
