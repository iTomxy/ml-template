Tune multiple hyper-parameters
(e.g. feature engineering, data augmentation, model size, optimisation)
in sequence.
See [1] for tutorial.

# General Principles

- Order: large impact first (e.g. data, defining the learning space),
specific later (e.g. loss weight).

- Control nuisance factors like learning rate and random seed.

- Collect all results in one / multiple table/s for insight. See [Result Table](#result-table).

- Fix the splitting (train/val/test) and epoch/iteration budget when tuning, including baseline.

- Reuse the baseline random seeds in tuning for paired comparison against the noise floor (baseline stddev).
But choose different seeds for consecutive hyper-parameter groups.
    - The statistical reason behind (by Claude):
    A run's val score is the configuration's true quality plus a random offset from the random seed (init, data order, dropout, augmentation).
    The seed offset is often larger than the difference between configurations.
    Sharing a seed makes that offset common to both runs,
    so it reduces comparison noise,
    and what remains is a cleaner estimate of the true difference.

- Fix the lr range test and lr suggestion protocol, including:
    - lr suggestion rule, e.g. (lr @ min loss) / 10.
    - start/end lr, iteration, smoothing EMA, stop criterion (e.g. loss > 4 x min_loss).
    - using the same fixed seed for that hyper-parameter group.

# Tuning Workflow (17 Sept 2026)

1. Begin with a simplest (e.g. no data augmentation, equal loss weight) yet reasonable (that can produce reasonable performance) recipe.
    - Default optimisation suite: AdamW + 1-cycle lr schedule + lr range test.
    - Consider enabling SyncBN whenever multiple GPUs are used for distributed training.
    - When affordable, run multiple (e.g. 3) repetitions,
    each with different random seeds and individual lr range test,
    to determine the noise floor.
    Any later tuning with *improvement* smaller than the noise floor risks being fake improvement,
    but can also be real but small.
    One heuristic (use standard deviation (stddev) instead of standard error (stderr) here, regarding the metric used for model selection):
        - delta <= 0:
        - 0 < delta < stddev: noise, potential fake improvement -> discard;
        - stddev <= delta < 2 * stddev: promising, worth confirming with repeated runs.
        - delta >= 2 * stddev: strong signal of improvement.

2. Determine *total* batch size for max efficiency,
then fix it for tuning and final running.
    - See [Determining Batch Size](#determining-batch-size).
    - Tune batch size only for training efficiency, not for performance.
    - No need to tune for each configuration / run unless the hardware environment is changed
    (e.g. changed to another gpu node with different gpu memory,
    or occupied by another user),
    or the model size is scaled up, causing OOM.
    - Leave some gpu memory headroom, e.g. for validation or temporary allocations.
    - One caveat needing total batch size consistency:
    different total batch size brings difference to gradient noise (implicit regularization) and BN.
    So keep it consistent across every run in a tuning campaign:
        - all repetitions: baseline (Step 1), tie-break candidate configurations (Step 3), final recipe (Step 4);
        - all candidate values (combinations) in a hyper-parameter group (Step 3).

3. Tune each hyper-parameter group in order.
    - A *hyper-parameter group* can be:
        - one hyper-parameter: sweep its candidate values, e.g. `lambda1 = a, b, c`; or
        - several closely related hyper-parameters: grid-search their candidate value combinations, e.g. `lambda2 x lambda3 = (d, e, f) x (g, h, i)`.
    - Choose one fixed seed for each hyper-parameter group across its different value (combination) from the baseline seeds (Step 1),
    but choose one different from the last group.
    E.g. seed1 for lambda1 = a, b, c; seed2 for lambda2 x lambda3 = (d, e, f) x (g, h, i).
    - Run range test to find lr for each configuration / run.
    I.e. seed1 for lambda1 = a, b, c;
    but individual range test for lambda1 = a, lambda1 = b, and lambda1 = c.
    Do not transfer the lr for one configuration (e.g. lambda1 = a) to others (e.g. lambda1 = b).
    - The paired/comparable *baseline* of a new group should be the inherited recipe from last group **run with the seed for the new group**.
    So if the inherited recipe is not run with the seed for the new group,
    add this run.
    This also brings a cheap verification of whether the *improvement (magnitude)* is just a random luck
    (compare the inherited recipe performance with different seeds).
    - Pick the best configuration on validation.
    - If the top several configurations have close (| delta | < baseline stddev, Step 1) val performance,
    randomness can change the selection.
    So sweep a same random seed set for each and select the winner with average performance across the seed set.
    E.g. both lambda1 = b, c are promising, with the same seed1.
    Run 2 more repetitions of each with the rest baseline seeds (Step 1) other than the used one for this group,
    all with individual lr range test.
    Then select the lambda1 value based on average val performance.
    - If a run diverges, spikes, or plateaus early,
    suspect its range-test LR before concluding the hyperparameter value is bad.
    Mark it unreliable and re-run if affordable.

4. Run 3 repetitions with the final tuned recipe with different seeds.
    - Using a *lr finding procedure* (e.g. range test) vs. a *fixed tuned lr* can also be a design choice.
    If this is tuned in Step 3, use the selected one here.
    - Use different seeds than the baseline ones.
    - Report in [Result Table](#result-table).

## known caveats

- 3 - 5 repetitions provide a very uncertain variability estimate.
Treat the noise floor as an order of magnitude, not a threshold.

- Sequential tuning assumes low interactions between hyper-parameters across groups.
May need to re-tune an old hyper-parameter jointly with a new one.

- Compare the final recipe (Step 4) stddev against the baseline (Step 1) noise floor.
If it grew substantially,
the recipe's variance changed during tuning,
and decisions made near the old floor are worth revisiting.

# Determining Batch Size

- Use real recipes when finding batch size.
- Ensure the batch size is a multiple of the number of gpus.
- When involving multiple clusters and configurations that can affect the batch size selection (e.g. backbone, input size),
consider recording a (specific) [batch size table](#batch-size-table),
so the selected batch size transfers to a new cluster node with seen hardware environment (same gpus).

Steps:
1. Find feasible batch size range [min batch size, max batch size].
    - Strange exceptions can occur besides OOM when a batch size does not fit.
    So catch all exceptions instead of OOM only.

2. Measure the throughput (end-to-end training examples per second) after warm-up (first several iterations) of each feasible batch size.
    - Consider logging the [throughput table](#throughput-table) in a json/csv file for later reference.

3. Heuristic: the selected batch size is not the one achieving largest throughput,
but the smallest one achieving roughly 90% - 95% of that.

## throughput table

| batch size | time per iteration (ms) | examples per second |
| ---: | ---: | ---: |
| 8 | 80 | 100 |

## batch size table

An example with pointcloud perception task.

| cluster | node | gpu type | gpu num. | used gpu num. | mem of each gpu (MiB) | backbone | point num. | selected total batch size |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| ihpc | mars28 | NVIDIA L4 | 2 | 2 | 23034 | PTv3 | 200000 | 16 |
| cbai | - | NVIDIA RTX A4000 | 4 | 4 | 16376 | PTv3 | 200000 | 8 |
| ihpc | mars28 | NVIDIA L4 | 2 | 2 | 23034 | DGCNN | 15000 | 8 |
| cbai | - | NVIDIA RTX A4000 | 4 | 4 | 16376 | DGCNN | 15000 | 4 |
| cbai | - | NVIDIA RTX A4000 | 4 | 2 | 16376 | DGCNN | 15000 | 2 |

## known caveats

- variable-size inputs.
E.g. PointTransformer with grid sampling.
Estimate the maximum feasible batch size against worst-case input size, not an average/random/empirical one.

# Result Table

- columns: include key info and design choices.
- rows: variants and/or repetitions.
- report only val set performance in tuning rows,
while also reporting test set for only final tuned recipe.
- report std**dev** of val set performance in tuning and final repetitions as noise floor,
but std**err** of test set performance in final repetitions to show the precision of the average test set performance.

| variants | log path | epoch | lr | data aug | input feat. | backbone | #layers | losses | loss | loss_term_1 | metric_1 |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| baseline (run 1, seed=42) | log/baseline/1/ | 79 | 0.0345 | N/A | xyz | PTv3 | 5 | ce + dice | 0.5 | 0.1 | $VAL_METRIC |
| baseline (run 2, seed=9527) |
| baseline (run 3, seed=1210) |
| baseline (aggregation, n=3) | | | | | | | | | | | $VAL_AVG +- $VAL_STDDEV |
| method 1 |
| final (run 1, seed=12138) |
| final (run 2, seed=13) |
| final (run 3, seed=65535) |
| final (aggregation, n=3) | | | | | | | | | | | $VAL_AVG +- $VAL_STDDEV /<br/>$TEST_AVG +- $TEST_STDERR |

# References

1. [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook)
