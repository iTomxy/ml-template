Tune multiple hyper-parameters
(e.g. feature engineering, data augmentation, model size, optimisation)
in sequence.
See [1] for tutorial.

- Order: large impact first (e.g. data, defining the learning space),
specific later (e.g. loss weight).
- Control nuisance factors like learning rate and random seed.

# Tuning Workflow (17 Sept 2026)

1. Begin with a simplest (e.g. no data augmentaiton, equal loss weight) yet reasonable (that can produce reasonable performance) recipe.
    - Default optimisation suite: AdamW + 1-cycle lr schedule + lr range test.
    - Enable SyncBN whenever multiple GPUs are used for distributed training.
    - When affordable, run multiple (e.g. 3 - 5) repeated runs with different random seeds to determine the noise floor.
    Any later tuning with ``improvement'' smaller than the noise floor should be treated as noise instead of real improvement.

2. Determine batch size for max efficiency,
then fix it for tuning and final running.
    - See [Determining Batch Size](#determining-batch-size).
    - No need to tune for each run unless the hardware environment is changed
    (e.g. changed to another gpu node with different gpu memory,
    or occupied by another user),
    or the model size is scaled up, causing OOM.

3. Tune each hyper-parameter once in a time.
    - One fixed random seed for each hyper-parameter across its different value.
    E.g. seed1 for lambda1 = a, b, c; seed2 for lambda2 = d, e, f.
    - Run range test to find lr for each configuration / run.
    I.e. seed1 for lambda1 = a, b, c;
    but individual range test for lambda1 = a, lambda1 = b, and lambda1 = c.
    Do not transfer the lr for one configuration (e.g. lambda1 = a) to others (e.g. lambda1 = b).
    - Pick the best configuration on validation.
    - When affordable, run a lr sweeping for each

# Determining Batch Size

|

# References

1. [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook)
