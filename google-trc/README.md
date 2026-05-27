Use Google TPU Research Cloud (TRC) for free to run machine learning experiments with PyTorch (without custom CUDA C++ codes).

My laptop system is Windows 11.

# Preparations

1. Create a Google account.
(I used my university email to login.)
2. Set up a payment method in the Google account.
(I set a Commonwealth Bank of Australia account.)
3. Install `gcloud`: [install-gcloud.ps1](install-gcloud.ps1)

# Application

Fill and submit the TRC application form in [1].
After that,
TRC will send a confirmation e-mail and tell you how to claim the Cloud TPU quota in 3 steps, each with a specific link:

1. create a Google Cloud project.
2. Turn on Cloud TPU API for it.
3. Send the project number to a form.

After submitting the form with project number,
TRC will send another e-mail with:
- project id of another project than the one created above,
and this new project has access to free TPU quota.
- quick start procedure [2].

# Next

- (10 May 2026) I don't know how to use yet.

# References

1. [TPU Research Cloud](https://sites.research.google/trc/about/)
2. [Set up a Google Cloud project for TPUs](https://docs.cloud.google.com/tpu/docs/setup-gcp-account)
