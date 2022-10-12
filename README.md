### SBM-Transformer

1. Our experiment environment is listed in `requirements.txt`.

2. Download code from the [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena/` in `datasets/`
3. Download and unzip [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) into `datasets/`. The resulting directory structure should be
```
datasets/long-range-arena
datasets/lra_release
```
4. Run `datasets/create_datasets.sh` to create the train-, dev-, and test-split pickle files for each LRA task.
5. To run baseline methods on LRA, run `code/run_baselines.sh`
6. To run SBM-transformer on LRA, run `code/run_tasks.sh`