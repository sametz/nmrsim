This folder will store conda environment.yml files tagged with nmrsim version numbers.
This is to support determining `nmrsim`'s allowed version ranges for its requirements.

Reminder: your current conda environment can be saved to an `environment.yml` file using:

```
conda env export > environment.yml
```

and a new conda environment can be created from an `environment.yml` file using:

```
conda env create -f environment.yml
```

Note that these were made using a developer install (`-e` option) of nmrsim.
For a non-dev environment, replace the nmrsim version in the .yml with the version in the fileneme.
