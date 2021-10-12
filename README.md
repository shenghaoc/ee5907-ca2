# EE5907 CA2
## Environment setup
Run
```
conda create -n ee5907-ca2 -c conda-forge tensorflow-gpu jupyterlab pillow matplotlib
conda activate ee5907-ca2
```
to create and activate a conda environment `ee5907-ca2` containing most of the dependencies.

Notes:

1. Replace `tensorflow-gpu` with `tensorflow` for CPU-only version.
1. The `conda-forge` channel is used instead of `default` because after adding `matplotlib`, the solution includes `tensorflow-gpu 2.3.0` and `matplotlib 3.4.2`. After changing to `conda-forge`, the solution includes `tensorflow-gpu 2.5.0` and `matplotlib 3.4.3`. See this [question](https://stackoverflow.com/q/65273118) on Stack Overflow for why `tensorflow-gpu 2.3.0` must not be used.
1. It takes a long time to solve environment, use [mamba](https://github.com/mamba-org/mamba) instead for better performance.

For LIBSVM, there is no official conda support, pip is necessary.

```
pip install -U libsvm-official
```
