# EE5907 CA2
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Running the code
The Python code was written with Jupyter Notebook and all plots and images are saved to disk in `report/figures`. There is a link to open the notebook on Colab.

It is assumed that the code runs from the same directory where the unzipped `PIE` folder and the `resized` folder containing the processed selfies reside.

The code assumes that each subject has the same number of images, which was the case for the subjects I chose.

## Environment setup
There is a [requirements file](requirements.txt) but to avoid having to install anything outside of virtual environments conda should be used.

Run
```
conda create -n ee5907-ca2 -c conda-forge tensorflow-gpu pillow matplotlib scikit-learn
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
