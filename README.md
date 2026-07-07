# SIP Decomposition Net

A complex-valued variational autoencoder (CVAE) for decomposing spectral
induced polarization (SIP) data into relaxation time distributions (RTD).
The decoder is a RTD-based complex conductivity model. The encoder 
maps a measured complex resistivity spectrum to a stochastic latent representation.

The repository includes the dataset and trained weights used in the paper,
so pretrained inference can be run immediately after installing the three
Python dependencies.

## Contents

- [Quick start](#quick-start)
- [What the model does](#what-the-model-does)
- [Running inference](#running-inference)
- [Training](#training)
- [Inputs and outputs](#inputs-and-outputs)
- [Using the model from Python](#using-the-model-from-python)
- [Repository layout](#repository-layout)
- [Reproducibility and limitations](#reproducibility-and-limitations)

## Quick start

Python 3.10 or newer is recommended. The current code has been tested with
Python 3.13, PyTorch 2.7, NumPy 2.2, and Matplotlib 3.10.

```bash
git clone <repository-url>
cd sip-decomposition-net

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib

python train_model.py
```

The default configuration loads `weights/best_weights.pt`, performs 100
stochastic predictions for every spectrum, and opens a two-panel fit figure
for four representative samples. The full example runs on CPU.


## What the model does

The encoder accepts a normalized complex resistivity spectrum and returns the
mean and log-variance of a real-valued latent Gaussian. A sampled latent vector
is decoded into:

- a normalized reference resistivity, `rho0`;
- total dimensionless chargeability, `m0`;
- a 128-component Gaussian mixture in natural-log relaxation time;
- a bounded high-frequency permittivity term, `epsilon`;
- a reconstructed complex spectrum.

The Gaussian mixture is evaluated on a 100-point quadrature grid. Its
normalized RTD is integrated through the Warburg or Debye kernels 
and the resulting conductivity is converted back to normalized complex
resistivity. See `CVAE.decode()` in `models.py` for the complete forward model.

The training objective combines a complex Gaussian negative log-likelihood,
using the supplied real and imaginary measurement uncertainties, with the
usual diagonal Gaussian latent KL divergence.

## Running inference

The main workflow is configured by constants near the top of
`train_model.py`:

| Setting | Default | Effect |
| --- | ---: | --- |
| `TRAIN_MODEL` | `False` | Train before inference |
| `LOAD_WEIGHTS` | `True` | Load `weights/best_weights.pt` |
| `SAVE_WEIGHTS` | `False` | Save the trained state dictionary |
| `SAVE_FIGURES` | `False` | Write PDFs to `figures/` instead of showing them |
| `n_epoch` | `100000` | Maximum training epochs |

For ordinary pretrained inference, leave the defaults unchanged and run:

```bash
python train_model.py
```

To change Monte Carlo sample count, edit `n_reps` in the call to `predict()`.
Larger values give smoother empirical intervals at proportionally higher
runtime and memory cost.

The example figure uses dataset indices `9`, `34`, `78`, and `93` and
apparatus-specific geometrical factors. Change both lists together when
plotting other samples. `plot_fit()` currently formats exactly four named
examples; callers wanting an arbitrary number of samples should adapt its
labels, markers, and z-order arrays.

## Training

To train a new model, edit these settings in `train_model.py`:

```python
TRAIN_MODEL = True
SAVE_WEIGHTS = True
LOAD_WEIGHTS = False
```

Then run:

```bash
python train_model.py
```

Training uses the complete 140-spectrum dataset as a single batch, AdamW with
a learning rate of `1e-3`, and CPU execution. Early stopping compares the mean
loss in two adjacent windows, each 1% of the configured maximum epoch count,
and stops when relative improvement falls below 1%.

When `SAVE_FIGURES = True`, training writes:

- `figures/LC.pdf` — negative log-likelihood and KL learning curves;
- `figures/FIT-examples.pdf` — measured and reconstructed spectra.

When `SAVE_WEIGHTS = True`, the final in-memory model state is written to
`weights/best_weights.pt`. Despite the filename, the training loop does not
currently checkpoint the lowest validation loss; there is no separate
validation split in this repository.

## Inputs and outputs

### Bundled dataset

`data/data_dict.pt` is a PyTorch-serialized dictionary. 

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `sample_series` | list of 11 integers | Available experimental series |
| `sample_numbers` | list of 11 lists | Sample numbers grouped by series |
| `flat_sample_list` | list of 140 tuples | `(series, sample)` lookup by row |
| `R0_all` | `(140,)`, `float32` | Per-sample reference resistance |
| `data_all` | `(140, 19)`, `complex64` | Normalized complex spectra |
| `freq_all` | `(140, 19)`, `complex64` | Log-transformed frequency grid; code uses `real.exp()` |
| `err_all` | `(140, 19)`, `complex64` | Real/imaginary standard deviations |

All samples currently share the same frequency grid. The model architecture
and pretrained weights expect exactly 19 complex input values.

### Prediction dictionary

`utilities.predict(model, dataloader, n_reps)` returns NumPy arrays. With `N`
samples, `M = n_reps`, latent dimension `L`, mixture size `R`, quadrature size
`J`, and frequency count `F`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `Z_pred` | `(N, M, F)` | Reconstructed complex spectra |
| `mu_pred` | `(N, M, L)` | Encoder means (repeated across draws) |
| `logvar_pred` | `(N, M, L)` | Encoder log-variances (repeated across draws) |
| `rho0_pred` | `(N, M, 1)` | Normalized reference resistivity |
| `m0_pred` | `(N, M, 1)` | Total chargeability |
| `epsilon_pred` | `(N, M, 1)` | Permittivity-related decoder output (`epsilon`) |
| `pi_pred` | `(N, M, R)` | Gaussian-mixture weights |
| `mu_rtd_pred` | `(N, M, R)` | Mixture centers in `ln(tau)` |
| `sigma_rtd_pred` | `(N, M, R)` | Mixture widths in `ln(tau)` |
| `tau_pred` | `(N, M, J)` | Relaxation-time quadrature grid |
| `m_pred` | `(N, M, J)` | Chargeability contribution per grid cell |

For the bundled architecture, `F=19`, `L=6`, `R=128`, and `J=100`.

## Using the model from Python

The architecture must match the saved state dictionary. This minimal example
loads the bundled data and weights and predicts ten stochastic realizations:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import CVAE, cCardioid
from utilities import predict

data = torch.load("data/data_dict.pt", map_location="cpu")
dataset = TensorDataset(data["data_all"], data["err_all"])
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

model = CVAE(
    input_dim=19,
    cond_dim=19,
    label_dim=0,
    num_hidden=2,
    hidden_dim=32,
    latent_dim=6,
    mixture_dim=128,
    quadrature_dim=100,
    activation=cCardioid(),
    frequencies=data["freq_all"][:1].real.exp(),
)
model.load_state_dict(
    torch.load("weights/best_weights.pt", map_location="cpu", weights_only=True)
)
model.eval()

results = predict(model, loader, n_reps=10)
mean_spectrum = results["Z_pred"].mean(axis=1)
```

For new data, preserve the training normalization, frequency ordering, tensor
shape, and `torch.complex64` dtype. Pair every spectrum with a complex
uncertainty tensor of the same shape: its real component is the standard
deviation of the real data and its imaginary component is the standard
deviation of the imaginary data. A different frequency grid changes the
decoder kernel and RTD bounds and therefore requires retraining.

## Repository layout

```text
.
├── data/data_dict.pt         Bundled SIP dataset
├── weights/best_weights.pt   Pretrained model state dictionary
├── train_model.py            End-to-end configuration and workflow
├── models.py                 Complex layers, CVAE, and physical decoder
├── utilities.py              Training, prediction, and numeric helpers
├── plotlib.py                Learning-curve and spectral-fit plots
├── LICENSE                   MIT license
└── README.md                 Project documentation
```

## Reproducibility and limitations

- `train_model.py` seeds PyTorch and NumPy with `11` and forces CPU execution.
- Inference is stochastic. Re-running the entire script with the same software
  stack and seed should reproduce its draws; calling `predict()` repeatedly in
  one process advances the random-number generator.
- The repository does not provide a pinned dependency file or automated test
  suite. Exact results may differ across PyTorch versions and platforms.
- The bundled weights are coupled to the architecture constants and the
  19-frequency grid in `train_model.py`.
- The plotting code covers the fit and learning-curve figures used by this
  workflow; it does not reproduce every manuscript figure, although all figures 
  can be reproduced using the results returned by `predict()`.

## License

Released under the [MIT License](LICENSE).
