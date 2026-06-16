# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_model.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: clberube <charles.berube@polymtl.ca>       +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/05/13 13:21:42 by clberube          #+#    #+#              #
#    Updated: 2026/06/16 16:31:47 by clberube         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import CVAE, cCardioid
from utilities import train, predict
from plotlib import plot_learning_curves, plot_fit

# For reproducibility
RANDOM_SEED = 11
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# There is no real advantage in using a GPU for this neural network
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # please use CPU for now

# Some user-defined flags for saving and loading results
TRAIN_MODEL = False
SAVE_WEIGHTS = False
LOAD_WEIGHTS = True
SAVE_FIGURES = False
SAVE_RESULTS = False

# Number of maximum training epochs (assuming no early stopping)
n_epoch = int(1e5)

# User-defined directories
wt_dir = Path("./weights")
fig_dir = Path("./figures") if SAVE_FIGURES else None
res_dir = Path("./results") if SAVE_RESULTS else None
data_dir = Path("./data")

os.makedirs(wt_dir, exist_ok=True) if (SAVE_WEIGHTS or LOAD_WEIGHTS) else None
os.makedirs(fig_dir, exist_ok=True) if SAVE_FIGURES else None
os.makedirs(res_dir, exist_ok=True) if SAVE_RESULTS else None

data_dict = torch.load(data_dir / "data_dict.pt")

# Useful variables that may be used later for processing results
sample_series = data_dict["sample_series"]
sample_numbers = data_dict["sample_numbers"]
flat_sample_list = data_dict["flat_sample_list"]
R0_all = data_dict["R0_all"]
data_all = data_dict["data_all"]
freq_all = data_dict["freq_all"]
err_all = data_dict["err_all"]

# Check imported data
print(f"Loaded {len(flat_sample_list)} samples with shape {data_all.shape}")

# Define the batch size (full dataset)
batch_size = data_all.shape[0]

# Define the PyTorch DataLoader
dataset = TensorDataset(data_all, err_all)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Model architecture is defined here using results of the model selection experiments
net_param = {
    "input_dim": dataset[:][0].shape[-1],  # input dimensions (number of frequencies)
    "cond_dim": dataset[:][1].shape[-1],  # condition dimensions (number of frequencies)
    "label_dim": 0,  # unused artefact from old code, to be removed
    "num_hidden": 2,  # number of hidden layers in the encoder
    "hidden_dim": 32,  # dimension of hidden layers in the encoder
    "latent_dim": 6,  # latent space dimension
    "mixture_dim": 128,  # number of Gaussian mixture components
    "quadrature_dim": 100,  # number of quadrature points to evaluate the integral
    "activation": cCardioid(),  # activation function
    "frequencies": freq_all[:1].real.exp(),  # used to determine the RTD bounds in init
}

# Initialize the model
model = CVAE(**net_param)
model.to(device)

# Calls the training loop with user-defined parameters
if TRAIN_MODEL:
    losses = train(
        model,
        dataloader,
        device=device,
        verbose=(n_epoch // 10),
        lr=1e-3,
        n_epoch=n_epoch,
        early_stopping=True,
    )


if TRAIN_MODEL:
    plot_learning_curves(losses, save=fig_dir)


if TRAIN_MODEL and SAVE_WEIGHTS:
    torch.save(model.state_dict(), wt_dir / "best_weights.pt")


if LOAD_WEIGHTS:
    model.load_state_dict(torch.load(wt_dir / "best_weights.pt", weights_only=True))
    model.eval()


# Predicts results using the trained model for all data in the dataloader
all_results = predict(
    model,
    dataloader,
    n_reps=100,  # number of stochastic realizations per predictions
)

# Add data in the results dictionary
all_results["frequencies"] = (
    freq_all[0].real.exp().unsqueeze(0).unsqueeze(0).cpu().numpy()
)  # shared frequency axis

all_results["R0"] = R0_all.unsqueeze(-1).unsqueeze(-1).numpy()

all_results["data_all"] = data_all.cpu().numpy()
all_results["err_all"] = err_all.cpu().numpy()


plot_fit(
    all_results,
    samples=[9, 34, 78, 93],
    geom_factors=[
        0.00647,
        0.00647,
        4 * np.pi,
        0.075,
    ],  # geometrical factors (specific to measurement apparatus),
    save=fig_dir,
)
