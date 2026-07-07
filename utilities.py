#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import math
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def to_latex_scientific_notation(mean, std, maxint=2):
    exponent_mean = int(np.floor(np.log10(abs(mean))))
    exponent_std = int(np.floor(np.log10(abs(std))))
    precision = abs(exponent_mean - exponent_std)
    coefficient_mean = round(mean / 10**exponent_mean, precision)
    coefficient_std = round(std / 10**exponent_mean, precision)
    if -maxint <= exponent_mean <= 0:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    elif 0 <= exponent_mean <= maxint and exponent_std >= 0:
        return f"${round(truncate(mean, -exponent_std))} \\pm {round(truncate(std, -exponent_std))}$"
    elif 0 <= exponent_mean <= maxint:
        return f"${truncate(mean, -exponent_std)} \\pm {truncate(std, -exponent_std)}$"
    else:
        if precision == 0:
            return (
                f"$({round(coefficient_mean)} \\pm {round(coefficient_std)}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{round(coefficient_mean)} \\pm {round(coefficient_std)}$"
            )
        else:
            return (
                f"$({coefficient_mean} \\pm {coefficient_std}) \\cdot 10^{{{exponent_mean}}}$"
                if exponent_mean != 0
                else f"{coefficient_mean} \\pm {coefficient_std}$"
            )


def str_with_err(value, error):
    if error > 0:
        digits = -int(math.floor(math.log10(error)))
    else:
        digits = 0
    if digits < 0:
        digits = 0
    err10digits = math.floor(error * 10**digits)
    return r"${0:.{2}f} \pm {1:.{2}f}$".format(value, error, digits)


def restore_minor_ticks_log_plot(ax, n_subticks=9, axis="both"):
    """For axes with a logrithmic scale where the span (max-min) exceeds
    10 orders of magnitude, matplotlib will not set logarithmic minor ticks.
    If you don't like this, call this function to restore minor ticks.

    All credit to Stack Overflow user importanceofbeingernest at
    https://stackoverflow.com/a/44079725/5972175

    Args:
        ax:
        n_subticks: Number of Should be either 4 or 9.

    Returns:
        None
    """
    if ax is None:
        ax = plt.gca()

    locmaj = mpl.ticker.LogLocator(base=10, numticks=1000)
    locmin = mpl.ticker.LogLocator(
        base=10.0, subs=np.linspace(0, 1.0, n_subticks + 2)[1:-1], numticks=1000
    )

    if axis == "x" or axis == "both":
        ax.xaxis.set_major_locator(locmaj)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if axis == "y" or axis == "both":
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def normalize(x, xmin, xmax, ymin, ymax):
    # x mapped from xmin, xmax to ymin, ymax
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def denormalize(x, xmin, xmax, ymin, ymax):
    # x mapped from ymin, ymax back to xmin, xmax
    return (x - ymin) * (xmax - xmin) / (ymax - ymin) + xmin


def softclip(tensor, min_val):
    return min_val + F.softplus(tensor - min_val)


def train(
    model,
    train_loader,
    verbose,
    lr,
    n_epoch,
    device=None,
    early_stopping=False,
):
    """My usual PyTorch training loop adapted for Complex Debye Net"""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_losses = ["NLL", "KLD", "train"]
    valid_losses = ["criterion"]

    history = {k: np.zeros(n_epoch) for k in train_losses}
    history.update({k: np.zeros(n_epoch) for k in valid_losses})

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = timer()
    model.to(device)

    for e in range(n_epoch):
        running_loss = {k: 0 for k in train_losses}  # reset running losses
        model.train()

        for X, dX in train_loader:

            X = X.to(device)  # normalized data
            dX = dX.to(device)  # uncertainty of normalized data

            optimizer.zero_grad()

            # Forward pass
            Xp, mu, logvar, *_ = model(X)

            # Compute loss
            total_loss, losses = model.vae_loss(
                Xp,
                X,
                dX,
                mu,
                logvar,
                beta=1.0,
            )

            running_loss["NLL"] += losses["rec"]
            running_loss["KLD"] += losses["kld"]
            running_loss["train"] += total_loss.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

        for k in train_losses:
            history[k][e] = running_loss[k] / len(train_loader.sampler)

        if early_stopping:
            # Only evaluate once we have at least 1% of max_iters recorded
            window_threshold = 0.01
            window = int(window_threshold * n_epoch)
            improv_threshold = 0.01
            # Need at least two complete windows to compare
            if e >= 2 * window:

                past = torch.tensor(history["train"][e - 2 * window : e - window])
                recent = torch.tensor(history["train"][e - window : e])

                past_mean = past.mean().item()
                recent_mean = recent.mean().item()

                # Relative improvement in loss (loss decreases = improvement > 0)
                if past_mean != 0:
                    rel_improv = (past_mean - recent_mean) / abs(past_mean)
                else:
                    rel_improv = 0.0

                history["criterion"][e] = rel_improv

                if rel_improv < improv_threshold:
                    if verbose:
                        print(
                            f"Stopping criterion reached (mean loss improved < {improv_threshold:.1%} over last {window_threshold:.1%} of epochs)."
                        )
                    break

        verbose_str = (
            f"Epoch: {(e+1):.0f}, "
            f"NLL: {history['NLL'][e]:.1e}, "
            f"KLD: {history['KLD'][e]:.1e}, "
            f"Train: {history['train'][e]:.1e}, "
            f"Criterion: {history['criterion'][e]:.1e}, "
        )

        if verbose:
            if (e + 1) % verbose == 0:
                print(verbose_str)

    end_time = timer()

    if verbose:
        print(f"Training time: {(end_time - start_time):.2f} s")

    return history


def predict(model, dataloader, n_reps=100):

    model.eval()

    all_results = {
        "Z_pred": [],  # (N, n_reps, n_freq)
        "mu_pred": [],  # encoder mean
        "logvar_pred": [],  # encoder logvar
        "rho0_pred": [],  # (N, n_reps)
        "m0_pred": [],  # (N, n_reps)
        "epsilon_pred": [],  # (N, n_reps)
        "pi_pred": [],  # (N, n_reps, R)
        "mu_rtd_pred": [],  # (N, n_reps, R)
        "sigma_rtd_pred": [],  # (N, n_reps, R)
        "tau_pred": [],  # (N, n_reps, J) fine grid
        "m_pred": [],  # (N, n_reps, J) fine grid
    }

    with torch.no_grad():
        for data_batch, _ in dataloader:

            n_samples = data_batch.shape[0]

            # storage lists for this batch
            Z_storage = []
            mu_storage = []
            logvar_storage = []
            rho0_storage = []
            m0_storage = []
            pi_storage = []
            mu_rtd_storage = []
            sigma_rtd_storage = []
            tau_storage = []
            m_storage = []
            epsilon_storage = []

            # -------------------------------------------
            # sample n_reps times for Monte Carlo inference
            # -------------------------------------------
            for _ in range(n_reps):

                xp, mu_enc, logvar_enc, param, rtd = model(data_batch)

                # unpack decoder outputs
                rho0, m0, pi, mu_rtd, sigma_rtd, epsilon = param
                tau_j, m_j = rtd

                # append to storage
                Z_storage.append(xp.cpu().numpy())
                mu_storage.append(mu_enc.cpu().numpy())
                logvar_storage.append(logvar_enc.cpu().numpy())

                rho0_storage.append(rho0.cpu().numpy())
                m0_storage.append(m0.cpu().numpy())
                epsilon_storage.append(epsilon.cpu().numpy())

                pi_storage.append(pi.cpu().numpy())
                mu_rtd_storage.append(mu_rtd.cpu().numpy())
                sigma_rtd_storage.append(sigma_rtd.cpu().numpy())

                tau_storage.append(tau_j.cpu().numpy())
                m_storage.append(m_j.cpu().numpy())

            # -------------------------------------------
            # stack along repetition axis
            # shapes → (n_samples, n_reps, ...)
            # -------------------------------------------
            all_results["Z_pred"].append(np.stack(Z_storage, axis=1))
            all_results["mu_pred"].append(np.stack(mu_storage, axis=1))
            all_results["logvar_pred"].append(np.stack(logvar_storage, axis=1))

            all_results["rho0_pred"].append(np.stack(rho0_storage, axis=1))
            all_results["m0_pred"].append(np.stack(m0_storage, axis=1))
            all_results["epsilon_pred"].append(np.stack(epsilon_storage, axis=1))

            all_results["pi_pred"].append(np.stack(pi_storage, axis=1))
            all_results["mu_rtd_pred"].append(np.stack(mu_rtd_storage, axis=1))
            all_results["sigma_rtd_pred"].append(np.stack(sigma_rtd_storage, axis=1))

            all_results["tau_pred"].append(np.stack(tau_storage, axis=1))
            all_results["m_pred"].append(np.stack(m_storage, axis=1))

    # -------------------------------------------
    # concatenate across batches
    # -------------------------------------------
    for k in all_results:
        all_results[k] = np.concatenate(all_results[k], axis=0)

    return all_results
