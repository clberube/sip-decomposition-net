#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#


import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from utilities import denormalize

log_two_pi = torch.tensor(2 * math.pi).log()


def complexCardioid(inp):
    return ((1 + torch.cos(inp.angle())) * inp) / 2


class cCardioid(nn.Module):
    @staticmethod
    def forward(inp):
        return complexCardioid(inp)


class cLinear(nn.Module):
    """Modified from
    https://github.com/wavefrontshaping/complexPyTorch/blob/master/complexPyTorch/complexLayers.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.cfloat)
        )
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias
            )
            nn.init.xavier_uniform_(self.bias)

        else:
            self.bias = None

    def forward(self, inp):
        if not inp.dtype == torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return F.linear(inp, self.weight, self.bias)


def v_delta_from_std(sigma_r, sigma_i, rho=None, cov_ri=None):
    """
    Compute empirical complex variance v and pseudo-variance delta
    from real/imag std-deviations and their correlation or covariance.

    sigma_r, sigma_i : real-valued std-devs
    rho              : correlation coefficient (optional)
    cov_ri           : covariance between real and imag parts (optional)

    returns v (real), delta (complex)
    """
    if cov_ri is None:
        if rho is None:
            rho = 0.0  # assume uncorrelated if not provided
        cov_ri = rho * sigma_r * sigma_i

    var_r = sigma_r**2
    var_i = sigma_i**2
    v = var_r + var_i
    delta = torch.complex(var_r - var_i, 2.0 * cov_ri)
    return v, delta


class CVAE(nn.Module):

    def __init__(
        self,
        input_dim,
        num_hidden=3,
        hidden_dim=128,
        latent_dim=2,
        cond_dim=8,
        label_dim=2,
        mixture_dim=32,
        quadrature_dim=128,
        activation=nn.Tanh(),
        frequencies=None,
    ):

        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.mixture_dim = mixture_dim
        self.quadrature_dim = quadrature_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.label_dim = label_dim
        self.activation = activation
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim

        if frequencies is not None:
            self.frequencies = frequencies

        self.R_mixture = mixture_dim  # number of Gaussians
        self.J_quad = quadrature_dim  # quadrature points

        # Precompute quadrature grid in log-tau
        self.w = 2 * torch.pi * self.frequencies.squeeze()

        # Consistent natural-log bounds for μ
        tau_min_log = (1 / self.w).log10().min().floor() - 2
        tau_max_log = (1 / self.w).log10().max().ceil() + 2

        self.u_min = tau_min_log / torch.log10(torch.tensor(torch.e))
        self.u_max = tau_max_log / torch.log10(torch.tensor(torch.e))

        J = self.J_quad

        u_grid = torch.linspace(self.u_min, self.u_max, J)
        du = (self.u_max - self.u_min) / (J - 1)
        quad_w = torch.ones(J) * du

        # Precompute Debye kernel
        kernel = []
        tau_grid = torch.exp(u_grid)
        for w_k in self.w:
            c = 1.0  # or user-defined
            kernel.append(1 / (1 + (1j * w_k * tau_grid) ** c))
        kernel = torch.stack(kernel, dim=0)

        self.register_buffer("u_grid", u_grid)
        self.register_buffer("quad_w", quad_w)
        self.register_buffer("kernel", kernel)

        self.encoder_layers = nn.ModuleList(
            [cLinear(input_dim, hidden_dim)]
            + [cLinear(hidden_dim, hidden_dim) for _ in range(num_hidden)]
        )

        self.mu_logvar_layers = nn.ModuleList(
            [
                cLinear(hidden_dim, latent_dim),
                cLinear(hidden_dim, latent_dim),
            ]
        )

        self.mixture_head = nn.ModuleList(
            [
                nn.Linear(latent_dim, 1),
                nn.Linear(latent_dim, 1),
                nn.Linear(latent_dim, 1),
                nn.Linear(latent_dim, mixture_dim),
                nn.Linear(latent_dim, mixture_dim),
                nn.Linear(latent_dim, mixture_dim),
            ]
        )

    def kld_real_diag(self, mu, logvar):
        # mu, logvar: [B, H] (réels)
        # KLD par échantillon = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        # forme équivalente au signe près :
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1)  # [B]

    def reconstruction_loss(self, x_hat, x, sigma=None):
        """
        Reconstruction loss: negative log-likelihood under complex Gaussian.
        """
        # Estimate per-feature std from data (optional)
        v_emp, delta_emp = v_delta_from_std(sigma.real, sigma.imag)
        rec = self.complex_gaussian_nll_adaptive(
            x_hat,
            x,
            v_emp,
            delta_emp,
        )
        return rec

    def vae_loss(self, x_hat, x, xerr, mu, logvar, beta=1.0):
        """
        Computes full complex-valued VAE loss (Nakashika et al. 2020).
        """
        # Reconstruction
        rec = self.reconstruction_loss(x_hat, x, xerr)

        kl = self.kld_real_diag(mu, logvar)
        kl = kl.mean()

        total_loss = rec + beta * kl
        return total_loss, {"rec": rec.item(), "kld": kl.item()}

    def complex_gaussian_nll_adaptive(
        self, x_hat, x, v_emp=None, delta_emp=None, eps=1e-12
    ):
        """
        Complex Gaussian NLL. Utilise les incertitudes fournies si disponibles,
        sinon les estime à partir des résidus complexes.

        Paramètres
        ----------
        x_hat : torch.cfloat [..., D]
            Moyenne prédite (complexe)
        x : torch.cfloat [..., D]
            Données mesurées (complexes)
        v_emp : torch.float [1, D] ou scalaire, optionnel
            Variance empirique (σ_r² + σ_i²)
        delta_emp : torch.cfloat [1, D] ou scalaire, optionnel
            Pseudo-variance empirique ((σ_r²−σ_i²)+2iρσ_rσ_i)
        eps : float
            Terme de stabilité numérique
        """

        # --- 1) Résidus complexes
        e = x - x_hat
        abs_e2 = e.real**2 + e.imag**2  # |e|²
        e2 = e * e  # e²

        # --- 2) Si les incertitudes ne sont pas fournies,
        # on les estime à partir des résidus
        if v_emp is None or delta_emp is None:
            # moyenne sur le batch
            v_est = torch.mean(abs_e2, dim=0, keepdim=True) + eps
            delta_est = torch.mean(e2, dim=0, keepdim=True)
            v = v_est.real.float()
            delta = delta_est
        else:
            v = v_emp.real.float()
            delta = delta_emp

        # --- 3) Calcul du dénominateur (positive definite)
        denom = v.pow(2) - (delta.real.pow(2) + delta.imag.pow(2))
        denom = torch.clamp(denom, min=1e-12)

        # --- 4) Terme quadratique : v|e|² − Re{δ e²}
        quad = (v * abs_e2) - torch.real(delta.conj() * e2)

        # --- 5) NLL complet
        nll = (quad / denom) + 0.5 * torch.log(denom) + math.log(math.pi)

        # --- 6) Moyenne (batch et features)
        return torch.mean(nll)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mu, logvar = (p(x) for p in self.mu_logvar_layers)
        return mu.real + mu.imag, logvar.real + logvar.imag

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, raw=False):
        """
        Continuous Debye in conductivity with RTD normalized by rho_0.
        Implements:
            - ρ₀ predicted directly
            - σ₀ = 1 / ρ₀
            - g(u) = σ₀ * m0 * φ(u)
            - σ*(ω) = σ₀ - ∫ g(u) K(ω,u) du
            - ρ*(ω) = 1 / σ*(ω)
            - z_out = ρ*(ω) / ρ₀   (same normalization structure)
        """

        # -----------------------------------------------------
        # 1. Network outputs
        # -----------------------------------------------------
        B = z.shape[0]
        R = self.R_mixture
        J = self.J_quad
        r_raw, m0_raw, eta_raw, pi_raw, mu_raw, logvar_raw = (
            p(z) for p in self.mixture_head
        )

        # -----------------------------------------------------
        # 2. Physical parameters
        # -----------------------------------------------------

        # ---- ρ0 (positive) ----
        r = denormalize(torch.tanh(r_raw), 0.9, 1.1, -1, 1)
        r_e = r.view(B, 1)

        # ---- dimensionless chargeability m0 in [0,1] ----
        m0 = torch.sigmoid(m0_raw)  # (B,)

        # eta = epsilon_inf * rho_ref, positive and bounded in log space
        # These bounds work slightly better than unbounded exponentiation
        eta_min = torch.as_tensor(1e-8, device=z.device, dtype=z.dtype)
        eta_max = torch.as_tensor(1e-3, device=z.device, dtype=z.dtype)

        log_eta = torch.log(eta_min) + torch.sigmoid(eta_raw) * (
            torch.log(eta_max) - torch.log(eta_min)
        )
        eta = torch.exp(log_eta).view(B, 1)  # (B,1)

        # ---- mixture weights ----
        pi = torch.nn.functional.softmax(pi_raw, dim=-1)  # (B,R)

        # ---- mixture centers in log(tau) ----
        mu = denormalize(torch.tanh(mu_raw), self.u_min, self.u_max, -1, 1)  # (B,R)
        # ---- mixture widths >0 ----
        sigma = torch.exp(0.5 * logvar_raw)

        # -----------------------------------------------------
        # 3. Evaluate φ(u) on quadrature grid
        # -----------------------------------------------------
        u = self.u_grid.view(1, 1, J)  # (1,1,J)
        mu_e = mu.unsqueeze(-1)  # (B,R,1)
        sig_e = sigma.unsqueeze(-1)  # (B,R,1)
        pi_e = pi.unsqueeze(-1)  # (B,R,1)

        gauss = torch.exp(-0.5 * ((u - mu_e) / sig_e) ** 2)  # (B,R,J)
        phi_raw = (pi_e * gauss).sum(dim=1)  # (B,J)

        # Normalisation correcte avec les poids de quadrature
        quad_w = self.quad_w.view(1, J)  # (1,J)
        Z = (phi_raw * quad_w).sum(dim=-1, keepdim=True)  # (B,1)
        phi_u = phi_raw / (Z + 1e-12)  # (B,J)

        # Dimensionless conductivity-domain RTD: integral equals m0
        m0_e = m0.view(B, 1)
        g_u = m0_e * phi_u
        g_quad = g_u * quad_w

        # Relaxation contribution:
        # sigma_relax = m0 * integral phi(u) lambda(omega, exp(u)) du
        sigma_relax = g_quad.to(self.kernel.dtype) @ self.kernel.T

        # -----------------------------------------------------
        # 5. Conductivity-domain forward model
        # -----------------------------------------------------
        w = self.w.view(1, -1).to(self.kernel.dtype)  # (1,n_freq)
        m0_c = m0_e.to(self.kernel.dtype)
        r_c = r_e.to(self.kernel.dtype)
        eta_c = eta.to(self.kernel.dtype)

        # Conductivity-domain forward model, normalized by sigma0
        sigma_star_norm_0 = (1.0 - sigma_relax) / (1.0 - m0_c) + 1j * w * eta_c * r_c

        # -----------------------------------------------------
        # 6. Conductivity to resistivity
        # -----------------------------------------------------
        rho_star_norm_0 = 1.0 / sigma_star_norm_0  # rho*(omega) / rho0

        # -----------------------------------------------------
        # 7. Output normalized by rho_ref
        # -----------------------------------------------------
        z_out = r_c * rho_star_norm_0

        # -----------------------------------------------------
        # 8. RAW return
        # -----------------------------------------------------
        if raw:
            return z_out, (r_raw, m0_raw, pi_raw, mu_raw, logvar_raw, eta_raw), ()

        # -------------------------------------------------
        # RTD Debye discrète fine sur la grille u_grid
        # -------------------------------------------------
        tau_j = torch.exp(self.u_grid).repeat(B, 1)  # (B,J)
        m_j = g_quad  # (B,J)

        return z_out, (r, m0, pi, mu, sigma, eta), (tau_j, m_j)

    def forward(self, x, raw=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xp, param, rtd = self.decode(z, raw=raw)
        return xp, mu, logvar, param, rtd
