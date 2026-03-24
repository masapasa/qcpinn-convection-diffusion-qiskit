from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim.lr_scheduler import CosineAnnealingLR


# =========================================================
# 1) Configuration
# =========================================================


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Architecture
    use_quantum: bool = True
    n_qubits: int = 6              # 14-16 is possible to request, but costly in practice
    n_q_layers: int = 2
    hidden_dim: int = 128
    trunk_layers: int = 4
    fourier_features: int = 32
    dropout: float = 0.0

    # Targets
    predict_phase: bool = False    # Turn on only if you have a real phase field target + PDE
    t_melt: float = 1687.0

    # Physics constants (Silicon melt placeholders)
    nu: float = 1.0e-6             # Kinematic viscosity
    alpha: float = 1.0e-5          # Thermal diffusivity
    rho: float = 2330.0            # Density
    beta_T: float = 0.0            # Thermal expansion coefficient if buoyancy is added later
    g_z: float = 0.0               # Axial gravity term if buoyancy is added later

    # Training
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-6
    epochs: int = 3000
    batch_data: int = 2048
    batch_collocation: int = 4096
    grad_clip: float = 1.0
    scheduler_tmax: int = 3000
    use_amp: bool = False          # keep False when quantum branch is active
    checkpoint_path: str = "best_si_q_pinn.pt"
    print_every: int = 50

    # Loss coefficients (starting priors; adaptive weights refine them)
    initial_loss_terms: Tuple[str, ...] = (
        "data",
        "continuity",
        "mom_r",
        "mom_theta",
        "mom_z",
        "energy",
    )

    # Data parsing
    coord_cols: Tuple[str, str] = ("r", "z")
    target_cols: Tuple[str, ...] = ("u", "v", "w", "p", "T")


# =========================================================
# 2) Utilities
# =========================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


@dataclass
class Scaler:
    x_min: torch.Tensor
    x_max: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @property
    def x_scale(self) -> torch.Tensor:
        # x_norm = 2 * (x - xmin) / (xmax - xmin) - 1
        # dx_norm / dx = 2 / (xmax - xmin)
        return 2.0 / (self.x_max - self.x_min).clamp_min(1e-12)

    def transform_x(self, x_phys: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x_phys - self.x_min) / (self.x_max - self.x_min).clamp_min(1e-12) - 1.0

    def inverse_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x_norm + 1.0) * (self.x_max - self.x_min) + self.x_min

    def transform_y(self, y_phys: torch.Tensor) -> torch.Tensor:
        return (y_phys - self.y_mean) / self.y_std

    def inverse_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std + self.y_mean


class AdaptiveMultiLoss(nn.Module):
    """
    Homoscedastic uncertainty-style weighting.
    Effective weight per term is exp(-log_var).
    """

    def __init__(self, names: Sequence[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in names}
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for name, loss in losses.items():
            lv = self.log_vars[name]
            total = total + torch.exp(-lv) * loss + lv
        return total

    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        return {name: float(torch.exp(-p).item()) for name, p in self.log_vars.items()}


# =========================================================
# 3) Data loading
# =========================================================


def _infer_dataframe_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Supports:
      - 7 columns:  r, z, u, v, w, p, T
      - 9 columns:  r_raw, z_raw, r, z, u, v, w, p, T
    """
    n = df.shape[1]
    if n == 7:
        df.columns = ["r", "z", "u", "v", "w", "p", "T"]
    elif n == 9:
        df.columns = ["r_raw", "z_raw", "r", "z", "u", "v", "w", "p", "T"]
    elif n > 9:
        # Keep the last 9 if extra bookkeeping columns exist.
        df = df.iloc[:, -9:].copy()
        df.columns = ["r_raw", "z_raw", "r", "z", "u", "v", "w", "p", "T"]
    else:
        raise ValueError(
            f"Unsupported column count {n}. Expected 7 or 9 columns for CZ data."
        )

    cols = list(cfg.coord_cols) + list(cfg.target_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


@dataclass
class DatasetBundle:
    x_phys: torch.Tensor
    x_norm: torch.Tensor
    y_phys: torch.Tensor
    y_norm: torch.Tensor
    scaler: Scaler
    boundary_masks: Dict[str, torch.Tensor] = field(default_factory=dict)



def build_scaler(x_phys: torch.Tensor, y_phys: torch.Tensor) -> Scaler:
    x_min = x_phys.min(dim=0).values
    x_max = x_phys.max(dim=0).values
    y_mean = y_phys.mean(dim=0)
    y_std = y_phys.std(dim=0).clamp_min(1e-6)
    return Scaler(x_min=x_min, x_max=x_max, y_mean=y_mean, y_std=y_std)



def infer_boundary_masks(x_phys: torch.Tensor) -> Dict[str, torch.Tensor]:
    r = x_phys[:, 0]
    z = x_phys[:, 1]
    r_min, r_max = r.min(), r.max()
    z_min, z_max = z.min(), z.max()
    rt = 1e-3 * (r_max - r_min).clamp_min(1e-12)
    zt = 1e-3 * (z_max - z_min).clamp_min(1e-12)
    return {
        "axis": (r - r_min).abs() <= rt,
        "outer_wall": (r - r_max).abs() <= rt,
        "inlet": (z - z_min).abs() <= zt,
        "outlet": (z - z_max).abs() <= zt,
    }



def load_cz_data(file_path: str | Path, cfg: Config) -> DatasetBundle:
    df = pd.read_csv(file_path, sep=r"\s+", comment="%", header=None)
    df = _infer_dataframe_columns(df, cfg)

    x_phys = torch.tensor(df[list(cfg.coord_cols)].values, dtype=cfg.dtype)
    y_phys = torch.tensor(df[list(cfg.target_cols)].values, dtype=cfg.dtype)

    if cfg.predict_phase:
        phi = torch.where(y_phys[:, 4:5] < cfg.t_melt, 1.0, -1.0)
        y_phys = torch.cat([y_phys, phi], dim=1)

    scaler = build_scaler(x_phys, y_phys)
    x_norm = scaler.transform_x(x_phys)
    y_norm = scaler.transform_y(y_phys)
    boundary_masks = infer_boundary_masks(x_phys)

    return DatasetBundle(
        x_phys=x_phys,
        x_norm=x_norm,
        y_phys=y_phys,
        y_norm=y_norm,
        scaler=scaler,
        boundary_masks=boundary_masks,
    )


# =========================================================
# 4) Quantum connector
# =========================================================


def create_quantum_layer(n_qubits: int, n_layers: int) -> nn.Module:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    qc = QuantumCircuit(n_qubits)
    inputs = ParameterVector("x", 4)
    weights = ParameterVector("w", n_qubits * n_layers * 2)

    w_idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(inputs[i % 4], i)
        for i in range(n_qubits):
            qc.rx(weights[w_idx], i)
            qc.rz(weights[w_idx + 1], i)
            w_idx += 2
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)

    obs = [
        SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)])
        for i in range(n_qubits)
    ]

    # Important for TorchConnector backprop through inputs.
    qnn = EstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=inputs,
        weight_params=weights,
        input_gradients=True,
    )
    return TorchConnector(qnn)


class QuantumFeatureHead(nn.Module):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 4),
        )
        self.q = create_quantum_layer(cfg.n_qubits, cfg.n_q_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q_in = math.pi * torch.tanh(self.pre(h))
        return self.q(q_in)


# =========================================================
# 5) Hybrid model
# =========================================================


class HybridSILBQPINN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ff_dim = cfg.fourier_features * 2
        self.ff = FourierFeatures(2, cfg.fourier_features, scale=1.0)

        in_dim = 2 + ff_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.trunk_layers)]
        )

        if cfg.use_quantum:
            self.quantum_head = QuantumFeatureHead(cfg.hidden_dim, cfg)
            fusion_in = cfg.hidden_dim + cfg.n_qubits
        else:
            self.quantum_head = None
            fusion_in = cfg.hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )

        out_dim = 5 + int(cfg.predict_phase)
        self.head = nn.Linear(cfg.hidden_dim, out_dim)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        ff = self.ff(x_norm)
        h = self.input_proj(torch.cat([x_norm, ff], dim=-1))
        for block in self.blocks:
            h = block(h)

        if self.quantum_head is not None:
            q = self.quantum_head(h)
            h = torch.cat([h, q], dim=-1)

        h = self.fusion(h)
        return self.head(h)


# =========================================================
# 6) Physics residuals in cylindrical (r, z) axisymmetric form
# =========================================================


def _grad_sum(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]



def _first_and_second(
    y_norm: torch.Tensor,
    x_norm: torch.Tensor,
    x_scale: torch.Tensor,
    y_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns physical derivatives:
      dy/dr, dy/dz, d2y/dr2, d2y/dz2

    x_scale = dx_norm/dx_phys.
    y_phys = y_norm * y_std + y_mean.
    """
    g1 = _grad_sum(y_norm, x_norm)
    dy_dr = y_std * g1[:, 0:1] * x_scale[0]
    dy_dz = y_std * g1[:, 1:2] * x_scale[1]

    d2y_dr2 = y_std * _grad_sum(g1[:, 0:1], x_norm)[:, 0:1] * (x_scale[0] ** 2)
    d2y_dz2 = y_std * _grad_sum(g1[:, 1:2], x_norm)[:, 1:2] * (x_scale[1] ** 2)
    return dy_dr, dy_dz, d2y_dr2, d2y_dz2



def get_residuals(
    model: nn.Module,
    x_norm: torch.Tensor,
    scaler: Scaler,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    x_norm = x_norm.detach().clone().requires_grad_(True)

    y_norm = model(x_norm)
    y_phys = scaler.inverse_y(y_norm)
    x_phys = scaler.inverse_x(x_norm)

    x_scale = scaler.x_scale
    r = x_phys[:, 0:1].clamp_min(1e-6)

    u = y_phys[:, 0:1]  # radial
    v = y_phys[:, 1:2]  # azimuthal / swirl
    w = y_phys[:, 2:3]  # axial
    p = y_phys[:, 3:4]
    T = y_phys[:, 4:5]

    u_r, u_z, u_rr, u_zz = _first_and_second(y_norm[:, 0:1], x_norm, x_scale, scaler.y_std[0:1])
    v_r, v_z, v_rr, v_zz = _first_and_second(y_norm[:, 1:2], x_norm, x_scale, scaler.y_std[1:2])
    w_r, w_z, w_rr, w_zz = _first_and_second(y_norm[:, 2:3], x_norm, x_scale, scaler.y_std[2:3])
    p_r, p_z, _, _ = _first_and_second(y_norm[:, 3:4], x_norm, x_scale, scaler.y_std[3:4])
    T_r, T_z, T_rr, T_zz = _first_and_second(y_norm[:, 4:5], x_norm, x_scale, scaler.y_std[4:5])

    continuity = u_r + u / r + w_z

    visc_r = u_rr + u_r / r - u / (r ** 2) + u_zz
    visc_theta = v_rr + v_r / r - v / (r ** 2) + v_zz
    visc_z = w_rr + w_r / r + w_zz
    visc_T = T_rr + T_r / r + T_zz

    mom_r = (u * u_r + w * u_z - (v ** 2) / r) + (1.0 / cfg.rho) * p_r - cfg.nu * visc_r
    mom_theta = (u * v_r + w * v_z + u * v / r) - cfg.nu * visc_theta
    mom_z = (u * w_r + w * w_z) + (1.0 / cfg.rho) * p_z - cfg.nu * visc_z - cfg.beta_T * cfg.g_z * (T - cfg.t_melt)
    energy = (u * T_r + w * T_z) - cfg.alpha * visc_T

    residuals = {
        "continuity": continuity,
        "mom_r": mom_r,
        "mom_theta": mom_theta,
        "mom_z": mom_z,
        "energy": energy,
    }

    if cfg.predict_phase:
        phi = y_phys[:, 5:6]
        phi_r, phi_z, phi_rr, phi_zz = _first_and_second(y_norm[:, 5:6], x_norm, x_scale, scaler.y_std[5:6])
        # Lightweight optional advection-diffusion regularizer for phi.
        residuals["phase"] = u * phi_r + w * phi_z - cfg.alpha * (phi_rr + phi_r / r + phi_zz)

    return residuals


# =========================================================
# 7) Boundary losses (soft, inferred from geometry extents)
# =========================================================


def boundary_loss(
    model: nn.Module,
    dataset: DatasetBundle,
    cfg: Config,
) -> torch.Tensor:
    # No hard BC assumptions if a boundary set is empty.
    x = dataset.x_norm.to(next(model.parameters()).device)
    y_norm = model(x)
    y_phys = dataset.scaler.inverse_y(y_norm)

    losses: List[torch.Tensor] = []
    masks = {k: v.to(x.device) for k, v in dataset.boundary_masks.items()}

    # Axis regularity: radial and swirl velocities should vanish at axis.
    if masks["axis"].any():
        axis_pred = y_phys[masks["axis"]]
        losses.append((axis_pred[:, 0:1] ** 2).mean())
        losses.append((axis_pred[:, 1:2] ** 2).mean())

    # Outer wall placeholder: penalize non-zero radial velocity.
    if masks["outer_wall"].any():
        wall_pred = y_phys[masks["outer_wall"]]
        losses.append((wall_pred[:, 0:1] ** 2).mean())

    if not losses:
        return torch.zeros((), device=x.device)
    return torch.stack(losses).mean()


# =========================================================
# 8) Training
# =========================================================


def sample_indices(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size >= n:
        return torch.arange(n, device=device)
    return torch.randint(0, n, (batch_size,), device=device)


@torch.no_grad()
def evaluate_data_mse(model: nn.Module, dataset: DatasetBundle, device: torch.device) -> float:
    x = dataset.x_norm.to(device)
    y = dataset.y_norm.to(device)
    pred = model(x)
    return float(F.mse_loss(pred, y).item())



def train_model(dataset: DatasetBundle, cfg: Config) -> Tuple[nn.Module, Dict[str, List[float]]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    scaler = Scaler(
        x_min=dataset.scaler.x_min.to(device),
        x_max=dataset.scaler.x_max.to(device),
        y_mean=dataset.scaler.y_mean.to(device),
        y_std=dataset.scaler.y_std.to(device),
    )
    dataset = DatasetBundle(
        x_phys=dataset.x_phys.to(device),
        x_norm=dataset.x_norm.to(device),
        y_phys=dataset.y_phys.to(device),
        y_norm=dataset.y_norm.to(device),
        scaler=scaler,
        boundary_masks={k: v.to(device) for k, v in dataset.boundary_masks.items()},
    )

    if cfg.predict_phase and "phase" not in cfg.initial_loss_terms:
        loss_terms = list(cfg.initial_loss_terms) + ["phase", "boundary"]
    else:
        loss_terms = list(cfg.initial_loss_terms) + ["boundary"]

    model = HybridSILBQPINN(cfg).to(device)
    loss_balancer = AdaptiveMultiLoss(loss_terms).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.scheduler_tmax)

    hist: Dict[str, List[float]] = {
        "total": [],
        "data": [],
        "physics": [],
        "boundary": [],
        "val_mse": [],
    }

    n = dataset.x_norm.shape[0]
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        data_idx = sample_indices(n, cfg.batch_data, device)
        col_idx = sample_indices(n, cfg.batch_collocation, device)

        x_data = dataset.x_norm[data_idx]
        y_data = dataset.y_norm[data_idx]
        x_col = dataset.x_norm[col_idx]

        pred = model(x_data)
        loss_data = F.mse_loss(pred, y_data)

        residuals = get_residuals(model, x_col, dataset.scaler, cfg)
        physics_losses = {name: (res ** 2).mean() for name, res in residuals.items()}
        physics_mean = torch.stack(list(physics_losses.values())).mean()

        loss_bc = boundary_loss(model, dataset, cfg)

        all_losses: Dict[str, torch.Tensor] = {
            "data": loss_data,
            **physics_losses,
            "boundary": loss_bc,
        }
        total_loss = loss_balancer(all_losses)
        total_loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        model.eval()
        val_mse = evaluate_data_mse(model, dataset, device)

        hist["total"].append(float(total_loss.item()))
        hist["data"].append(float(loss_data.item()))
        hist["physics"].append(float(physics_mean.item()))
        hist["boundary"].append(float(loss_bc.item()))
        hist["val_mse"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "loss_balancer_state_dict": loss_balancer.state_dict(),
                    "config": cfg.__dict__,
                    "scaler": {
                        "x_min": dataset.scaler.x_min.detach().cpu(),
                        "x_max": dataset.scaler.x_max.detach().cpu(),
                        "y_mean": dataset.scaler.y_mean.detach().cpu(),
                        "y_std": dataset.scaler.y_std.detach().cpu(),
                    },
                    "best_val_mse": best_val,
                },
                cfg.checkpoint_path,
            )

        if epoch % cfg.print_every == 0 or epoch == 1:
            w = loss_balancer.weights()
            print(
                f"Epoch {epoch:5d} | total={total_loss.item():.4e} | "
                f"data={loss_data.item():.4e} | phys={physics_mean.item():.4e} | "
                f"bc={loss_bc.item():.4e} | val_mse={val_mse:.4e} | weights={w}"
            )

    return model, hist


# =========================================================
# 9) Inference helpers
# =========================================================


def predict_physical(
    model: nn.Module,
    scaler: Scaler,
    coords_phys: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    x = scaler.transform_x(coords_phys.to(device))
    with torch.no_grad():
        y_norm = model(x)
        y_phys = scaler.inverse_y(y_norm)
    return y_phys


# =========================================================
# 10) Main
# =========================================================


def main(data_path: str = "crystal_data.txt") -> None:
    cfg = Config()
    if cfg.use_quantum and cfg.use_amp:
        raise ValueError("Set use_amp=False when the quantum branch is enabled.")

    dataset = load_cz_data(data_path, cfg)
    model, history = train_model(dataset, cfg)

    print("Training finished.")
    print(f"Best checkpoint saved to: {cfg.checkpoint_path}")
    print(f"Final data MSE: {history['val_mse'][-1]:.6e}")


if __name__ == "__main__":
    main()
