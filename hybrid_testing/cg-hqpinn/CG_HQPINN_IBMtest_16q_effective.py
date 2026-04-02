from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pennylane as qml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DataStats:
    length_scale: float
    velocity_scale: float
    pressure_scale: float
    temp_min: float
    temp_max: float
    pressure_coeff: float

    @property
    def temp_scale(self) -> float:
        return max(self.temp_max - self.temp_min, 1e-12)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class TrainArtifacts:
    x: torch.Tensor
    y: torch.Tensor
    stats: DataStats


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 2, mapping_size: int = 24, scale: float = 6.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, mapping_size) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class Hybrid16QPINN(nn.Module):
    """A pragmatic 16-qubit hybrid PINN.

    The quantum block is shallow enough to be transpilation-friendly while still
    injecting 16 learned quantum features into the classical post-network.
    """

    def __init__(self, q_layer: nn.Module, n_qubits: int = 16):
        super().__init__()
        self.n_qubits = n_qubits
        self.detach_quantum = False

        self.ff = FourierFeatures(in_dim=2, mapping_size=24, scale=6.0)
        self.coord_proj = nn.Sequential(
            nn.Linear(2 + 48, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.res1 = ResidualMLP(128)
        self.res2 = ResidualMLP(128)

        self.to_quantum = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_qubits),
        )
        self.classical_skip = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        self.q_layer = q_layer
        self.q_norm = nn.LayerNorm(n_qubits)
        self.post = nn.Sequential(
            nn.Linear(64 + n_qubits + 2, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.ff(x)], dim=-1)
        h = self.coord_proj(h)
        h = self.res1(h)
        h = self.res2(h)
        return h

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        q_in = math.pi * torch.tanh(self.to_quantum(h))
        # TorchLayer with remote devices expects (batch, n_qubits) — ensure no extra dims
        q_in = q_in.view(-1, self.n_qubits)
        q_out = self.q_layer(q_in)
        if self.detach_quantum:
            q_out = q_out.detach()
        c_skip = self.classical_skip(h)
        return torch.cat([c_skip, self.q_norm(q_out), x], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.quantum_features(x)
        raw = self.post(fused)

        # Hard axis constraints: radial velocity and swirl vanish on the axis.
        r = x[:, 0:1]
        u_r = r * raw[:, 0:1]
        u_z = raw[:, 1:2]
        u_theta = r * raw[:, 2:3]
        p = raw[:, 3:4]
        T = raw[:, 4:5]
        return torch.cat([u_r, u_z, u_theta, p, T], dim=1)

    def freeze_for_ibm_head_tuning(self) -> None:
        for module in [self.ff, self.coord_proj, self.res1, self.res2, self.to_quantum, self.q_layer, self.q_norm, self.classical_skip]:
            for p in module.parameters():
                p.requires_grad = False
        for p in self.post.parameters():
            p.requires_grad = True
        self.detach_quantum = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self.detach_quantum = False


# -----------------------------------------------------------------------------
# Quantum device builders
# -----------------------------------------------------------------------------

def build_aer_device(n_qubits: int):
    # Analytic Aer simulation is used for the 2000-epoch pretraining stage so the
    # spatial PINN derivatives remain tractable.
    return qml.device(
        "qiskit.aer",
        wires=n_qubits,
        shots=None,
        backend="aer_simulator",
        method="statevector",
    )


def build_fake_device(n_qubits: int, shots: int):
    """Noisy local simulator using IBM's FakeSherbrooke (127q Eagle noise model)."""
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    backend = FakeSherbrooke()
    dev = qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=1,
    )
    return dev, backend


def build_ibm_device(n_qubits: int, shots: int, channel: str, backend_name: str, instance: Optional[str], token: Optional[str]):
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {"channel": channel, "token": token}
    if instance:
        kwargs["instance"] = instance

    service = QiskitRuntimeService(**kwargs)

    if backend_name and backend_name != "least_busy":
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=n_qubits,
        )

    dev = qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=shots,
        optimization_level=1,
    )
    return dev, backend


# -----------------------------------------------------------------------------
# Quantum layer
# -----------------------------------------------------------------------------

def make_quantum_layer(device, n_qubits: int, n_layers: int, diff_method: str) -> nn.Module:
    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs shape: (n_qubits,) — one sample, TorchLayer handles batching externally
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RZ(0.5 * inputs[(i + layer) % n_qubits], wires=i)
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            for i in range(0, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    # Wrap in a batching shim: TorchLayer passes the full (batch, n_qubits) tensor
    # to the qnode at once. We manually loop over samples so the circuit always
    # receives a 1D (n_qubits,) input, which is what the remote device expects.
    base_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    class BatchedQLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = base_layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n_qubits) — run one sample at a time
            return torch.stack([self.layer(x[i]) for i in range(x.shape[0])], dim=0)

    return BatchedQLayer()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_cz_data(file_path: str, device: torch.device) -> TrainArtifacts:
    data = pd.read_csv(file_path, comment="%", sep=r"\s+", header=None)

    if data.shape[1] >= 9:
        # Keep the convention already used in your file.
        frame = data.iloc[:, [0, 1, 4, 5, 6, 7, 8]].copy()
        frame.columns = ["r", "z", "u", "w", "vtheta", "p", "T"]
    elif data.shape[1] == 7:
        frame = data.copy()
        frame.columns = ["r", "z", "u", "w", "vtheta", "p", "T"]
    else:
        raise ValueError(
            f"Unsupported data shape {tuple(data.shape)}. Expected 7 or at least 9 columns."
        )

    length_scale = float(max(frame["r"].abs().max(), frame["z"].abs().max(), 1e-12))
    velocity_scale = float(max(frame[["u", "w", "vtheta"]].abs().max().max(), 1e-12))
    pressure_scale = float(max(frame["p"].abs().max(), 1e-12))
    temp_min = float(frame["T"].min())
    temp_max = float(frame["T"].max())
    temp_scale = max(temp_max - temp_min, 1e-12)

    frame["r"] = frame["r"] / length_scale
    frame["z"] = frame["z"] / length_scale
    frame["u"] = frame["u"] / velocity_scale
    frame["w"] = frame["w"] / velocity_scale
    frame["vtheta"] = frame["vtheta"] / velocity_scale
    frame["p"] = frame["p"] / pressure_scale
    frame["T"] = (frame["T"] - temp_min) / temp_scale

    # If density is normalized to 1, pressure in the nondimensional Navier-Stokes
    # equations should scale with V^2. This coefficient maps your p/p_ref target to
    # that convention more cleanly than using p_ref blindly.
    pressure_coeff = pressure_scale / max(velocity_scale ** 2, 1e-12)

    X = torch.tensor(frame[["r", "z"]].values, dtype=torch.float32, device=device)
    Y = torch.tensor(frame[["u", "w", "vtheta", "p", "T"]].values, dtype=torch.float32, device=device)
    stats = DataStats(
        length_scale=length_scale,
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        temp_min=temp_min,
        temp_max=temp_max,
        pressure_coeff=pressure_coeff,
    )
    return TrainArtifacts(x=X, y=Y, stats=stats)


# -----------------------------------------------------------------------------
# PINN residuals
# -----------------------------------------------------------------------------

def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def physics_loss(
    x: torch.Tensor,
    model: Hybrid16QPINN,
    stats: DataStats,
    re: float,
    pr: float,
    gr: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = x.clone().detach().requires_grad_(True)
    pred = model(x)

    # Predicted fields are already dimensionless except pressure, which uses your
    # dataset pressure scale. pressure_coeff maps it back toward the V^2 scaling.
    u_r = pred[:, 0:1]
    u_z = pred[:, 1:2]
    u_theta = pred[:, 2:3]
    p_hat = pred[:, 3:4]
    theta = pred[:, 4:5]

    r = torch.clamp(x[:, 0:1], min=1e-4)

    du_r = gradients(u_r, x)
    du_z = gradients(u_z, x)
    du_t = gradients(u_theta, x)
    dp = gradients(p_hat, x)
    dT = gradients(theta, x)

    ur_r, ur_z = du_r[:, 0:1], du_r[:, 1:2]
    uz_r, uz_z = du_z[:, 0:1], du_z[:, 1:2]
    ut_r, ut_z = du_t[:, 0:1], du_t[:, 1:2]
    p_r, p_z = dp[:, 0:1], dp[:, 1:2]
    T_r, T_z = dT[:, 0:1], dT[:, 1:2]

    ur_rr = gradients(ur_r, x)[:, 0:1]
    ur_zz = gradients(ur_z, x)[:, 1:2]
    uz_rr = gradients(uz_r, x)[:, 0:1]
    uz_zz = gradients(uz_z, x)[:, 1:2]
    ut_rr = gradients(ut_r, x)[:, 0:1]
    ut_zz = gradients(ut_z, x)[:, 1:2]
    T_rr = gradients(T_r, x)[:, 0:1]
    T_zz = gradients(T_z, x)[:, 1:2]

    p_coeff = stats.pressure_coeff

    continuity = ur_r + u_r / r + uz_z
    mom_r = (
        u_r * ur_r
        + u_z * ur_z
        - (u_theta ** 2) / r
        + p_coeff * p_r
        - (1.0 / re) * (ur_rr + ur_r / r - u_r / (r ** 2) + ur_zz)
    )
    mom_z = (
        u_r * uz_r
        + u_z * uz_z
        + p_coeff * p_z
        - (1.0 / re) * (uz_rr + uz_r / r + uz_zz)
        - (gr / (re ** 2)) * theta
    )
    swirl = (
        u_r * ut_r
        + u_z * ut_z
        + (u_r * u_theta) / r
        - (1.0 / re) * (ut_rr + ut_r / r - u_theta / (r ** 2) + ut_zz)
    )
    energy = u_r * T_r + u_z * T_z - (1.0 / (pr * re)) * (T_rr + T_r / r + T_zz)

    terms = {
        "cont": continuity.pow(2).mean(),
        "mom_r": mom_r.pow(2).mean(),
        "mom_z": mom_z.pow(2).mean(),
        "swirl": swirl.pow(2).mean(),
        "energy": energy.pow(2).mean(),
    }
    total = sum(terms.values())
    return total, terms


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMAWeights:
    def __init__(self, beta: float = 0.95):
        self.beta = beta
        self.values: Dict[str, float] = {}

    def update(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        scalar_losses = {k: float(v.detach().item()) for k, v in losses.items()}
        avg = max(sum(scalar_losses.values()) / len(scalar_losses), 1e-12)
        for k, value in scalar_losses.items():
            target = value / avg
            prev = self.values.get(k, 1.0)
            self.values[k] = self.beta * prev + (1.0 - self.beta) * target
            out[k] = self.values[k]
        return out


def make_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x.detach().cpu(), y.detach().cpu())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def choose_calibration_subset(x: torch.Tensor, y: torch.Tensor, subset_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    subset_size = min(subset_size, x.shape[0])
    if subset_size >= x.shape[0]:
        return x, y

    # A simple coverage heuristic: sort by radial coordinate, then pick evenly spaced points.
    order = torch.argsort(x[:, 0] + 0.25 * x[:, 1])
    picks = torch.linspace(0, len(order) - 1, steps=subset_size).round().long()
    idx = order[picks]
    return x[idx], y[idx]


def save_checkpoint(path: str, model: nn.Module, stats: DataStats, args: argparse.Namespace) -> None:
    payload = {
        "model_state": model.state_dict(),
        "stats": asdict(stats),
        "args": vars(args),
    }
    torch.save(payload, path)



def load_checkpoint(path: str, model: nn.Module) -> Optional[DataStats]:
    payload = torch.load(path, map_location="cpu")
    if "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=True)
        stats_dict = payload.get("stats")
        if stats_dict is not None:
            return DataStats(**stats_dict)
        return None

    # Backward-compatible plain state_dict support.
    model.load_state_dict(payload, strict=False)
    return None


# -----------------------------------------------------------------------------
# SPSA Optimizer
# -----------------------------------------------------------------------------

class SPSAOptimizer:
    """SPSA gradient estimator — 2 circuit evals per step regardless of param count."""

    def __init__(self, quantum_params: List[torch.Tensor], a=0.1, c=0.02, alpha=0.602, gamma=0.101):
        self.params = quantum_params
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma
        self.k = 1

    def step(self, loss_fn) -> float:
        ak = self.a / (self.k ** self.alpha)
        ck = self.c / (self.k ** self.gamma)
        self.k += 1

        deltas = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1 for p in self.params]

        for p, d in zip(self.params, deltas):
            p.data.add_(d * ck)
        loss_plus = loss_fn()

        for p, d in zip(self.params, deltas):
            p.data.add_(d * -2 * ck)
        loss_minus = loss_fn()

        with torch.no_grad():
            for p, d in zip(self.params, deltas):
                p.data.add_(d * ck)  # restore
                p.data.sub_(ak * (loss_plus - loss_minus) / (2 * ck * d))

        return (loss_plus + loss_minus) / 2


# -----------------------------------------------------------------------------
# Pre-training diagnostic plots
# -----------------------------------------------------------------------------

def plot_pretrain_diagnostics(
    artifacts: TrainArtifacts,
    model: Hybrid16QPINN,
    device: torch.device,
    save_dir: str,
    x_calib: Optional[torch.Tensor] = None,
    y_calib: Optional[torch.Tensor] = None,
) -> None:
    """Save diagnostic plots before training starts."""
    os.makedirs(save_dir, exist_ok=True)
    x_np = artifacts.x.detach().cpu().numpy()
    y_np = artifacts.y.detach().cpu().numpy()
    field_names = ["u_r", "u_z", "u_θ", "p", "T"]

    # 1. Data distribution: scatter of (r, z) coloured by each field
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, (ax, name) in enumerate(zip(axes, field_names)):
        sc = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np[:, i], s=1, cmap="viridis")
        ax.set_xlabel("r"); ax.set_ylabel("z"); ax.set_title(name)
        plt.colorbar(sc, ax=ax)
    fig.suptitle("Training data fields (normalised)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "data_fields.png"), dpi=150)
    plt.close(fig)

    # 2. Calibration subset coverage (if provided)
    if x_calib is not None:
        xc = x_calib.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_np[:, 0], x_np[:, 1], s=1, alpha=0.3, label="full data")
        ax.scatter(xc[:, 0], xc[:, 1], s=30, c="red", marker="x", label=f"calib ({len(xc)})")
        ax.set_xlabel("r"); ax.set_ylabel("z"); ax.set_title("Calibration subset coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "calib_coverage.png"), dpi=150)
        plt.close(fig)

    # 3. Initial model predictions vs ground truth (on a random subset)
    model.eval()
    n_plot = min(2000, x_np.shape[0])
    idx = np.random.choice(x_np.shape[0], n_plot, replace=False)
    with torch.no_grad():
        pred = model(artifacts.x[idx].to(device)).cpu().numpy()
    gt = y_np[idx]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, (ax, name) in enumerate(zip(axes, field_names)):
        lo = min(gt[:, i].min(), pred[:, i].min())
        hi = max(gt[:, i].max(), pred[:, i].max())
        ax.scatter(gt[:, i], pred[:, i], s=2, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("ground truth"); ax.set_ylabel("prediction"); ax.set_title(name)
    fig.suptitle("Initial predictions vs ground truth (before fine-tuning)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "initial_pred_vs_gt.png"), dpi=150)
    plt.close(fig)

    # 4. Quantum weight histogram
    q_weights = torch.cat([p.detach().cpu().flatten() for p in model.q_layer.parameters()])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(q_weights.numpy(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("weight value"); ax.set_ylabel("count"); ax.set_title("Quantum layer weight distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "quantum_weights_hist.png"), dpi=150)
    plt.close(fig)

    print(f"Diagnostic plots saved to {save_dir}/")


# -----------------------------------------------------------------------------
# Main training loops
# -----------------------------------------------------------------------------

def run_aer_stage(args: argparse.Namespace, device: torch.device) -> None:
    artifacts = load_cz_data(args.data, device)

    qdev = build_aer_device(args.n_qubits)
    q_layer = make_quantum_layer(qdev, args.n_qubits, args.n_layers, diff_method="best")
    model = Hybrid16QPINN(q_layer=q_layer, n_qubits=args.n_qubits).to(device)
    model.unfreeze_all()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ema = EMAWeights(beta=0.95)

    loader = make_dataloader(artifacts.x, artifacts.y, batch_size=args.batch_size, shuffle=True)

    print("=" * 80)
    print("AER PRETRAINING")
    print(f"trainable params: {count_trainable_params(model):,}")
    print(f"stats: {artifacts.stats.to_json()}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_phys = 0.0

        phys_ramp = min(1.0, max(0.0, (epoch - args.physics_warmup) / max(args.physics_ramp, 1)))
        target_phys_weight = args.physics_weight * phys_ramp

        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            data_loss = torch.mean((pred - yb) ** 2)
            phys_loss, phys_terms = physics_loss(xb, model, artifacts.stats, args.re, args.pr, args.gr)

            weights = ema.update({"data": data_loss, **phys_terms})
            mean_phys_weight = sum(weights[k] for k in ["cont", "mom_r", "mom_z", "swirl", "energy"]) / 5.0
            scaled_phys = target_phys_weight * (phys_loss / max(mean_phys_weight, 1e-12))
            total_loss = data_loss + scaled_phys

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += float(total_loss.item())
            epoch_data += float(data_loss.item())
            epoch_phys += float(phys_loss.item())

        scheduler.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            n_batches = max(len(loader), 1)
            print(
                f"[AER] epoch {epoch:04d}/{args.epochs} | "
                f"loss={epoch_total / n_batches:.4e} | "
                f"data={epoch_data / n_batches:.4e} | "
                f"phys={epoch_phys / n_batches:.4e} | "
                f"phys_w={target_phys_weight:.3e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    save_checkpoint(args.save, model, artifacts.stats, args)
    stats_path = str(Path(args.save).with_suffix(".stats.json"))
    Path(stats_path).write_text(artifacts.stats.to_json())
    print(f"Saved AER checkpoint to {args.save}")
    print(f"Saved normalization stats to {stats_path}")



def run_ibm_stage(args: argparse.Namespace, device: torch.device) -> None:
    if not args.load:
        raise ValueError("IBM phase requires --load with the AER pretrained checkpoint.")

    artifacts = load_cz_data(args.data, device)

    if args.phase == "ibm-sim":
        qdev, backend = build_fake_device(args.n_qubits, args.shots)
    else:
        qdev, backend = build_ibm_device(
            n_qubits=args.n_qubits,
            shots=args.shots,
            channel=args.ibm_channel,
            backend_name=args.backend,
            instance=args.ibm_instance,
            token=args.ibm_token,
        )

    # diff_method=None — SPSA handles quantum gradients externally
    q_layer = make_quantum_layer(qdev, args.n_qubits, args.n_layers, diff_method=None)
    model = Hybrid16QPINN(q_layer=q_layer, n_qubits=args.n_qubits).to(device)

    ckpt_stats = load_checkpoint(args.load, model)
    if ckpt_stats is not None:
        artifacts.stats = ckpt_stats

    quantum_params = list(model.q_layer.parameters())
    classical_params = list(model.post.parameters())

    if args.ibm_train_scope == "head":
        model.freeze_for_ibm_head_tuning()
        q_optimizer = None
    elif args.ibm_train_scope == "full":
        model.unfreeze_all()
        q_optimizer = SPSAOptimizer(quantum_params, a=0.05, c=0.02)
    else:
        raise ValueError(f"Unsupported --ibm-train-scope {args.ibm_train_scope!r}")

    c_optimizer = torch.optim.Adam(classical_params, lr=args.lr)

    x_calib, y_calib = choose_calibration_subset(artifacts.x, artifacts.y, args.ibm_calib_size)
    loader = make_dataloader(x_calib, y_calib, batch_size=min(args.batch_size, 1), shuffle=True)

    # --- Diagnostic plots before training ---
    plot_dir = str(Path(args.save).parent / "diagnostics")
    plot_pretrain_diagnostics(artifacts, model, device, plot_dir, x_calib, y_calib)

    n_ps = 2 * count_trainable_params(model.q_layer) if args.ibm_train_scope == "full" else 0
    print("=" * 80)
    print("IBM HARDWARE FINE-TUNING (SPSA)")
    print(f"backend: {getattr(backend, 'name', str(backend))}")
    print(f"train scope: {args.ibm_train_scope}")
    print(f"SPSA: {'Active' if q_optimizer else 'Disabled (head-only)'}")
    print(f"Cost/step: 2 circuit evals (vs {n_ps} with param-shift)")
    print(f"calibration subset: {x_calib.shape[0]} pts")
    print("=" * 80)

    import time
    t0 = time.time()
    save_every = max(args.save_every, 1)

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)

            def get_loss():
                pred = model(xb)
                return torch.mean((pred - yb) ** 2).item()

            if q_optimizer:
                batch_loss_val = q_optimizer.step(get_loss)
            else:
                batch_loss_val = get_loss()

            # Classical head update via Adam
            c_optimizer.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            c_optimizer.step()

            epoch_loss += batch_loss_val
            n_batches += 1

        elapsed = (time.time() - t0) / 60.0
        eta = elapsed / max(epoch, 1) * (args.epochs - epoch)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[IBM] epoch {epoch:04d}/{args.epochs} | "
                f"loss={epoch_loss / max(n_batches, 1):.4e} | "
                f"elapsed={elapsed:.1f}m eta={eta:.1f}m"
            )

        if epoch % save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.save, model, artifacts.stats, args)

    print(f"Final IBM checkpoint saved to {args.save}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="16-qubit hybrid PINN with Aer pretraining and IBM fine-tuning.")
    p.add_argument("--phase", choices=["aer", "ibm", "ibm-sim"], required=True)
    p.add_argument("--data", type=str, required=True, help="Path to cz_melt_raw.txt or equivalent data file.")
    p.add_argument("--save", type=str, required=True, help="Where to save the checkpoint.")
    p.add_argument("--load", type=str, default="", help="Checkpoint to load for IBM fine-tuning.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-qubits", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--re", type=float, default=15.0)
    p.add_argument("--pr", type=float, default=28.463)
    p.add_argument("--gr", type=float, default=8000.0)
    p.add_argument("--physics-weight", type=float, default=0.05)
    p.add_argument("--physics-warmup", type=int, default=150)
    p.add_argument("--physics-ramp", type=int, default=400)

    p.add_argument("--shots", type=int, default=4096, help="IBM shot count. Ignored in AER analytic mode.")
    p.add_argument("--backend", type=str, default="least_busy", help="IBM backend name or 'least_busy'.")
    p.add_argument("--ibm-channel", type=str, default="ibm_quantum_platform")
    p.add_argument("--ibm-instance", type=str, default=os.getenv("IBM_INSTANCE", ""))
    p.add_argument("--ibm-token", type=str, default=os.getenv("IBM_TOKEN", ""))
    p.add_argument("--ibm-calib-size", type=int, default=8)
    p.add_argument("--ibm-train-scope", choices=["head", "full"], default="head")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs.")
    p.add_argument("--start-epoch", type=int, default=1, help="Starting epoch (for resuming).")
    p.add_argument("--quick-check", action="store_true", help="2-epoch IBM smoke test — skips AER pretraining, uses random weights.")

    return p



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}")

    if args.quick_check:
        args.phase = "ibm"
        args.epochs = 2
        args.ibm_calib_size = 4
        args.ibm_train_scope = "head"
        args.shots = 512
        args.batch_size = 1   # remote device processes one sample at a time
        print("=== QUICK IBM CONNECTION CHECK (2 epochs, 512 shots, 4 samples, batch=1) ===")
        # Build a dummy checkpoint so load_checkpoint doesn't fail
        if not args.load:
            artifacts = load_cz_data(args.data, device)
            qdev_tmp = build_aer_device(args.n_qubits)
            q_tmp = make_quantum_layer(qdev_tmp, args.n_qubits, args.n_layers, diff_method="best")
            m_tmp = Hybrid16QPINN(q_layer=q_tmp, n_qubits=args.n_qubits).to(device)
            _dummy = "/tmp/quick_check_dummy.pt"
            save_checkpoint(_dummy, m_tmp, artifacts.stats, args)
            args.load = _dummy
            print(f"No --load provided; created dummy checkpoint at {_dummy}")

    if args.phase == "aer":
        run_aer_stage(args, device)
    elif args.phase in ("ibm", "ibm-sim"):
        run_ibm_stage(args, device)
    else:
        raise ValueError(f"Unsupported phase {args.phase!r}")


if __name__ == "__main__":
    main()
