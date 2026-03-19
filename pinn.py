import argparse
import math
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

class SmileiLoader:
    def __init__(self, hdf5_path: str, dim: int = 2, mode: str = "TM", max_snapshots: int = 50):
        self.path = hdf5_path
        self.dim = dim
        self.mode = mode.upper() if dim == 2 else "FULL"
        self.max_snapshots = max_snapshots
        self.data = {}
        self.meta = {}

    @staticmethod
    def _find_dataset(h5file, candidates):
        for key in candidates:
            if key in h5file:
                return h5file[key]
        for key in h5file.keys():
            for cand in candidates:
                if cand.lower() in key.lower():
                    return h5file[key]
        return None

    def load(self):
        with h5py.File(self.path, "r") as f:
            root = f['data'] if 'data' in f else f
            
            for xkey in ["x", "X", "grid_x", "axes/x"]:
                if xkey in f:
                    self.meta["x"] = np.array(f[xkey])
                    break
            for ykey in ["y", "Y", "grid_y", "axes/y"]:
                if ykey in f:
                    self.meta["y"] = np.array(f[ykey])
                    break
            if self.dim == 3:
                for zkey in ["z", "Z", "grid_z", "axes/z"]:
                    if zkey in f:
                        self.meta["z"] = np.array(f[zkey])
                        break

            if self.dim == 2:
                field_names = (
                    ["Ez", "Bx", "By"] if self.mode == "TM" else ["Ex", "Ey", "Bz"]
                )
            else:
                field_names = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

            snapshots = {}
            times = []

            snapshot_keys = sorted(
                [k for k in root.keys() if k.isdigit()],
                key=lambda k: int(k)
            )[: self.max_snapshots]

            if snapshot_keys:
                for sk in snapshot_keys:
                    grp = root[sk]
                    t_val = float(sk)
                    times.append(t_val)
                    for fn in field_names:
                        if fn in grp:
                            arr = np.array(grp[fn])
                            snapshots.setdefault(fn, []).append(arr)
            else:
                times = [0.0]
                for fn in field_names:
                    ds = self._find_dataset(root, [fn, fn + "_mode_0"])
                    snapshots[fn] = [np.array(ds) if ds is not None else None]

            self.meta["times"] = np.array(times)
            for fn in field_names:
                arrs = snapshots.get(fn, [])
                valid = [a for a in arrs if a is not None]
                if valid:
                    self.data[fn] = np.stack(valid, axis=0)

        return self

    def get_training_tensors(self, n_collocation: int = 20_000):
        fields_present = list(self.data.keys())
        times = self.meta["times"]
        T = len(times)

        data_arr = list(self.data.values())[0]
        
        if self.dim == 2:
            Nx, Ny = data_arr.shape[1], data_arr.shape[2]
            x_coords = self.meta.get("x", np.linspace(0, 1, Nx))
            y_coords = self.meta.get("y", np.linspace(0, 1, Ny))
            xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

            all_x, all_y, all_t, all_f = [], [], [], []
            for ti, t in enumerate(times):
                all_x.append(xx.ravel())
                all_y.append(yy.ravel())
                all_t.append(np.full(xx.size, t))
                row = [self.data[fn][ti].ravel() for fn in fields_present]
                all_f.append(np.stack(row, axis=-1))

            coords = np.column_stack([
                np.concatenate(all_x),
                np.concatenate(all_y),
                np.concatenate(all_t),
            ])
            
        else:
            Nx, Ny, Nz = data_arr.shape[1], data_arr.shape[2], data_arr.shape[3]
            x_coords = self.meta.get("x", np.linspace(0, 1, Nx))
            y_coords = self.meta.get("y", np.linspace(0, 1, Ny))
            z_coords = self.meta.get("z", np.linspace(0, 1, Nz))
            xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

            all_x, all_y, all_z, all_t, all_f = [], [], [], [], []
            for ti, t in enumerate(times):
                all_x.append(xx.ravel())
                all_y.append(yy.ravel())
                all_z.append(zz.ravel())
                all_t.append(np.full(xx.size, t))
                row = [self.data[fn][ti].ravel() for fn in fields_present]
                all_f.append(np.stack(row, axis=-1))

            coords = np.column_stack([
                np.concatenate(all_x),
                np.concatenate(all_y),
                np.concatenate(all_z),
                np.concatenate(all_t),
            ])
            
        fields = np.concatenate(all_f, axis=0)

        norm = {
            "x_mean": coords[:, 0].mean(), "x_std": coords[:, 0].std() + 1e-8,
            "y_mean": coords[:, 1].mean(), "y_std": coords[:, 1].std() + 1e-8,
        }
        
        if self.dim == 2:
            norm["t_mean"] = coords[:, 2].mean()
            norm["t_std"] = coords[:, 2].std() + 1e-8
            coords_n = np.column_stack([
                (coords[:, 0] - norm["x_mean"]) / norm["x_std"],
                (coords[:, 1] - norm["y_mean"]) / norm["y_std"],
                (coords[:, 2] - norm["t_mean"]) / norm["t_std"],
            ])
        else:
            norm["z_mean"] = coords[:, 2].mean()
            norm["z_std"] = coords[:, 2].std() + 1e-8
            norm["t_mean"] = coords[:, 3].mean()
            norm["t_std"] = coords[:, 3].std() + 1e-8
            coords_n = np.column_stack([
                (coords[:, 0] - norm["x_mean"]) / norm["x_std"],
                (coords[:, 1] - norm["y_mean"]) / norm["y_std"],
                (coords[:, 2] - norm["z_mean"]) / norm["z_std"],
                (coords[:, 3] - norm["t_mean"]) / norm["t_std"],
            ])
        
        norm["f_mean"] = fields.mean(axis=0)
        norm["f_std"] = fields.std(axis=0) + 1e-8
        fields_n = (fields - norm["f_mean"]) / norm["f_std"]

        rng = np.random.default_rng(42)
        n_spatial = self.dim + 1
        collocation = np.zeros((n_collocation, n_spatial))
        for i in range(n_spatial):
            collocation[:, i] = rng.uniform(coords_n[:, i].min(), coords_n[:, i].max(), n_collocation)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        to = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)

        return (
            to(coords_n), to(fields_n),
            to(collocation),
            norm, fields_present,
        )

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, n_freqs: int = 64, scale: float = 1.0):
        super().__init__()
        B = torch.randn(input_dim, n_freqs) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
class MaxwellPINN(nn.Module):
    def __init__(
        self,
        field_names: list,
        input_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int   = 6,
        n_freqs: int    = 128,
        fourier_scale: float = 10.0,
    ):
        super().__init__()
        self.field_names = field_names
        self.input_dim = input_dim
        n_out = len(field_names)

        self.embed = FourierEmbedding(input_dim, n_freqs, fourier_scale)
        in_dim = 2 * n_freqs

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, n_out))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords):
        return self.net(self.embed(coords))
def compute_gradients(u, coords):
    grads = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    if coords.shape[1] == 3:
        return grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
    else:
        return grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]
def maxwell_residuals_TM(model, coords, norm_params):
    coords = coords.requires_grad_(True)
    pred = model(coords)

    Ez = pred[:, 0:1]
    Bx = pred[:, 1:2]
    By = pred[:, 2:3]

    dEz_dx, dEz_dy, dEz_dt = compute_gradients(Ez, coords)
    dBx_dx, dBx_dy, dBx_dt = compute_gradients(Bx, coords)
    dBy_dx, dBy_dy, dBy_dt = compute_gradients(By, coords)

    r_faraday_x  = dBx_dt + dEz_dy
    r_faraday_y  = dBy_dt - dEz_dx
    r_ampere_z   = dEz_dt - dBy_dx + dBx_dy
    r_gauss_B    = dBx_dx + dBy_dy

    return r_faraday_x, r_faraday_y, r_ampere_z, r_gauss_B
def maxwell_residuals_TE(model, coords, norm_params):
    coords = coords.requires_grad_(True)
    pred = model(coords)

    Ex = pred[:, 0:1]
    Ey = pred[:, 1:2]
    Bz = pred[:, 2:3]

    dEx_dx, dEx_dy, dEx_dt = compute_gradients(Ex, coords)
    dEy_dx, dEy_dy, dEy_dt = compute_gradients(Ey, coords)
    dBz_dx, dBz_dy, dBz_dt = compute_gradients(Bz, coords)

    r_ampere_x   = dEx_dt - dBz_dy
    r_ampere_y   = dEy_dt + dBz_dx
    r_faraday_z  = dBz_dt - dEx_dy + dEy_dx
    r_gauss_E    = dEx_dx + dEy_dy

    return r_ampere_x, r_ampere_y, r_faraday_z, r_gauss_E
def maxwell_residuals_3D(model, coords, norm_params):
    coords = coords.requires_grad_(True)
    pred = model(coords)
    
    Ex = pred[:, 0:1]
    Ey = pred[:, 1:2]
    Ez = pred[:, 2:3]
    Bx = pred[:, 3:4]
    By = pred[:, 4:5]
    Bz = pred[:, 5:6]
    
    dEx_dx, dEx_dy, dEx_dz, dEx_dt = compute_gradients(Ex, coords)
    dEy_dx, dEy_dy, dEy_dz, dEy_dt = compute_gradients(Ey, coords)
    dEz_dx, dEz_dy, dEz_dz, dEz_dt = compute_gradients(Ez, coords)
    dBx_dx, dBx_dy, dBx_dz, dBx_dt = compute_gradients(Bx, coords)
    dBy_dx, dBy_dy, dBy_dz, dBy_dt = compute_gradients(By, coords)
    dBz_dx, dBz_dy, dBz_dz, dBz_dt = compute_gradients(Bz, coords)
    
    r_faraday_x = dBx_dt + dEz_dy - dEy_dz
    r_faraday_y = dBy_dt + dEx_dz - dEz_dx
    r_faraday_z = dBz_dt + dEy_dx - dEx_dy
    
    r_ampere_x = dEx_dt - dBz_dy + dBy_dz
    r_ampere_y = dEy_dt - dBx_dz + dBz_dx
    r_ampere_z = dEz_dt - dBy_dx + dBx_dy
    
    r_gauss_E = dEx_dx + dEy_dy + dEz_dz
    r_gauss_B = dBx_dx + dBy_dy + dBz_dz
    
    return (r_faraday_x, r_faraday_y, r_faraday_z,
            r_ampere_x, r_ampere_y, r_ampere_z,
            r_gauss_E, r_gauss_B)
class PINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_phys: float = 0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys

    def forward(self, pred_data, true_data, residuals):
        l_data = nn.functional.mse_loss(pred_data, true_data)
        l_phys = sum(r.pow(2).mean() for r in residuals)
        total  = self.lambda_data * l_data + self.lambda_phys * l_phys
        return total, l_data.item(), l_phys.item()
class PINNTrainer:
    def __init__(
        self,
        model: MaxwellPINN,
        dim: int,
        mode: str,
        coords_data: torch.Tensor,
        fields_data: torch.Tensor,
        coords_phys: torch.Tensor,
        norm_params: dict,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 4096,
        adaptive_weights: bool = True,
    ):
        self.model       = model
        self.dim         = dim
        self.mode        = mode.upper() if dim == 2 else "FULL"
        self.c_data      = coords_data
        self.f_data      = fields_data
        self.c_phys      = coords_phys
        self.norm_params = norm_params
        self.loss_fn     = PINNLoss(lambda_data, lambda_phys)
        self.optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
        self.batch_size  = batch_size
        self.history     = {"total": [], "data": [], "phys": []}
        self.adaptive_weights = adaptive_weights
        self.lambda_phys_init = lambda_phys
        
        if dim == 2:
            self.residual_fn = (
                maxwell_residuals_TM if self.mode == "TM" else maxwell_residuals_TE
            )
        else:
            self.residual_fn = maxwell_residuals_3D

    def _sample_batch(self, coords, fields, n):
        idx = torch.randint(0, coords.shape[0], (n,), device=coords.device)
        return coords[idx], fields[idx]

    def step(self):
        self.model.train()
        self.optimizer.zero_grad()

        c_b, f_b = self._sample_batch(self.c_data, self.f_data, self.batch_size)
        pred_b   = self.model(c_b)

        c_p, _ = self._sample_batch(
            self.c_phys, self.c_phys,
            min(self.batch_size, self.c_phys.shape[0]),
        )
        c_p = c_p.requires_grad_(True)
        residuals = self.residual_fn(self.model, c_p, self.norm_params)

        total, l_d, l_p = self.loss_fn(pred_b, f_b, residuals)
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total.item(), l_d, l_p

    def train(self, epochs: int = 20_000, log_every: int = 500, scheduler=None, warmup_epochs: int = 5000):
        t0 = time.time()
        for ep in range(1, epochs + 1):
            if self.adaptive_weights and warmup_epochs > 0:
                progress = min(ep / warmup_epochs, 1.0)
                self.loss_fn.lambda_phys = self.lambda_phys_init * progress
            
            total, l_d, l_p = self.step()
            self.history["total"].append(total)
            self.history["data"].append(l_d)
            self.history["phys"].append(l_p)

            if scheduler:
                scheduler.step()

            if ep % log_every == 0 or ep == 1:
                phys_weight = self.loss_fn.lambda_phys if self.adaptive_weights else self.lambda_phys_init
                print(
                    f"Epoch {ep:6d}/{epochs} | "
                    f"Total {total:.4e} | "
                    f"Data {l_d:.4e} | "
                    f"Phys {l_p:.4e} | "
                    f"λ_phys {phys_weight:.3f}"
                )
class MaxwellSurrogate:
    def __init__(self, model: MaxwellPINN, norm_params: dict, field_names: list, dim: int = 2):
        self.model       = model
        self.norm_params = norm_params
        self.field_names = field_names
        self.dim         = dim
        self.device      = next(model.parameters()).device

    def predict(self, x, y, t, z=None):
        if self.dim == 2:
            x, y, t = map(np.atleast_1d, (x, y, t))
            np_m = self.norm_params

            xn = (x - np_m["x_mean"]) / np_m["x_std"]
            yn = (y - np_m["y_mean"]) / np_m["y_std"]
            tn = (t - np_m["t_mean"]) / np_m["t_std"]

            coords = torch.tensor(
                np.column_stack([xn, yn, tn]), dtype=torch.float32, device=self.device
            )
        else:
            if z is None:
                raise ValueError("z coordinate required for 3D surrogate")
            x, y, z, t = map(np.atleast_1d, (x, y, z, t))
            np_m = self.norm_params

            xn = (x - np_m["x_mean"]) / np_m["x_std"]
            yn = (y - np_m["y_mean"]) / np_m["y_std"]
            zn = (z - np_m["z_mean"]) / np_m["z_std"]
            tn = (t - np_m["t_mean"]) / np_m["t_std"]

            coords = torch.tensor(
                np.column_stack([xn, yn, zn, tn]), dtype=torch.float32, device=self.device
            )
            
        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(coords).cpu().numpy()

        pred = pred_n * np_m["f_std"] + np_m["f_mean"]
        return {fn: pred[:, i] for i, fn in enumerate(self.field_names)}

    def compression_ratio(self, original_bytes: int) -> float:
        n_params = sum(p.numel() for p in self.model.parameters())
        model_bytes = n_params * 4
        return original_bytes / model_bytes
def plot_loss_history(history: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(history["total"], label="Total", lw=1.5)
    axes[0].semilogy(history["data"],  label="Data",  lw=1.5, linestyle="--")
    axes[0].semilogy(history["phys"],  label="Physics", lw=1.5, linestyle=":")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].plot(
        np.array(history["phys"]) / (np.array(history["data"]) + 1e-12),
        lw=1.5, color="purple"
    )
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Physics / Data loss ratio")
    axes[1].set_title("Loss Balance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_field_comparison(surrogate: MaxwellSurrogate, loader: SmileiLoader,
                          snapshot_idx: int = 0, field: str = "Ez",
                          save_path: str = None):
    if field not in loader.data:
        return

    data_arr = loader.data[field]
    times    = loader.meta["times"]
    t_val    = times[snapshot_idx]
    Nx, Ny   = data_arr.shape[1], data_arr.shape[2]

    x_coords = loader.meta.get("x", np.linspace(0, 1, Nx))
    y_coords = loader.meta.get("y", np.linspace(0, 1, Ny))
    xx, yy   = np.meshgrid(x_coords, y_coords, indexing="ij")

    true_field = data_arr[snapshot_idx]

    pred_dict  = surrogate.predict(xx.ravel(), yy.ravel(), np.full(xx.size, t_val))
    pred_field = pred_dict[field].reshape(Nx, Ny)

    vmin = min(true_field.min(), pred_field.min())
    vmax = max(true_field.max(), pred_field.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    kw = dict(vmin=vmin, vmax=vmax, cmap="RdBu_r", origin="lower",
              extent=[y_coords[0], y_coords[-1], x_coords[0], x_coords[-1]])

    im0 = axes[0].imshow(true_field, **kw)
    axes[0].set_title(f"Smilei HDF5  |  {field}  t={t_val:.2f}")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_field, **kw)
    axes[1].set_title(f"PINN Surrogate  |  {field}  t={t_val:.2f}")
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(pred_field - true_field)
    im2 = axes[2].imshow(err, cmap="hot", origin="lower",
                         extent=[y_coords[0], y_coords[-1], x_coords[0], x_coords[-1]])
    axes[2].set_title(f"Absolute Error  (max={err.max():.2e})")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_field_evolution(surrogate: MaxwellSurrogate, x_slice: float,
                         y_slice: float, t_range: tuple, n_steps: int = 200,
                         field: str = "Ez", save_path: str = None):
    t_vals = np.linspace(*t_range, n_steps)
    pred   = surrogate.predict(
        np.full(n_steps, x_slice),
        np.full(n_steps, y_slice),
        t_vals
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_vals, pred[field], lw=2)
    ax.set_xlabel("t"); ax.set_ylabel(field)
    ax.set_title(f"PINN field evolution at (x={x_slice:.2f}, y={y_slice:.2f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
def save_checkpoint(model, norm_params, field_names, dim, path: str):
    torch.save({
        "model_state":  model.state_dict(),
        "norm_params":  norm_params,
        "field_names":  field_names,
        "dim":          dim,
        "model_kwargs": {
            "field_names": field_names,
            "input_dim":   model.input_dim,
            "hidden_dim":  model.net[0].out_features,
            "n_layers":    sum(1 for m in model.net if isinstance(m, nn.Linear)) - 1,
            "n_freqs":     model.embed.B.shape[1],
            "fourier_scale": 1.0,
        }
    }, path)


def load_checkpoint(path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MaxwellPINN(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    dim = ckpt.get("dim", 2)
    return MaxwellSurrogate(model, ckpt["norm_params"], ckpt["field_names"], dim)
def run_synthetic_demo(epochs: int = 3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kx, omega = 2 * math.pi, 2 * math.pi
    n_x, n_y, n_t = 40, 40, 20
    x_vals = np.linspace(0, 1, n_x)
    y_vals = np.linspace(0, 1, n_y)
    t_vals = np.linspace(0, 2, n_t)

    xx, yy, tt = np.meshgrid(x_vals, y_vals, t_vals, indexing="ij")
    Ez = np.sin(kx * xx - omega * tt)
    Bx = np.zeros_like(Ez)
    By = np.sin(kx * xx - omega * tt)

    coords = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
    fields = np.column_stack([Ez.ravel(), Bx.ravel(), By.ravel()])

    c_mean = coords.mean(0); c_std = coords.std(0) + 1e-8
    f_mean = fields.mean(0); f_std = fields.std(0) + 1e-8
    norm_params = dict(
        x_mean=c_mean[0], x_std=c_std[0],
        y_mean=c_mean[1], y_std=c_std[1],
        t_mean=c_mean[2], t_std=c_std[2],
        f_mean=f_mean,    f_std=f_std,
    )
    coords_n = (coords - c_mean) / c_std
    fields_n = (fields - f_mean) / f_std

    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)

    rng  = np.random.default_rng(0)
    col  = rng.uniform([coords_n.min(0)], [coords_n.max(0)], (10_000, 3)).squeeze()
    field_names = ["Ez", "Bx", "By"]

    model   = MaxwellPINN(field_names, input_dim=3, hidden_dim=128, n_layers=4).to(device)
    trainer = PINNTrainer(
        model, dim=2, mode="TM",
        coords_data=to(coords_n), 
        fields_data=to(fields_n), 
        coords_phys=to(col),
        norm_params=norm_params,
        lambda_data=1.0, lambda_phys=0.05,
        lr=1e-3, batch_size=2048,
    )
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=epochs, eta_min=1e-5)
    trainer.train(epochs=epochs, log_every=500, scheduler=scheduler)

    surrogate = MaxwellSurrogate(model, norm_params, field_names, dim=2)

    x_test = np.linspace(0, 1, 100)
    t_test = np.full(100, 1.0)
    y_test = np.full(100, 0.5)
    pred   = surrogate.predict(x_test, y_test, t_test)
    true_Ez = np.sin(kx * x_test - omega * t_test)
    rel_err = np.abs(pred["Ez"] - true_Ez).mean() / (np.abs(true_Ez).mean() + 1e-8)
    print(f"Relative L1 error: {rel_err*100:.2f}%")

    save_checkpoint(model, norm_params, field_names, dim=2, path="maxwell_pinn_demo.pt")
    plot_loss_history(trainer.history, save_path="loss_history.png")
    plot_field_evolution(
        surrogate, x_slice=0.5, y_slice=0.5,
        t_range=(0, 2), field="Ez",
        save_path="field_evolution.png"
    )

    return surrogate, trainer
def main():
    parser = argparse.ArgumentParser(description="Maxwell PINN for Smilei data")
    parser.add_argument("--hdf5",        type=str,   default=None,
                        help="Path to Smilei Fields*.h5 file")
    parser.add_argument("--dim",         type=int,   default=2,
                        choices=[2, 3], help="Simulation dimensionality (2 or 3)")
    parser.add_argument("--mode",        type=str,   default="TM",
                        choices=["TM", "TE"], help="2D polarisation mode (ignored for 3D)")
    parser.add_argument("--epochs",      type=int,   default=20_000)
    parser.add_argument("--hidden",      type=int,   default=None,
                        help="Hidden dim (default: 256 for 2D, 512 for 3D)")
    parser.add_argument("--layers",      type=int,   default=None,
                        help="Number of layers (default: 6 for 2D, 8 for 3D)")
    parser.add_argument("--n_freqs",     type=int,   default=128,
                        help="Number of Fourier frequencies")
    parser.add_argument("--fourier_scale", type=float, default=10.0,
                        help="Fourier feature scale")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lambda_phys", type=float, default=None,
                        help="Physics loss weight (default: 0.1 for 2D, 0.5 for 3D)")
    parser.add_argument("--batch",       type=int,   default=4096)
    parser.add_argument("--colloc",      type=int,   default=None,
                        help="Collocation points (default: 20k for 2D, 100k for 3D)")
    parser.add_argument("--snapshots",   type=int,   default=50,
                        help="Max HDF5 snapshots to load")
    parser.add_argument("--save",        type=str,   default="maxwell_pinn.pt",
                        help="Path to save model checkpoint")
    parser.add_argument("--demo",        action="store_true",
                        help="Run synthetic 2D demo (no HDF5 required)")
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Disable adaptive physics loss weighting")
    parser.add_argument("--warmup",      type=int,   default=5000,
                        help="Warmup epochs for adaptive weighting")
    args = parser.parse_args()

    if args.demo or args.hdf5 is None:
        run_synthetic_demo(epochs=args.epochs)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = args.hidden if args.hidden is not None else (512 if args.dim == 3 else 256)
    layers = args.layers if args.layers is not None else (8 if args.dim == 3 else 6)
    lambda_phys = args.lambda_phys if args.lambda_phys is not None else (0.5 if args.dim == 3 else 0.1)
    colloc = args.colloc if args.colloc is not None else (100_000 if args.dim == 3 else 20_000)

    loader = SmileiLoader(
        args.hdf5, 
        dim=args.dim,
        mode=args.mode if args.dim == 2 else "FULL",
        max_snapshots=args.snapshots
    ).load()

    coords_data, fields_data, coords_phys, norm_params, field_names = \
        loader.get_training_tensors(n_collocation=colloc)

    input_dim = 3 if args.dim == 2 else 4
    model = MaxwellPINN(
        field_names,
        input_dim=input_dim,
        hidden_dim=hidden,
        n_layers=layers,
        n_freqs=args.n_freqs,
        fourier_scale=args.fourier_scale,
    ).to(device)

    trainer = PINNTrainer(
        model, 
        dim=args.dim,
        mode=args.mode if args.dim == 2 else "FULL",
        coords_data=coords_data, 
        fields_data=fields_data, 
        coords_phys=coords_phys,
        norm_params=norm_params,
        lambda_data=1.0,
        lambda_phys=lambda_phys,
        lr=args.lr,
        batch_size=args.batch,
        adaptive_weights=not args.no_adaptive,
    )
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-5)
    trainer.train(epochs=args.epochs, log_every=1000, scheduler=scheduler, warmup_epochs=args.warmup)

    surrogate = MaxwellSurrogate(model, norm_params, field_names, dim=args.dim)
    original_bytes = Path(args.hdf5).stat().st_size
    cr = surrogate.compression_ratio(original_bytes)
    print(f"Compression ratio: {cr:.1f}×")

    save_checkpoint(model, norm_params, field_names, dim=args.dim, path=args.save)

    plot_loss_history(trainer.history)
    if args.dim == 2:
        plot_field_comparison(surrogate, loader, snapshot_idx=0,
                              field=field_names[0])


if __name__ == "__main__":
    main()