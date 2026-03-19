import numpy as np
import matplotlib.pyplot as plt
from pinn import load_checkpoint
import h5py

surrogate = load_checkpoint('results/pinn_models/fields0_stable.pt')

with h5py.File('Fields0.h5', 'r') as f:
    root = f['data']
    
    snapshot_keys = sorted([k for k in root.keys() if k.isdigit()], key=int)
    
    snapshot_idx = len(snapshot_keys) // 2
    snapshot_key = snapshot_keys[snapshot_idx]
    print(f"Using snapshot: {snapshot_key}")
    
    Ex_true = np.array(root[f'{snapshot_key}/Ex'])
    t_val = float(snapshot_key)

print(f"HDF5 shape: {Ex_true.shape}")

nx, ny, nz = Ex_true.shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny) 
z = np.linspace(0, 1, nz)

slice_idx = nz // 2
xx, yy = np.meshgrid(x, y, indexing='ij')

pred = surrogate.predict(
    xx.ravel(), 
    yy.ravel(), 
    np.full(xx.size, t_val),
    z=np.full(xx.size, z[slice_idx])
)

Ex_pred = pred['Ex'].reshape(nx, ny)
Ex_slice = Ex_true[:, :, slice_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin, vmax = Ex_slice.min(), Ex_slice.max()

im0 = axes[0].imshow(Ex_slice.T, vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[0].set_title('Ground Truth')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(Ex_pred.T, vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[1].set_title('PINN')
plt.colorbar(im1, ax=axes[1])

err = np.abs(Ex_slice - Ex_pred)
im2 = axes[2].imshow(err.T, cmap='hot')
axes[2].set_title(f'Error (max={err.max():.2e})')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)

rel_err = np.linalg.norm(Ex_slice - Ex_pred) / np.linalg.norm(Ex_slice) * 100
print(f"Relative error: {rel_err:.1f}%")