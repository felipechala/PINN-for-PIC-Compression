import h5py
with h5py.File('Fields0.h5', 'r') as f:
    print(f['0']['Ex'].shape)