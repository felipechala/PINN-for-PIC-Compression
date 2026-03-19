import h5py

with h5py.File('Fields0.h5', 'r') as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}")
        else:
            print(f"{name}/ (group)")
    
    f.visititems(show)
    print("\nTop-level keys:", list(f.keys()))