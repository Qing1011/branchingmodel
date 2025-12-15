# minimal_create_timevarying.py
import os, gzip, numpy as np, torch, h5py
from torch_geometric.data import Data

def load_gz(path):
    with gzip.open(path, 'rb') as f:
        return np.load(f)

def build_dataset(export_dir, r_list, W_path, t0=10, T=50, g_max=300, round_r=3):
    # 1) load mobility window once (dense): A -> (T, N, N)
    with h5py.File(W_path, 'r') as f:
        A = f['WM'][t0:t0+T]              # (T,N,N) numpy array
    A_t = torch.from_numpy(A.astype(np.float32, copy=False)).contiguous()

    dataset = []
    for r in r_list:
        r_tag = f"{np.round(r, round_r)}"
        subdir = os.path.join(export_dir, f"branching_r-{r_tag}")
        for g in range(g_max):
            # fn = os.path.join(subdir, f"NewInf_post-{r_tag}_{g}.npy.gz")
            fn = os.path.join(subdir, f"NewInf_r-{r_tag}_{g}.npy.gz")
            if not os.path.exists(fn):  # minimal skip
                continue
            gi = load_gz(fn)                                # (N, T_total)
            x = torch.from_numpy(gi[:, t0:t0+T].astype(np.float32, copy=False)).contiguous()  # (N,T)
            y = torch.log(torch.tensor([float(r)], dtype=torch.float32))  # (1,)
            dataset.append(Data(x=x, y=y))
    return dataset, A_t

if __name__ == "__main__":
    # example usage
    ds, A_t = build_dataset(
        export_dir="./",
        r_list=[10.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 5.,  2.5  , 3.333,  1.333,  0.667,  0.286,  0.133,
        0.067,  0.033, 0.37,7.4,20,13.333],
        W_path="NMM_t_3142.hdf5",
        t0=0, T=21, g_max=300
    )
    # Save A_t ONCE, separately
    torch.save(A_t, "./training/A_t_1.pt", _use_new_zipfile_serialization=False)
    
     # Save dataset WITHOUT A_t; use legacy serializer for NFS robustness
    tmp = "./training/.simulation_real_p1.tmp"
    out = "./training/simulation_real_p1.pt"
    torch.save(ds, tmp, _use_new_zipfile_serialization=False)
    os.replace(tmp, out)
    print("saved", len(ds), "samples and A_t once")
