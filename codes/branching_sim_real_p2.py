import numpy as np
import pandas as pd
import scipy.stats as SSA
import h5py, gzip, os, sys

def superspreading_T_Loc_mobility_varrying(T, Initials_inf, pop, paras, WM_t, rand_seed, r_t, Ptx_t):
    """
    T: int total days
    Initials_inf: (N, T0) seeds
    pop: (N,)
    paras: (Z, Zb, D, Db)
    WM_t: list/array of length T, each (N, N) mobility probability matrix (row = dest, col = origin)
    rand_seed: np.random.SeedSequence
    r_t: (T,) piecewise-constant dispersion parameter over time
    Ptx_t: (T, N) daily p parameters for NB at each node
    """
    Z, Zb, D, Db = paras
    child_seeds = rand_seed.spawn(T)
    num_fips = len(pop)
    NewInf = np.zeros((num_fips, T), dtype=np.int64)
    TotInf = np.zeros((num_fips, T), dtype=np.int64)
    initial_days = Initials_inf.shape[1] #Initials_inf.shape[1]
    NewInf[:, :initial_days] = Initials_inf
    TotInf[:, :initial_days] = Initials_inf

    for ti in range(T):
        WM = WM_t[ti]                 # (N,N)
        infectors = np.int64(NewInf[:, ti])
        pop_immu = 1.0 - TotInf[:, ti] / pop
        pop_immu[pop_immu < 0] = 0.0
        rng = np.random.default_rng(child_seeds[ti])

        r_now = float(r_t[ti])
        P_t = Ptx_t[ti, :]            # (N,)

        # Draw secondary counts per infector at its origin node
        total_new = []
        new_locs = []
        for loc_idx, size in enumerate(infectors):
            if size <= 0:
                continue
            sam_i = rng.negative_binomial(n=r_now, p=P_t[loc_idx], size=size)
            real_new = np.int64(np.round(sam_i * pop_immu[loc_idx]))
            if real_new.sum() == 0:
                continue
            total_new.extend(real_new)
            new_locs.extend([loc_idx] * int(real_new.sum()))

        z_num = int(np.sum(total_new))
        if z_num == 0:
            # still need to carry forward totals
            TotInf[:, ti] = (TotInf[:, ti - 1] if ti > 0 else 0) + NewInf[:, ti]
            continue

        # Infection delays
        latency = SSA.gamma.rvs(a=Z, scale=Zb, size=z_num, random_state=rng)
        infectious = SSA.gamma.rvs(a=D, scale=Db, size=z_num, random_state=rng)
        delay_days = latency + rng.random(z_num) * infectious
        future_times = np.ceil(delay_days + ti).astype(np.int64)

        # Assign destinations via WM (columns are origins)
        o_cols = np.array(new_locs, dtype=np.int64)
        valid = future_times <= (T - 1)
        if not np.any(valid):
            TotInf[:, ti] = (TotInf[:, ti - 1] if ti > 0 else 0) + NewInf[:, ti]
            continue

        ft = future_times[valid]
        oc = o_cols[valid]
        # For each event, draw destination according to column oc
        dests = np.fromiter(
            (rng.choice(np.arange(num_fips), p=WM[:, oc[k]]) for k in range(len(oc))),
            dtype=np.int64,
            count=len(oc)
        )
        # Accumulate arrivals
        np.add.at(NewInf, (dests, ft), 1)
        TotInf[:, ti] = (TotInf[:, ti - 1] if ti > 0 else 0) + NewInf[:, ti]

    return NewInf, TotInf


def main():
    # args: s (index into r_list), es_idx (ensemble index)
    s = int(sys.argv[1])
    es_idx = int(sys.argv[2])
    T = 56 #56  #Rtx.shape[0]

    # Load time-varying mobility, population, Rt, seeds
    with h5py.File('NMM_t_3142.hdf5', 'r') as f:
        WM_t = f['WM'][:T]  # shape (T, N, N), starting 2020-02-23
    pop = np.loadtxt('pop_new.csv')
    Rtx = np.loadtxt('Rt_real.csv')
    Rtx = Rtx[:T]             # expect shape (T, N)
    Ini_seed = np.loadtxt('seed_real.csv', delimiter=',')

    # Horizon
    N = pop.shape[0]
    assert Rtx.shape == (T, N), "Rt_real.csv must be T x N"
    assert WM_t.shape[0] == T and WM_t.shape[1] == N and WM_t.shape[2] == N, "NMM_t_3142.hdf5[WM] must be T x N x N"

    # Parameters
    r_list = np.array([20, 10, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 5., 2.5, 13.333, 3.333, 1.333, 0.667, 0.286, 0.133, 0.067, 0.033, 0.37, 7.4])
    r_pre = 0.05
    r_post = float(r_list[s])

    # Build r_t: first 21 days r=0.02, then r=r_post
    r_t = np.full(T, r_post, dtype=float)
    pre_days = min(21, T)
    r_t[:pre_days] = r_pre

    # Compute daily per-node NB "p" parameters: P_t = r_t/(Rt + r_t)
    # broadcast r_t (T,) to (T,N)
    Ptx_t = (r_t[:, None]) / (Rtx + (r_t[:, None]))

    # Pathogen characteristics
    Z, Zb, D, Db = 3.0, 1.0, 5.0, 1.0

    # Seeds
    ss = np.random.SeedSequence(es_idx)

    # Run one ensemble member
    E_NewInf, E_TotInf = superspreading_T_Loc_mobility_varrying(
        T=T,
        Initials_inf=Ini_seed,
        pop=pop,
        paras=(Z, Zb, D, Db),
        WM_t=WM_t,
        rand_seed=ss,
        r_t=r_t,
        Ptx_t=Ptx_t
    )

    # Save
    save_dir = f'branching_pre{np.round(r_pre,3)}_post-{np.round(r_post,3)}/'
    os.makedirs(save_dir, exist_ok=True)
    with gzip.GzipFile(save_dir + f"NewInf_post-{np.round(r_post,3)}_{es_idx}.npy.gz", "w") as f:
        np.save(file=f, arr=E_NewInf)


if __name__ == "__main__":
    main()
