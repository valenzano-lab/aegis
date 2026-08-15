"""Pre-evolve one population, subsample to a range of Ne, run each forward at CONSTANT K=N.

Dario's design: a single burn-in gives every Ne treatment an IDENTICAL genetic ancestor
(same standing variation, same evolved life history). Subsampling to N and holding K=N
sets the ongoing Ne. Forward divergence in lifespan is then attributable to Ne alone --
no independent trajectories, no oscillation/extrinsic-mortality confound (birth-regulated,
no starvation). At the end of each run we RE-MEASURE Ne (it drifts from the subsample size)
and relate realized Ne -> evolved lifespan.

Everything under __main__ guard: aegis_sim.sim() starts the ticker via multiprocessing
(spawn on macOS), which re-imports this module; without the guard the child would re-run
the pipeline and rmtree outputs.

Usage: preevolve.py K_BURN BURN_STEPS FWD_STEPS N1 N2 N3 ...
"""
import sys, pathlib, yaml
import numpy as np
sys.path.insert(0, "/Users/dvalenzano/Dropbox/Lab/git/projects/aegis/runs")
from ne_ma_ap_configs import build

HERE = pathlib.Path(__file__).parent
ARCH = dict(AGE_LIMIT=30, MATURATION_AGE=6, MAX_OFFSPRING_NUMBER=3,
            STARVATION_PENALTY=0.0, REPRODUCTION_REGULATION=True,
            POPGENSTATS_SAMPLE_SIZE=100)


def write_cfg(path, K, steps, **extra):
    cfg = build(arm="MA", ne=K, mode="asexual", seed=1, steps=steps)
    cfg.update(ARCH)
    cfg.update(dict(INITIAL_POPULATION_SIZE=K, RESOURCE_MAXIMUM_AMOUNT=K,
                    RESOURCE_ADDITIVE_GROWTH=K, STEPS_PER_SIMULATION=steps,
                    POPGENSTATS_RATE=max(1000, steps // 20), SNAPSHOT_RATE=steps,
                    PICKLE_RATE=steps))
    cfg.update(extra)
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=True)
    return cfg


def main():
    K_BURN, BURN, FWD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    TARGETS = [int(x) for x in sys.argv[4:]]
    assert max(TARGETS) <= K_BURN, "cannot subsample above the burn-in size"

    import aegis_sim
    from aegis_sim.dataclasses.population import Population

    # 1. BURN-IN --------------------------------------------------------------
    burn_yml = HERE / "burn.yml"
    write_cfg(burn_yml, K=K_BURN, steps=BURN)
    print(f"[burn-in] K={K_BURN} steps={BURN}")
    import shutil
    if (HERE / "burn").exists():
        shutil.rmtree(HERE / "burn")
    aegis_sim.run(custom_config_path=burn_yml, pickle_path=None, overwrite=True, custom_input_params={})
    burn_pickle = HERE / "burn" / "pickles" / str(BURN)
    assert burn_pickle.exists(), f"no burn-in pickle at {burn_pickle}"

    # 2. SUBSAMPLE (identical ancestor) --------------------------------------
    n_burn = len(Population.load_pickle_from(burn_pickle))
    print(f"[subsample] burn-in final population = {n_burn} individuals")
    rng = np.random.default_rng(0)
    subpaths = {}
    for N in TARGETS:
        assert N <= n_burn, f"burn-in only has {n_burn}, cannot draw {N}"
        pop = Population.load_pickle_from(burn_pickle)   # fresh copy per N
        idx = rng.choice(n_burn, size=N, replace=False)
        pop *= idx                                       # in-place subset (uses phenotypes.keep)
        assert len(pop) == N
        sp = HERE / f"sub_{N}.pkl"
        pop.save_pickle_to(sp)
        subpaths[N] = sp
        print(f"   subsampled {N}")

    # 3. FORWARD at constant K=N ---------------------------------------------
    for N in TARGETS:
        fwd_yml = HERE / f"fwd{N}.yml"
        write_cfg(fwd_yml, K=N, steps=FWD)
        if (HERE / f"fwd{N}").exists():
            shutil.rmtree(HERE / f"fwd{N}")
        print(f"[forward] N={N} K={N} steps={FWD} (from common ancestor)")
        aegis_sim.run(custom_config_path=fwd_yml, pickle_path=subpaths[N],
                      overwrite=True, custom_input_params={})
    print("ALL DONE ->", " ".join(f"fwd{N}" for N in TARGETS), "(+ burn)")


if __name__ == "__main__":
    main()
