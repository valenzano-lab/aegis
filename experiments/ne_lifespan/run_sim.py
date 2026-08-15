"""Driver: build a config dict, write yaml, run the aegis_sim engine (no GUI).
Usage: run_sim.py <out.yml> key=val key=val ...  (vals are eval'd as python literals)
Builds on ne_ma_ap_configs.build (MA, asexual) then applies overrides.

NOTE: everything runs under __main__ guard. aegis_sim.sim() starts the ticker as a
multiprocessing subprocess; on macOS (spawn) the child re-imports this module, and
without the guard it would re-run aegis_sim.run(overwrite=True) -> rmtree the output
dir out from under the parent.
"""
import sys, ast, pathlib, yaml
sys.path.insert(0, "/Users/dvalenzano/Dropbox/Lab/git/projects/aegis/runs")
from ne_ma_ap_configs import build


def main():
    out = pathlib.Path(sys.argv[1]).absolute()
    cfg = build(arm="MA", ne=100, mode="asexual", seed=1, steps=3000)
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    print("config:", out)
    print("regime:", {k: cfg.get(k) for k in
          ["RESOURCE_MAXIMUM_AMOUNT", "STARVATION_PENALTY", "REPRODUCTION_REGULATION",
           "FRAILTY_MODIFIER", "STEPS_PER_SIMULATION", "AGE_LIMIT", "POPGENSTATS_RATE"]})

    import aegis_sim
    aegis_sim.run(custom_config_path=out, pickle_path=None, overwrite=True, custom_input_params={})
    print("DONE:", out.parent / out.stem)


if __name__ == "__main__":
    main()
